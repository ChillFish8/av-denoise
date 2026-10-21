use cubecl::server::Handle;

use super::engine::{noisy_frames, run_engine_on_planes, temporal_fast};
use super::fixtures::{WindowSpec, code_format, quantised_codes, upload_codes, window_site};
use super::runtime::{TestRuntime, test_client};
use crate::Nlmeans;
use crate::engine::{Geometry, SampleFormat};
use crate::nlmeans::{ChannelMode, MotionCompensationMode, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::gpu::{read_f32, read_u16, sync};
use crate::tune::sites::{
    AccumOutputs,
    NLM_WINDOW,
    SEPARABLE_MAX_ABS,
    SEPARABLE_MIN_PSNR,
    WindowCandidate,
    WindowPass,
    WindowSite,
    estimate_planes,
};
use crate::tune::{
    EngineKernels,
    Fidelity,
    IntegerCodes,
    KernelPlan,
    PipelineProfile,
    PlaneDiff,
    SiteCandidate,
    VariantSite,
    diff_plane,
};

const EIGHT_BIT: IntegerCodes = IntegerCodes { depth: 8 };
const TEN_BIT: IntegerCodes = IntegerCodes { depth: 10 };

/// The largest gap any normalised estimate may show against the host evaluation.
const HOST_TOLERANCE: f64 = 1.0e-5;

const ENGINE_WIDTH: u32 = 96;
const ENGINE_HEIGHT: u32 = 64;
const ENGINE_FRAMES: usize = 6;

/// Single and pair windows, the pair with and without confidence, in every channel mode and at
/// both code depths.
fn integer_window_specs(width: u32, height: u32) -> Vec<WindowSpec> {
    let mut specs = Vec::new();

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        for codes in [EIGHT_BIT, TEN_BIT] {
            for (pair, confidence) in [(false, false), (true, false), (true, true)] {
                let mut spec = WindowSpec::single(width, height, channels);
                spec.pair = pair;
                spec.confidence = confidence;
                spec.codes = Some(codes);
                specs.push(spec);
            }
        }
    }

    for pair in [false, true] {
        let mut spec = WindowSpec::single(width, height, ChannelMode::Luma);
        spec.patch_radius = 4;
        spec.search_radius = 2;
        spec.pair = pair;
        spec.confidence = pair;
        spec.codes = Some(TEN_BIT);
        specs.push(spec);
    }

    specs
}

/// Runs `candidate` into fresh outputs.
fn run_candidate(site: &WindowSite<TestRuntime>, candidate: WindowCandidate) -> AccumOutputs {
    let outputs = site.scratch_outputs();
    site.launch(candidate, &outputs);

    outputs
}

/// Reads `outputs` back as normalised estimate planes, one per stored lane and then the largest weight.
fn gpu_estimates(site: &WindowSite<TestRuntime>, outputs: &AccumOutputs) -> Vec<Vec<f32>> {
    let client = &site.client;
    let accum = read_f32(client, &outputs.accum, outputs.accum_len).expect("accum reads back");
    let weight_sum = read_f32(client, &outputs.weight_sum, outputs.pixels).expect("weights read back");
    let max_weight = read_f32(client, &outputs.max_weight, outputs.pixels).expect("maxima read back");

    estimate_planes(&accum, &weight_sum, &max_weight, site.stored_ch)
}

/// Everything a host evaluation of a window site reads, copied back from the device.
struct HostWindow {
    width: i32,
    height: i32,
    channels: usize,
    stored_ch: usize,
    patch_radius: i32,
    search_radius: i32,
    frame_t: usize,
    input: Vec<f32>,
    codes: Vec<u16>,
    inv_scale2: f64,
    h2_inv_norm: f64,
}

impl HostWindow {
    fn read(site: &WindowSite<TestRuntime>) -> Self {
        let window_codes = site.codes.as_ref().expect("an integer window site has codes");
        let input = read_f32(&site.client, &site.input, site.ring_len).expect("the input reads back");
        let codes = read_u16(&site.client, &window_codes.ring, site.ring_len).expect("the codes read back");

        Self {
            width: site.width as i32,
            height: site.height as i32,
            channels: site.channels as usize,
            stored_ch: site.stored_ch as usize,
            patch_radius: site.patch_radius as i32,
            search_radius: site.search_radius as i32,
            frame_t: site.frame_t as usize,
            input,
            codes,
            inv_scale2: f64::from(window_codes.inv_scale2),
            h2_inv_norm: f64::from(site.h2_inv_norm),
        }
    }

    /// The first stored lane of the pixel at `(x, y)` in `frame`, clamped to the image edges.
    fn lane_start(&self, frame: usize, x: i32, y: i32) -> usize {
        let clamped_x = x.clamp(0, self.width - 1) as usize;
        let clamped_y = y.clamp(0, self.height - 1) as usize;
        let row = frame * self.height as usize + clamped_y;
        let pixel = row * self.width as usize + clamped_x;

        pixel * self.stored_ch
    }

    /// The exact SSD between the patch at `(x, y)` in the centre frame and the one shifted by `q` in
    /// `frame_q`, each read clamped.
    fn ssd(&self, frame_q: usize, x: i32, y: i32, q: (i32, i32)) -> i64 {
        let (q_x, q_y) = q;
        let radius = self.patch_radius;
        let mut sum = 0i64;

        for offset_y in -radius..=radius {
            for offset_x in -radius..=radius {
                let tap_x = x + offset_x;
                let tap_y = y + offset_y;
                let centre = self.lane_start(self.frame_t, tap_x, tap_y);
                let neighbour = self.lane_start(frame_q, tap_x + q_x, tap_y + q_y);

                for lane in 0..self.channels {
                    let centre_code = i64::from(self.codes[centre + lane]);
                    let neighbour_code = i64::from(self.codes[neighbour + lane]);
                    let diff = centre_code - neighbour_code;
                    sum += diff * diff;
                }
            }
        }

        sum
    }

    fn weight(&self, ssd: i64, offset: f64) -> f64 {
        let distance = ssd as f64 * self.inv_scale2;
        let excess = (distance - offset).max(0.0);

        (-excess * self.h2_inv_norm).exp()
    }
}

/// One pixel's running sums in f64.
struct HostSums {
    accum: Vec<f64>,
    weight_sum: f64,
    max_weight: f64,
}

impl HostSums {
    fn new(stored_ch: usize) -> Self {
        Self {
            accum: vec![0.0; stored_ch],
            weight_sum: 0.0,
            max_weight: 0.0,
        }
    }

    fn add(&mut self, window: &HostWindow, lane_start: usize, weight: f64) {
        for (lane, sum) in self.accum.iter_mut().enumerate() {
            let value = f64::from(window.input[lane_start + lane]);
            *sum += value * weight;
        }

        self.weight_sum += weight;
        self.max_weight = self.max_weight.max(weight);
    }
}

/// The normalised estimates of `site`, evaluated in plain f64 Rust from exact integer SSDs.
///
/// The planes follow [estimate_planes], one per stored lane and then the largest weight.
fn host_window_estimates(site: &WindowSite<TestRuntime>) -> Vec<Vec<f64>> {
    let window = HostWindow::read(site);
    let side = 2 * window.search_radius + 1;
    let radius = window.search_radius;
    let pixels = (window.width * window.height) as usize;
    let mut planes: Vec<Vec<f64>> = (0..=window.stored_ch)
        .map(|_| Vec::with_capacity(pixels))
        .collect();

    let single_offsets = match &site.pass {
        WindowPass::Single { offsets, offsets_len } => {
            let values = read_f32(&site.client, offsets, *offsets_len).expect("the offsets read back");
            Some(values)
        },
        WindowPass::Pair { .. } => None,
    };
    let pair_confidence = match &site.pass {
        WindowPass::Pair { confidence, .. } if confidence.enabled => {
            let forward =
                read_f32(&site.client, &confidence.forward, confidence.len).expect("forward reads back");
            let backward =
                read_f32(&site.client, &confidence.backward, confidence.len).expect("backward reads back");
            Some((
                forward,
                backward,
                confidence.step,
                confidence.blocks_x,
                confidence.blocks_y,
            ))
        },
        _ => None,
    };

    for y in 0..window.height {
        for x in 0..window.width {
            let mut sums = HostSums::new(window.stored_ch);

            for q_y in -radius..=radius {
                for q_x in -radius..=radius {
                    match &site.pass {
                        WindowPass::Single { .. } => {
                            if q_x == 0 && q_y == 0 {
                                continue;
                            }

                            let offsets = single_offsets.as_ref().expect("a single pass has offsets");
                            let lut_index = ((q_y + radius) * side + (q_x + radius)) as usize;
                            let offset = f64::from(offsets[lut_index]);
                            let ssd = window.ssd(window.frame_t, x, y, (q_x, q_y));
                            let weight = window.weight(ssd, offset);
                            let lane_start = window.lane_start(window.frame_t, x + q_x, y + q_y);
                            sums.add(&window, lane_start, weight);
                        },
                        WindowPass::Pair {
                            frame_fwd,
                            frame_bwd,
                            noise_offset,
                            ..
                        } => {
                            let forward_frame = *frame_fwd as usize;
                            let backward_frame = *frame_bwd as usize;
                            let offset = f64::from(*noise_offset);
                            let forward_ssd = window.ssd(forward_frame, x, y, (q_x, q_y));
                            let backward_ssd = window.ssd(backward_frame, x, y, (-q_x, -q_y));
                            let mut forward_weight = window.weight(forward_ssd, offset);
                            let mut backward_weight = window.weight(backward_ssd, offset);

                            if let Some((forward, backward, step, blocks_x, blocks_y)) = &pair_confidence {
                                let block_col = (x as u32 / step).min(blocks_x - 1);
                                let block_row = (y as u32 / step).min(blocks_y - 1);
                                let block = (block_row * blocks_x + block_col) as usize;
                                forward_weight *= f64::from(forward[block]);
                                backward_weight *= f64::from(backward[block]);
                            }

                            let forward_start = window.lane_start(forward_frame, x + q_x, y + q_y);
                            let backward_start = window.lane_start(backward_frame, x - q_x, y - q_y);
                            sums.add(&window, forward_start, forward_weight);
                            sums.add(&window, backward_start, backward_weight);
                        },
                    }
                }
            }

            let normalise = |value: f64| {
                if sums.weight_sum == 0.0 {
                    return 0.0;
                }

                value / sums.weight_sum
            };

            for (lane, &sum) in sums.accum.iter().enumerate() {
                let estimate = normalise(sum);
                planes[lane].push(estimate);
            }

            let largest = normalise(sums.max_weight);
            planes[window.stored_ch].push(largest);
        }
    }

    planes
}

/// Checks every GPU estimate lies within [HOST_TOLERANCE] of the host evaluation.
fn assert_matches_host(gpu: &[Vec<f32>], host: &[Vec<f64>], label: &str) {
    assert_eq!(gpu.len(), host.len(), "{label}: plane counts differ");

    for (plane, (gpu_plane, host_plane)) in gpu.iter().zip(host).enumerate() {
        let worst = gpu_plane
            .iter()
            .zip(host_plane)
            .map(|(&gpu_value, &host_value)| (f64::from(gpu_value) - host_value).abs())
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let (pixel, gap) = worst.expect("a plane holds pixels");

        assert!(
            gap <= HOST_TOLERANCE,
            "{label}: plane {plane} pixel {pixel} is {gap} from the host evaluation"
        );
    }
}

fn assert_scalar_int_matches_host(specs: &[WindowSpec]) {
    let client = test_client();

    for spec in specs {
        let site = window_site(&client, spec);
        let label = format!("{spec:?}");
        let outputs = run_candidate(&site, WindowCandidate::ScalarInt);
        let gpu = gpu_estimates(&site, &outputs);
        let host = host_window_estimates(&site);

        let weights = &gpu[site.stored_ch as usize];
        let has_signal = weights.iter().any(|&weight| weight > 0.0);
        assert!(has_signal, "{label}: the integer window accumulated nothing");

        assert_matches_host(&gpu, &host, &label);
    }
}

#[test]
fn scalar_int_windows_match_the_host_reference() {
    let specs = integer_window_specs(40, 24);
    assert_scalar_int_matches_host(&specs);
}

#[test]
fn scalar_int_windows_stay_within_the_window_bound_of_the_baseline() {
    let client = test_client();
    let fidelity = WindowCandidate::ScalarInt.fidelity();
    let mut specs = integer_window_specs(40, 24);
    let odd_specs = integer_window_specs(75, 53);
    specs.extend(odd_specs);

    for spec in &specs {
        let site = window_site(&client, spec);
        let label = format!("{spec:?}");
        let baseline_candidate = site.baseline();
        let integer_reference = site.integer_reference();
        assert_eq!(integer_reference, Some(WindowCandidate::ScalarInt), "{label}");

        let baseline = run_candidate(&site, baseline_candidate);
        let scalar_int = run_candidate(&site, WindowCandidate::ScalarInt);
        let verdict = site
            .compare(&baseline, &scalar_int, fidelity)
            .expect("the outputs read back");

        assert!(verdict.passed(), "{label}: {verdict:?}");
    }
}

#[test]
fn scalar_int_windows_handle_odd_frames_and_edges() {
    let specs = integer_window_specs(75, 53);
    assert_scalar_int_matches_host(&specs);
}

/// A motion-compensated temporal NLMeans run, whose temporal windows read the compensated rings.
fn motion_compensated_fast() -> NlmeansAlgorithm {
    let NlmeansAlgorithm::Fast(fast) = temporal_fast() else {
        unreachable!("temporal_fast builds the fast variant");
    };
    let options = NlmeansOptions {
        motion_compensation: MotionCompensationMode::mvtools_default(),
        ..fast
    };

    NlmeansAlgorithm::Fast(options)
}

/// Frames of code planes for `channels`, rounded from noisy frames to `codes`.
fn code_frames(channels: ChannelMode, codes: IntegerCodes) -> Vec<Vec<Vec<u16>>> {
    let planes = channels.count() as usize;
    let frames = noisy_frames(ENGINE_WIDTH, ENGINE_HEIGHT, planes, ENGINE_FRAMES);

    frames
        .iter()
        .map(|frame| frame.iter().map(|plane| quantised_codes(plane, codes)).collect())
        .collect()
}

/// Every frame an NLMeans engine under the integer search and `plan` emits for code frames.
///
/// The engine must keep a code ring, or the integer windows went untested.
fn integer_nlmeans_outputs(
    algorithm: NlmeansAlgorithm,
    channels: ChannelMode,
    codes: IntegerCodes,
    frames: &[Vec<Vec<u16>>],
    plan: KernelPlan,
) -> Vec<Vec<f32>> {
    let client = test_client();
    let geometry = Geometry {
        width: ENGINE_WIDTH,
        height: ENGINE_HEIGHT,
        channels,
        input: code_format(codes),
        output: SampleFormat::F32,
    };
    let kernels = EngineKernels {
        profile: PipelineProfile::IntegerSearch,
        plan,
        ..EngineKernels::default()
    };
    let mut engine = Nlmeans::with_kernels(&client, algorithm, geometry, kernels).expect("the engine builds");
    let kept = engine.front_for_test().code_ring().map(|(_, kept)| kept);
    assert_eq!(kept, Some(codes), "the engine keeps the codes");

    let mut uploaded = Vec::with_capacity(frames.len());
    for frame in frames {
        let handles: Vec<Handle> = frame
            .iter()
            .map(|plane| upload_codes(&client, plane, codes))
            .collect();
        uploaded.push(handles);
    }

    run_engine_on_planes(&client, &mut engine, ENGINE_WIDTH, ENGINE_HEIGHT, &uploaded)
        .expect("the frames denoise")
}

/// Whether every plane of every frame of `forced` lies within the window bound of `baseline`, and
/// whether any frame differs at all.
fn within_window_bound(forced: &[Vec<f32>], baseline: &[Vec<f32>], label: &str) -> bool {
    let bound = Fidelity::Rounding {
        min_psnr: SEPARABLE_MIN_PSNR,
        max_abs: SEPARABLE_MAX_ABS,
    };
    let pixels = (ENGINE_WIDTH * ENGINE_HEIGHT) as usize;
    assert_eq!(
        forced.len(),
        baseline.len(),
        "{label}: the runs emitted different frame counts"
    );

    let mut differs = false;
    for (frame, (forced_frame, baseline_frame)) in forced.iter().zip(baseline).enumerate() {
        let forced_planes = forced_frame.chunks(pixels);
        let baseline_planes = baseline_frame.chunks(pixels);
        let diffs: Vec<PlaneDiff> = baseline_planes
            .zip(forced_planes)
            .map(|(expected, actual)| diff_plane(expected, actual))
            .collect();
        let verdict = bound.judge(&diffs);
        differs |= diffs.iter().any(|diff| !diff.identical);

        assert!(verdict.passed(), "{label} frame {frame}: {verdict:?}");
    }

    differs
}

/// Whether `left` and `right` are the same buffer.
///
/// A handle prints its unique memory id, so two handles to one buffer print the same text.
fn same_buffer(left: &Handle, right: &Handle) -> bool {
    let left_text = format!("{left:?}");
    let right_text = format!("{right:?}");

    left_text == right_text
}

#[test]
fn integer_nlmeans_windows_read_the_code_copy_of_the_ring_they_read() {
    let client = test_client();
    let geometry = Geometry {
        width: ENGINE_WIDTH,
        height: ENGINE_HEIGHT,
        channels: ChannelMode::Luma,
        input: SampleFormat::U8,
        output: SampleFormat::F32,
    };
    let algorithms = [
        ("temporal", temporal_fast(), false),
        ("motion compensated", motion_compensated_fast(), true),
    ];

    for (name, algorithm, compensated) in algorithms {
        let kernels = EngineKernels {
            profile: PipelineProfile::IntegerSearch,
            ..EngineKernels::default()
        };
        let engine = Nlmeans::with_kernels(&client, algorithm, geometry, kernels).expect("the engine builds");
        let front = engine.front_for_test();
        let (code_ring, _) = front.code_ring().expect("the engine keeps a code ring");
        let plain = front.plain_codes().expect("the single window reads codes");
        let temporal = front.temporal_codes().expect("the pair windows read codes");

        assert_eq!(plain.codes, EIGHT_BIT, "{name}");
        assert_eq!(temporal.codes, EIGHT_BIT, "{name}");

        let plain_reads_code_ring = same_buffer(&plain.ring, code_ring);
        assert!(
            plain_reads_code_ring,
            "{name}: the single window reads the code ring"
        );

        let compensated_ring = front.compensated_code_ring();
        let expected_temporal = if compensated {
            compensated_ring.expect("motion compensation keeps compensated codes")
        } else {
            code_ring
        };
        let temporal_reads_expected = same_buffer(&temporal.ring, expected_temporal);
        let temporal_reads_plain = same_buffer(&temporal.ring, code_ring);

        assert!(
            temporal_reads_expected,
            "{name}: the pair windows read the wrong code ring"
        );
        assert_eq!(temporal_reads_plain, !compensated, "{name}");
    }

    // A ROCm test that uploads but never launches needs a sync before exit.
    sync(&client).expect("the device syncs");
}

#[test]
fn forcing_the_integer_window_on_integer_nlmeans_stays_within_the_window_bound() {
    let algorithms = [
        ("temporal", temporal_fast()),
        ("motion compensated", motion_compensated_fast()),
    ];
    let scalar_int = WindowCandidate::ScalarInt.name();

    for (name, algorithm) in algorithms {
        let mut differs_somewhere = false;

        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            for codes in [EIGHT_BIT, TEN_BIT] {
                let label = format!("{name} {channels:?} {}-bit", codes.depth);
                let frames = code_frames(channels, codes);
                let baseline_plan = KernelPlan::baseline();
                let forced_plan = KernelPlan::baseline().with_site(NLM_WINDOW, scalar_int);
                let baseline = integer_nlmeans_outputs(algorithm, channels, codes, &frames, baseline_plan);
                let forced = integer_nlmeans_outputs(algorithm, channels, codes, &frames, forced_plan);

                differs_somewhere |= within_window_bound(&forced, &baseline, &label);
            }
        }

        assert!(differs_somewhere, "{name}: the forced integer window never ran");
    }
}
