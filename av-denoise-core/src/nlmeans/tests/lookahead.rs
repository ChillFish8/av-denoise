use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::*;
use crate::noise::NOISE_WINDOW_RADIUS;
use crate::tune::tests::fixtures::noisy_field_over;

const PAN_WIDTH: u32 = 256;
const PAN_HEIGHT: u32 = 192;
const PAN_SHIFT: u32 = 3;
const PAN_FRAMES: usize = 12;
const PAN_SIGMA: f32 = 2.0 / 255.0;
const PAN_SEED: u32 = 7;

/// Each pan frame's sigma read along its own motion, the reference the look-ahead readings match.
/// `Direct` at radius 2 and `Chained` at radius 3 read the same values.
const PASS_TIME_SIGMAS: [f32; PAN_FRAMES] = [
    0.007815467,
    0.007805687,
    0.007863555,
    0.007839382,
    0.0077624097,
    0.007782205,
    0.007812833,
    0.0077841855,
    0.007855936,
    0.007838592,
    0.007886594,
    0.007867605,
];

fn nl4d_pan_params(temporal_radius: u32) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        max_search_escalation: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Auto,
    };
    let nlm = NlmParams {
        channels: ChannelMode::Luma,
        motion_compensation,
        ..defaults.nlm
    };

    Nl4dParams {
        nlm,
        temporal_radius,
        ..defaults
    }
}

fn hq_params(temporal_radius: u32, motion_compensation: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: Some(HqParams::default()),
    }
}

/// Pushes and submits every frame of `frames`, as an engine does.
fn run_nl4d(params: Nl4dParams, frames: &[Vec<f32>]) -> Nl4dDenoiser {
    let client = make_client();
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");

    for frame in frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");
    }

    denoiser
}

/// The temporal sample sigma the history holds for stream frame `frame`, `None` without a sample.
fn history_sigma(front: &NlmDenoiser, frame: u32) -> Option<f32> {
    let noise = front.measured_noise().expect("the noise is measured");
    let (reading, _) = noise.read_history_frame(frame);

    reading.sample.map(|sample| sample.sigma[0])
}

fn assert_within(value: f32, reference: f32, tolerance: f32, label: &str) {
    let rel_err = (value - reference).abs() / reference;
    assert!(
        rel_err <= tolerance,
        "{label}: {value} vs {reference} (rel err {rel_err:.4})"
    );
}

#[test]
fn lookahead_readings_match_pass_time_readings() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, PAN_SEED);

    for (radius, label) in [(2, "direct"), (3, "chained")] {
        let params = nl4d_pan_params(radius);
        let denoiser = run_nl4d(params, &frames);
        let front = denoiser.front_for_test();
        assert_eq!(front.is_chained(), radius == 3, "{label}: estimation");

        for (frame, &reference) in PASS_TIME_SIGMAS.iter().enumerate() {
            let sigma = history_sigma(front, frame as u32).expect("a pan frame has a sample");
            let frame_label = format!("{label} frame {frame}");
            assert_within(sigma, reference, 0.02, &frame_label);
        }
    }
}

#[test]
fn lookahead_readings_do_not_depend_on_the_field_lambda() {
    // On flat grain each block's best match fits the grain, which the truemotion stage would smooth.
    let flat = vec![0.5f32; (PAN_WIDTH * PAN_HEIGHT) as usize];
    let frames: Vec<Vec<f32>> = (0..PAN_FRAMES as u32)
        .map(|index| noisy_field_over(&flat, PAN_WIDTH, PAN_HEIGHT, PAN_SIGMA, PAN_SEED * 1000 + index))
        .collect();

    for radius in [2, 3] {
        let smoothed_params = Nl4dParams {
            field_lambda: 4.0,
            ..nl4d_pan_params(radius)
        };
        let raw_params = Nl4dParams {
            field_lambda: 0.0,
            ..nl4d_pan_params(radius)
        };
        let smoothed = run_nl4d(smoothed_params, &frames);
        let raw = run_nl4d(raw_params, &frames);

        for frame in 0..PAN_FRAMES as u32 {
            let smoothed_sigma =
                history_sigma(smoothed.front_for_test(), frame).expect("a flat frame has a sample");
            let raw_sigma = history_sigma(raw.front_for_test(), frame).expect("a flat frame has a sample");
            assert_eq!(smoothed_sigma, raw_sigma, "radius {radius} frame {frame}");
        }
    }
}

#[test]
fn scene_first_frame_reads_against_the_next_frame() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, 2, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let params = nl4d_pan_params(2);
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");

    denoiser.push_frame(&frames[0]);
    let front = denoiser.front_for_test();
    let alone = history_sigma(front, 0);
    assert_eq!(alone, None, "the first frame alone has nothing to read against");

    denoiser.push_frame(&frames[1]);
    let front = denoiser.front_for_test();
    let first = history_sigma(front, 0).expect("the first frame has a sample");
    assert_within(first, PASS_TIME_SIGMAS[0], 0.02, "first frame");

    let raw = raw_reading_sigma(&frames[1], &frames[0], PAN_WIDTH, PAN_HEIGHT, 1)
        .expect("the raw reading has a sample");
    assert!(
        first <= 0.5 * raw[0],
        "the first frame read {first}, not well below the uncompensated {}",
        raw[0]
    );
}

#[test]
fn held_frames_do_not_enter_the_window() {
    const HELD: [usize; 2] = [3, 7];
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, 16, PAN_SIGMA, PAN_SEED);
    let mut with_held = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        with_held.push(frame.clone());
        if HELD.contains(&index) {
            with_held.push(frame.clone());
        }
    }

    let plain_params = nl4d_pan_params(2);
    let held_params = nl4d_pan_params(2);
    let plain = run_nl4d(plain_params, &frames);
    let held = run_nl4d(held_params, &with_held);

    // Each held copy follows its frame, so it lands one stream frame later per earlier copy.
    let front = held.front_for_test();
    for (copies_before, &index) in HELD.iter().enumerate() {
        let copy = (index + copies_before + 1) as u32;
        let sigma = history_sigma(front, copy);
        assert_eq!(sigma, None, "the held copy at stream frame {copy} has a sample");
    }

    let plain_snapshot = plain.noise_snapshot().expect("the noise state reads back");
    let held_snapshot = held.noise_snapshot().expect("the noise state reads back");
    let plain_sigma = plain_snapshot.derived.sigma[0];
    let held_sigma = held_snapshot.derived.sigma[0];
    assert_within(held_sigma, plain_sigma, 0.01, "derived sigma with held frames");
}

#[test]
fn motion_off_reads_uncompensated_at_push() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, 0, 4, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let params = hq_params(2, MotionCompensationMode::None);
    let mut denoiser = NlmDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT);

    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);

        let frame_index = index as u32;
        let sigma = history_sigma(&denoiser, frame_index);
        let Some(previous) = index.checked_sub(1) else {
            assert_eq!(sigma, None, "the first frame has nothing to read against");
            continue;
        };

        let sigma = sigma.expect("a static frame has a sample at push");
        let reference = raw_reading_sigma(&frames[previous], frame, PAN_WIDTH, PAN_HEIGHT, 1)
            .expect("the raw reading has a sample");
        let label = format!("frame {index}");
        assert_within(sigma, reference[0], 1e-5, &label);
    }

    let spatial_params = hq_params(0, MotionCompensationMode::None);
    let mut spatial = NlmDenoiser::new(&client, spatial_params, PAN_WIDTH, PAN_HEIGHT);
    spatial.push_frame(&frames[0]);
    spatial.push_frame(&frames[1]);

    let noise = spatial.measured_noise().expect("the noise is measured");
    let (reading, _) = noise.read_history_frame(1);
    assert_eq!(reading.sample, None, "a spatial front reads no temporal sample");
    assert!(
        reading.immerkaer_low[0] > 0.0,
        "a spatial front still reads its Immerkær sigma at push"
    );
}

#[test]
fn released_frames_are_bit_identical_to_pushed_frames() {
    let pushes = LOOKAHEAD_FRAMES as usize + 4;
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, pushes, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let motion = MotionCompensationMode::mvtools_default();
    let params = hq_params(2, motion);
    let total_frames = params.total_frames();
    let mut denoiser = NlmDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT);
    let depth = denoiser.lookahead_depth();
    assert_eq!(depth, LOOKAHEAD_FRAMES as usize, "a measuring front looks ahead");

    let frame_len = (PAN_WIDTH * PAN_HEIGHT) as usize;
    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");

        let Some(released) = index.checked_sub(depth) else {
            assert_eq!(denoiser.real_pushes(), 0, "push {index} released a frame early");
            continue;
        };

        assert_eq!(denoiser.real_pushes(), released + 1, "one release per push");

        let newest_slot = denoiser.ring_slot(total_frames - 1) as usize;
        let ring = denoiser.input_ring().clone();
        let ring_bytes = client.read_one(ring).expect("the ring reads back");
        let ring_values = f32::from_bytes(&ring_bytes);
        let start = newest_slot * frame_len;
        let stored = &ring_values[start..start + frame_len];

        let stored_bits: Vec<u32> = stored.iter().map(|value| value.to_bits()).collect();
        let pushed_bits: Vec<u32> = frames[released].iter().map(|value| value.to_bits()).collect();
        assert!(
            stored_bits == pushed_bits,
            "released frame {released} differs from the pushed frame"
        );
    }
}

/// A measuring radius 2 front with motion compensation, at the pan size.
fn scene_front() -> NlmDenoiser {
    let client = make_client();
    let motion = MotionCompensationMode::mvtools_default();
    let params = hq_params(2, motion);

    NlmDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT)
}

/// Pushes one scene and flushes it, returning every output in emit order.
fn scene_outputs(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut denoiser = scene_front();
    let mut outputs = Vec::new();

    for frame in frames {
        denoiser.push_frame(frame);

        let output = denoiser.denoise().expect("denoise failed");
        outputs.extend(output);
    }

    denoiser
        .flush(|output| outputs.push(output.to_vec()))
        .expect("flush failed");

    outputs
}

/// Pushes one scene and flushes it, returning the derived luma sigma of each pass.
fn scene_pass_sigmas(frames: &[Vec<f32>]) -> Vec<f32> {
    let mut denoiser = scene_front();
    let mut sigmas = Vec::new();
    let mut sample = |front: &NlmDenoiser| {
        let snapshot = front.noise_snapshot().expect("the noise state reads back");
        sigmas.push(snapshot.derived.sigma[0]);
    };

    for frame in frames {
        denoiser.push_frame(frame);

        let output = denoiser.denoise_submit_gpu().expect("denoise failed");
        if output.is_some() {
            sample(&denoiser);
        }
    }

    let mut flushed = 0;
    let target = denoiser.flush_target();
    while flushed < target {
        let step = denoiser.flush_step_gpu().expect("flush step failed");
        if step.is_none() {
            continue;
        }

        sample(&denoiser);
        flushed += 1;
    }

    sigmas
}

/// The index of the frame in `frames` closest to `output`.
fn nearest_frame(output: &[f32], frames: &[Vec<f32>]) -> usize {
    let distance = |frame: &Vec<f32>| -> f32 {
        let pairs = frame.iter().zip(output);
        pairs.map(|(input, denoised)| (input - denoised).powi(2)).sum()
    };
    let distances: Vec<f32> = frames.iter().map(distance).collect();
    let nearest = distances
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1));

    nearest.map(|(index, _)| index).expect("a scene has frames")
}

#[test]
fn short_scene_windows_cover_the_whole_scene() {
    let longest = 20;
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, longest, PAN_SIGMA, PAN_SEED);
    let whole_scene_limit = NOISE_WINDOW_RADIUS as usize + 1;
    let first_window = &frames[..whole_scene_limit];
    let first_window_sigmas = scene_pass_sigmas(first_window);

    for length in [1, 8, longest] {
        let scene = &frames[..length];
        let outputs = scene_outputs(scene);
        assert_eq!(
            outputs.len(),
            length,
            "a {length} frame scene emits each frame once"
        );

        // Each pan frame is shifted `PAN_SHIFT` pixels from the last over texture, so a denoised frame is
        // far closer to its own input than to any other, and the nearest input gives its index.
        for (index, output) in outputs.iter().enumerate() {
            let nearest = nearest_frame(output, scene);
            assert_eq!(
                nearest, index,
                "output {index} of a {length} frame scene is out of order"
            );
        }

        let sigmas = scene_pass_sigmas(scene);
        assert_eq!(
            sigmas.len(),
            length,
            "a {length} frame scene runs one pass per frame"
        );

        if length <= whole_scene_limit {
            let first = sigmas[0];
            assert!(
                sigmas.iter().all(|&sigma| sigma == first),
                "every window of a {length} frame scene is the whole scene: {sigmas:?}"
            );
            continue;
        }

        // The first centre's window stops at the scene start, so it holds the scene's first
        // `NOISE_WINDOW_RADIUS + 1` frames and reads like a scene of only those.
        assert_eq!(sigmas[0], first_window_sigmas[0]);
    }
}
