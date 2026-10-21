use cubecl::server::Handle;

use super::engine::{nl4d_outputs_with, noisy_frames, run_engine_on_planes};
use super::fixtures::{CollabSpec, code_format, collab_site, integer_split_sites, upload_codes};
use super::runtime::{TestRuntime, packed_supported, test_client};
use crate::bench_api::HostIo;
use crate::collab::needs_warp_uniform_search;
use crate::engine::{Geometry, SampleFormat};
use crate::nl4d::denoiser::Nl4dDenoiser;
use crate::nl4d::resolve_params;
use crate::nlmeans::ChannelMode;
use crate::tune::sites::{COLLAB_SEARCH, IntegerVariant, SearchVariant};
use crate::tune::{
    DeviceLimits,
    EngineKernels,
    IntegerCodes,
    KernelPlan,
    KernelSelector,
    Legality,
    PipelineProfile,
    SiteCandidate,
    VariantSite,
};
use crate::{Nl4d, Nl4dOptions};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const CODE_FRAMES: usize = 8;

const EIGHT_BIT: IntegerCodes = IntegerCodes { depth: 8 };
const TEN_BIT: IntegerCodes = IntegerCodes { depth: 10 };

/// The lowest PSNR any plane of the integer search may sit at against the f32 search.
const INTEGER_MIN_PSNR: f64 = 45.0;

/// Frames of code planes, each plane `WIDTH * HEIGHT` codes.
type CodeFrames = Vec<Vec<Vec<u16>>>;

/// One source an integer search test denoises.
struct CodeCase {
    label: String,
    channels: ChannelMode,
    codes: IntegerCodes,
    frames: CodeFrames,
}

fn plane_codes(plane: &[f32], codes: IntegerCodes) -> Vec<u16> {
    let max_value = codes.max_code() as f32;

    plane
        .iter()
        .map(|&value| {
            let scaled = (value * max_value).round();
            scaled.clamp(0.0, max_value) as u16
        })
        .collect()
}

fn noisy_code_frames(planes: usize, codes: IntegerCodes) -> CodeFrames {
    let frames = noisy_frames(WIDTH, HEIGHT, planes, CODE_FRAMES);
    let mut code_frames = Vec::with_capacity(frames.len());

    for frame in &frames {
        let code_planes = frame.iter().map(|plane| plane_codes(plane, codes)).collect();
        code_frames.push(code_planes);
    }

    code_frames
}

fn flat_code_frames(planes: usize, codes: IntegerCodes) -> CodeFrames {
    let middle_code = codes.max_code().div_ceil(2) as u16;
    let plane = vec![middle_code; (WIDTH * HEIGHT) as usize];
    let frame = vec![plane; planes];

    vec![frame; CODE_FRAMES]
}

/// Noisy and flat 8- and 10-bit sources in every channel mode.
fn code_cases() -> Vec<CodeCase> {
    let modes = [
        (ChannelMode::Luma, 1usize),
        (ChannelMode::Chroma, 2),
        (ChannelMode::Yuv, 3),
    ];
    let mut cases = Vec::new();

    for (channels, planes) in modes {
        for codes in [EIGHT_BIT, TEN_BIT] {
            let depth = codes.depth;
            let noisy = noisy_code_frames(planes, codes);
            let flat = flat_code_frames(planes, codes);

            for (content, frames) in [("noisy", noisy), ("flat", flat)] {
                let label = format!("{channels:?} {content} {depth}-bit");
                let case = CodeCase {
                    label,
                    channels,
                    codes,
                    frames,
                };
                cases.push(case);
            }
        }
    }

    cases
}

/// Every frame an nl4d engine under `profile` and `plan` emits for `case`.
///
/// An integer search engine must keep the case's codes, or the integer route went untested.
fn code_outputs(profile: PipelineProfile, plan: KernelPlan, case: &CodeCase) -> Vec<Vec<f32>> {
    let client = test_client();
    let geometry = Geometry {
        width: WIDTH,
        height: HEIGHT,
        channels: case.channels,
        input: code_format(case.codes),
        output: SampleFormat::F32,
    };
    let kernels = EngineKernels {
        profile,
        plan,
        ..EngineKernels::default()
    };
    let options = Nl4dOptions::default();
    let mut engine = Nl4d::with_kernels(&client, options, geometry, kernels).expect("the engine builds");

    if profile == PipelineProfile::IntegerSearch {
        let kept = engine.denoiser_for_test().integer_codes();
        assert_eq!(kept, Some(case.codes), "{}: the engine keeps codes", case.label);
    }

    let mut uploaded = Vec::with_capacity(case.frames.len());
    for frame in &case.frames {
        let handles: Vec<Handle> = frame
            .iter()
            .map(|plane| upload_codes(&client, plane, case.codes))
            .collect();
        uploaded.push(handles);
    }

    run_engine_on_planes(&client, &mut engine, WIDTH, HEIGHT, &uploaded).expect("the frames denoise")
}

/// A plan that forces `variant` on every collab search key.
fn forced_search(variant: IntegerVariant) -> KernelPlan {
    let name = variant.name();
    KernelPlan::baseline().with_site(COLLAB_SEARCH, name)
}

/// Every integer search variant that is legal on this device for a search of `case`'s geometry and
/// fits it.
///
/// The search is the fixture's at the engine's frame size and default spatial radius.
fn runnable_integer_variants(case: &CodeCase) -> Vec<IntegerVariant> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let legality = Legality::default();
    let options = Nl4dOptions::default();
    let spec = CollabSpec {
        spatial_radius: options.spatial_radius,
        ..CollabSpec::new(WIDTH, HEIGHT, case.channels, 1)
    };
    let fused = collab_site(&client, &spec);
    let (search, _) = integer_split_sites(&fused, case.codes);

    SearchVariant::ALL
        .iter()
        .copied()
        .filter(|&candidate| search.fits(candidate, &limits, &legality))
        .filter_map(|candidate| match candidate {
            SearchVariant::Integer(variant) => Some(variant),
            SearchVariant::Float(_) => None,
        })
        .collect()
}

fn integer_reference() -> IntegerVariant {
    let client = test_client();
    let uniform_only = needs_warp_uniform_search(&client);

    IntegerVariant::reference_for(uniform_only)
}

/// The PSNR of every plane of `got` against `want`, over every frame, with a peak of 1.
fn plane_psnrs(want: &[Vec<f32>], got: &[Vec<f32>], planes: usize) -> Vec<f64> {
    let pixels = (WIDTH * HEIGHT) as usize;
    let mut squared_sums = vec![0.0f64; planes];
    let mut counts = vec![0usize; planes];

    for (want_frame, got_frame) in want.iter().zip(got) {
        let plane_pairs = want_frame.chunks(pixels).zip(got_frame.chunks(pixels));

        for (plane, (want_plane, got_plane)) in plane_pairs.enumerate() {
            for (&expected, &actual) in want_plane.iter().zip(got_plane) {
                let error = f64::from(expected) - f64::from(actual);
                squared_sums[plane] += error * error;
            }

            counts[plane] += want_plane.len();
        }
    }

    squared_sums
        .iter()
        .zip(&counts)
        .map(|(&squared_sum, &count)| {
            let mse = squared_sum / count as f64;
            -10.0 * mse.log10()
        })
        .collect()
}

fn outputs_under(profile: PipelineProfile, channels: ChannelMode, frames: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    let kernels = EngineKernels {
        profile,
        ..EngineKernels::default()
    };
    let options = Nl4dOptions::default();

    nl4d_outputs_with(options, channels, WIDTH, HEIGHT, frames, kernels)
}

fn constant_frames(planes: usize, count: usize) -> Vec<Vec<Vec<f32>>> {
    let plane = vec![0.5f32; (WIDTH * HEIGHT) as usize];
    let frame = vec![plane; planes];

    vec![frame; count]
}

fn interleave(planes: &[Vec<f32>]) -> Vec<f32> {
    let pixels = planes[0].len();
    let mut samples = Vec::with_capacity(pixels * planes.len());

    for pixel in 0..pixels {
        for plane in planes {
            samples.push(plane[pixel]);
        }
    }

    samples
}

/// Every frame one denoiser under `profile` emits for `first` then `second`, each stream's
/// flushed tail included.
fn two_streams(
    profile: PipelineProfile,
    channels: ChannelMode,
    first: &[Vec<Vec<f32>>],
    second: &[Vec<Vec<f32>>],
) -> Vec<Vec<f32>> {
    let client = test_client();
    let options = Nl4dOptions::default();
    let params = resolve_params(&options, channels).expect("the default options resolve");
    let kernels = EngineKernels {
        profile,
        ..EngineKernels::default()
    };
    let selector = KernelSelector::new(&client, kernels);
    let mut denoiser = Nl4dDenoiser::<TestRuntime>::with_kernels(&client, params, WIDTH, HEIGHT, selector)
        .expect("the denoiser builds");
    let mut outputs = Vec::new();

    for stream in [first, second] {
        for frame in stream {
            let samples = interleave(frame);
            denoiser.push_frame(&samples);
            let denoised = denoiser.denoise().expect("the frame denoises");
            outputs.extend(denoised);
        }

        let sink = |frame: &[f32]| {
            let owned = frame.to_vec();
            outputs.push(owned);
        };
        denoiser.flush(sink).expect("the stream flushes");
    }

    outputs
}

#[test]
fn the_split_search_profile_matches_the_baseline_bit_for_bit() {
    let modes = [
        (ChannelMode::Luma, 1usize),
        (ChannelMode::Chroma, 2),
        (ChannelMode::Yuv, 3),
    ];

    for (channels, planes) in modes {
        let noisy = noisy_frames(WIDTH, HEIGHT, planes, 8);
        let flat = constant_frames(planes, 8);

        for (content, frames) in [("noisy", &noisy), ("flat", &flat)] {
            let want = outputs_under(PipelineProfile::Baseline, channels, frames);
            let got = outputs_under(PipelineProfile::SplitSearch, channels, frames);

            assert!(!want.is_empty(), "{channels:?} {content} emits frames");
            assert_eq!(got, want, "{channels:?} {content}");
        }
    }
}

#[test]
fn the_packed_atomics_profile_matches_the_baseline_bit_for_bit() {
    if !packed_supported() {
        eprintln!("skipped, this device has no 64-bit atomic add");
        return;
    }

    let modes = [
        (ChannelMode::Luma, 1usize),
        (ChannelMode::Chroma, 2),
        (ChannelMode::Yuv, 3),
    ];

    for (channels, planes) in modes {
        let noisy = noisy_frames(WIDTH, HEIGHT, planes, 8);
        let flat = constant_frames(planes, 8);

        for (content, frames) in [("noisy", &noisy), ("flat", &flat)] {
            let want = outputs_under(PipelineProfile::Baseline, channels, frames);
            let got = outputs_under(PipelineProfile::PackedAtomics, channels, frames);

            assert!(!want.is_empty(), "{channels:?} {content} emits frames");
            assert_eq!(got, want, "{channels:?} {content}");
        }
    }
}

/// A finished stream leaves words in the accumulator ring, which the next stream's first pass
/// must clear in the packed layout as in the split one.
#[test]
fn packed_atomics_match_the_baseline_across_a_stream_reset() {
    if !packed_supported() {
        eprintln!("skipped, this device has no 64-bit atomic add");
        return;
    }

    let modes = [
        (ChannelMode::Luma, 1usize),
        (ChannelMode::Chroma, 2),
        (ChannelMode::Yuv, 3),
    ];

    for (channels, planes) in modes {
        let first = noisy_frames(WIDTH, HEIGHT, planes, 6);
        let second = noisy_frames(WIDTH, HEIGHT, planes, 7);
        let want = two_streams(PipelineProfile::Baseline, channels, &first, &second);
        let got = two_streams(PipelineProfile::PackedAtomics, channels, &first, &second);

        assert!(!want.is_empty(), "{channels:?} emits frames");
        assert_eq!(got, want, "{channels:?}");
    }
}

#[test]
fn profile_names_round_trip_with_the_new_profiles() {
    for profile in PipelineProfile::ALL {
        let name = profile.name();
        let parsed = PipelineProfile::from_name(name);

        assert_eq!(parsed, Some(profile), "{name}");
    }

    assert!(PipelineProfile::ALL.contains(&PipelineProfile::SplitSearch));
    assert!(PipelineProfile::ALL.contains(&PipelineProfile::PackedAtomics));
    assert!(PipelineProfile::ALL.contains(&PipelineProfile::IntegerSearch));
    assert_eq!(PipelineProfile::IntegerSearch.name(), "integer_search");
    assert_eq!(PipelineProfile::ALL.len(), 5);
}

#[test]
fn the_integer_search_profile_runs_the_f32_baseline_when_nothing_is_tuned() {
    for case in code_cases() {
        let label = &case.label;
        let float_plan = KernelPlan::baseline();
        let integer_plan = KernelPlan::baseline();
        let want = code_outputs(PipelineProfile::F32Search, float_plan, &case);
        let got = code_outputs(PipelineProfile::IntegerSearch, integer_plan, &case);

        assert!(!want.is_empty(), "{label} emits frames");
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn every_integer_candidate_gives_the_same_frames() {
    let reference = integer_reference();

    for case in code_cases() {
        let label = &case.label;
        let variants = runnable_integer_variants(&case);
        let followers: Vec<IntegerVariant> = variants
            .into_iter()
            .filter(|&variant| variant != reference)
            .collect();
        assert!(
            !followers.is_empty(),
            "{label}: some integer variant follows the reference"
        );

        let reference_plan = forced_search(reference);
        let want = code_outputs(PipelineProfile::IntegerSearch, reference_plan, &case);
        assert!(!want.is_empty(), "{label} emits frames");

        for &variant in &followers {
            let name = variant.name();
            let plan = forced_search(variant);
            let got = code_outputs(PipelineProfile::IntegerSearch, plan, &case);

            assert_eq!(got, want, "{label} {name}");
        }
    }
}

/// Some noisy source must come out different, or the forced plan never reached the integer search.
#[test]
fn the_integer_search_stays_close_to_the_f32_search() {
    let reference = integer_reference();
    let mut any_differs = false;

    for case in code_cases() {
        let label = &case.label;
        let planes = case.channels.count() as usize;
        let float_plan = KernelPlan::baseline();
        let want = code_outputs(PipelineProfile::F32Search, float_plan, &case);
        let reference_plan = forced_search(reference);
        let got = code_outputs(PipelineProfile::IntegerSearch, reference_plan, &case);
        assert_eq!(got.len(), want.len(), "{label} frame count");

        any_differs |= got != want;

        let psnrs = plane_psnrs(&want, &got, planes);
        for (plane, psnr) in psnrs.iter().enumerate() {
            assert!(*psnr >= INTEGER_MIN_PSNR, "{label} plane {plane}: {psnr:.2} dB");
        }
    }

    assert!(any_differs, "the integer search ran on some source");
}
