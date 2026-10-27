use super::engine::{nl4d_outputs_with, noisy_frames};
use super::runtime::{packed_supported, test_client};
use crate::Nl4dOptions;
use crate::bench_api::HostIo;
use crate::nl4d::denoiser::Nl4dDenoiser;
use crate::nl4d::resolve_params;
use crate::nlmeans::ChannelMode;
use crate::tune::{EngineKernels, KernelSelector, PipelineProfile};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;

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
    let mut denoiser =
        Nl4dDenoiser::with_kernels(&client, params, WIDTH, HEIGHT, selector).expect("the denoiser builds");
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
}
