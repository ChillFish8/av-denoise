use super::engine::{nl4d_outputs_with, noisy_frames};
use crate::Nl4dOptions;
use crate::nlmeans::ChannelMode;
use crate::tune::{EngineKernels, PipelineProfile};

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
fn profile_names_round_trip_with_the_new_profiles() {
    for profile in PipelineProfile::ALL {
        let name = profile.name();
        let parsed = PipelineProfile::from_name(name);

        assert_eq!(parsed, Some(profile), "{name}");
    }

    assert!(PipelineProfile::ALL.contains(&PipelineProfile::SplitSearch));
}
