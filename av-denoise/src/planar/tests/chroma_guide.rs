use super::pan_reseed::{FRAMES, clips, pan_layout, stream};
use super::reseed::test_plane_options;
use super::*;
use crate::{MotionCompensationMode, MotionSearch, Nl4dOptions, NlmeansHqOptions, NlmeansOptions};

/// Temporal nl4d and motion-compensated HQ nlmeans, both cleaning luma and chroma.
fn motion_options() -> [PlaneOptions; 2] {
    let nl4d = PlaneOptions {
        algorithm: Algorithm::Nl4d(Nl4dOptions::default()),
        ..test_plane_options(2)
    };

    let compensated = NlmeansOptions {
        motion_compensation: MotionCompensationMode::from(MotionSearch::default()),
        ..NlmeansOptions::default()
    };
    let hq = NlmeansHqOptions {
        nlm: compensated,
        ..NlmeansHqOptions::default()
    };
    let nlmeans = PlaneOptions {
        algorithm: Algorithm::NlmeansHq(hq),
        ..test_plane_options(2)
    };

    [nl4d, nlmeans]
}

#[test]
fn chroma_only_matches_the_luma_plus_chroma_chroma() {
    for options in motion_options() {
        let chroma_only = PlaneOptions {
            intent: ChannelIntent::Chroma,
            ..options.clone()
        };

        for (name, frames) in clips() {
            let mut both_denoiser = PlanarDenoiser::create(&options, pan_layout()).unwrap();
            let expected = stream(&mut both_denoiser, &frames);
            let mut denoiser = PlanarDenoiser::create(&chroma_only, pan_layout()).unwrap();
            let got = stream(&mut denoiser, &frames);
            assert_eq!(got.len(), FRAMES, "{name}");

            for (index, planes) in got.iter().enumerate() {
                let label = format!("{:?} {name} frame {index}", options.algorithm);
                assert!(planes.y == frames[index].y, "{label}: luma is not the source");
                assert!(planes.u == expected[index].u, "{label}: u differs");
                assert!(planes.v == expected[index].v, "{label}: v differs");
            }
        }
    }
}
