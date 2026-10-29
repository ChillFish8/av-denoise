use av_denoise_core::{Nl4dOptions, NlmeansHqOptions};

use super::reseed::{
    layout,
    ramp_clip,
    reseed_shifted,
    shifted_output_count,
    stream_all,
    test_plane_options,
    window_of_span,
};
use crate::planar::{ChannelIntent, PlanarDenoiser, PlaneOptions};
use crate::{Algorithm, DenoisingMode};

/// Temporal nl4d at `radius`, measuring the noise.
fn nl4d_measured_options(radius: u32) -> PlaneOptions {
    let nl4d_options = Nl4dOptions::default();

    PlaneOptions {
        algorithm: Algorithm::Nl4d(nl4d_options),
        ..test_plane_options(radius)
    }
}

/// Temporal nlmeans HQ at `radius`, measuring the noise.
fn nlmeans_hq_measured_options(radius: u32) -> PlaneOptions {
    let hq_options = NlmeansHqOptions::default();

    PlaneOptions {
        algorithm: Algorithm::NlmeansHq(hq_options),
        ..test_plane_options(radius)
    }
}

/// `options` for each intent, so the passthrough planes are checked as well as the denoised ones.
fn every_intent(options: PlaneOptions) -> [PlaneOptions; 3] {
    let intents = [
        ChannelIntent::LumaChroma,
        ChannelIntent::Luma,
        ChannelIntent::Chroma,
    ];

    intents.map(|intent| PlaneOptions {
        intent,
        ..options.clone()
    })
}

#[test]
fn a_measured_nl4d_reseed_matches_streaming_at_every_frame() {
    let frame_layout = layout();

    let base_options = nl4d_measured_options(2);
    for options in every_intent(base_options) {
        for clip_len in [7usize, 30] {
            let frames = ramp_clip(&frame_layout, clip_len);
            let streamed = stream_all(&options, &frames);
            assert_eq!(streamed.len(), clip_len);

            for target_frame in 0..clip_len {
                let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
                let got = reseed_shifted(&mut denoiser, &frames, target_frame);

                let intent = options.intent;
                let expected_len = shifted_output_count(&denoiser, clip_len, target_frame);
                assert_eq!(
                    got.len(),
                    expected_len,
                    "{intent:?} len={clip_len} k={target_frame}"
                );

                for (offset, planes) in got.iter().enumerate() {
                    let index = target_frame + offset;
                    let label = format!("{intent:?} len={clip_len} k={target_frame} frame {index}");
                    assert_eq!(planes.y, streamed[index].y, "{label} luma");
                    assert_eq!(planes.u, streamed[index].u, "{label} u");
                    assert_eq!(planes.v, streamed[index].v, "{label} v");
                }
            }
        }
    }
}

#[test]
fn a_measured_nlmeans_hq_reseed_matches_the_streaming_output_mid_clip() {
    let frame_layout = layout();

    let base_options = nlmeans_hq_measured_options(2);
    for options in every_intent(base_options) {
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let target_frame = span.behind + 2;
        let frames = ramp_clip(&frame_layout, target_frame + span.ahead + 3);
        let streamed = stream_all(&options, &frames);

        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        let intent = options.intent;
        assert_eq!(got.y, streamed[target_frame].y, "{intent:?} luma");
        assert_eq!(got.u, streamed[target_frame].u, "{intent:?} u");
        assert_eq!(got.v, streamed[target_frame].v, "{intent:?} v");
    }
}

#[test]
fn held_frames_from_the_options_match_the_built_denoiser() {
    let frame_layout = layout();
    let nl4d_pinned = Nl4dOptions {
        sigma: Some(0.03),
        ..Nl4dOptions::default()
    };
    let option_sets = [
        test_plane_options(2),
        nlmeans_hq_measured_options(2),
        nl4d_measured_options(2),
        PlaneOptions {
            algorithm: Algorithm::Nl4d(nl4d_pinned),
            ..test_plane_options(3)
        },
        PlaneOptions {
            intent: ChannelIntent::Chroma,
            ..nl4d_measured_options(1)
        },
        PlaneOptions {
            mode: DenoisingMode::Spacial,
            ..nlmeans_hq_measured_options(0)
        },
    ];

    for options in option_sets {
        let denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let held = options.max_held_frames(frame_layout);

        assert_eq!(held, span.ahead, "{:?} at {:?}", options.algorithm, options.mode);
    }
}
