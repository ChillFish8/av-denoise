use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nl4d::tests::helpers::{noisy_copy_of, textured_base};
use crate::nlmeans::*;

const RADIUS: u32 = 2;
const SIZE: u32 = 256;
const GRAIN: f32 = 6.0 / 255.0;

fn edge_params(windowed: bool) -> NlmParams {
    NlmParams {
        temporal_radius: RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: Some(HqParams {
            windowed_noise_estimation: windowed,
            ..HqParams::default()
        }),
    }
}

fn shifted_denoiser(client: &ComputeClient<R>, windowed: bool) -> NlmDenoiser<R> {
    let params = edge_params(windowed);
    let mut denoiser = NlmDenoiser::<R>::new(client, params, SIZE, SIZE);
    denoiser.set_shifted_edges(true);
    denoiser.set_luma_noise_fields(true);
    denoiser
}

fn push_grain(denoiser: &mut NlmDenoiser<R>, count: u32) {
    let base = textured_base(SIZE, SIZE);
    for seed in 0..count {
        let frame = noisy_copy_of(&base, SIZE, SIZE, GRAIN, seed);
        denoiser.push_frame(&frame);
    }
}

#[test]
fn shifted_edges_skip_the_leading_copies() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client, false);

    push_grain(&mut denoiser, 1);

    assert_eq!(denoiser.frames_loaded_for_test(), 1);
    assert!(!denoiser.window_ready());
}

#[test]
fn centre_zero_borrows_the_next_frames_reading() {
    for windowed in [false, true] {
        let client = make_client();
        let total_frames = 2 * RADIUS + 1;

        let mut at_zero = shifted_denoiser(&client, windowed);
        push_grain(&mut at_zero, total_frames);
        at_zero.submit_machinery(0).unwrap().unwrap();

        let mut at_one = shifted_denoiser(&client, windowed);
        push_grain(&mut at_one, total_frames);
        at_one.submit_machinery(1).unwrap().unwrap();

        let borrowed = at_zero.current_sigmas_temporal_only();
        let own = at_one.current_sigmas_temporal_only();
        let spatial = at_zero.current_sigmas_low_unboosted();
        assert_eq!(borrowed, own, "windowed={windowed}");
        assert_ne!(borrowed, spatial, "windowed={windowed} fell back to spatial");
        assert!(at_zero.current_noise_curve().is_some(), "windowed={windowed}");
    }
}

#[test]
fn centre_zero_falls_back_when_every_reading_ahead_is_rejected() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client, true);
    let base = textured_base(SIZE, SIZE);
    let frozen = noisy_copy_of(&base, SIZE, SIZE, GRAIN, 0);
    for _ in 0..(2 * RADIUS + 1) {
        denoiser.push_frame(&frozen);
    }

    denoiser.submit_machinery(0).unwrap().unwrap();

    let temporal_only = denoiser.current_sigmas_temporal_only();
    let low_unboosted = denoiser.current_sigmas_low_unboosted();
    assert_eq!(temporal_only, low_unboosted);
    assert!(denoiser.current_noise_curve().is_none());
}

#[test]
fn a_second_stream_never_reads_the_first_streams_f0_stats() {
    let client = make_client();
    let total_frames = 2 * RADIUS + 1;
    let mut denoiser = shifted_denoiser(&client, true);
    // One more than the ring, so the first stream writes a real record into the slot the second
    // stream's f0 lands in.
    push_grain(&mut denoiser, total_frames + 1);
    denoiser.reset_stream_state();

    let base = textured_base(SIZE, SIZE);
    let frozen = noisy_copy_of(&base, SIZE, SIZE, GRAIN, 0);
    for _ in 0..total_frames {
        denoiser.push_frame(&frozen);
    }

    denoiser.submit_machinery(0).unwrap().unwrap();

    let temporal_only = denoiser.current_sigmas_temporal_only();
    let low_unboosted = denoiser.current_sigmas_low_unboosted();
    assert_eq!(temporal_only, low_unboosted);
}

#[test]
fn fill_ring_with_last_frame_makes_a_short_stream_ready() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client, false);
    push_grain(&mut denoiser, 2);

    denoiser.fill_ring_with_last_frame().expect("fill ring");

    assert!(denoiser.window_ready());
    assert_eq!(denoiser.real_pushes(), 2);
}

#[test]
fn nlm_mode_keeps_the_leading_copies() {
    let client = make_client();
    let params = edge_params(false);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, SIZE, SIZE);

    push_grain(&mut denoiser, 1);

    assert_eq!(denoiser.frames_loaded_for_test(), 1 + RADIUS as usize);
}
