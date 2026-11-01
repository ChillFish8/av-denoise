use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nl4d::tests::helpers::textured_base;
use crate::nlmeans::*;

const RADIUS: u32 = 2;
const SIZE: u32 = 256;
const GRAIN: f32 = 6.0 / 255.0;

fn edge_params() -> NlmParams {
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
        hq: Some(HqParams::default()),
    }
}

fn shifted_denoiser(client: &Client) -> NlmDenoiser {
    let params = edge_params();
    let mut denoiser = NlmDenoiser::new(client, params, SIZE, SIZE);
    denoiser.set_shifted_edges(true);
    denoiser.set_luma_noise_fields(true);
    denoiser
}

/// Moves every frame the noise look-ahead holds into the ring, as a stream's end does.
fn release_held(denoiser: &mut NlmDenoiser) {
    while denoiser.release_held_frame().expect("release failed") {}
}

/// Pushes `count` grain frames and releases each into the ring.
fn push_grain(denoiser: &mut NlmDenoiser, count: u32) {
    let base = textured_base(SIZE, SIZE);
    for seed in 0..count {
        let frame = noisy_field_over(&base, SIZE, SIZE, GRAIN, seed);
        denoiser.push_frame(&frame);
    }

    release_held(denoiser);
}

fn push_frozen(denoiser: &mut NlmDenoiser, count: u32) {
    let base = textured_base(SIZE, SIZE);
    let frozen = noisy_field_over(&base, SIZE, SIZE, GRAIN, 0);
    for _ in 0..count {
        denoiser.push_frame(&frozen);
    }

    release_held(denoiser);
}

#[test]
fn shifted_edges_skip_the_leading_copies() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client);

    push_grain(&mut denoiser, 1);

    assert_eq!(denoiser.frames_loaded_for_test(), 1);
    assert!(!denoiser.window_ready());
}

/// Centre 0 has no previous frame, so it is measured against the next one. Centre 1 measures the
/// same pair the other way round.
#[test]
fn centre_zero_reads_its_own_sample_against_the_next_frame() {
    let client = make_client();
    let total_frames = 2 * RADIUS + 1;

    let mut at_zero = shifted_denoiser(&client);
    push_grain(&mut at_zero, total_frames);
    at_zero.submit_machinery(0).unwrap().unwrap();

    let mut at_one = shifted_denoiser(&client);
    push_grain(&mut at_one, total_frames);
    at_one.submit_machinery(1).unwrap().unwrap();

    let noise = at_zero.measured_noise().expect("the noise is measured");
    let (reading, _) = noise.read_history_frame(0);
    assert!(reading.sample.is_some(), "centre 0 has its own sample");

    let from_next = at_zero.current_sigmas_temporal_only();
    let from_previous = at_one.current_sigmas_temporal_only();
    let spatial = at_zero.current_sigmas_low_unboosted();
    let rel_err = (from_next[0] - from_previous[0]).abs() / from_previous[0];
    assert!(
        rel_err <= 0.05,
        "the pair read {} forwards and {} backwards",
        from_next[0],
        from_previous[0]
    );
    assert_ne!(from_next, spatial, "fell back to spatial");
    assert!(at_zero.current_noise_curve().is_some());
}

#[test]
fn centre_zero_falls_back_when_every_reading_ahead_is_rejected() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client);
    push_frozen(&mut denoiser, 2 * RADIUS + 1);

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
    let mut denoiser = shifted_denoiser(&client);
    // One more than the ring, so the first stream writes a real record into the slot the second
    // stream's f0 lands in.
    push_grain(&mut denoiser, total_frames + 1);
    denoiser.reset_stream_state();
    push_frozen(&mut denoiser, total_frames);

    denoiser.submit_machinery(0).unwrap().unwrap();

    let temporal_only = denoiser.current_sigmas_temporal_only();
    let low_unboosted = denoiser.current_sigmas_low_unboosted();
    assert_eq!(temporal_only, low_unboosted);
}

#[test]
fn fill_ring_with_last_frame_makes_a_short_stream_ready() {
    let client = make_client();
    let mut denoiser = shifted_denoiser(&client);
    push_grain(&mut denoiser, 2);

    denoiser.fill_ring_with_last_frame().expect("fill ring");

    assert!(denoiser.window_ready());
    assert_eq!(denoiser.real_pushes(), 2);
}

#[test]
fn nlm_mode_keeps_the_leading_copies() {
    let client = make_client();
    let params = edge_params();
    let mut denoiser = NlmDenoiser::new(&client, params, SIZE, SIZE);

    push_grain(&mut denoiser, 1);

    assert_eq!(denoiser.frames_loaded_for_test(), 1 + RADIUS as usize);
}
