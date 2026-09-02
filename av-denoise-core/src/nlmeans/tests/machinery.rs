//! `submit_machinery` and `flush_step_machinery` run the NLM denoiser's
//! ring, motion, and confidence machinery without launching any NLM
//! denoising kernel, so a separate collaborative stage can read the same
//! ring, motion fields, and confidence scores the NLM path builds.
//!
//! These tests pin that the returned [`RingView`] geometry and content
//! line up with what a real, non-trivial motion sequence produces.

use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::motion::neighbour_idx_for_k;
use crate::nlmeans::*;

const RADIUS: u32 = 2;
const SIZE: u32 = 128;

/// A single `size x size` noisy world pattern, read at a shifting
/// horizontal offset and clamped at the edges, so a sequence built from
/// increasing `shift` values translates the same content one pixel to
/// the right per frame.
///
/// A flat gradient (varying only along x) was tried first and rejected.
/// Its SAD is identical at every vertical candidate offset, the same
/// degeneracy `block_match.rs`'s tie-break comments describe for a
/// uniform block, and the fine pass's tie-break resolves that to
/// whatever the coarse pass seeded rather than to zero, letting a wrong
/// vertical offset slip through the "within 1 px" assertion below
/// undetected (confirmed empirically while writing this test). Dense 2D
/// noise gives every candidate a distinct score, so the block match has
/// a genuine, unambiguous minimum at the planted shift.
fn translating_frame(size: u32, shift: i32) -> Vec<f32> {
    let world = noisy_copy(size, 0.5, 0.2, 777);
    let mut frame = vec![0.0f32; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let sx = (x as i32 - shift).clamp(0, size as i32 - 1) as u32;
            frame[(y * size + x) as usize] = world[(y * size + sx) as usize];
        }
    }
    frame
}

fn machinery_params() -> NlmParams {
    NlmParams {
        temporal_radius: RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: DEFAULT_BLKSIZE_FOR_TEST,
            overlap: DEFAULT_OVERLAP_FOR_TEST,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: Some(HqParams::with_sigma(4.0 / 255.0)),
    }
}

// Mirrors `motion::DEFAULT_BLKSIZE`/`DEFAULT_OVERLAP`, spelled out locally
// so this file does not need to reach into the `motion` module just for
// two constants already fixed by the geometry the assertions below
// assume (`step = blksize - overlap = 8`).
const DEFAULT_BLKSIZE_FOR_TEST: u32 = 16;
const DEFAULT_OVERLAP_FOR_TEST: u32 = 8;

/// Pushes `2 * RADIUS + 1` frames of a one-pixel-per-frame translating
/// world, exactly filling the temporal window, and returns the built
/// denoiser.
fn push_translating_sequence(client: &ComputeClient<R>) -> NlmDenoiser<R> {
    let mut d = NlmDenoiser::<R>::new(client, machinery_params(), SIZE, SIZE);
    let total_frames = 2 * RADIUS + 1;
    for n in 0..total_frames {
        let frame = translating_frame(SIZE, n as i32);
        d.push_frame(&frame);
    }
    d
}

#[test]
fn submit_machinery_reports_ring_view_with_correct_motion_and_confidence() {
    let client = make_client();
    let mut d = push_translating_sequence(&client);

    let view = d
        .submit_machinery()
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full, submit_machinery should report Some");

    // The centre slot differs from every neighbour slot. With exactly
    // `2 * RADIUS + 1` pushes into a same-sized ring, every frame landed
    // in its own distinct physical slot, so this also confirms the ring
    // never doubled a slot up.
    for &slot in &view.neighbour_slots {
        assert_ne!(
            slot, view.centre_slot,
            "a neighbour slot must never equal the centre slot"
        );
    }

    assert_eq!(
        view.neighbour_slots.len(),
        (2 * RADIUS) as usize,
        "one neighbour slot per non-zero k in -RADIUS..=RADIUS"
    );

    let mc = d.motion_ctx();
    let bx = (64 / mc.step).min(mc.blocks_x - 1);
    let by = (64 / mc.step).min(mc.blocks_y - 1);

    let nidx = neighbour_idx_for_k(RADIUS, 1);
    let mv_idx = (nidx * view.mv_stride + (by * mc.blocks_x + bx) * 2) as usize;
    let mv_bytes = d
        .compute_client()
        .read_one(view.mv_field.clone())
        .expect("mv_field readback failed");
    let mv = i32::from_bytes(&mv_bytes);

    // The sequence translates by exactly one pixel per frame, so the
    // immediate forward neighbour (k = 1) moved by exactly (1, 0)
    // relative to the centre.
    assert!(
        (mv[mv_idx] - 1).abs() <= 1,
        "expected mv.x within 1px of the planted shift of 1, got {}",
        mv[mv_idx]
    );
    assert!(
        mv[mv_idx + 1].abs() <= 1,
        "expected mv.y within 1px of the planted shift of 0, got {}",
        mv[mv_idx + 1]
    );

    let conf_idx = (nidx * view.conf_stride + (by * mc.blocks_x + bx)) as usize;
    let conf_bytes = d
        .compute_client()
        .read_one(view.confidence.clone())
        .expect("confidence readback failed");
    let confidence = f32::from_bytes(&conf_bytes)[conf_idx];

    assert!(
        confidence.is_finite() && (0.0..=1.0).contains(&confidence),
        "confidence must be finite and in [0, 1], got {confidence}"
    );
    assert!(
        confidence > 0.5,
        "clean translating content should match with confidence above 0.5, got {confidence}"
    );
}

/// A window that has not filled yet reports `None`, the same convention
/// `denoise_submit_gpu` uses.
#[test]
fn submit_machinery_none_while_window_is_filling() {
    let client = make_client();
    let mut d = NlmDenoiser::<R>::new(&client, machinery_params(), SIZE, SIZE);

    // Fewer than `2 * RADIUS + 1` pushes, so the window never fills.
    for n in 0..RADIUS {
        let frame = translating_frame(SIZE, n as i32);
        d.push_frame(&frame);
    }

    let result = d.submit_machinery().expect("submit_machinery dispatch failed");
    assert!(
        result.is_none(),
        "a partially-filled window must report None, the same as denoise_submit_gpu"
    );
}

/// `flush_step_machinery` drains the trailing frames the same way
/// `flush_step_gpu` does, minus the NLM launches, reporting `Some` at
/// every step once the window has ever been full.
#[test]
fn flush_step_machinery_drains_the_tail() {
    let client = make_client();
    let mut d = push_translating_sequence(&client);

    // Consume the one output the fully-loaded window already owes,
    // mirroring how a real caller drains `submit_machinery` during
    // pushing before it ever reaches `flush`.
    d.submit_machinery()
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full, submit_machinery should report Some");

    let target = d.flush_target();
    assert_eq!(
        target, RADIUS as usize,
        "flush_target should ask for exactly RADIUS trailing frames"
    );

    for _ in 0..target {
        let view = d
            .flush_step_machinery()
            .expect("flush_step_machinery dispatch failed")
            .expect("every flush step past the initial fill should report Some");
        assert_eq!(view.neighbour_slots.len(), (2 * RADIUS) as usize);
    }
}

/// Priming a whole window with [`Denoiser::push_frame_priming`], then
/// submitting only the last real push, must produce the same frame the
/// streaming path emits for the window's centre. Filling the window this
/// way is what a caller with random-order access to a fixed window, such
/// as a VapourSynth plugin, needs to reseed on every frame request
/// instead of pushing one frame at a time in order.
#[cfg(feature = "vulkan")]
#[test]
fn priming_pushes_then_one_submit_matches_the_streaming_centre() {
    let r = 2u32;
    let window: Vec<Vec<f32>> = (0..(2 * r + 1) as usize).map(|i| ramp_frame(64, 64, i)).collect();

    let mut windowed = test_denoiser(r, 64, 64);
    for frame in &window[..(2 * r) as usize] {
        windowed.push_frame_priming(frame).unwrap();
    }
    windowed.push_frame(&window[(2 * r) as usize]).unwrap();
    let got = windowed.recv_frame().unwrap().expect("one frame");

    let mut streamed = test_denoiser(r, 64, 64);
    let mut emitted = Vec::new();
    for frame in &window {
        streamed.push_frame(frame).unwrap();
        if let Some(out) = streamed.recv_frame().unwrap() {
            emitted.push(out);
        }
    }

    assert_eq!(emitted.len(), (r + 1) as usize);
    assert_eq!(got, emitted[r as usize]);
}

#[cfg(feature = "vulkan")]
#[test]
fn try_recv_frame_returns_none_when_nothing_is_in_flight() {
    let mut d = test_denoiser(2, 64, 64);
    assert_eq!(d.try_recv_frame().unwrap(), None);
}

#[cfg(feature = "vulkan")]
#[test]
fn try_recv_frame_observes_a_landed_readback_within_a_bounded_poll() {
    // A poll count is the wrong proxy for the wall-clock interval this
    // test needs to cover (cold pipeline compile plus dispatch plus
    // readback), since a faster CPU makes each poll cheaper and so
    // needs *more* of them for the same GPU latency. A deadline covers
    // both a slow GPU and a fast CPU the same way.
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

    let r = 2u32;
    // `temporal_radius + 1` pushes is exactly enough to prime the window
    // and submit one denoise, leaving exactly one readback in flight.
    let window: Vec<Vec<f32>> = (0..(r + 1) as usize).map(|i| ramp_frame(64, 64, i)).collect();

    let mut polled = test_denoiser(r, 64, 64);
    for frame in &window {
        polled.push_frame(frame).unwrap();
    }

    let start = std::time::Instant::now();
    let mut got = None;
    let mut polls = 0;
    while start.elapsed() < DEADLINE {
        polls += 1;
        if let Some(frame) = polled.try_recv_frame().unwrap() {
            got = Some(frame);
            break;
        }
    }
    let got = got.unwrap_or_else(|| panic!("readback never landed within {DEADLINE:?} ({polls} polls)"));
    eprintln!(
        "try_recv_frame landed after {polls} poll(s), {:?}",
        start.elapsed()
    );

    let mut blocking = test_denoiser(r, 64, 64);
    for frame in &window {
        blocking.push_frame(frame).unwrap();
    }
    let expected = blocking
        .recv_frame()
        .unwrap()
        .expect("blocking denoiser should have a frame ready");

    assert_eq!(got, expected);
}
