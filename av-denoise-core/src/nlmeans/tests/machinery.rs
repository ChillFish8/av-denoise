use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::motion::neighbour_idx_for_k;
use crate::nlmeans::*;

const RADIUS: u32 = 2;
const SIZE: u32 = 128;

// The motion module's default block size and overlap, giving the `step = blksize - overlap = 8`
// geometry the assertions assume.
const DEFAULT_BLKSIZE_FOR_TEST: u32 = 16;
const DEFAULT_OVERLAP_FOR_TEST: u32 = 8;

/// A noisy world read at a horizontal offset of `shift` and clamped at the edges.
///
/// Increasing `shift` by one moves the content one pixel right. Dense 2D noise gives every candidate
/// offset a distinct SAD, so the block match has one clear minimum at the planted shift. Content that
/// varies only along x ties every vertical offset, and the tie-break can then let a wrong vertical
/// offset pass the 1 px checks.
fn translating_frame(size: u32, shift: i32) -> Vec<f32> {
    let world = noisy_copy(size, 0.5, 0.2, 777);
    let mut frame = vec![0.0f32; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let source_x = (x as i32 - shift).clamp(0, size as i32 - 1) as u32;
            frame[(y * size + x) as usize] = world[(y * size + source_x) as usize];
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

/// Pushes `2 * RADIUS + 1` frames of a world translating one pixel per frame, exactly filling the
/// window.
fn push_translating_sequence(client: &ComputeClient<R>) -> NlmDenoiser<R> {
    let params = machinery_params();
    let mut denoiser = NlmDenoiser::<R>::new(client, params, SIZE, SIZE);
    let total_frames = 2 * RADIUS + 1;
    for frame_index in 0..total_frames {
        let frame = translating_frame(SIZE, frame_index as i32);
        denoiser.push_frame(&frame);
    }

    denoiser
}

#[test]
fn submit_machinery_reports_ring_view_with_correct_motion_and_confidence() {
    let client = make_client();
    let mut denoiser = push_translating_sequence(&client);

    let view = denoiser
        .submit_machinery(RADIUS)
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full, submit_machinery should report Some");

    // With exactly `2 * RADIUS + 1` pushes into a ring of that size, every frame lands in its own slot,
    // so this also confirms the ring never doubled a slot up.
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

    let motion = denoiser.motion_ctx();
    let block_x = (64 / motion.step).min(motion.blocks_x - 1);
    let block_y = (64 / motion.step).min(motion.blocks_y - 1);

    let neighbour_idx = neighbour_idx_for_k(RADIUS, 1);
    let mv_idx = (neighbour_idx * view.mv_stride + (block_y * motion.blocks_x + block_x) * 2) as usize;
    let mv_field = view.mv_field.clone();
    let mv_bytes = denoiser
        .compute_client()
        .read_one(mv_field)
        .expect("mv_field readback failed");
    let motion_vectors = i32::from_bytes(&mv_bytes);

    // The world shifts one pixel per frame, so the forward neighbour at k = 1 moved by exactly (1, 0).
    assert!(
        (motion_vectors[mv_idx] - 1).abs() <= 1,
        "expected mv.x within 1px of the planted shift of 1, got {}",
        motion_vectors[mv_idx]
    );
    assert!(
        motion_vectors[mv_idx + 1].abs() <= 1,
        "expected mv.y within 1px of the planted shift of 0, got {}",
        motion_vectors[mv_idx + 1]
    );

    let confidence_idx = (neighbour_idx * view.conf_stride + (block_y * motion.blocks_x + block_x)) as usize;
    let confidence_field = view.confidence.clone();
    let confidence_bytes = denoiser
        .compute_client()
        .read_one(confidence_field)
        .expect("confidence readback failed");
    let confidence = f32::from_bytes(&confidence_bytes)[confidence_idx];

    assert!(
        confidence.is_finite() && (0.0..=1.0).contains(&confidence),
        "confidence must be finite and in [0, 1], got {confidence}"
    );
    assert!(
        confidence > 0.5,
        "clean translating content should match with confidence above 0.5, got {confidence}"
    );
}

#[test]
fn submit_machinery_at_centre_zero_lists_every_later_slot() {
    let client = make_client();
    let mut denoiser = push_translating_sequence(&client);
    let total_frames = 2 * RADIUS + 1;

    let view = denoiser
        .submit_machinery(0)
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full");

    let expected: Vec<u32> = (1..total_frames)
        .map(|logical| denoiser.ring_slot(logical))
        .collect();
    let centre_slot = denoiser.ring_slot(0);
    assert_eq!(view.centre_slot, centre_slot);
    assert_eq!(view.neighbour_slots, expected);
}

#[test]
fn submit_machinery_at_the_last_slot_lists_every_earlier_slot() {
    let client = make_client();
    let mut denoiser = push_translating_sequence(&client);
    let last = 2 * RADIUS;

    let view = denoiser
        .submit_machinery(last)
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full");

    let expected: Vec<u32> = (0..last).map(|logical| denoiser.ring_slot(logical)).collect();
    let centre_slot = denoiser.ring_slot(last);
    assert_eq!(view.centre_slot, centre_slot);
    assert_eq!(view.neighbour_slots, expected);
}

/// From centre 0 the neighbour at logical `2 * RADIUS` sits `2 * RADIUS` pixels to the right.
///
/// It is the last neighbour submitted, so it lands at field index `2 * RADIUS - 1`.
#[test]
fn submit_machinery_at_centre_zero_finds_motion_at_the_far_offset() {
    let client = make_client();
    let mut denoiser = push_translating_sequence(&client);
    let far = 2 * RADIUS;

    let view = denoiser
        .submit_machinery(0)
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full");

    let motion = denoiser.motion_ctx();
    let block_x = (64 / motion.step).min(motion.blocks_x - 1);
    let block_y = (64 / motion.step).min(motion.blocks_y - 1);

    let neighbour_idx = far - 1;
    let mv_idx = (neighbour_idx * view.mv_stride + (block_y * motion.blocks_x + block_x) * 2) as usize;
    let mv_field = view.mv_field.clone();
    let mv_bytes = denoiser
        .compute_client()
        .read_one(mv_field)
        .expect("mv_field readback failed");
    let motion_vectors = i32::from_bytes(&mv_bytes);

    assert!(
        (motion_vectors[mv_idx] - far as i32).abs() <= 1,
        "expected mv.x within 1px of the planted shift of {far}, got {}",
        motion_vectors[mv_idx]
    );
}

#[test]
fn ring_view_exposes_the_analysed_pyramid_and_the_noise_floor() {
    let client = make_client();
    let mut denoiser = push_translating_sequence(&client);
    let view = denoiser
        .submit_machinery(RADIUS)
        .expect("submit_machinery dispatch failed")
        .expect("window is exactly full, submit_machinery should report Some");

    let frames = machinery_params().total_frames();
    assert_eq!(view.frame_count, frames);

    // The pyramid holds every level of every slot, so it is at least one full-resolution luma plane
    // per slot.
    let pyramid = view.pyramid.clone();
    let bytes = client.read_one(pyramid).expect("pyramid readback failed");
    let plane = bytes.len() / (frames as usize * size_of::<f32>());
    assert!(plane > 0, "the pyramid must hold at least one plane per slot");
    assert!(denoiser.sad_noise_floor_value() >= 0.0);
}

#[test]
fn submit_machinery_none_while_window_is_filling() {
    let client = make_client();
    let params = machinery_params();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, SIZE, SIZE);

    // Fewer than `2 * RADIUS + 1` pushes, so the window never fills.
    for frame_index in 0..RADIUS {
        let frame = translating_frame(SIZE, frame_index as i32);
        denoiser.push_frame(&frame);
    }

    let result = denoiser
        .submit_machinery(RADIUS)
        .expect("submit_machinery dispatch failed");
    assert!(
        result.is_none(),
        "a partially-filled window must report None, the same as denoise_submit_gpu"
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn priming_pushes_then_one_submit_matches_the_streaming_centre() {
    let radius = 2u32;
    let window: Vec<Vec<f32>> = (0..(2 * radius + 1) as usize)
        .map(|i| ramp_frame(64, 64, i))
        .collect();

    let mut windowed = test_denoiser(radius, 64, 64);
    for frame in &window[..(2 * radius) as usize] {
        windowed.push_frame(frame);
    }

    windowed.push_frame(&window[(2 * radius) as usize]);
    let got = windowed.denoise().unwrap().expect("one frame");

    let mut streamed = test_denoiser(radius, 64, 64);
    let mut emitted = Vec::new();
    for frame in &window {
        streamed.push_frame(frame);

        if let Some(output) = streamed.denoise().unwrap() {
            emitted.push(output);
        }
    }

    assert_eq!(emitted.len(), (radius + 1) as usize);
    assert_eq!(got, emitted[radius as usize]);
}
