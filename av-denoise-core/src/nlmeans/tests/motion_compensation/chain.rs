use cubecl::prelude::*;
use cubecl::server::Handle;

use super::frame_shifted_by;
use crate::bench_api::HostIo;
use crate::nlmeans::motion::{
    CHAINED_RADIUS_THRESHOLD,
    MotionCtx,
    mv_field_byte_offset,
    neighbour_idx_for_k,
    pair_byte_offset,
};
use crate::nlmeans::tests::helpers::*;
use crate::nlmeans::*;

const CHAIN_TEST_RADIUS: u32 = 2;
const CHAIN_TEST_SIZE: u32 = 64;

/// Temporal radius large enough to reach `k = 4`.
const K4_RADIUS: u32 = 4;
/// Frame side, larger than any offset the search can reach, so a wrapped shift is never confused
/// with its alias on the other side.
const K4_SIZE: u32 = 128;
/// Diagonal shift per frame in pixels.
///
/// Direct reaches about 12 px at this geometry, 8 px from the coarse pass plus 4 px from the fine
/// pass. So `k = 1` sits inside its reach and `k = 4` (16 px) does not.
const K4_V: i32 = 4;

fn chained_params(refine_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius: CHAIN_TEST_RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained { refine_radius },
        },
        hq: None,
    }
}

/// Like [frame_shifted_by](crate::nlmeans::tests::motion_compensation::frame_shifted_by) but wraps
/// at the edges, so a growing shift never leaves a clamped border.
fn frame_shifted_wrapped(world: &[f32], width: u32, height: u32, shift_x: i32, shift_y: i32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x as i32 - shift_x).rem_euclid(width as i32) as u32;
            let source_y = (y as i32 - shift_y).rem_euclid(height as i32) as u32;
            frame[(y * width + x) as usize] = world[(source_y * width + source_x) as usize];
        }
    }
    frame
}

/// Pushes a sequence moving `velocity` pixels per frame on both axes through a `Chained` denoiser.
///
/// A pair-ring slot lives for `2 * radius` pushes (see
/// [pair_ring_slot_count](crate::nlmeans::motion::pair_ring_slot_count)), so `1 + 3 * radius`
/// pushes replace every priming duplicate with a real analyse. Two more frames give margin.
fn push_constant_velocity(client: &ComputeClient<R>, radius: u32, velocity: i32) -> NlmDenoiser<R> {
    let width = CHAIN_TEST_SIZE;
    let height = CHAIN_TEST_SIZE;
    let world = noisy_copy(width, 0.5, 0.2, 99);

    let params = chained_params(2);
    let mut denoiser = NlmDenoiser::<R>::new(client, params, width, height);

    let real_pushes = 1 + 3 * radius as i32 + 2;
    for frame_number in 0..real_pushes {
        let shift = frame_number * velocity;
        let frame = frame_shifted_by(&world, width, height, shift, shift);
        denoiser.push_frame(&frame);
    }
    denoiser
}

/// Runs chain compose for offset `k` and reads back the composed vector at the centre block.
fn composed_centre_mv(denoiser: &NlmDenoiser<R>, center_t: u32, k: i32, neighbour_idx: u32) -> (i32, i32) {
    denoiser
        .run_chain_compose(center_t, k, neighbour_idx)
        .expect("chain compose dispatch failed");

    let motion_ctx = MotionCtx::new(
        denoiser.params.motion_compensation,
        denoiser.width,
        denoiser.height,
        denoiser.align,
    )
    .unwrap();
    let mv_field = denoiser
        .mv_field_buf
        .as_ref()
        .expect("mv_field allocated when mc_ctx is Some");
    let offset = mv_field_byte_offset(&motion_ctx, neighbour_idx);
    let sliced = mv_field.clone().offset_start(offset);
    let bytes = denoiser.client.read_one(sliced).expect("mv readback failed");
    let data = i32::from_bytes(&bytes);

    let block_x = motion_ctx.blocks_x / 2;
    let block_y = motion_ctx.blocks_y / 2;
    let mv_index = ((block_y * motion_ctx.blocks_x + block_x) * 2) as usize;
    (data[mv_index], data[mv_index + 1])
}

/// Asserts the `radius` pair-ring writes starting at `ring_head_before` are zero in both directions.
///
/// `ring_head_before` is the head the first write saw, before it advanced. Duplicated slots hold
/// zero motion by definition.
fn assert_pair_ring_zero_from(denoiser: &NlmDenoiser<R>, ring_head_before: i32, radius: u32) {
    let motion_ctx = MotionCtx::new(
        denoiser.params.motion_compensation,
        denoiser.width,
        denoiser.height,
        denoiser.align,
    )
    .unwrap();
    let pair_ring = denoiser
        .pair_ring_buf
        .as_ref()
        .expect("pair_ring allocated when Chained is active");
    let pair_ring_slots = 2 * radius as i32;
    let direction_len = motion_ctx.pair_direction_len() as usize;

    for i in 0..radius as i32 {
        let slot = (ring_head_before + i).rem_euclid(pair_ring_slots) as u32;
        for direction in 0..2u32 {
            let offset = pair_byte_offset(&motion_ctx, slot, direction);
            let sliced = pair_ring.clone().offset_start(offset);
            let bytes = denoiser
                .client
                .read_one(sliced)
                .expect("pair ring readback failed");
            let data = i32::from_bytes(&bytes);
            assert!(
                data[..direction_len].iter().all(|&value| value == 0),
                "duplicate pair slot {slot} direction {direction} should be zero-filled, got {:?}",
                &data[..direction_len],
            );
        }
    }
}

#[test]
fn chain_compose_zero_motion_gives_zero_mv() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let denoiser = push_constant_velocity(&client, radius, 0);

    for k in 1..=radius as i32 {
        let forward_idx = neighbour_idx_for_k(radius, k);
        let forward_mv = composed_centre_mv(&denoiser, radius, k, forward_idx);
        assert_eq!(
            forward_mv,
            (0, 0),
            "forward k={k} should compose to zero motion on a static sequence"
        );

        let backward_idx = neighbour_idx_for_k(radius, -k);
        let backward_mv = composed_centre_mv(&denoiser, radius, -k, backward_idx);
        assert_eq!(
            backward_mv,
            (0, 0),
            "backward k={k} should compose to zero motion on a static sequence"
        );
    }
}

/// A velocity of 1 is half a pixel at the coarse level, ambiguous enough for the coarse pass to
/// lock onto the wrong candidate, so this uses 2.
#[test]
fn chain_compose_constant_velocity_matches_k_times_v() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let velocity = 2;
    let denoiser = push_constant_velocity(&client, radius, velocity);

    for k in 1..=radius as i32 {
        let forward_idx = neighbour_idx_for_k(radius, k);
        let forward_mv = composed_centre_mv(&denoiser, radius, k, forward_idx);
        assert_eq!(
            forward_mv,
            (k * velocity, k * velocity),
            "forward k={k} should compose to exactly k*v = ({}, {})",
            k * velocity,
            k * velocity
        );
    }
}

/// From centre 0 the walk reaches `k = 2 * radius`, the far end of the ring, and lands in the last
/// neighbour slot.
#[test]
fn chain_compose_reaches_twice_the_radius_from_the_ring_start() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let velocity = 2;
    let denoiser = push_constant_velocity(&client, radius, velocity);
    let far = 2 * radius as i32;

    let composed = composed_centre_mv(&denoiser, 0, far, 2 * radius - 1);

    assert_eq!(composed, (far * velocity, far * velocity));
}

#[test]
fn chain_compose_backward_direction_matches_negative_k_times_v() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let velocity = 2;
    let denoiser = push_constant_velocity(&client, radius, velocity);

    for k in 1..=radius as i32 {
        let backward_idx = neighbour_idx_for_k(radius, -k);
        let backward_mv = composed_centre_mv(&denoiser, radius, -k, backward_idx);
        assert_eq!(
            backward_mv,
            (-k * velocity, -k * velocity),
            "backward k={k} should compose to exactly -k*v = ({}, {})",
            -k * velocity,
            -k * velocity
        );
    }
}

/// Moving frames are pushed between priming and flush, so the check cannot pass on a still-zeroed
/// buffer.
#[test]
fn chain_compose_duplicated_slot_pairs_are_zero_filled() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let width = CHAIN_TEST_SIZE;
    let height = CHAIN_TEST_SIZE;
    let world = noisy_copy(width, 0.5, 0.2, 7);

    let params = chained_params(2);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    // The first push has no older partner, so the `radius` primed duplicates that follow fill the
    // first gap. Their pair writes start with `ring_head` at 1.
    denoiser.push_frame(&world);
    assert_pair_ring_zero_from(&denoiser, 1, radius);

    // Real motion in every pair slot makes the flush's zero-fill distinguishable.
    for frame_number in 1..=(3 * radius) {
        let shift = frame_number as i32;
        let frame = frame_shifted_by(&world, width, height, shift, shift);
        denoiser.push_frame(&frame);
    }

    let ring_head_before_flush = denoiser.ring_head as i32;
    denoiser.flush(|_| {}).expect("flush failed");
    assert_pair_ring_zero_from(&denoiser, ring_head_before_flush, radius);
}

fn auto_params(temporal_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        },
        hq: None,
    }
}

#[test]
fn auto_estimation_at_high_radius_allocates_pair_ring() {
    let client = make_client();
    let params = auto_params(CHAINED_RADIUS_THRESHOLD);

    let denoiser = NlmDenoiser::<R>::new(&client, params, 32, 32);

    assert!(
        denoiser.pair_ring_buf.is_some(),
        "Auto at radius {CHAINED_RADIUS_THRESHOLD} (>= CHAINED_RADIUS_THRESHOLD) should \
         resolve to Chained and allocate the pair ring"
    );
}

#[test]
fn auto_estimation_at_low_radius_does_not_allocate_pair_ring() {
    let client = make_client();
    let params = auto_params(CHAINED_RADIUS_THRESHOLD - 1);

    let denoiser = NlmDenoiser::<R>::new(&client, params, 32, 32);

    assert!(
        denoiser.pair_ring_buf.is_none(),
        "Auto at radius {} (< CHAINED_RADIUS_THRESHOLD) should resolve to Direct \
         and skip the pair ring",
        CHAINED_RADIUS_THRESHOLD - 1
    );
}

fn k4_params(estimation: MotionEstimation) -> NlmParams {
    NlmParams {
        temporal_radius: K4_RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        },
        hq: None,
    }
}

/// Reads frame `slot` from a ring buffer of `height * width * stored_channels` f32 frames.
fn read_frame_slot(
    client: &ComputeClient<R>,
    buf: &Handle,
    slot: u32,
    width: u32,
    height: u32,
    stored_channels: u32,
) -> Vec<f32> {
    let frame_size = (width * height * stored_channels) as usize;
    let byte_offset = (slot as u64) * (frame_size as u64) * (size_of::<f32>() as u64);
    let sliced = buf.clone().offset_start(byte_offset);
    let bytes = client.read_one(sliced).expect("frame readback failed");
    f32::from_bytes(&bytes)[..frame_size].to_vec()
}

/// Returns the mean absolute residual between the centre frame and the forward `k` neighbour's
/// warped copy, over a wrapped sequence moving `K4_V` pixels per frame.
///
/// `2 * K4_RADIUS + 4` pushes clear the leading priming duplicates, which need more than
/// `2 * radius + 1` real pushes, with a small margin.
fn k4_compensated_residual(estimation: MotionEstimation, k: i32) -> f32 {
    let client = make_client();
    let width = K4_SIZE;
    let height = K4_SIZE;
    let world = noisy_copy(width, 0.5, 0.2, 55);

    let params = k4_params(estimation);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    let real_pushes = 2 * K4_RADIUS as i32 + 4;
    for frame_number in 0..real_pushes {
        let shift = frame_number * K4_V;
        let frame = frame_shifted_wrapped(&world, width, height, shift, shift);
        denoiser.push_frame(&frame);
    }
    denoiser.denoise().unwrap();

    let radius = denoiser.params.temporal_radius;
    let stored_channels = denoiser.params.channels.storage_count();
    let centre_slot = denoiser.phys_frame(radius as i32);
    let neighbour_slot = denoiser.phys_frame(radius as i32 + k);
    let compensated = denoiser
        .compensated_input_buf
        .as_ref()
        .expect("compensated buf allocated when MC is active");

    let centre_frame = read_frame_slot(
        &denoiser.client,
        &denoiser.input_buf,
        centre_slot,
        width,
        height,
        stored_channels,
    );
    let warped = read_frame_slot(
        &denoiser.client,
        compensated,
        neighbour_slot,
        width,
        height,
        stored_channels,
    );

    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            sum += (centre_frame[index] - warped[index]).abs();
            count += 1;
        }
    }
    sum / count as f32
}

/// At `k = 4` the 16 px motion is beyond direct's 12 px reach, while chained composes four exact
/// steps and only needs a refine radius of 2.
#[test]
fn chained_beats_direct_at_k4_beyond_direct_window() {
    let direct_residual = k4_compensated_residual(MotionEstimation::Direct, 4);
    let chained_residual = k4_compensated_residual(MotionEstimation::Chained { refine_radius: 2 }, 4);

    assert!(
        direct_residual > 0.02,
        "expected direct's k=4 match to show a real misalignment residual \
         (window reach ≈12px, true motion 16px), got {direct_residual}"
    );
    assert!(
        chained_residual < direct_residual * 0.5,
        "chained's composed+refined k=4 alignment should beat direct's by a \
         wide margin: chained={chained_residual}, direct={direct_residual}"
    );
}

/// Pins the `k = 4` result on the window size rather than chained always winning.
#[test]
fn direct_already_aligns_at_k1_inside_its_window() {
    let direct_residual = k4_compensated_residual(MotionEstimation::Direct, 1);
    assert!(
        direct_residual < 0.02,
        "direct should align cleanly at k=1 (motion {K4_V}px, well inside its ~12px reach), got {direct_residual}"
    );
}
