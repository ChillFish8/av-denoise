use cubecl::prelude::*;

use super::{frame_shifted_by, shift_clamped};
use crate::bench_api::HostIo;
use crate::nlmeans::motion::{
    DEFAULT_BLKSIZE,
    DEFAULT_OVERLAP,
    DEFAULT_SEARCH_RADIUS,
    MotionCtx,
    mv_field_byte_offset,
    neighbour_idx_for_k,
};
use crate::nlmeans::tests::helpers::*;
use crate::nlmeans::*;

/// Builds a frame from two independent halves, each shifted within its own half.
///
/// Clamping per half keeps a block deep inside one half from ever depending on the other.
fn split_half_frame(
    width: u32,
    height: u32,
    half: u32,
    left: &[f32],
    right: &[f32],
    left_shift: i32,
    right_shift: i32,
) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if x < half {
                let left_x = shift_clamped(x as i32, left_shift, half as i32) as u32;
                frame[index] = left[(y * half + left_x) as usize];
            } else {
                let right_x = shift_clamped((x - half) as i32, right_shift, half as i32) as u32;
                frame[index] = right[(y * half + right_x) as usize];
            }
        }
    }
    frame
}

/// Pushes `base` twice then `neighbour` through a `Direct` denoiser and reads back the forward
/// neighbour's motion field.
///
/// The second push is the centre frame, so the field matches `base` against `neighbour`.
fn direct_mv_field_for_forward_neighbour(
    mode: MotionCompensationMode,
    width: u32,
    height: u32,
    base: &[f32],
    neighbour: &[f32],
) -> Vec<i32> {
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: mode,
        hq: None,
    };

    let client = make_client();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(base);
    denoiser.push_frame(base);
    denoiser.push_frame(neighbour);
    denoiser.denoise().unwrap();

    let align = test_align();
    let motion_ctx = MotionCtx::new(mode, width, height, align).unwrap();
    let neighbour_idx = neighbour_idx_for_k(1, 1);
    let mv_field = denoiser
        .mv_field_buf
        .as_ref()
        .expect("mv_field allocated when mc_ctx is Some");
    let offset = mv_field_byte_offset(&motion_ctx, neighbour_idx);
    let sliced = mv_field.clone().offset_start(offset);
    let bytes = denoiser.client.read_one(sliced).expect("mv readback failed");
    i32::from_bytes(&bytes).to_vec()
}

/// At this geometry the coarse and fine grids have the same block count, so seeding must map by
/// position rather than by doubling the index.
///
/// Each half moves by its own amount, and a block seeded from the wrong half lands outside the
/// fine search window. Both shifts are multiples of the pyramid scale, so the coarse level sees an
/// exact half-size copy. Left-half blocks always seed from left-half blocks here, so only the right
/// half shows the defect.
#[test]
fn coarse_seeding_handles_equal_grids() {
    let width = 128u32;
    let height = 64u32;
    let half = 64u32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: 8,
        overlap: 4,
        search_radius: 3,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let align = test_align();
    let motion_ctx = MotionCtx::new(mode, width, height, align).unwrap();

    // Recomputes the coarse grid width to confirm this geometry gives equal grids.
    let coarse_scale = 1u32 << (motion_ctx.pyramid_levels - 1);
    let coarse_step = (motion_ctx.step / coarse_scale).max(1);
    let coarse_width = width / coarse_scale;
    let coarse_blocks_x = coarse_width.div_ceil(coarse_step).max(1);
    assert_eq!(
        coarse_blocks_x, motion_ctx.blocks_x,
        "test premise: this geometry must give equal coarse/fine grids"
    );

    let left = noisy_copy(half, 0.5, 0.2, 201);
    let right = noisy_copy(half, 0.5, 0.2, 202);
    let left_shift = 4i32;
    let right_shift = -4i32;

    let base = split_half_frame(width, height, half, &left, &right, 0, 0);
    let shifted = split_half_frame(width, height, half, &left, &right, left_shift, right_shift);

    let data = direct_mv_field_for_forward_neighbour(mode, width, height, &base, &shifted);

    // Deep inside each half, clear of the boundary at x=64 and the frame edges by well over
    // `search_radius + blksize` (11).
    let block_y = 4u32;
    let left_block_x = 8u32;
    let right_block_x = 24u32;
    let left_index = ((block_y * motion_ctx.blocks_x + left_block_x) * 2) as usize;
    let right_index = ((block_y * motion_ctx.blocks_x + right_block_x) * 2) as usize;

    assert_eq!(
        (data[left_index], data[left_index + 1]),
        (left_shift, 0),
        "left-half block should recover the left half's own motion ({left_shift}, 0), got ({}, {})",
        data[left_index],
        data[left_index + 1],
    );
    assert_eq!(
        (data[right_index], data[right_index + 1]),
        (right_shift, 0),
        "right-half block should recover the right half's own motion ({right_shift}, 0), got \
         ({}, {}); a wrong value here means it was seeded from the wrong (left-half) coarse block",
        data[right_index],
        data[right_index + 1],
    );
}

/// At a step of 1 the coarse step floors to 1, giving a genuine 2:1 fine to coarse grid.
///
/// Position-based seeding must then reduce to plain index doubling, both for uniform motion and
/// for halves moving in opposite directions.
#[test]
fn coarse_seeding_still_correct_at_half_grid() {
    let width = 48u32;
    let height = 16u32;
    let half = 24u32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: 4,
        overlap: 3,
        search_radius: 2,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let align = test_align();
    let motion_ctx = MotionCtx::new(mode, width, height, align).unwrap();

    let coarse_scale = 1u32 << (motion_ctx.pyramid_levels - 1);
    let coarse_step = (motion_ctx.step / coarse_scale).max(1);
    let coarse_width = width / coarse_scale;
    let coarse_height = height / coarse_scale;
    let coarse_blocks_x = coarse_width.div_ceil(coarse_step).max(1);
    let coarse_blocks_y = coarse_height.div_ceil(coarse_step).max(1);
    assert_eq!(
        motion_ctx.step, 1,
        "test premise: step must floor-clamp coarse_step to 1"
    );
    assert_eq!(
        motion_ctx.blocks_x,
        2 * coarse_blocks_x,
        "test premise: this geometry must give a genuine 2:1 fine:coarse ratio in x"
    );
    assert_eq!(
        motion_ctx.blocks_y,
        2 * coarse_blocks_y,
        "test premise: this geometry must give a genuine 2:1 fine:coarse ratio in y"
    );

    let left = noisy_copy(half, 0.5, 0.2, 301);
    let right = noisy_copy(half, 0.5, 0.2, 302);

    // Deep inside each half, clear of the boundary at x=24 and the frame edges by well over
    // `search_radius + blksize` (6).
    let block_y = 6u32;
    let left_block_x = 10u32;
    let right_block_x = 32u32;

    let mv_at = |left_shift: i32, right_shift: i32| -> ((i32, i32), (i32, i32)) {
        let base = split_half_frame(width, height, half, &left, &right, 0, 0);
        let shifted = split_half_frame(width, height, half, &left, &right, left_shift, right_shift);
        let data = direct_mv_field_for_forward_neighbour(mode, width, height, &base, &shifted);
        let left_index = ((block_y * motion_ctx.blocks_x + left_block_x) * 2) as usize;
        let right_index = ((block_y * motion_ctx.blocks_x + right_block_x) * 2) as usize;
        (
            (data[left_index], data[left_index + 1]),
            (data[right_index], data[right_index + 1]),
        )
    };

    // Both halves share one vector here, so any coarse block seeds any fine block correctly.
    let (uniform_left, uniform_right) = mv_at(2, 2);
    assert_eq!(
        uniform_left,
        (2, 0),
        "uniform motion: left block got {uniform_left:?}"
    );
    assert_eq!(
        uniform_right,
        (2, 0),
        "uniform motion: right block got {uniform_right:?}"
    );

    // The halves move oppositely, so a fine block only recovers its own half's motion when it is
    // seeded from the coarse block covering the same region.
    let (varying_left, varying_right) = mv_at(2, -2);
    assert_eq!(
        varying_left,
        (2, 0),
        "varying motion: left block got {varying_left:?}"
    );
    assert_eq!(
        varying_right,
        (-2, 0),
        "varying motion: right block got {varying_right:?}"
    );
}

/// With a single pyramid level the fine pass runs unseeded straight off level 0, so a skipped
/// level 0 extraction leaves the recovered vector unrelated to the shift.
#[test]
fn pyramid_level0_extracted_at_one_level() {
    let width = 64u32;
    let height = 64u32;
    let shift_x = 2i32;
    let shift_y = 1i32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: DEFAULT_SEARCH_RADIUS,
        pyramid_levels: 1,
        estimation: MotionEstimation::Direct,
    };
    let align = test_align();
    let motion_ctx = MotionCtx::new(mode, width, height, align).unwrap();

    let world = noisy_copy(width, 0.5, 0.2, 77);
    let shifted = frame_shifted_by(&world, width, height, shift_x, shift_y);

    let data = direct_mv_field_for_forward_neighbour(mode, width, height, &world, &shifted);

    // An interior block, well clear of the frame edges.
    let block_x = motion_ctx.blocks_x / 2;
    let block_y = motion_ctx.blocks_y / 2;
    let mv_index = ((block_y * motion_ctx.blocks_x + block_x) * 2) as usize;

    assert_eq!(
        (data[mv_index], data[mv_index + 1]),
        (shift_x, shift_y),
        "a clean ({shift_x}, {shift_y}) shift with pyramid_levels=1 should give exactly that MV at an \
         interior block once level-0 luma is actually extracted, got ({}, {})",
        data[mv_index],
        data[mv_index + 1],
    );
}

/// The coarse and fine block counts round up over different widths and steps, so at some sizes
/// the coarse grid ends one fine block short and its last block must reach the trailing edge.
///
/// An unseeded trailing block keeps whatever the motion field already held. The shift escapes the
/// unseeded search window but is reachable through a correct seed. It is negative because a
/// trailing-edge block can only tell apart offsets that pull content in from the interior, so a
/// positive shift there is unrecoverable by any search.
#[test]
fn coarse_seeding_covers_ragged_last_block() {
    let shift = -6i32;
    let mode = MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };

    // A ragged side of 57 is one more than a multiple of the step (8), so the coarse grid ends one
    // block short on that axis. The even side of 64 is an exact multiple. Each case is ragged on
    // one axis only, because two odd sides can never give a pixel count that meets the ring
    // buffers' 32-byte frame-stride alignment.
    let ragged_side = 57u32;
    let even_side = 64u32;

    let build_mv_field = |width: u32, height: u32| -> (MotionCtx, Vec<i32>) {
        let align = test_align();
        let motion_ctx = MotionCtx::new(mode, width, height, align).unwrap();
        let world = make_noisy_gaussian_frame(width, height, 1, 0.5, &[0.2]);
        let shifted = frame_shifted_by(&world, width, height, shift, shift);
        let data = direct_mv_field_for_forward_neighbour(mode, width, height, &world, &shifted);
        (motion_ctx, data)
    };
    let mv_at = |motion_ctx: &MotionCtx, data: &[i32], block_x: u32, block_y: u32| -> (i32, i32) {
        let mv_index = ((block_y * motion_ctx.blocks_x + block_x) * 2) as usize;
        (data[mv_index], data[mv_index + 1])
    };
    // Recomputes the coarse grid to confirm the named axis is ragged and the other is equal.
    let assert_ragged_on = |motion_ctx: &MotionCtx, width: u32, height: u32, ragged_axis_is_x: bool| {
        let coarse_scale = 1u32 << (motion_ctx.pyramid_levels - 1);
        let coarse_step = (motion_ctx.step / coarse_scale).max(1);
        let coarse_blocks_x = (width / coarse_scale).div_ceil(coarse_step).max(1);
        let coarse_blocks_y = (height / coarse_scale).div_ceil(coarse_step).max(1);

        if ragged_axis_is_x {
            assert_eq!(
                width % motion_ctx.step,
                1,
                "test premise: width must be step*k + 1"
            );
            assert_eq!(
                coarse_blocks_x,
                motion_ctx.blocks_x - 1,
                "test premise: ragged coarse grid, one block short in x"
            );
            assert_eq!(
                coarse_blocks_y, motion_ctx.blocks_y,
                "test premise: y axis is an ordinary equal grid here"
            );
        } else {
            assert_eq!(
                height % motion_ctx.step,
                1,
                "test premise: height must be step*k + 1"
            );
            assert_eq!(
                coarse_blocks_y,
                motion_ctx.blocks_y - 1,
                "test premise: ragged coarse grid, one block short in y"
            );
            assert_eq!(
                coarse_blocks_x, motion_ctx.blocks_x,
                "test premise: x axis is an ordinary equal grid here"
            );
        }
    };

    // X-axis case, the last column at a non-edge row.
    let (x_case_ctx, x_case_field) = build_mv_field(ragged_side, even_side);
    assert_ragged_on(&x_case_ctx, ragged_side, even_side, true);
    let middle_x = x_case_ctx.blocks_x / 2;
    let middle_y = x_case_ctx.blocks_y / 2;
    let interior_mv = mv_at(&x_case_ctx, &x_case_field, middle_x, middle_y);
    assert_eq!(
        interior_mv,
        (shift, shift),
        "interior control block (x-axis case) should recover ({shift}, {shift})"
    );
    let last_column_mv = mv_at(&x_case_ctx, &x_case_field, x_case_ctx.blocks_x - 1, middle_y);
    assert_eq!(
        last_column_mv,
        (shift, shift),
        "last-column block (x-axis coverage gap) should recover ({shift}, {shift})"
    );

    // Y-axis case, the last row at a non-edge column, using the transposed frame size.
    let (y_case_ctx, y_case_field) = build_mv_field(even_side, ragged_side);
    assert_ragged_on(&y_case_ctx, even_side, ragged_side, false);
    let middle_x = y_case_ctx.blocks_x / 2;
    let middle_y = y_case_ctx.blocks_y / 2;
    let interior_mv = mv_at(&y_case_ctx, &y_case_field, middle_x, middle_y);
    assert_eq!(
        interior_mv,
        (shift, shift),
        "interior control block (y-axis case) should recover ({shift}, {shift})"
    );
    let last_row_mv = mv_at(&y_case_ctx, &y_case_field, middle_x, y_case_ctx.blocks_y - 1);
    assert_eq!(
        last_row_mv,
        (shift, shift),
        "last-row block (y-axis coverage gap) should recover ({shift}, {shift})"
    );
}
