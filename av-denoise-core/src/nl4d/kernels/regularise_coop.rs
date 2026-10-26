use cubecl::prelude::*;

use super::regularise::{
    REGULARISE_CANDIDATES,
    candidate_cost,
    gather_candidates,
    load_centre_tile,
    write_winner,
};
use crate::noise::DERIVED_SAD_FLOOR;

/// Clamps a coordinate into `0..limit` without a branch.
#[cube]
fn clamp_index(value: i32, limit: i32) -> i32 {
    let floored = i32::max(value, 0i32);
    i32::min(floored, limit - 1i32)
}

/// Re-scores each block's motion vector with every thread of the cube sharing every candidate's SAD.
///
/// It picks the same winner as [nl4d_mv_regularise](crate::nl4d::kernels::nl4d_mv_regularise) up
/// to rounding. Launch one cube of `threads` threads along x per block per neighbour, where
/// `threads` is a power of two of at least 8. Each thread sums the absolute differences of every
/// `threads`-th pixel of the block for all seven candidates, so neighbouring threads read
/// neighbouring pixels. The partial sums meet in shared memory through a tree with one barrier per
/// level. Thread 0 then costs the candidates and writes the winner and its confidence, scored
/// against the SAD floor of the derived noise values `noise`.
///
/// Every frame is a full-resolution luma slot of `pyramid`, `slot_pixels` values apart. The
/// neighbour runs on z. Neighbour `n` re-scores the field `n * mv_stride` values into `mv_in`
/// against slot `neighbour_slots[n]`, and writes `n * mv_stride` values into `mv_out` and
/// `n * conf_stride` values into `confidence_out`.
///
/// Candidates whose differences match pixel for pixel tie exactly, because every candidate goes
/// through the same tree.
#[cube(launch_unchecked)]
pub fn nl4d_mv_regularise_coop(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_in: &[i32],
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    centre_slot: u32,
    lambda_pixel: f32,
    noise: &[f32],
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] threads: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let neighbour_index = CUBE_POS_Z;
    let block = block_row * blocks_x + block_col;
    let thread_id = UNIT_POS_X;

    let neighbour_slot = neighbour_slots[neighbour_index as usize];
    let centre_base = centre_slot * slot_pixels;
    let neighbour_base = neighbour_slot * slot_pixels;
    let mv_base = neighbour_index * mv_stride;
    let conf_base = neighbour_index * conf_stride;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = Shared::<[f32]>::new_slice(block_pixels as usize);
    let mut candidates = Shared::<[i32]>::new_slice(comptime!(2 * REGULARISE_CANDIDATES) as usize);
    let mut median = Shared::<[i32]>::new_slice(2usize);
    let mut sad_scratch = Shared::<[f32]>::new_slice(REGULARISE_CANDIDATES as usize);
    let mut cost = Shared::<[f32]>::new_slice(REGULARISE_CANDIDATES as usize);
    let mut partials = Shared::<[f32]>::new_slice(comptime!(REGULARISE_CANDIDATES * threads) as usize);

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    load_centre_tile(
        pyramid,
        &mut centre_smem,
        centre_base,
        block_origin_x,
        block_origin_y,
        width,
        height,
        blksize,
    );

    if thread_id == 0u32 {
        gather_candidates(
            mv_in,
            mv_base,
            &mut candidates,
            &mut median,
            block_col,
            block_row,
            blocks_x,
            blocks_y,
        );
    }

    sync_cube();

    let mut partial = Array::<f32>::new(REGULARISE_CANDIDATES as usize);
    for candidate in 0..REGULARISE_CANDIDATES {
        partial[candidate as usize] = 0.0f32;
    }

    let mut pixel = thread_id;
    while pixel < block_pixels {
        let pixel_x = pixel % blksize;
        let pixel_y = pixel / blksize;
        let centre_x = block_origin_x + pixel_x as i32;
        let centre_y = block_origin_y + pixel_y as i32;
        let centre_val = centre_smem[pixel as usize];

        for candidate in 0..REGULARISE_CANDIDATES {
            let mvx = candidates[(candidate * 2u32) as usize];
            let mvy = candidates[(candidate * 2u32 + 1u32) as usize];
            let neighbour_x = clamp_index(centre_x + mvx, width as i32);
            let neighbour_y = clamp_index(centre_y + mvy, height as i32);
            let pixel_index = (neighbour_y * width as i32 + neighbour_x) as u32;
            let neighbour_val = pyramid[(neighbour_base + pixel_index) as usize];
            let diff = centre_val - neighbour_val;
            partial[candidate as usize] += f32::abs(diff);
        }

        pixel += threads;
    }

    for candidate in 0..REGULARISE_CANDIDATES {
        partials[(candidate * threads + thread_id) as usize] = partial[candidate as usize];
    }

    sync_cube();

    // Every thread walks the same strides, so every thread reaches every barrier.
    let half_threads = comptime!(threads / 2u32);
    let mut stride = half_threads.runtime();
    while stride > 0u32 {
        if thread_id < stride {
            for candidate in 0..REGULARISE_CANDIDATES {
                let own = (candidate * threads + thread_id) as usize;
                let other = (candidate * threads + thread_id + stride) as usize;
                let other_sum = partials[other];
                partials[own] += other_sum;
            }
        }

        sync_cube();
        stride /= 2u32;
    }

    if thread_id == 0u32 {
        for candidate in 0..REGULARISE_CANDIDATES {
            let sad = partials[(candidate * threads) as usize];
            let mvx = candidates[(candidate * 2u32) as usize];
            let mvy = candidates[(candidate * 2u32 + 1u32) as usize];
            sad_scratch[candidate as usize] = sad;
            cost[candidate as usize] = candidate_cost(sad, mvx, mvy, &median, lambda_pixel);
        }

        write_winner(
            &candidates,
            &sad_scratch,
            &cost,
            mv_out,
            confidence_out,
            mv_base,
            conf_base,
            block,
            noise[DERIVED_SAD_FLOOR as usize],
            thsad,
        );
    }
}
