use cubecl::prelude::*;

use super::{LIST_CAPACITY, PassBlock, clamp_coord};

/// Sums the SADs of the listed vectors from entry `first` up to `count` over the block into
/// `sads`, one row of one vector per thread at a time.
///
/// Each row sums its pixels left to right into `row_sums`, then one thread per vector adds its rows
/// top to bottom. The order is the same for every thread count, so every layout gives the same
/// sums. An empty range still reaches the barrier. Every thread must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the scoring reads"
)]
pub(super) fn score_serial(
    pyramid: &[f32],
    centre_smem: &Shared<[f32]>,
    list: &Shared<[i32]>,
    sads: &mut Shared<[f32]>,
    row_sums: &mut Shared<[f32]>,
    first: u32,
    count: u32,
    block: &PassBlock,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
) {
    let rows = (count - first) * blksize;
    let mut item = block.thread_id;
    while item < rows {
        let entry = first + item / blksize;
        let iy = item % blksize;
        let mvx = list[(entry * 2u32) as usize];
        let mvy = list[(entry * 2u32 + 1u32) as usize];
        let centre_y = block.origin_y + iy as i32;
        let neighbour_y = clamp_coord(centre_y + mvy, height as i32);
        let row_base = block.neighbour_base as i32 + neighbour_y * width as i32;

        let mut row_sum = 0.0f32;
        #[unroll]
        for ix in 0..blksize {
            let centre_x = block.origin_x + ix as i32;
            let centre_val = centre_smem[(iy * blksize + ix) as usize];
            let neighbour_x = clamp_coord(centre_x + mvx, width as i32);
            let diff = centre_val - pyramid[(row_base + neighbour_x) as usize];
            let abs_diff = if diff < 0.0f32 { -diff } else { diff };
            row_sum += abs_diff;
        }

        row_sums[item as usize] = row_sum;
        item += block.threads;
    }

    sync_cube();

    let mut entry = first + block.thread_id;
    while entry < count {
        let mut sad = 0.0f32;
        for iy in 0..blksize {
            sad += row_sums[((entry - first) * blksize + iy) as usize];
        }

        sads[entry as usize] = sad;
        entry += block.threads;
    }
}

/// Sums the SADs of the listed vectors from entry `first` up to `count` over the block into
/// `sads`, with every thread sharing every vector.
///
/// Each of `threads` threads sums every `threads`-th pixel, and the partial sums meet through a tree
/// in `partials` with one barrier per level. An empty range still reaches every barrier. Every
/// thread must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the scoring reads"
)]
pub(super) fn score_cooperative(
    pyramid: &[f32],
    centre_smem: &Shared<[f32]>,
    list: &Shared<[i32]>,
    sads: &mut Shared<[f32]>,
    partials: &mut Shared<[f32]>,
    first: u32,
    count: u32,
    block: &PassBlock,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] threads: u32,
) {
    let thread_id = block.thread_id;
    let block_pixels = comptime!(blksize * blksize);
    let mut partial = Array::<f32>::new(LIST_CAPACITY as usize);
    for entry in 0..LIST_CAPACITY {
        partial[entry as usize] = 0.0f32;
    }

    let mut pixel = thread_id;
    while pixel < block_pixels {
        let pixel_x = pixel % blksize;
        let pixel_y = pixel / blksize;
        let centre_x = block.origin_x + pixel_x as i32;
        let centre_y = block.origin_y + pixel_y as i32;
        let centre_val = centre_smem[pixel as usize];

        let mut entry = first;
        while entry < count {
            let mvx = list[(entry * 2u32) as usize];
            let mvy = list[(entry * 2u32 + 1u32) as usize];
            let neighbour_x = clamp_coord(centre_x + mvx, width as i32);
            let neighbour_y = clamp_coord(centre_y + mvy, height as i32);
            let pixel_index = (neighbour_y * width as i32 + neighbour_x) as u32;
            let neighbour_val = pyramid[(block.neighbour_base + pixel_index) as usize];
            let diff = centre_val - neighbour_val;
            partial[entry as usize] += f32::abs(diff);
            entry += 1u32;
        }

        pixel += threads;
    }

    let mut entry = first;
    while entry < count {
        partials[(entry * threads + thread_id) as usize] = partial[entry as usize];
        entry += 1u32;
    }

    sync_cube();

    // Every thread walks the same strides, so every thread reaches every barrier.
    let half_threads = comptime!(threads / 2u32);
    let mut stride = half_threads.runtime();
    while stride > 0u32 {
        if thread_id < stride {
            let mut entry = first;
            while entry < count {
                let own = (entry * threads + thread_id) as usize;
                let other = (entry * threads + thread_id + stride) as usize;
                let other_sum = partials[other];
                partials[own] += other_sum;
                entry += 1u32;
            }
        }

        sync_cube();
        stride /= 2u32;
    }

    if thread_id == 0u32 {
        let mut entry = first;
        while entry < count {
            sads[entry as usize] = partials[(entry * threads) as usize];
            entry += 1u32;
        }
    }
}
