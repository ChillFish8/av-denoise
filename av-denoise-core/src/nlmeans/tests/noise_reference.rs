use cubecl::prelude::*;

use crate::nlmeans::kernels::helpers::read_line;
use crate::nlmeans::noise::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTERS,
};

/// The single-thread reduction the production kernel must reproduce.
#[cube(launch_unchecked)]
pub(super) fn reference_noise_partial<N: Size>(
    input: &Array<Vector<f32, N>>,
    partials: &mut Array<f32>,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let threads = comptime!(block_x * block_y);
    let mut scratch = SharedMemory::<f32>::new(comptime!(block_x * block_y * 4) as usize);

    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    let thread_id = UNIT_POS_Y * block_x + UNIT_POS_X;

    let interior = x >= 1 && x < width - 1 && y >= 1 && y < height - 1;

    let mut response = Vector::<f32, N>::empty();
    if interior {
        let centre = read_line(input, x, y, frame, width, height);
        let left = read_line(input, x - 1, y, frame, width, height);
        let right = read_line(input, x + 1, y, frame, width, height);
        let up = read_line(input, x, y - 1, frame, width, height);
        let down = read_line(input, x, y + 1, frame, width, height);
        let up_left = read_line(input, x - 1, y - 1, frame, width, height);
        let up_right = read_line(input, x + 1, y - 1, frame, width, height);
        let down_left = read_line(input, x - 1, y + 1, frame, width, height);
        let down_right = read_line(input, x + 1, y + 1, frame, width, height);
        let four = Vector::<f32, N>::empty().fill(4.0f32);
        let two = Vector::<f32, N>::empty().fill(2.0f32);
        response =
            centre * four - (left + right + up + down) * two + (up_left + up_right + down_left + down_right);
    }

    #[unroll]
    for channel in 0..channels {
        scratch[(thread_id * 4 + channel) as usize] = f32::abs(response[channel as usize]);
    }

    #[unroll]
    for channel in channels..4u32 {
        scratch[(thread_id * 4 + channel) as usize] = 0.0f32;
    }

    sync_cube();

    if thread_id == 0 {
        let cube_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;

        #[unroll]
        for channel in 0..4u32 {
            let mut sum = 0.0f32;
            for t in 0..threads {
                sum += scratch[(t * 4 + channel) as usize];
            }

            partials[(cube_index * 4 + channel) as usize] = sum;
        }
    }
}

/// The single-thread reduction the production kernel must reproduce.
#[cube(launch_unchecked)]
pub(super) fn reference_noise_reduce(
    partials: &Array<f32>,
    results: &mut Array<f32>,
    slot: u32,
    num_partials: u32,
    #[comptime] block: u32,
) {
    let mut scratch = SharedMemory::<f32>::new(comptime!(block * 4) as usize);
    let thread_id = UNIT_POS_X;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = thread_id;
    while i < num_partials {
        sum0 += partials[(i * 4) as usize];
        sum1 += partials[(i * 4 + 1) as usize];
        sum2 += partials[(i * 4 + 2) as usize];
        sum3 += partials[(i * 4 + 3) as usize];
        i += block;
    }

    scratch[(thread_id * 4) as usize] = sum0;
    scratch[(thread_id * 4 + 1) as usize] = sum1;
    scratch[(thread_id * 4 + 2) as usize] = sum2;
    scratch[(thread_id * 4 + 3) as usize] = sum3;

    sync_cube();

    if thread_id == 0 {
        #[unroll]
        for channel in 0..4u32 {
            let mut total = 0.0f32;
            for t in 0..block {
                total += scratch[(t * 4 + channel) as usize];
            }

            results[(slot * 4 + channel) as usize] = total;
        }
    }
}

/// The single-thread reduction the production kernel must reproduce.
#[cube(launch_unchecked)]
pub(super) fn reference_temporal_noise_stats<N: Size>(
    input: &Array<Vector<f32, N>>,
    stats: &mut Array<f32>,
    slot_new: u32,
    slot_prev: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
    #[comptime] block: u32,
    #[comptime] luma_fields: bool,
) {
    // The scratch is sized by `scratch_len`, not `record_len`, or its unwritten tail would enter
    // the reduction as uninitialised memory. The luma branch reuses it for three `threads`-long
    // tensor runs, which fit because `scratch_len` is at least 3.
    let quarter_lanes = comptime!(TEMPORAL_QUARTERS * TEMPORAL_QUARTER_FIELDS);
    let record_len = comptime!(2 * stored_ch + TEMPORAL_QUARTER_BASE + quarter_lanes);
    let scratch_len = comptime!(2 * stored_ch + 1);
    let threads = comptime!(block * block);

    let mut scratch = SharedMemory::<f32>::new(comptime!(threads * scratch_len) as usize);
    let mut d0_tile = SharedMemory::<f32>::new(threads as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let thread_id = local_y * block + local_x;

    let block_origin_x = CUBE_POS_X * block;
    let block_origin_y = CUBE_POS_Y * block;
    let global_x = block_origin_x + local_x;
    let global_y = block_origin_y + local_y;

    let valid = global_x < width && global_y < height;

    // The in-frame extent of this block, so a pair never reaches past its slice of the frame.
    let block_w = u32::min(block, width - block_origin_x);
    let block_h = u32::min(block, height - block_origin_y);

    // These are never read when `luma_fields` is off. Only the luma tiles below cost enough to sit
    // behind the flag.
    let mut new_luma_raw = 0.0f32;
    let mut mean_luma_raw = 0.0f32;

    // Sentinels outside the 0..=1 luma range, so an invalid pixel never wins the min or max
    // reduction. cubecl folds a negative literal `let` initialiser into a plain Rust constant that
    // cannot unify with `f32::min`/`f32::max`, so the negative one is built by a subtraction.
    let mut min_seed = 1.0e30f32;
    let mut max_seed = 1.0e30f32;
    max_seed -= 2.0e30f32;

    let mut residual = Vector::<f32, N>::empty();
    if valid {
        let new_pixel = read_line(input, global_x, global_y, slot_new, width, height);
        let prev_pixel = read_line(input, global_x, global_y, slot_prev, width, height);
        residual = new_pixel - prev_pixel;

        if comptime!(luma_fields) {
            new_luma_raw = new_pixel[0];
            mean_luma_raw = 0.5f32 * (new_pixel[0] + prev_pixel[0]);
            min_seed = new_pixel[0];
            max_seed = new_pixel[0];
        }
    }

    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let d0 = if valid { residual[0] } else { 0.0f32.into() };
    d0_tile[thread_id as usize] = d0;

    #[unroll]
    for channel in 0..stored_ch {
        #[expect(
            clippy::useless_conversion,
            reason = "both branches have to expand to the same cubecl native type, which the \
                      conversion supplies"
        )]
        let lane_residual = if valid {
            residual[channel as usize]
        } else {
            0.0f32.into()
        };
        scratch[(thread_id * scratch_len + channel) as usize] = lane_residual;
        scratch[(thread_id * scratch_len + stored_ch + channel) as usize] = lane_residual * lane_residual;
    }

    sync_cube();

    let pair_valid = valid && local_x + 1 < block_w;
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let lag = if pair_valid {
        d0 * d0_tile[(thread_id + 1) as usize]
    } else {
        0.0f32.into()
    };
    scratch[(thread_id * scratch_len + 2 * stored_ch) as usize] = lag;

    sync_cube();

    if thread_id == 0 {
        let block_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        let out_base = block_index * record_len;

        #[unroll]
        for lane in 0..scratch_len {
            let mut total = 0.0f32;
            for t in 0..threads {
                total += scratch[(t * scratch_len + lane) as usize];
            }

            stats[(out_base + lane) as usize] = total;
        }

        // The quarter lanes are written explicitly so a reader never sees an earlier frame's data.
        if comptime!(!luma_fields) {
            let quarters_start = out_base + 2 * stored_ch + TEMPORAL_QUARTER_BASE;

            #[unroll]
            for lane in 0..quarter_lanes {
                stats[(quarters_start + lane) as usize] = 0.0f32;
            }
        }
    }

    // The branch is comptime, so every thread reaches every barrier inside it.
    if comptime!(luma_fields) {
        // Each quarter's 64 pixels sit in one contiguous segment of the reduction tiles, so one tree
        // reduction folds all four quarters at once.
        let quarter = (local_y / 8u32) * 2u32 + local_x / 8u32;
        let quarter_pos = (local_y % 8u32) * 8u32 + local_x % 8u32;
        let slot = quarter * 64u32 + quarter_pos;

        let mut sum_tile = SharedMemory::<f32>::new(threads as usize);
        let mut min_tile = SharedMemory::<f32>::new(threads as usize);
        let mut max_tile = SharedMemory::<f32>::new(threads as usize);
        let mut residual_tile = SharedMemory::<f32>::new(threads as usize);
        let mut residual_sq_tile = SharedMemory::<f32>::new(threads as usize);
        let mut cells = SharedMemory::<f32>::new(64usize);
        let mut pair_tile = SharedMemory::<f32>::new(64usize);

        sum_tile[slot as usize] = new_luma_raw;
        min_tile[slot as usize] = min_seed;
        max_tile[slot as usize] = max_seed;
        residual_tile[slot as usize] = d0;
        residual_sq_tile[slot as usize] = d0 * d0;

        // `d0_tile` is reused for the temporal mean, because a separate tile lowers occupancy enough
        // to cost about 40% of this variant's throughput.
        d0_tile[thread_id as usize] = mean_luma_raw;

        sync_cube();

        // Each 2x2 window inside its quarter and the frame adds one gradient of the temporal mean to
        // the structure tensor. `scratch` holds the three sums, one `threads`-long run each.
        let quarter_col = local_x % 8u32;
        let quarter_row = local_y % 8u32;
        let window_in_quarter = quarter_col < 7u32 && quarter_row < 7u32;
        let window_in_frame = global_x + 1u32 < width && global_y + 1u32 < height;
        let mut grad_x = 0.0f32;
        let mut grad_y = 0.0f32;
        if window_in_quarter && window_in_frame {
            let top_left = d0_tile[thread_id as usize];
            let top_right = d0_tile[(thread_id + 1u32) as usize];
            let bottom_left = d0_tile[(thread_id + block) as usize];
            let bottom_right = d0_tile[(thread_id + block + 1u32) as usize];
            grad_x = 0.5f32 * ((top_right + bottom_right) - (top_left + bottom_left));
            grad_y = 0.5f32 * ((bottom_left + bottom_right) - (top_left + top_right));
        }

        scratch[slot as usize] = grad_x * grad_x;
        scratch[(threads + slot) as usize] = grad_y * grad_y;
        scratch[(2u32 * threads + slot) as usize] = grad_x * grad_y;

        sync_cube();

        // Six halving rounds reduce each 64-slot segment to its first slot. `stride` is a per-round
        // comptime constant because cubecl panics at JIT time on a `mut` seeded from a comptime
        // value. Every thread reaches every `sync_cube()`, the halving only guards the updates.
        #[unroll]
        for step in 0..6u32 {
            let stride = comptime!(32u32 >> step);
            if quarter_pos < stride {
                let here = slot as usize;
                let partner = (slot + stride) as usize;
                sum_tile[here] += sum_tile[partner];
                residual_tile[here] += residual_tile[partner];
                residual_sq_tile[here] += residual_sq_tile[partner];
                min_tile[here] = f32::min(min_tile[here], min_tile[partner]);
                max_tile[here] = f32::max(max_tile[here], max_tile[partner]);

                let yy_here = (threads + slot) as usize;
                let yy_partner = (threads + slot + stride) as usize;
                let xy_here = (2u32 * threads + slot) as usize;
                let xy_partner = (2u32 * threads + slot + stride) as usize;
                scratch[here] += scratch[partner];
                scratch[yy_here] += scratch[yy_partner];
                scratch[xy_here] += scratch[xy_partner];
            }

            sync_cube();
        }

        // 64 threads each average a 2x2 group of `d0_tile` into one cell of a row-major 8x8 grid.
        if thread_id < 64u32 {
            let cell_y = thread_id / 8u32;
            let cell_x = thread_id % 8u32;
            let mut cell = 0.0f32;

            #[unroll]
            for dy in 0..2u32 {
                #[unroll]
                for dx in 0..2u32 {
                    let idx = (2u32 * cell_y + dy) * block + (2u32 * cell_x + dx);
                    cell += d0_tile[idx as usize];
                }
            }

            cells[thread_id as usize] = cell / 4.0f32;
        }

        sync_cube();

        // Each cell scores its right and down neighbour within its quarter, giving 24 pairs per
        // quarter. The scores are stored quarter-major, 16 cells per quarter.
        if thread_id < 64u32 {
            let cell_y = thread_id / 8u32;
            let cell_x = thread_id % 8u32;
            let here = cells[thread_id as usize];
            let mut local_energy = 0.0f32;
            if cell_x % 4u32 != 3u32 {
                let right = cells[(thread_id + 1u32) as usize];
                local_energy += (here - right) * (here - right);
            }

            if cell_y % 4u32 != 3u32 {
                let below = cells[(thread_id + 8u32) as usize];
                local_energy += (here - below) * (here - below);
            }

            let cell_quarter = (cell_y / 4u32) * 2u32 + cell_x / 4u32;
            let cell_pos = (cell_y % 4u32) * 4u32 + cell_x % 4u32;
            pair_tile[(cell_quarter * 16u32 + cell_pos) as usize] = local_energy;
        }

        sync_cube();

        // Four halving rounds reduce each quarter's 16 scores to its first slot.
        #[unroll]
        for step in 0..4u32 {
            let energy_stride = comptime!(8u32 >> step);
            if thread_id < 64u32 && thread_id % 16u32 < energy_stride {
                pair_tile[thread_id as usize] += pair_tile[(thread_id + energy_stride) as usize];
            }

            sync_cube();
        }

        if quarter_pos == 0u32 {
            let quarter_x = quarter % 2u32;
            let quarter_y = quarter / 2u32;
            let full_width = block_w >= (quarter_x + 1u32) * 8u32;
            let full_height = block_h >= (quarter_y + 1u32) * 8u32;

            let mut flatness = 3.0e38f32;
            if full_width && full_height {
                flatness = pair_tile[(quarter * 16u32) as usize] / 24.0f32;
            }

            let block_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
            let quarter_offset = TEMPORAL_QUARTER_BASE + quarter * TEMPORAL_QUARTER_FIELDS;
            let out_base = block_index * record_len + 2 * stored_ch + quarter_offset;
            stats[(out_base + QUARTER_SUM_D) as usize] = residual_tile[slot as usize];
            stats[(out_base + QUARTER_SUM_D2) as usize] = residual_sq_tile[slot as usize];
            stats[(out_base + QUARTER_LUMA_SUM) as usize] = sum_tile[slot as usize];
            stats[(out_base + QUARTER_FLATNESS) as usize] = flatness;
            stats[(out_base + QUARTER_LUMA_MIN) as usize] = min_tile[slot as usize];
            stats[(out_base + QUARTER_LUMA_MAX) as usize] = max_tile[slot as usize];
            stats[(out_base + QUARTER_TENSOR_XX) as usize] = scratch[slot as usize];
            stats[(out_base + QUARTER_TENSOR_YY) as usize] = scratch[(threads + slot) as usize];
            stats[(out_base + QUARTER_TENSOR_XY) as usize] = scratch[(2u32 * threads + slot) as usize];
        }
    }
}
