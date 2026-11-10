use cubecl::prelude::*;

use super::grain::luma_at;
use crate::nl4d::grain::consts::{
    CELL,
    CLIP_HIGH,
    CLIP_LOW,
    FLAT_RANGE,
    HISTORY_ENTRY_LEN,
    HISTORY_FRAMES,
    LAG_RECORD_LEN,
    LUMA_HIGH,
    LUMA_LOW,
    PERSIST_BLOCK_CELLS,
    PERSIST_CELLS_PER_THREAD,
    PERSIST_THREADS,
    PROJECTIONS,
    REDUCE_THREADS,
};

/// Threads in one projecting cube, one per pixel of a cell.
const THREADS: u32 = CELL * CELL;
/// Halving rounds that reduce [THREADS] values to one.
const PROJECT_ROUNDS: u32 = THREADS.ilog2();
/// Halving rounds that reduce [PERSIST_THREADS] values to one.
const BLOCK_ROUNDS: u32 = PERSIST_THREADS.ilog2();
/// Halving rounds that reduce [REDUCE_THREADS] values to one.
const REDUCE_ROUNDS: u32 = REDUCE_THREADS.ilog2();
/// Bits in one pattern word.
const WORD_BITS: u32 = 32;
/// Pattern words per projection.
const WORDS_PER_PATTERN: u32 = 2;
/// Shared sum lanes, one per projection and one for the denoised luma.
const SUM_LANES: u32 = PROJECTIONS as u32 + 1;
/// Shared range lanes, one for the denoised luma and one for the source luma.
const RANGE_LANES: u32 = 2;
/// Scales a cell's summed squared projections of both frames to one pixel's variance.
const AUTO_SCALE: f32 = 1.0 / (2 * PROJECTIONS as u32 * THREADS) as f32;
/// Scales a cell's summed projection products to one pixel's covariance.
const CROSS_SCALE: f32 = 1.0 / (PROJECTIONS as u32 * THREADS) as f32;

/// The sign of a pixel in a projection, from the pattern's two words.
#[cube]
fn pattern_sign(patterns: &[u32], projection: u32, bit: u32) -> f32 {
    let word_index = projection * WORDS_PER_PATTERN + bit / WORD_BITS;
    let word = patterns[word_index as usize];
    let set = (word >> (bit % WORD_BITS)) & 1u32;
    f32::cast_from(set) * 2.0f32 - 1.0f32
}

/// Writes one completed frame's cell projections and flat flags into history slot `history_slot`.
///
/// Launch one `CELL x CELL` cube per cell. Each projection sums the cell's source luma from ring
/// slot `frame_slot`, adding pixel `y * 8 + x` where that bit of the pattern is set and taking it
/// away where it is clear. `patterns` holds two words per pattern, the low bits first. The flag is
/// 1 when the denoised cell in `output` is flat and mid-luma and no source pixel is near clipping,
/// otherwise 0. Every cell writes its entry.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, ring index or comptime shape the kernel binds"
)]
pub fn grain_project<N: Size>(
    input: &[Vector<f32, N>],
    output: &[f32],
    patterns: &[u32],
    history: &mut [f32],
    frame_slot: u32,
    history_slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let mut sums = Shared::<[f32]>::new_slice((SUM_LANES * THREADS) as usize);
    let mut lows = Shared::<[f32]>::new_slice((RANGE_LANES * THREADS) as usize);
    let mut highs = Shared::<[f32]>::new_slice((RANGE_LANES * THREADS) as usize);

    let cells_x = comptime!(width / CELL);
    let cells = comptime!((width / CELL) * (height / CELL));
    let thread_id = UNIT_POS_Y * CELL + UNIT_POS_X;
    let x = CUBE_POS_X * CELL + UNIT_POS_X;
    let y = CUBE_POS_Y * CELL + UNIT_POS_Y;
    let luma = luma_at(input, x, y, frame_slot, width, height);
    let clean = output[((y * width + x) * channels) as usize];

    #[unroll]
    for projection in 0..PROJECTIONS as u32 {
        let sign = pattern_sign(patterns, projection, thread_id);
        sums[(projection * THREADS + thread_id) as usize] = sign * luma;
    }

    sums[(PROJECTIONS as u32 * THREADS + thread_id) as usize] = clean;
    lows[thread_id as usize] = clean;
    lows[(THREADS + thread_id) as usize] = luma;
    highs[thread_id as usize] = clean;
    highs[(THREADS + thread_id) as usize] = luma;
    sync_cube();

    #[unroll]
    for round in 0..PROJECT_ROUNDS {
        let stride = comptime!(THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for lane in 0..SUM_LANES {
                let here = (lane * THREADS + thread_id) as usize;
                let there = (lane * THREADS + thread_id + stride) as usize;
                sums[here] += sums[there];
            }

            #[unroll]
            for lane in 0..RANGE_LANES {
                let here = (lane * THREADS + thread_id) as usize;
                let there = (lane * THREADS + thread_id + stride) as usize;
                lows[here] = f32::min(lows[here], lows[there]);
                highs[here] = f32::max(highs[here], highs[there]);
            }
        }

        sync_cube();
    }

    if thread_id == 0 {
        let cell = CUBE_POS_Y * cells_x + CUBE_POS_X;
        let base = (history_slot * cells + cell) * HISTORY_ENTRY_LEN as u32;

        #[unroll]
        for projection in 0..PROJECTIONS as u32 {
            history[(base + projection) as usize] = sums[(projection * THREADS) as usize];
        }

        let mean = sums[(PROJECTIONS as u32 * THREADS) as usize] / THREADS as f32;
        let range = highs[0] - lows[0];
        let flat = range < FLAT_RANGE
            && mean > LUMA_LOW
            && mean < LUMA_HIGH
            && lows[THREADS as usize] >= CLIP_LOW
            && highs[THREADS as usize] <= CLIP_HIGH;
        let mut flag = 0.0f32;
        if flat {
            flag = 1.0f32;
        }

        history[(base + PROJECTIONS as u32) as usize] = flag;
    }
}

/// Writes the persistence sums of history slot `now_slot` against each earlier slot, per block of
/// cells, into `partials`.
///
/// Launch a grid of `blocks` by lags, with `PERSIST_THREADS` threads per cube. The cube at
/// `(block, n)` covers the `PERSIST_BLOCK_CELLS` cells from `block * PERSIST_BLOCK_CELLS` at lag
/// `n + 1`. Launch only as many lags as the stream has earlier frames. Each cube writes the cell
/// count, auto sum and cross sum of its block at `(n * blocks + block) * LAG_RECORD_LEN`. A cell
/// counts when it is flagged flat in both slots.
#[cube(launch_unchecked)]
pub fn grain_persist(history: &[f32], partials: &mut [f32], now_slot: u32, cells: u32, blocks: u32) {
    let mut scratch = Shared::<[f32]>::new_slice((LAG_RECORD_LEN as u32 * PERSIST_THREADS) as usize);
    let block = CUBE_POS_X;
    let lag_index = CUBE_POS_Y;
    let thread_id = UNIT_POS_X;
    let earlier_slot = (now_slot + HISTORY_FRAMES - 1 - lag_index) % HISTORY_FRAMES;
    let entry_len = HISTORY_ENTRY_LEN as u32;
    let first_cell = block * PERSIST_BLOCK_CELLS + thread_id;
    let mut cell_count = 0.0f32;
    let mut auto_sum = 0.0f32;
    let mut cross_sum = 0.0f32;

    #[unroll]
    for step in 0..PERSIST_CELLS_PER_THREAD {
        let cell = first_cell + step * PERSIST_THREADS;
        if cell < cells {
            let now_base = (now_slot * cells + cell) * entry_len;
            let earlier_base = (earlier_slot * cells + cell) * entry_len;
            let now_flag = history[(now_base + PROJECTIONS as u32) as usize];
            let earlier_flag = history[(earlier_base + PROJECTIONS as u32) as usize];

            if now_flag * earlier_flag > 0.5f32 {
                let mut squares = 0.0f32;
                let mut products = 0.0f32;

                #[unroll]
                for projection in 0..PROJECTIONS as u32 {
                    let now_value = history[(now_base + projection) as usize];
                    let earlier_value = history[(earlier_base + projection) as usize];
                    squares += now_value * now_value + earlier_value * earlier_value;
                    products += now_value * earlier_value;
                }

                cell_count += 1.0f32;
                auto_sum += squares * AUTO_SCALE;
                cross_sum += products * CROSS_SCALE;
            }
        }
    }

    scratch[thread_id as usize] = cell_count;
    scratch[(PERSIST_THREADS + thread_id) as usize] = auto_sum;
    scratch[(2 * PERSIST_THREADS + thread_id) as usize] = cross_sum;
    sync_cube();

    #[unroll]
    for round in 0..BLOCK_ROUNDS {
        let stride = comptime!(PERSIST_THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for lane in 0..LAG_RECORD_LEN as u32 {
                let here = (lane * PERSIST_THREADS + thread_id) as usize;
                let there = (lane * PERSIST_THREADS + thread_id + stride) as usize;
                scratch[here] += scratch[there];
            }
        }

        sync_cube();
    }

    if thread_id < LAG_RECORD_LEN as u32 {
        let target = (lag_index * blocks + block) * LAG_RECORD_LEN as u32 + thread_id;
        partials[target as usize] = scratch[(thread_id * PERSIST_THREADS) as usize];
    }
}

/// Adds each lag's block partials from [grain_persist] into `record`.
///
/// Launch one cube of `REDUCE_THREADS` threads per lag, as many lags as [grain_persist] ran. Each
/// lag's cell count, auto sum and cross sum add into `record` at `record_offset` plus the lag's
/// offset.
#[cube(launch_unchecked)]
pub fn grain_persist_reduce(partials: &[f32], record: &mut [f32], record_offset: u32, blocks: u32) {
    let mut scratch = Shared::<[f32]>::new_slice((LAG_RECORD_LEN as u32 * REDUCE_THREADS) as usize);
    let lag_index = CUBE_POS_X;
    let thread_id = UNIT_POS_X;
    let lag_base = lag_index * blocks;

    #[unroll]
    for lane in 0..LAG_RECORD_LEN as u32 {
        scratch[(lane * REDUCE_THREADS + thread_id) as usize] = 0.0f32;
    }

    let mut block = thread_id;
    while block < blocks {
        let base = (lag_base + block) * LAG_RECORD_LEN as u32;

        #[unroll]
        for lane in 0..LAG_RECORD_LEN as u32 {
            scratch[(lane * REDUCE_THREADS + thread_id) as usize] += partials[(base + lane) as usize];
        }

        block += REDUCE_THREADS;
    }

    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let stride = comptime!(REDUCE_THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for lane in 0..LAG_RECORD_LEN as u32 {
                let here = (lane * REDUCE_THREADS + thread_id) as usize;
                let there = (lane * REDUCE_THREADS + thread_id + stride) as usize;
                scratch[here] += scratch[there];
            }
        }

        sync_cube();
    }

    if thread_id < LAG_RECORD_LEN as u32 {
        let target = record_offset + lag_index * LAG_RECORD_LEN as u32 + thread_id;
        record[target as usize] += scratch[(thread_id * REDUCE_THREADS) as usize];
    }
}
