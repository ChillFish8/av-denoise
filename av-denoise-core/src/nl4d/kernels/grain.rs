use cubecl::prelude::*;

use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    BUCKETS_PER_GROUP,
    CELL,
    CLIP_HIGH,
    CLIP_LOW,
    CONF_MIN,
    FLAT_RANGE,
    HIST_LEN,
    LUMA_BINS,
    LUMA_HIGH,
    LUMA_LOW,
    PARTIAL_LEN,
    REDUCE_THREADS,
    STD_BUCKETS,
    STD_MIN,
    STRENGTH_GROUPS,
};

const THREADS: u32 = 64;
const TILE_W: u32 = 20;
const TILE_H: u32 = 11;
const TILE_LEN: u32 = TILE_W * TILE_H;
const HALO_X: u32 = 6;
const LAG_LANES: u32 = 46;
/// Halving rounds that reduce `THREADS` values to one.
const REDUCE_ROUNDS: u32 = THREADS.ilog2();
/// Halving rounds that reduce `REDUCE_THREADS` values to one.
const CHUNK_ROUNDS: u32 = REDUCE_THREADS.ilog2();
/// Shared lanes for sums of the grain, its square and the clean luma.
const SUM_LANES: u32 = 3;

/// Copies one neighbour's motion vectors and confidence into a saved ring entry.
///
/// `mv_offset` and `conf_offset` are the neighbour's element offsets in the motion field and
/// confidence. Each thread strides by `total_threads`, so a clamped grid still covers every block.
#[cube(launch_unchecked)]
pub fn grain_save_vectors(
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    saved_mv: &mut Array<i32>,
    saved_conf: &mut Array<f32>,
    mv_offset: u32,
    conf_offset: u32,
    entry: u32,
    #[comptime] blocks: u32,
    #[comptime] total_threads: u32,
) {
    let mut block = ABSOLUTE_POS_X;

    while block < blocks {
        let saved_block = entry * blocks + block;
        saved_mv[(saved_block * 2) as usize] = mv_field[(mv_offset + block * 2) as usize];
        saved_mv[(saved_block * 2 + 1) as usize] = mv_field[(mv_offset + block * 2 + 1) as usize];
        saved_conf[saved_block as usize] = confidence[(conf_offset + block) as usize];
        block += total_threads;
    }
}

/// The lag lane of a half-plane offset, in the order of [LAGS](crate::nl4d::grain::consts::LAGS).
///
/// `dx_index` runs between 0 and 12 and is the horizontal offset plus [HALO_X].
const fn lag_lane(dy: u32, dx_index: u32) -> u32 {
    if dy == 0 {
        dx_index - HALO_X
    } else {
        (HALO_X + 1) + (dy - 1) * (2 * HALO_X + 1) + dx_index
    }
}

#[cube]
fn luma_at<N: Size>(
    input: &Array<Vector<f32, N>>,
    x: u32,
    y: u32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> f32 {
    let index = (slot * height + y) * width + x;
    let value = input[index as usize];
    value[0]
}

/// Moves `value` by `shift` and clamps it into `0..limit`.
#[cube]
fn clamp_shift(value: u32, shift: i32, #[comptime] limit: u32) -> u32 {
    let moved = value as i32 + shift;
    let low = i32::max(moved, 0i32);
    let clamped = i32::min(low, limit as i32 - 1i32);
    clamped as u32
}

#[cube]
fn block_for(
    cell_x: u32,
    cell_y: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] step: u32,
) -> u32 {
    let block_x = u32::min(cell_x * CELL / step, blocks_x - 1);
    let block_y = u32::min(cell_y * CELL / step, blocks_y - 1);
    block_y * blocks_x + block_x
}

/// The motion-compensated source grain at one pixel, using the vector of the pixel's own cell.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, ring index or comptime shape the grain read needs"
)]
fn source_grain_at<N: Size>(
    input: &Array<Vector<f32, N>>,
    saved_mv: &Array<i32>,
    x: u32,
    y: u32,
    slot_t: u32,
    slot_next: u32,
    entry_base: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] step: u32,
) -> f32 {
    let block = block_for(x / CELL, y / CELL, blocks_x, blocks_y, step);
    let mv_index = entry_base + block * 2;
    let dx = saved_mv[mv_index as usize];
    let dy = saved_mv[(mv_index + 1) as usize];
    let next_x = clamp_shift(x, dx, width);
    let next_y = clamp_shift(y, dy, height);
    let next = luma_at(input, next_x, next_y, slot_next, width, height);
    let current = luma_at(input, x, y, slot_t, width, height);
    (next - current) * std::f32::consts::FRAC_1_SQRT_2
}

/// The highest bucket whose lower edge is at or below `grain_std`.
#[cube]
fn bucket_for(grain_std: f32, edges: &Array<f32>) -> u32 {
    let mut bucket = 0u32;

    for index in 1..STD_BUCKETS as u32 {
        if grain_std >= edges[index as usize] {
            bucket = index;
        }
    }

    bucket
}

#[cube]
fn luma_bin(mean: f32) -> u32 {
    let scaled = u32::cast_from(mean * comptime!(LUMA_BINS as f32));
    u32::min(scaled, comptime!(LUMA_BINS as u32 - 1))
}

/// The sample standard deviation of `THREADS` values from their sum and sum of squares.
#[cube]
fn std_of(sum: f32, sum_sq: f32) -> f32 {
    let count = THREADS as f32;
    let variance = (sum_sq - sum * sum / count) / (count - 1.0f32);
    f32::sqrt(f32::max(variance, 0.0f32))
}

/// Measures one completed frame's source grain and kept grain over one 8x8 cell.
///
/// Source grain is the motion-compensated difference of noisy frames `t` and `t + 1`, and kept
/// grain the same between outputs `t - 1` and `t`, both divided by sqrt 2. Each accepted cell adds
/// one count to its histogram. An accepted source cell writes its 46 lag sums, pixel count and
/// strength group to `partials`, and every other cell writes zeros there. The lag sums take the
/// cell's mean source grain off every pixel and halo neighbour first.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, ring index or comptime shape the kernel binds"
)]
pub fn grain_measure<N: Size>(
    input: &Array<Vector<f32, N>>,
    out_t: &Array<f32>,
    out_prev: &Array<f32>,
    saved_mv: &Array<i32>,
    saved_conf: &Array<f32>,
    edges: &Array<f32>,
    hist: &mut Array<Atomic<i32>>,
    partials: &mut Array<f32>,
    slot_t: u32,
    slot_next: u32,
    source_entry: u32,
    kept_entry: u32,
    has_source: u32,
    has_kept: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] step: u32,
) {
    let mut tile = SharedMemory::<f32>::new(TILE_LEN as usize);
    let mut sums = SharedMemory::<f32>::new((SUM_LANES * THREADS) as usize);
    let mut lows = SharedMemory::<f32>::new((2 * THREADS) as usize);
    let mut highs = SharedMemory::<f32>::new((2 * THREADS) as usize);
    let mut lags = SharedMemory::<f32>::new((LAG_LANES * THREADS) as usize);
    let mut accepted = SharedMemory::<f32>::new(1usize);

    let cells_x = comptime!(width / CELL);
    let cells_y = comptime!(height / CELL);
    let blocks = comptime!(blocks_x * blocks_y);
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let tid = local_y * CELL + local_x;
    let cell_x = CUBE_POS_X;
    let cell_y = CUBE_POS_Y;
    let x0 = cell_x * CELL;
    let y0 = cell_y * CELL;
    let x = x0 + local_x;
    let y = y0 + local_y;
    let block = block_for(cell_x, cell_y, blocks_x, blocks_y, step);
    let source_base = source_entry * blocks * 2;
    let kept_base = kept_entry * blocks * 2;
    let mut group = 0u32;

    // Source grain over the cell and its halo, every coordinate clamped into the frame.
    let mut fill = tid;
    while fill < TILE_LEN {
        let tile_x = fill % TILE_W;
        let tile_y = fill / TILE_W;
        let raw_x = x0 as i32 + tile_x as i32 - HALO_X as i32;
        let read_x = clamp_shift(0u32, raw_x, width);
        let read_y = clamp_shift(y0 + tile_y, 0i32, height);
        tile[fill as usize] = source_grain_at(
            input,
            saved_mv,
            read_x,
            read_y,
            slot_t,
            slot_next,
            source_base,
            width,
            height,
            blocks_x,
            blocks_y,
            step,
        );
        fill += THREADS;
    }

    let own_grain = source_grain_at(
        input,
        saved_mv,
        x,
        y,
        slot_t,
        slot_next,
        source_base,
        width,
        height,
        blocks_x,
        blocks_y,
        step,
    );
    let clean = out_t[((y * width + x) * stored_ch) as usize];
    let current = luma_at(input, x, y, slot_t, width, height);
    let dx = saved_mv[(source_base + block * 2) as usize];
    let dy = saved_mv[(source_base + block * 2 + 1) as usize];
    let next_x = clamp_shift(x, dx, width);
    let next_y = clamp_shift(y, dy, height);
    let next = luma_at(input, next_x, next_y, slot_next, width, height);

    sums[tid as usize] = own_grain;
    sums[(THREADS + tid) as usize] = own_grain * own_grain;
    sums[(2 * THREADS + tid) as usize] = clean;
    lows[tid as usize] = clean;
    lows[(THREADS + tid) as usize] = f32::min(current, next);
    highs[tid as usize] = clean;
    highs[(THREADS + tid) as usize] = f32::max(current, next);
    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let stride = comptime!(THREADS >> (round + 1));
        if tid < stride {
            #[unroll]
            for lane in 0..SUM_LANES {
                let here = (lane * THREADS + tid) as usize;
                let there = (lane * THREADS + tid + stride) as usize;
                sums[here] = sums[here] + sums[there];
            }

            #[unroll]
            for lane in 0..2u32 {
                let here = (lane * THREADS + tid) as usize;
                let there = (lane * THREADS + tid + stride) as usize;
                lows[here] = f32::min(lows[here], lows[there]);
                highs[here] = f32::max(highs[here], highs[there]);
            }
        }

        sync_cube();
    }

    if tid == 0 {
        let count = THREADS as f32;
        let grain_std = std_of(sums[0], sums[THREADS as usize]);
        let mean = sums[(2 * THREADS) as usize] / count;
        let range = highs[0] - lows[0];
        let interior = cell_x >= 1 && cell_x + 2 <= cells_x && cell_y + 2 <= cells_y;
        let conf = saved_conf[(source_entry * blocks + block) as usize];
        let ok = has_source != 0
            && interior
            && conf >= CONF_MIN
            && range < FLAT_RANGE
            && mean > LUMA_LOW
            && mean < LUMA_HIGH
            && lows[THREADS as usize] >= CLIP_LOW
            && highs[THREADS as usize] <= CLIP_HIGH
            && grain_std > STD_MIN;
        accepted[0] = 0.0f32;

        if ok {
            accepted[0] = 1.0f32;
            let bucket = bucket_for(grain_std, edges);
            let slot = luma_bin(mean) * STD_BUCKETS as u32 + bucket;
            group = bucket / BUCKETS_PER_GROUP as u32;
            Atomic::fetch_add(&hist[slot as usize], 1i32);
        }
    }

    sync_cube();

    let take = accepted[0];
    let grain_mean = sums[0] / THREADS as f32;
    let centre_index = local_y * TILE_W + local_x + HALO_X;
    let centre = tile[centre_index as usize] - grain_mean;

    #[unroll]
    for dy in 0..4u32 {
        #[unroll]
        for dx_index in 0..13u32 {
            if comptime!(dy > 0 || dx_index >= HALO_X) {
                let lane = comptime!(lag_lane(dy, dx_index));
                let neighbour_index = (local_y + dy) * TILE_W + local_x + dx_index;
                let neighbour = tile[neighbour_index as usize] - grain_mean;
                lags[(lane * THREADS + tid) as usize] = take * centre * neighbour;
            }
        }
    }

    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let lag_stride = comptime!(THREADS >> (round + 1));
        if tid < lag_stride {
            #[unroll]
            for index in 0..LAG_LANES {
                let here = (index * THREADS + tid) as usize;
                let there = (index * THREADS + tid + lag_stride) as usize;
                lags[here] = lags[here] + lags[there];
            }
        }

        sync_cube();
    }

    if tid == 0 {
        let cell_index = cell_y * cells_x + cell_x;
        let base = cell_index * PARTIAL_LEN as u32;

        #[unroll]
        for index in 0..LAG_LANES {
            partials[(base + index) as usize] = lags[(index * THREADS) as usize];
        }

        partials[(base + LAG_LANES) as usize] = take * THREADS as f32;
        partials[(base + AUTOCOV_LEN as u32) as usize] = group as f32;
    }

    // Kept grain on the same cell of output `t - 1`.
    let kept_dx = saved_mv[(kept_base + block * 2) as usize];
    let kept_dy = saved_mv[(kept_base + block * 2 + 1) as usize];
    let warped_x = clamp_shift(x, kept_dx, width);
    let warped_y = clamp_shift(y, kept_dy, height);
    let warped = out_t[((warped_y * width + warped_x) * stored_ch) as usize];
    let previous = out_prev[((y * width + x) * stored_ch) as usize];
    let kept = (warped - previous) * std::f32::consts::FRAC_1_SQRT_2;

    sums[tid as usize] = kept;
    sums[(THREADS + tid) as usize] = kept * kept;
    sums[(2 * THREADS + tid) as usize] = previous;
    lows[tid as usize] = previous;
    highs[tid as usize] = previous;
    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let kept_stride = comptime!(THREADS >> (round + 1));
        if tid < kept_stride {
            #[unroll]
            for index in 0..SUM_LANES {
                let here = (index * THREADS + tid) as usize;
                let there = (index * THREADS + tid + kept_stride) as usize;
                sums[here] = sums[here] + sums[there];
            }

            let here = tid as usize;
            let there = (tid + kept_stride) as usize;
            lows[here] = f32::min(lows[here], lows[there]);
            highs[here] = f32::max(highs[here], highs[there]);
        }

        sync_cube();
    }

    if tid == 0 {
        let count = THREADS as f32;
        let kept_std = std_of(sums[0], sums[THREADS as usize]);
        let mean = sums[(2 * THREADS) as usize] / count;
        let low = lows[0];
        let high = highs[0];
        let conf = saved_conf[(kept_entry * blocks + block) as usize];
        let ok = has_kept != 0
            && conf >= CONF_MIN
            && high - low < FLAT_RANGE
            && mean > LUMA_LOW
            && mean < LUMA_HIGH
            && low >= CLIP_LOW
            && high <= CLIP_HIGH
            && kept_std > STD_MIN;

        if ok {
            let bucket = bucket_for(kept_std, edges);
            let slot = HIST_LEN as u32 + luma_bin(mean) * STD_BUCKETS as u32 + bucket;
            Atomic::fetch_add(&hist[slot as usize], 1i32);
        }
    }
}

/// Adds every cell's partial for one lane into its strength group's chunk record, one cube per lane.
///
/// Each thread adds its cells into its own column of a per-group scratch, and each group's column
/// sums then reduce to one value.
#[cube(launch_unchecked)]
pub fn grain_reduce_partials(partials: &Array<f32>, chunk: &mut Array<f32>, cells: u32) {
    let mut scratch = SharedMemory::<f32>::new((STRENGTH_GROUPS as u32 * REDUCE_THREADS) as usize);
    let lane = CUBE_POS_X;
    let tid = UNIT_POS_X;

    #[unroll]
    for group in 0..STRENGTH_GROUPS as u32 {
        scratch[(group * REDUCE_THREADS + tid) as usize] = 0.0f32;
    }

    let mut cell = tid;
    while cell < cells {
        let base = cell * PARTIAL_LEN as u32;
        let raw_group = u32::cast_from(partials[(base + AUTOCOV_LEN as u32) as usize]);
        let group = u32::min(raw_group, comptime!(STRENGTH_GROUPS as u32 - 1));
        let slot = (group * REDUCE_THREADS + tid) as usize;
        scratch[slot] = scratch[slot] + partials[(base + lane) as usize];
        cell += REDUCE_THREADS;
    }

    sync_cube();

    #[unroll]
    for round in 0..CHUNK_ROUNDS {
        let stride = comptime!(REDUCE_THREADS >> (round + 1));
        if tid < stride {
            #[unroll]
            for group in 0..STRENGTH_GROUPS as u32 {
                let here = (group * REDUCE_THREADS + tid) as usize;
                let there = (group * REDUCE_THREADS + tid + stride) as usize;
                scratch[here] = scratch[here] + scratch[there];
            }
        }

        sync_cube();
    }

    if tid < STRENGTH_GROUPS as u32 {
        let target = (tid * AUTOCOV_LEN as u32 + lane) as usize;
        chunk[target] = chunk[target] + scratch[(tid * REDUCE_THREADS) as usize];
    }
}
