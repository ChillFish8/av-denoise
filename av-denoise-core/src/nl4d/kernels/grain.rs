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
use crate::noise::TEMPORAL_NOISE_BLOCK;
use crate::noise::colour::{COLOUR_BRIGHTNESS_STEP, COLOUR_MEAN_FIELDS};

/// Threads in one measuring cube, one per pixel of a cell.
const THREADS: u32 = 64;
/// The shared tile width, one cell plus [HALO_X] pixels each side.
const TILE_W: u32 = 20;
/// The shared tile height, one cell plus the 3 lag rows below it.
const TILE_H: u32 = 11;
const TILE_LEN: u32 = TILE_W * TILE_H;
/// The largest horizontal lag.
const HALO_X: u32 = 6;
const LAG_LANES: u32 = 46;
/// Halving rounds that reduce [THREADS] values to one.
const REDUCE_ROUNDS: u32 = THREADS.ilog2();
/// Halving rounds that reduce [REDUCE_THREADS] values to one.
const CHUNK_ROUNDS: u32 = REDUCE_THREADS.ilog2();
/// Shared lanes for sums of the grain, its square, the clean sample, the centred source sample and
/// its square.
const SUM_LANES: u32 = 5;

/// Copies one neighbour's motion vectors and confidence into saved ring entry `entry`.
///
/// `mv_offset` and `conf_offset` are the neighbour's element offsets in the motion field and
/// confidence. Each thread strides by `total_threads`, so a clamped grid still covers every block.
#[cube(launch_unchecked)]
pub fn grain_save_vectors(
    mv_field: &[i32],
    confidence: &[f32],
    saved_mv: &mut [i32],
    saved_conf: &mut [f32],
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
/// `dx_index` runs between 0 and 12 and is the horizontal offset plus [HALO_X]. Only half the plane
/// is measured because a lag and its mirror share one autocovariance.
const fn lag_lane(dy: u32, dx_index: u32) -> u32 {
    if dy == 0 {
        dx_index - HALO_X
    } else {
        (HALO_X + 1) + (dy - 1) * (2 * HALO_X + 1) + dx_index
    }
}

/// The sample of channel `channel` at pixel `(x, y)` of ring slot `slot`.
#[cube]
pub(super) fn channel_at<N: Size>(
    input: &[Vector<f32, N>],
    x: u32,
    y: u32,
    slot: u32,
    #[comptime] channel: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> f32 {
    let index = (slot * height + y) * width + x;
    let value = input[index as usize];
    value.extract(channel as usize)
}

/// Moves `value` by `shift` and clamps it into `0..limit`.
#[cube]
fn clamp_shift(value: u32, shift: i32, #[comptime] limit: u32) -> u32 {
    let moved = value as i32 + shift;
    let low = i32::max(moved, 0i32);
    let clamped = i32::min(low, limit as i32 - 1i32);
    clamped as u32
}

/// The motion block of a cell, clamped into the block grid.
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

/// The motion-compensated source grain of channel `channel` at one pixel, using the vector of the
/// pixel's own cell.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, ring index or comptime shape the grain read needs"
)]
fn source_grain_at<N: Size>(
    input: &[Vector<f32, N>],
    saved_mv: &[i32],
    x: u32,
    y: u32,
    slot_t: u32,
    slot_next: u32,
    entry_base: u32,
    #[comptime] channel: u32,
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
    let next = channel_at(input, next_x, next_y, slot_next, channel, width, height);
    let current = channel_at(input, x, y, slot_t, channel, width, height);
    (next - current) * std::f32::consts::FRAC_1_SQRT_2
}

/// Whether a cell's denoised `mean` sits inside the luma band, always true without `luma_band`.
#[cube]
fn in_band(mean: f32, #[comptime] luma_band: bool) -> bool {
    let mut inside = true;
    if luma_band {
        inside = mean > LUMA_LOW && mean < LUMA_HIGH;
    }

    inside
}

/// The highest bucket whose lower edge is at or below `grain_std`.
#[cube]
fn bucket_for(grain_std: f32, edges: &[f32]) -> u32 {
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

/// Whether a one-frame cell has a brightness to bin by, always true without a colour grid.
#[cube]
fn means_ready(has_luma_means: u32, #[comptime] means_blocks_x: u32) -> bool {
    let mut ready = true;
    if comptime!(means_blocks_x > 0) {
        ready = has_luma_means != 0;
    }

    ready
}

/// The brightness bin of a one-frame cell.
///
/// Without a colour grid it is the bin of the cell's denoised `mean`. Otherwise it is the bin of
/// the mean luma of the colour block the cell origin falls in, on the 8-bit scale.
#[cube]
fn frame_bin(
    luma_means: &[f32],
    mean: f32,
    cell_x: u32,
    cell_y: u32,
    #[comptime] means_blocks_x: u32,
) -> u32 {
    let mut bin = luma_bin(mean);
    if comptime!(means_blocks_x > 0) {
        let block_x = cell_x * CELL / TEMPORAL_NOISE_BLOCK;
        let block_y = cell_y * CELL / TEMPORAL_NOISE_BLOCK;
        let block = block_y * means_blocks_x + block_x;
        let block_luma = luma_means[(block * COLOUR_MEAN_FIELDS) as usize];
        let scaled = u32::cast_from(block_luma / COLOUR_BRIGHTNESS_STEP);
        bin = u32::min(scaled, comptime!(LUMA_BINS as u32 - 1));
    }

    bin
}

/// The sample standard deviation of [THREADS] values from their sum and sum of squares.
#[cube]
fn std_of(sum: f32, sum_sq: f32) -> f32 {
    let count = THREADS as f32;
    let variance = (sum_sq - sum * sum / count) / (count - 1.0f32);
    f32::sqrt(f32::max(variance, 0.0f32))
}

/// Measures one completed frame's source grain and one-frame grain over one 8x8 cell.
///
/// Launch one `CELL x CELL` cube per cell. Every sample is read from channel `channel`. Source
/// grain is the motion-compensated difference of the noisy ring slots `slot_t` and `slot_next`,
/// divided by sqrt 2 so it carries one frame's grain std. `source_entry` picks the difference's
/// saved vectors and confidence. One-frame grain is the cell's own std in slot `slot_t`, with its
/// mean taken off. A zero `has_source` rejects every cell.
///
/// Every cell needs a flat `out_t`, a mean inside the luma band when `luma_band` is set, and no
/// source pixel of `t` or `t + 1` near clipping. A source cell must also sit off the left, right
/// and bottom edges, so its lag halo never reads a clamped pixel, and needs a confident vector and
/// a grain std above the floor. A one-frame cell needs only a std above the floor, and with a
/// colour grid a set `has_luma_means`.
///
/// Each accepted cell adds one count to its brightness bin and std bucket of the frame's
/// histograms, which start at `hist_offset` in `hist`. The first `HIST_LEN` counts are for source
/// grain and the next for one-frame grain. A source cell bins by its `out_t` mean, and so does a
/// one-frame cell when `means_blocks_x` is 0. Otherwise a one-frame cell bins by the mean luma in
/// `luma_means` of the colour block its origin falls in, on a grid `means_blocks_x` blocks wide.
///
/// Every cell writes its partial, the 46 lag sums, pixel count and strength group of its source
/// grain, as zeros when rejected. The lag sums take the cell's mean source grain off every pixel
/// and halo neighbour first, so a uniform brightness flicker between the two frames never reaches
/// them. The strength group keeps cells of outlying grain strength in a record of their own.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, ring index or comptime shape the kernel binds"
)]
pub fn grain_measure<N: Size>(
    input: &[Vector<f32, N>],
    out_t: &[f32],
    saved_mv: &[i32],
    saved_conf: &[f32],
    edges: &[f32],
    luma_means: &[f32],
    hist: &mut [Atomic<i32>],
    partials: &mut [f32],
    slot_t: u32,
    slot_next: u32,
    source_entry: u32,
    has_source: u32,
    has_luma_means: u32,
    hist_offset: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] step: u32,
    #[comptime] channel: u32,
    #[comptime] luma_band: bool,
    #[comptime] means_blocks_x: u32,
) {
    let mut tile = Shared::<[f32]>::new_slice(TILE_LEN as usize);
    let mut sums = Shared::<[f32]>::new_slice((SUM_LANES * THREADS) as usize);
    let mut lows = Shared::<[f32]>::new_slice((2 * THREADS) as usize);
    let mut highs = Shared::<[f32]>::new_slice((2 * THREADS) as usize);
    let mut lags = Shared::<[f32]>::new_slice((LAG_LANES * THREADS) as usize);
    let mut accepted = Shared::<[f32]>::new_slice(1usize);

    let cells_x = comptime!(width / CELL);
    let cells_y = comptime!(height / CELL);
    let blocks = comptime!(blocks_x * blocks_y);
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let thread_id = local_y * CELL + local_x;
    let cell_x = CUBE_POS_X;
    let cell_y = CUBE_POS_Y;
    let origin_x = cell_x * CELL;
    let origin_y = cell_y * CELL;
    let x = origin_x + local_x;
    let y = origin_y + local_y;
    let block = block_for(cell_x, cell_y, blocks_x, blocks_y, step);
    let source_base = source_entry * blocks * 2;
    let mut group = 0u32;

    // Source grain over the cell and its halo, every coordinate clamped into the frame.
    let mut fill = thread_id;
    while fill < TILE_LEN {
        let tile_x = fill % TILE_W;
        let tile_y = fill / TILE_W;
        let raw_x = origin_x as i32 + tile_x as i32 - HALO_X as i32;
        let read_x = clamp_shift(0u32, raw_x, width);
        let read_y = clamp_shift(origin_y + tile_y, 0i32, height);
        tile[fill as usize] = source_grain_at(
            input,
            saved_mv,
            read_x,
            read_y,
            slot_t,
            slot_next,
            source_base,
            channel,
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
        channel,
        width,
        height,
        blocks_x,
        blocks_y,
        step,
    );
    let clean = out_t[((y * width + x) * channels + channel) as usize];
    let current = channel_at(input, x, y, slot_t, channel, width, height);
    let dx = saved_mv[(source_base + block * 2) as usize];
    let dy = saved_mv[(source_base + block * 2 + 1) as usize];
    let next_x = clamp_shift(x, dx, width);
    let next_y = clamp_shift(y, dy, height);
    let next = channel_at(input, next_x, next_y, slot_next, channel, width, height);
    // The cell origin's sample comes off first, so the f32 sums of squares keep their precision.
    let origin = channel_at(input, origin_x, origin_y, slot_t, channel, width, height);
    let centred = current - origin;

    sums[thread_id as usize] = own_grain;
    sums[(THREADS + thread_id) as usize] = own_grain * own_grain;
    sums[(2 * THREADS + thread_id) as usize] = clean;
    sums[(3 * THREADS + thread_id) as usize] = centred;
    sums[(4 * THREADS + thread_id) as usize] = centred * centred;
    lows[thread_id as usize] = clean;
    lows[(THREADS + thread_id) as usize] = f32::min(current, next);
    highs[thread_id as usize] = clean;
    highs[(THREADS + thread_id) as usize] = f32::max(current, next);
    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let stride = comptime!(THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for lane in 0..SUM_LANES {
                let here = (lane * THREADS + thread_id) as usize;
                let there = (lane * THREADS + thread_id + stride) as usize;
                sums[here] += sums[there];
            }

            #[unroll]
            for lane in 0..2u32 {
                let here = (lane * THREADS + thread_id) as usize;
                let there = (lane * THREADS + thread_id + stride) as usize;
                lows[here] = f32::min(lows[here], lows[there]);
                highs[here] = f32::max(highs[here], highs[there]);
            }
        }

        sync_cube();
    }

    if thread_id == 0 {
        let count = THREADS as f32;
        let grain_std = std_of(sums[0], sums[THREADS as usize]);
        let mean = sums[(2 * THREADS) as usize] / count;
        let frame_std = std_of(sums[(3 * THREADS) as usize], sums[(4 * THREADS) as usize]);
        let range = highs[0] - lows[0];
        let interior = cell_x >= 1 && cell_x + 2 <= cells_x && cell_y + 2 <= cells_y;
        let confidence = saved_conf[(source_entry * blocks + block) as usize];
        let usable = has_source != 0
            && range < FLAT_RANGE
            && in_band(mean, luma_band)
            && lows[THREADS as usize] >= CLIP_LOW
            && highs[THREADS as usize] <= CLIP_HIGH;
        let passes = usable && interior && confidence >= CONF_MIN && grain_std > STD_MIN;
        accepted[0] = 0.0f32;

        if passes {
            accepted[0] = 1.0f32;
            let bucket = bucket_for(grain_std, edges);
            let slot = hist_offset + luma_bin(mean) * STD_BUCKETS as u32 + bucket;
            group = bucket / BUCKETS_PER_GROUP as u32;
            Atomic::fetch_add(&hist[slot as usize], 1i32);
        }

        let frame_passes = usable && frame_std > STD_MIN && means_ready(has_luma_means, means_blocks_x);
        if frame_passes {
            let bucket = bucket_for(frame_std, edges);
            let bin = frame_bin(luma_means, mean, cell_x, cell_y, means_blocks_x);
            let frame_offset = hist_offset + HIST_LEN as u32;
            let slot = frame_offset + bin * STD_BUCKETS as u32 + bucket;
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
                lags[(lane * THREADS + thread_id) as usize] = take * centre * neighbour;
            }
        }
    }

    sync_cube();

    #[unroll]
    for round in 0..REDUCE_ROUNDS {
        let lag_stride = comptime!(THREADS >> (round + 1));
        if thread_id < lag_stride {
            #[unroll]
            for index in 0..LAG_LANES {
                let here = (index * THREADS + thread_id) as usize;
                let there = (index * THREADS + thread_id + lag_stride) as usize;
                lags[here] += lags[there];
            }
        }

        sync_cube();
    }

    if thread_id == 0 {
        let cell_index = cell_y * cells_x + cell_x;
        let base = cell_index * PARTIAL_LEN as u32;

        #[unroll]
        for index in 0..LAG_LANES {
            partials[(base + index) as usize] = lags[(index * THREADS) as usize];
        }

        partials[(base + LAG_LANES) as usize] = take * THREADS as f32;
        partials[(base + AUTOCOV_LEN as u32) as usize] = group as f32;
    }
}

/// Adds one lane of every cell's partial into its strength group's record in `chunk`.
///
/// Launch one cube of `REDUCE_THREADS` threads per lane, `AUTOCOV_LEN` cubes in all. The frame's
/// records start at `record_offset` in `chunk`, one per strength group, and keep their earlier
/// sums. Each thread adds its cells into its own column of a per-group scratch, and the columns
/// then reduce to one value per group. A cell's group is clamped to the last group, so a bad value
/// never indexes past the scratch.
#[cube(launch_unchecked)]
pub fn grain_reduce_partials(partials: &[f32], chunk: &mut [f32], record_offset: u32, cells: u32) {
    let mut scratch = Shared::<[f32]>::new_slice((STRENGTH_GROUPS as u32 * REDUCE_THREADS) as usize);
    let lane = CUBE_POS_X;
    let thread_id = UNIT_POS_X;

    #[unroll]
    for group in 0..STRENGTH_GROUPS as u32 {
        scratch[(group * REDUCE_THREADS + thread_id) as usize] = 0.0f32;
    }

    let mut cell = thread_id;
    while cell < cells {
        let base = cell * PARTIAL_LEN as u32;
        let raw_group = u32::cast_from(partials[(base + AUTOCOV_LEN as u32) as usize]);
        let group = u32::min(raw_group, comptime!(STRENGTH_GROUPS as u32 - 1));
        let slot = (group * REDUCE_THREADS + thread_id) as usize;
        scratch[slot] += partials[(base + lane) as usize];
        cell += REDUCE_THREADS;
    }

    sync_cube();

    #[unroll]
    for round in 0..CHUNK_ROUNDS {
        let stride = comptime!(REDUCE_THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for group in 0..STRENGTH_GROUPS as u32 {
                let here = (group * REDUCE_THREADS + thread_id) as usize;
                let there = (group * REDUCE_THREADS + thread_id + stride) as usize;
                scratch[here] += scratch[there];
            }
        }

        sync_cube();
    }

    if thread_id < STRENGTH_GROUPS as u32 {
        let target = (record_offset + thread_id * AUTOCOV_LEN as u32 + lane) as usize;
        chunk[target] += scratch[(thread_id * REDUCE_THREADS) as usize];
    }
}
