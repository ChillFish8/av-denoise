use cubecl::prelude::*;

use super::helpers::read_line;
use crate::nlmeans::noise::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTERS,
};

/// The per-block stage of the Immerkær noise estimate.
///
/// Every interior thread applies a 3x3 mask to its pixel, which cancels
/// out smooth content and leaves mostly noise. The block then sums the
/// absolute responses per channel into one partial total.
///
/// Border pixels contribute zero, as do threads that land outside the
/// image because the grid overshoots on the last row or column of
/// blocks.
///
/// Results are written as `partials[block_index * 4 + lane]`, with any
/// unused lane left at zero.
#[cube(launch_unchecked)]
pub fn nlm_noise_partial<N: Size>(
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
    let tid = UNIT_POS_Y * block_x + UNIT_POS_X;

    let interior = x >= 1 && x < width - 1 && y >= 1 && y < height - 1;

    let mut response = Vector::<f32, N>::empty();
    if interior {
        let c = read_line(input, x, y, frame, width, height);
        let l = read_line(input, x - 1, y, frame, width, height);
        let r = read_line(input, x + 1, y, frame, width, height);
        let u = read_line(input, x, y - 1, frame, width, height);
        let d = read_line(input, x, y + 1, frame, width, height);
        let ul = read_line(input, x - 1, y - 1, frame, width, height);
        let ur = read_line(input, x + 1, y - 1, frame, width, height);
        let dl = read_line(input, x - 1, y + 1, frame, width, height);
        let dr = read_line(input, x + 1, y + 1, frame, width, height);
        let four = Vector::<f32, N>::empty().fill(4.0f32);
        let two = Vector::<f32, N>::empty().fill(2.0f32);
        response = c * four - (l + r + u + d) * two + (ul + ur + dl + dr);
    }

    #[unroll]
    for ch in 0..channels {
        scratch[(tid * 4 + ch) as usize] = f32::abs(response[ch as usize]);
    }
    #[unroll]
    for ch in channels..4u32 {
        scratch[(tid * 4 + ch) as usize] = 0.0f32;
    }

    sync_cube();

    if tid == 0 {
        let cube_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        #[unroll]
        for ch in 0..4u32 {
            let mut sum = 0.0f32;
            for t in 0..threads {
                sum += scratch[(t * 4 + ch) as usize];
            }
            partials[(cube_index * 4 + ch) as usize] = sum;
        }
    }
}

/// The final stage of the Immerkær noise estimate.
///
/// A single block sums every partial into the per-channel totals for the
/// given ring slot. Each thread adds up a strided share of the partials,
/// then thread zero folds those shares together.
#[cube(launch_unchecked)]
pub fn nlm_noise_reduce(
    partials: &Array<f32>,
    results: &mut Array<f32>,
    slot: u32,
    num_partials: u32,
    #[comptime] block: u32,
) {
    let mut scratch = SharedMemory::<f32>::new(comptime!(block * 4) as usize);
    let tid = UNIT_POS_X;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = tid;
    while i < num_partials {
        sum0 += partials[(i * 4) as usize];
        sum1 += partials[(i * 4 + 1) as usize];
        sum2 += partials[(i * 4 + 2) as usize];
        sum3 += partials[(i * 4 + 3) as usize];
        i += block;
    }
    scratch[(tid * 4) as usize] = sum0;
    scratch[(tid * 4 + 1) as usize] = sum1;
    scratch[(tid * 4 + 2) as usize] = sum2;
    scratch[(tid * 4 + 3) as usize] = sum3;

    sync_cube();

    if tid == 0 {
        #[unroll]
        for ch in 0..4u32 {
            let mut total = 0.0f32;
            for t in 0..block {
                total += scratch[(t * 4 + ch) as usize];
            }
            results[(slot * 4 + ch) as usize] = total;
        }
    }
}

/// Gathers temporal residual statistics for each spatial block, with one
/// GPU block per `block x block` region.
///
/// For every pixel it computes the difference between the new slot and
/// the previous one, then reduces those differences into a single
/// record. The lag-1 product of neighbouring channel-0 differences is
/// what reveals grain correlated across nearby pixels.
///
/// Each record also carries six fields per 8x8 quarter of the block, in
/// top-left, top-right, bottom-left, bottom-right order. Every quarter
/// field reads channel 0 over the quarter's valid pixels.
///
/// - `sum_d` and `sum_d2` of the residual.
/// - `luma_sum`, `luma_min` and `luma_max` of the new frame.
/// - `flatness`, the mean squared neighbour difference of a 4x4 grid.
///   Each grid cell averages a 2x2 group of pixels, and each pixel
///   averages the new and previous frames. A quarter smaller than 8x8
///   writes `3.0e38` instead, so a flat gate downstream always rejects
///   it.
///
/// With `luma_fields` off, the kernel skips every quarter tile, barrier
/// and reduction, and writes 0 to each quarter lane instead.
///
/// A block that runs past the frame edge uses only its in-frame part.
/// Pixels outside the frame contribute nothing, and a pair only forms
/// when its second pixel is still inside that part, so a pair never
/// crosses a block boundary.
///
/// # Layout
///
/// Records go into `stats` one per block, at
/// `stats[block_index * (2 * stored_ch + 25) ..]`, laid out as every
/// `sum_d`, then every `sum_d2`, then `sum_lag`. Quarter `q` follows at
/// `2 * stored_ch + 1 + 6 * q`, with its fields at the offsets from
/// [QUARTER_SUM_D](crate::nlmeans::noise::QUARTER_SUM_D) to
/// [QUARTER_LUMA_MAX](crate::nlmeans::noise::QUARTER_LUMA_MAX). That
/// stride never depends on `luma_fields`.
///
/// `stats` should already be sliced down to the new slot's own region of
/// the larger ring buffer. See `noise::run_temporal_noise_stats`.
///
/// That is the same convention the motion-compensation kernels use for
/// their per-neighbour slices, and it means this kernel never needs to
/// know about the ring's other slots or the padding between them.
#[cube(launch_unchecked)]
pub fn nlm_temporal_noise_stats<N: Size>(
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
    // `record_len` is only ever used for `stats`' own output stride. The
    // reduction scratch below only ever holds the `sum_d`, `sum_d2` and
    // `sum_lag` lanes, so it is sized by `scratch_len` instead, or its
    // unwritten tail would enter the reduction as uninitialised memory.
    let quarter_lanes = comptime!(TEMPORAL_QUARTERS * TEMPORAL_QUARTER_FIELDS);
    let record_len = comptime!(2 * stored_ch + TEMPORAL_QUARTER_BASE + quarter_lanes);
    let scratch_len = comptime!(2 * stored_ch + 1);
    let threads = comptime!(block * block);

    let mut scratch = SharedMemory::<f32>::new(comptime!(threads * scratch_len) as usize);
    let mut d0_tile = SharedMemory::<f32>::new(threads as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let tid = local_y * block + local_x;

    let block_origin_x = CUBE_POS_X * block;
    let block_origin_y = CUBE_POS_Y * block;
    let gx = block_origin_x + local_x;
    let gy = block_origin_y + local_y;

    let valid = gx < width && gy < height;

    // The in-block extent, truncated so a pair never reaches past this
    // block's own slice of the frame. Ragged right and bottom edges use
    // the truncated extent, the same way the block matcher's coarse
    // kernel seeds its ragged last block from its position rather than
    // from a fixed block size.
    let block_w = u32::min(block, width - block_origin_x);
    let block_h = u32::min(block, height - block_origin_y);

    // These four stay at their harmless defaults, and are never read,
    // when `luma_fields` is off. Declaring them either way costs a few
    // registers at most, so only the real resource cost, the luma
    // tiles below, is behind the flag.
    let mut new_luma_raw = 0.0f32;
    let mut mean_luma_raw = 0.0f32;
    // Sentinels far above and far below any normalised luma value
    // (luma runs between 0 and 1), so an invalid pixel never wins the
    // min or max reduction below. The negative one is built from a `mut`
    // variable and a later compound assignment rather than a negative
    // literal initializer. cubecl folds a negative literal used as a
    // `let` initializer into a plain Rust constant. That constant cannot
    // unify with the `f32::min`/`f32::max` calls below.
    let mut min_seed = 1.0e30f32;
    let mut max_seed = 1.0e30f32;
    max_seed -= 2.0e30f32;

    let mut d = Vector::<f32, N>::empty();
    if valid {
        let c = read_line(input, gx, gy, slot_new, width, height);
        let p = read_line(input, gx, gy, slot_prev, width, height);
        d = c - p;
        if comptime!(luma_fields) {
            new_luma_raw = c[0];
            mean_luma_raw = 0.5f32 * (c[0] + p[0]);
            min_seed = c[0];
            max_seed = c[0];
        }
    }

    // The `.into()` calls are what let cubecl unify the two branches,
    // because both arms have to expand to the same `NativeExpand<f32>`.
    // Clippy cannot see that requirement.
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let d0 = if valid { d[0] } else { 0.0f32.into() };
    d0_tile[tid as usize] = d0;

    #[unroll]
    for ch in 0..stored_ch {
        #[expect(
            clippy::useless_conversion,
            reason = "both branches have to expand to the same cubecl native type, which the \
                      conversion supplies"
        )]
        let v = if valid { d[ch as usize] } else { 0.0f32.into() };
        scratch[(tid * scratch_len + ch) as usize] = v;
        scratch[(tid * scratch_len + stored_ch + ch) as usize] = v * v;
    }

    sync_cube();

    let pair_valid = valid && local_x + 1 < block_w;
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let lag = if pair_valid {
        d0 * d0_tile[(tid + 1) as usize]
    } else {
        0.0f32.into()
    };
    scratch[(tid * scratch_len + 2 * stored_ch) as usize] = lag;

    sync_cube();

    if tid == 0 {
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

        // With `luma_fields` off, the quarter lanes get an explicit 0
        // rather than whatever `stats` already held there, so a reader
        // never sees stale data left over from an earlier frame's slot.
        if comptime!(!luma_fields) {
            let quarters_start = out_base + 2 * stored_ch + TEMPORAL_QUARTER_BASE;

            #[unroll]
            for lane in 0..quarter_lanes {
                stats[(quarters_start + lane) as usize] = 0.0f32;
            }
        }
    }

    // Everything from here on computes the quarter lanes. With
    // `luma_fields` off, none of it is compiled. `luma_fields` never
    // varies within a launch, so every thread takes this branch
    // identically and reaches every barrier inside it.
    if comptime!(luma_fields) {
        // Each quarter's 64 pixels sit in one contiguous 64-slot segment
        // of the reduction tiles, so one tree reduction folds all four
        // quarters at once.
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
        // The lag pairs above are done with `d0_tile`, so it holds the
        // temporal mean instead. A separate tile costs about 40% of this
        // variant's throughput, because the extra shared memory lowers
        // how many workgroups run at once.
        d0_tile[tid as usize] = mean_luma_raw;

        sync_cube();

        // Six halving rounds reduce each 64-slot segment to its first
        // slot. `stride` is a per-round compile-time constant, because
        // cubecl panics at JIT time on a `mut` seeded from a comptime
        // value. The halving only guards which threads update a slot, so
        // every thread reaches every `sync_cube()`.
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
            }
            sync_cube();
        }

        // 64 threads each average their own 2x2 group of `d0_tile` into
        // one cell of an 8x8 grid, laid out row-major.
        if tid < 64u32 {
            let cell_y = tid / 8u32;
            let cell_x = tid % 8u32;
            let mut cell = 0.0f32;
            #[unroll]
            for dy in 0..2u32 {
                #[unroll]
                for dx in 0..2u32 {
                    let idx = (2u32 * cell_y + dy) * block + (2u32 * cell_x + dx);
                    cell += d0_tile[idx as usize];
                }
            }
            cells[tid as usize] = cell / 4.0f32;
        }
        sync_cube();

        // The same 64 threads each score their cell's right and down
        // neighbour. A pair that would cross into another quarter is
        // skipped, which leaves 24 pairs per quarter. The scores are
        // stored quarter-major, 16 cells per quarter.
        if tid < 64u32 {
            let cell_y = tid / 8u32;
            let cell_x = tid % 8u32;
            let here = cells[tid as usize];
            let mut local_energy = 0.0f32;
            if cell_x % 4u32 != 3u32 {
                let right = cells[(tid + 1u32) as usize];
                local_energy += (here - right) * (here - right);
            }

            if cell_y % 4u32 != 3u32 {
                let below = cells[(tid + 8u32) as usize];
                local_energy += (here - below) * (here - below);
            }

            let cell_quarter = (cell_y / 4u32) * 2u32 + cell_x / 4u32;
            let cell_pos = (cell_y % 4u32) * 4u32 + cell_x % 4u32;
            pair_tile[(cell_quarter * 16u32 + cell_pos) as usize] = local_energy;
        }
        sync_cube();

        // Four halving rounds reduce each quarter's 16 scores to its
        // first slot.
        #[unroll]
        for step in 0..4u32 {
            let energy_stride = comptime!(8u32 >> step);
            if tid < 64u32 && tid % 16u32 < energy_stride {
                pair_tile[tid as usize] += pair_tile[(tid + energy_stride) as usize];
            }
            sync_cube();
        }

        // The first thread of each quarter writes that quarter's record.
        if quarter_pos == 0u32 {
            let quarter_x = quarter % 2u32;
            let quarter_y = quarter / 2u32;
            let full_width = block_w >= (quarter_x + 1u32) * 8u32;
            let full_height = block_h >= (quarter_y + 1u32) * 8u32;

            // A quarter smaller than 8x8 keeps this sentinel, far above
            // any real gradient energy, so a flat gate always rejects it.
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
        }
    }
}

/// Fills a slice of the temporal-stats ring with zeroes.
///
/// A duplicated ring slot holds exactly the same pixels as the one
/// before it, so measuring the difference would only ever produce an
/// all-zero record. Writing the zeroes directly is cheaper and gives the
/// aggregation step the same "nothing to measure here" signal.
#[cube(launch_unchecked)]
pub fn nlm_temporal_stats_zero(
    dst: &mut Array<f32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}
