use cubecl::prelude::*;

use crate::engine::kernels::read_line;
use crate::noise::spatial::IMMERKAER_LANES;
use crate::noise::{
    BLOCK_MOVED_LANE,
    BLOCK_TEXTURE_LANE,
    COMPENSATION_GAIN_MAX,
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTERS,
    UNMEASURED_FLATNESS,
};

// nlm_noise_reduce keeps one named accumulator per lane.
const _: () = assert!(IMMERKAER_LANES == 2);

/// The per-block stage of the Immerkær noise estimate.
///
/// Each interior pixel applies a 3x3 mask that cancels smooth content and leaves mostly noise. The
/// block sums the absolute responses of each channel into its own lane of the block's
/// [IMMERKAER_LANES](crate::noise::spatial::IMMERKAER_LANES) partials. Border pixels, threads past
/// the image and unused lanes contribute zero. Thread `l` sums lane `l` over every thread in order.
#[cube(launch_unchecked)]
pub fn nlm_noise_partial<N: Size>(
    input: &[Vector<f32, N>],
    partials: &mut [f32],
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let threads = comptime!(block_x * block_y);

    comptime! {
        assert!(threads >= IMMERKAER_LANES, "nlm_noise_partial needs a thread per lane");
        assert!(channels <= IMMERKAER_LANES, "nlm_noise_partial needs a lane per channel");
    }

    let mut scratch = Shared::<[f32]>::new_slice(comptime!(threads * IMMERKAER_LANES) as usize);

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
        let lane = response.extract(channel as usize);
        scratch[(thread_id * IMMERKAER_LANES + channel) as usize] = f32::abs(lane);
    }

    #[unroll]
    for channel in channels..IMMERKAER_LANES {
        scratch[(thread_id * IMMERKAER_LANES + channel) as usize] = 0.0f32;
    }

    sync_cube();

    if thread_id < IMMERKAER_LANES {
        let cube_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        let channel = thread_id;
        let mut sum = 0.0f32;
        for t in 0..threads {
            sum += scratch[(t * IMMERKAER_LANES + channel) as usize];
        }

        partials[(cube_index * IMMERKAER_LANES + channel) as usize] = sum;
    }
}

/// The final stage of the Immerkær noise estimate.
///
/// A single block of `block` threads sums every partial into the per-channel totals in row `slot`
/// of `results`. Thread `l` sums lane `l` over every thread in order.
#[cube(launch_unchecked)]
pub fn nlm_noise_reduce(
    partials: &[f32],
    results: &mut [f32],
    slot: u32,
    num_partials: u32,
    #[comptime] block: u32,
) {
    comptime! {
        assert!(block >= IMMERKAER_LANES, "nlm_noise_reduce needs a thread per lane");
    }

    let mut scratch = Shared::<[f32]>::new_slice(comptime!(block * IMMERKAER_LANES) as usize);
    let thread_id = UNIT_POS_X;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut i = thread_id;
    while i < num_partials {
        sum0 += partials[(i * IMMERKAER_LANES) as usize];
        sum1 += partials[(i * IMMERKAER_LANES + 1) as usize];
        i += block;
    }

    scratch[(thread_id * IMMERKAER_LANES) as usize] = sum0;
    scratch[(thread_id * IMMERKAER_LANES + 1) as usize] = sum1;

    sync_cube();

    if thread_id < IMMERKAER_LANES {
        let channel = thread_id;
        let mut total = 0.0f32;
        for t in 0..block {
            total += scratch[(t * IMMERKAER_LANES + channel) as usize];
        }

        results[(slot * IMMERKAER_LANES + channel) as usize] = total;
    }
}

/// Writes one temporal residual record per `block x block` region of the frame.
///
/// The residual is the new slot minus the previous slot. Each record holds the per-channel `sum_d`
/// and `sum_d2`, then `sum_lag`, the lag-1 product of neighbouring channel-0 residuals, which
/// reveals grain correlated across nearby pixels. Pixels outside the frame contribute nothing, and
/// a lag pair never crosses a block boundary.
///
/// `texture` follows. The block's in-frame pixels split into a 4x4 grid of 4x4-pixel cells of the
/// new frame's channel 0, and `texture` is the mean squared difference between the means of
/// horizontally and vertically adjacent cells that lie fully inside the frame. A block with fewer
/// than two such cells writes 0. Then `moved`, 1.0 when the block followed a non-zero motion
/// vector, else 0.0.
///
/// Each record also carries six channel-0 fields per 8x8 quarter, in top-left, top-right,
/// bottom-left, bottom-right order.
///
/// - `sum_d` and `sum_d2` of the residual.
/// - `luma_sum`, `luma_min` and `luma_max` of the new frame.
/// - `flatness`, the mean squared neighbour difference of a 4x4 grid of 2x2 cells over the mean of
///   both frames. A quarter smaller than 8x8 writes
///   [UNMEASURED_FLATNESS](crate::noise::UNMEASURED_FLATNESS).
///
/// With `luma_fields` off, every quarter lane is written as 0 and the quarter work is skipped.
/// `block` must be 16, because the quarter reductions assume 8x8 quarters and 64-slot segments.
///
/// # Motion
///
/// With `compensated` on, each block reads the whole-pixel vector of the motion block covering its
/// origin, from the `mv_step` grid of `mv_blocks_x` by `mv_blocks_y` blocks starting at
/// `mv_field[mv_offset..]`. The block follows the vector only when its channel-0 residual variance
/// along it, over the pixels whose match stays in the frame, is at most
/// [COMPENSATION_GAIN_MAX](crate::noise::COMPENSATION_GAIN_MAX) times the zero-vector variance.
/// Pixel `(x, y)` of the new slot is then differenced against `(x + mvx, y + mvy)` of the previous
/// slot, and the temporal mean uses the same shifted pixel. A block with no match inside the frame
/// follows its vector unchecked. A followed block whose shifted extent leaves the frame writes every
/// lane as 0. A block that keeps the zero vector writes exactly what the uncompensated variant
/// writes. With `compensated` off, `mv_field` is never read.
///
/// # Layout
///
/// Records sit at `stats[block_index * (2 * channels + 27)..]` as every `sum_d`, every `sum_d2`,
/// then `sum_lag`, `texture` and `moved`. Quarter `q` follows at `2 * channels + 3 + 6 * q`, with
/// its fields at the `QUARTER_*` offsets in `noise`. The stride never depends on `luma_fields`.
/// `stats` must be bound to the new slot's own region of the ring. Thread `l` sums record lane `l`
/// over every thread in order.
#[cube(launch_unchecked)]
pub fn nlm_temporal_noise_stats<N: Size>(
    input: &[Vector<f32, N>],
    stats: &mut [f32],
    mv_field: &[i32],
    slot_new: u32,
    slot_prev: u32,
    mv_offset: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] block: u32,
    #[comptime] luma_fields: bool,
    #[comptime] compensated: bool,
    #[comptime] mv_step: u32,
    #[comptime] mv_blocks_x: u32,
    #[comptime] mv_blocks_y: u32,
) {
    // The scratch holds `scratch_len` interleaved lanes per thread, then a `threads`-long tile at
    // `tile_base`. It is sized by `scratch_len`, not `record_len`, or its unwritten tail would
    // enter the reduction as uninitialised memory.
    //
    // The tile holds the new frame's channel 0 for the texture cells, and later the temporal mean.
    // A separate tile lowers occupancy enough to cost about 40% of the luma variant's throughput.
    // The compensated variant first uses four `threads`-long runs for its variance check.
    let quarter_lanes = comptime!(TEMPORAL_QUARTERS * TEMPORAL_QUARTER_FIELDS);
    let record_len = comptime!(2 * channels + TEMPORAL_QUARTER_BASE + quarter_lanes);
    let scratch_len = comptime!(2 * channels + 1);
    let texture_lane = comptime!(2 * channels + BLOCK_TEXTURE_LANE);
    let threads = comptime!(block * block);
    let tile_base = comptime!(threads * scratch_len);

    comptime! {
        assert!(threads > texture_lane, "nlm_temporal_noise_stats needs a thread per lane");
        assert!(threads == 256, "the gain check's eight halving rounds need 256 threads");
    }

    let mut scratch = Shared::<[f32]>::new_slice(comptime!(tile_base + threads) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let thread_id = local_y * block + local_x;

    let block_origin_x = CUBE_POS_X * block;
    let block_origin_y = CUBE_POS_Y * block;
    let global_x = block_origin_x + local_x;
    let global_y = block_origin_y + local_y;

    // The in-frame extent of this block, so a pair never reaches past its slice of the frame.
    let block_w = u32::min(block, width - block_origin_x);
    let block_h = u32::min(block, height - block_origin_y);

    // The shift into the previous slot, and whether the shifted block stays inside the frame. Both
    // are uniform across the cube, so no barrier ever sits behind a divergent branch.
    let mut shift_x = 0i32;
    let mut shift_y = 0i32;
    let mut matched = true;
    let mut moved = 0.0f32;
    if comptime!(compensated) {
        let mv_col = u32::min(block_origin_x / mv_step, mv_blocks_x - 1);
        let mv_row = u32::min(block_origin_y / mv_step, mv_blocks_y - 1);
        let mv_idx = mv_offset + (mv_row * mv_blocks_x + mv_col) * 2;
        let vector_x = mv_field[mv_idx as usize];
        let vector_y = mv_field[(mv_idx + 1) as usize];

        let in_frame = global_x < width && global_y < height;
        let match_x = global_x as i32 + vector_x;
        let match_y = global_y as i32 + vector_y;
        let match_in_frame =
            match_x >= 0 && match_x < width as i32 && match_y >= 0 && match_y < height as i32;

        let mut zero_residual = 0.0f32;
        let mut vector_residual = 0.0f32;
        if in_frame {
            let new_pixel = read_line(input, global_x, global_y, slot_new, width, height);
            let still_pixel = read_line(input, global_x, global_y, slot_prev, width, height);
            let new_luma = new_pixel.extract(0usize);
            zero_residual = new_luma - still_pixel.extract(0usize);

            if match_in_frame {
                let match_pixel = read_line(input, match_x as u32, match_y as u32, slot_prev, width, height);
                vector_residual = new_luma - match_pixel.extract(0usize);
            }
        }

        scratch[thread_id as usize] = zero_residual;
        scratch[(threads + thread_id) as usize] = zero_residual * zero_residual;
        scratch[(2u32 * threads + thread_id) as usize] = vector_residual;
        scratch[(3u32 * threads + thread_id) as usize] = vector_residual * vector_residual;

        sync_cube();

        // Eight halving rounds fold the four runs into their first slots.
        #[unroll]
        for step in 0..8u32 {
            let check_stride = comptime!(128u32 >> step);
            if thread_id < check_stride {
                #[unroll]
                for run in 0..4u32 {
                    let here = (run * threads + thread_id) as usize;
                    let partner = (run * threads + thread_id + check_stride) as usize;
                    scratch[here] += scratch[partner];
                }
            }

            sync_cube();
        }

        let zero_sum = scratch[0usize];
        let zero_sum_sq = scratch[threads as usize];
        let vector_sum = scratch[(2u32 * threads) as usize];
        let vector_sum_sq = scratch[(3u32 * threads) as usize];

        // Every thread has read the totals before the runs are overwritten below.
        sync_cube();

        let left = block_origin_x as i32 + vector_x;
        let top = block_origin_y as i32 + vector_y;
        let right = left + block_w as i32;
        let bottom = top + block_h as i32;
        let span_w = i32::min(right, width as i32) - i32::max(left, 0i32);
        let span_h = i32::min(bottom, height as i32) - i32::max(top, 0i32);
        let overlap_w = i32::max(span_w, 0i32);
        let overlap_h = i32::max(span_h, 0i32);
        let matched_pixels = (overlap_w * overlap_h) as f32;
        let block_pixels = (block_w * block_h) as f32;

        let zero_mean = zero_sum / block_pixels;
        let zero_variance = zero_sum_sq / block_pixels - zero_mean * zero_mean;
        let safe_pixels = f32::max(matched_pixels, 1.0f32);
        let vector_mean = vector_sum / safe_pixels;
        let vector_variance = vector_sum_sq / safe_pixels - vector_mean * vector_mean;

        // A block with no match inside the frame cannot check its vector, so it follows it and
        // zeroes below rather than read a zero-vector residual the vector says is moving.
        let has_vector = vector_x != 0 || vector_y != 0;
        let unchecked = matched_pixels == 0.0f32;
        let earns_keep = vector_variance <= COMPENSATION_GAIN_MAX * zero_variance;
        let follows = has_vector && (unchecked || earns_keep);
        if follows {
            shift_x = vector_x;
            shift_y = vector_y;

            let fits_x = left >= 0 && right <= width as i32;
            let fits_y = top >= 0 && bottom <= height as i32;
            matched = fits_x && fits_y;
        }

        if follows && matched {
            moved = 1.0f32;
        }
    }

    let valid = global_x < width && global_y < height && matched;

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
    let mut cell_luma = 0.0f32;
    if valid {
        let new_pixel = read_line(input, global_x, global_y, slot_new, width, height);
        let mut prev_x = global_x;
        let mut prev_y = global_y;
        if comptime!(compensated) {
            prev_x = (global_x as i32 + shift_x) as u32;
            prev_y = (global_y as i32 + shift_y) as u32;
        }

        let prev_pixel = read_line(input, prev_x, prev_y, slot_prev, width, height);
        residual = new_pixel - prev_pixel;

        let new_luma = new_pixel.extract(0usize);
        cell_luma = new_luma;

        if comptime!(luma_fields) {
            let prev_luma = prev_pixel.extract(0usize);
            new_luma_raw = new_luma;
            mean_luma_raw = 0.5f32 * (new_luma + prev_luma);
            min_seed = new_luma;
            max_seed = new_luma;
        }
    }

    let d0 = if valid { residual.extract(0usize) } else { 0.0f32 };
    scratch[(tile_base + thread_id) as usize] = cell_luma;

    #[unroll]
    for channel in 0..channels {
        let lane_residual = if valid {
            residual.extract(channel as usize)
        } else {
            0.0f32
        };
        scratch[(thread_id * scratch_len + channel) as usize] = lane_residual;
        scratch[(thread_id * scratch_len + channels + channel) as usize] = lane_residual * lane_residual;
    }

    sync_cube();

    // Lane 0 of the next thread holds its channel-0 residual.
    let pair_valid = valid && local_x + 1 < block_w;
    let lag = if pair_valid {
        d0 * scratch[((thread_id + 1) * scratch_len) as usize]
    } else {
        0.0f32
    };
    scratch[(thread_id * scratch_len + 2 * channels) as usize] = lag;

    sync_cube();

    let block_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let out_base = block_index * record_len;

    if thread_id < scratch_len {
        let lane = thread_id;
        let mut total = 0.0f32;
        for t in 0..threads {
            total += scratch[(t * scratch_len + lane) as usize];
        }

        stats[(out_base + lane) as usize] = total;
    }

    // Every lane total is read before the cell means overwrite the first lanes.
    sync_cube();

    if thread_id < 16u32 {
        let cell_x = thread_id % 4u32;
        let cell_y = thread_id / 4u32;
        let mut cell = 0.0f32;

        #[unroll]
        for dy in 0..4u32 {
            #[unroll]
            for dx in 0..4u32 {
                let idx = (4u32 * cell_y + dy) * block + (4u32 * cell_x + dx);
                cell += scratch[(tile_base + idx) as usize];
            }
        }

        scratch[thread_id as usize] = cell / 16.0f32;
    }

    sync_cube();

    if thread_id == 0 {
        let full_cols = block_w / 4u32;
        let full_rows = block_h / 4u32;
        let mut energy = 0.0f32;
        let mut pairs = 0u32;
        for cell_y in 0..full_rows {
            for cell_x in 0..full_cols {
                let here = scratch[(cell_y * 4u32 + cell_x) as usize];
                if cell_x + 1u32 < full_cols {
                    let right_cell = scratch[(cell_y * 4u32 + cell_x + 1u32) as usize];
                    let step_x = here - right_cell;
                    energy += step_x * step_x;
                    pairs += 1u32;
                }

                if cell_y + 1u32 < full_rows {
                    let below_cell = scratch[((cell_y + 1u32) * 4u32 + cell_x) as usize];
                    let step_y = here - below_cell;
                    energy += step_y * step_y;
                    pairs += 1u32;
                }
            }
        }

        let mut texture = 0.0f32;
        if pairs > 0u32 {
            texture = energy / pairs as f32;
        }

        stats[(out_base + texture_lane) as usize] = texture;
        stats[(out_base + 2 * channels + BLOCK_MOVED_LANE) as usize] = moved;
    }

    // The quarter lanes are written explicitly so a reader never sees an earlier frame's data.
    if comptime!(!luma_fields) && thread_id == 0 {
        let quarters_start = out_base + 2 * channels + TEMPORAL_QUARTER_BASE;

        #[unroll]
        for lane in 0..quarter_lanes {
            stats[(quarters_start + lane) as usize] = 0.0f32;
        }
    }

    // The branch is comptime, so every thread reaches every barrier inside it.
    if comptime!(luma_fields) {
        // Each quarter's 64 pixels sit in one contiguous segment of the reduction tiles, so one tree
        // reduction folds all four quarters at once.
        let quarter = (local_y / 8u32) * 2u32 + local_x / 8u32;
        let quarter_pos = (local_y % 8u32) * 8u32 + local_x % 8u32;
        let slot = quarter * 64u32 + quarter_pos;

        let mut sum_tile = Shared::<[f32]>::new_slice(threads as usize);
        let mut min_tile = Shared::<[f32]>::new_slice(threads as usize);
        let mut max_tile = Shared::<[f32]>::new_slice(threads as usize);
        let mut residual_tile = Shared::<[f32]>::new_slice(threads as usize);
        let mut residual_sq_tile = Shared::<[f32]>::new_slice(threads as usize);
        let mut cells = Shared::<[f32]>::new_slice(64usize);
        let mut pair_tile = Shared::<[f32]>::new_slice(64usize);

        sum_tile[slot as usize] = new_luma_raw;
        min_tile[slot as usize] = min_seed;
        max_tile[slot as usize] = max_seed;
        residual_tile[slot as usize] = d0;
        residual_sq_tile[slot as usize] = d0 * d0;

        scratch[(tile_base + thread_id) as usize] = mean_luma_raw;

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
            }

            sync_cube();
        }

        // 64 threads each average a 2x2 group of the tile into one cell of a row-major 8x8 grid.
        if thread_id < 64u32 {
            let cell_y = thread_id / 8u32;
            let cell_x = thread_id % 8u32;
            let mut cell = 0.0f32;

            #[unroll]
            for dy in 0..2u32 {
                #[unroll]
                for dx in 0..2u32 {
                    let idx = (2u32 * cell_y + dy) * block + (2u32 * cell_x + dx);
                    cell += scratch[(tile_base + idx) as usize];
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

            let mut flatness = UNMEASURED_FLATNESS.runtime();
            if full_width && full_height {
                flatness = pair_tile[(quarter * 16u32) as usize] / 24.0f32;
            }

            let quarter_offset = TEMPORAL_QUARTER_BASE + quarter * TEMPORAL_QUARTER_FIELDS;
            let quarter_base = out_base + 2 * channels + quarter_offset;

            // An unmatched block reads as no measurable noise, so its seeds and sentinel give way
            // to zeroes.
            if matched {
                stats[(quarter_base + QUARTER_SUM_D) as usize] = residual_tile[slot as usize];
                stats[(quarter_base + QUARTER_SUM_D2) as usize] = residual_sq_tile[slot as usize];
                stats[(quarter_base + QUARTER_LUMA_SUM) as usize] = sum_tile[slot as usize];
                stats[(quarter_base + QUARTER_FLATNESS) as usize] = flatness;
                stats[(quarter_base + QUARTER_LUMA_MIN) as usize] = min_tile[slot as usize];
                stats[(quarter_base + QUARTER_LUMA_MAX) as usize] = max_tile[slot as usize];
            } else {
                #[unroll]
                for field in 0..TEMPORAL_QUARTER_FIELDS {
                    stats[(quarter_base + field) as usize] = 0.0f32;
                }
            }
        }
    }
}

/// Fills a slice of the temporal-stats ring with zeroes.
///
/// A duplicated ring slot would only ever measure an all-zero record, so writing the zeroes is
/// cheaper and gives the same result.
#[cube(launch_unchecked)]
pub fn nlm_temporal_stats_zero(dst: &mut [f32], #[comptime] length: u32, #[comptime] total_threads: u32) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}
