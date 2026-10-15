use cubecl::prelude::*;
use cubecl::terminate;

/// The threads in one block-match workgroup.
pub const BLOCK_MATCH_THREADS: u32 = 32;

/// How many horizontally adjacent candidate offsets one thread scores together.
const OFFSETS_PER_THREAD: u32 = 3;

/// Planes one block-match cube may span, which fits a cube of up to 256 threads at a plane width of 8.
const ARGMIN_MAX_PLANES: u32 = 32;

/// The index [parallel_argmin] returns when no SAD is below its `1.0e30` start.
const NO_CANDIDATE: u32 = u32::MAX;

const _: () = assert!(
    BLOCK_MATCH_THREADS <= ARGMIN_MAX_PLANES * 8,
    "the argmin's per-plane buffers must cover a block-match cube at a plane width of 8"
);

/// The row stride of the staged search window.
///
/// It is wide enough for the last group of offsets in a row to read a full register row.
fn window_stride(blksize: u32, search_radius: u32) -> u32 {
    let window_side = 2 * search_radius + 1;
    let groups_per_row = window_side.div_ceil(OFFSETS_PER_THREAD);
    let last_read = groups_per_row * OFFSETS_PER_THREAD + blksize - 1;
    let reach = blksize + 2 * search_radius;
    reach.max(last_read)
}

/// Finds each block's motion on one pyramid level and seeds the fine grid with it.
///
/// One GPU block handles one image block. Every thread scores a share of the
/// `(2 * search_radius + 1)^2` candidates, and the lowest SAD is picked. A tie picks the first in
/// row-major order, and the zero offset wins a tie with that pick.
///
/// `blksize` is capped at [MAX_BLKSIZE](crate::nlmeans::motion::MAX_BLKSIZE) so the shared centre
/// tile stays within 1024 values. The neighbour pixels the search reaches are staged into shared
/// memory once, clamped to the level.
///
/// This level of every frame is a slot of `pyramid`, the first `level_offset` values in and
/// `slot_pixels` values apart. The launch runs one cube per block per neighbour, with the neighbour
/// on z. Neighbour `n` matches slot `neighbour_slots[n]` against `centre_slot` and writes its field
/// `n * mv_stride` values in.
///
/// The winner is scaled by `level_scale`, 2 raised to the coarse level, and written to every fine
/// block inside this block's region. `step` and `fine_step` are the coarse and fine block spacings.
#[cube(launch_unchecked)]
pub fn nlm_mc_block_match_coarse(
    pyramid: &Array<f32>,
    neighbour_slots: &Array<u32>,
    mv_field: &mut Array<i32>,
    centre_slot: u32,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
    #[comptime] level_offset: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    #[comptime] level_scale: u32,
    #[comptime] fine_blocks_x: u32,
    #[comptime] fine_blocks_y: u32,
    #[comptime] fine_step: u32,
    #[comptime] mv_stride: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let neighbour_index = CUBE_POS_Z;

    let neighbour_slot = neighbour_slots[neighbour_index as usize];
    let centre_base = level_offset + centre_slot * slot_pixels;
    let neighbour_base = level_offset + neighbour_slot * slot_pixels;
    let mv_base = neighbour_index * mv_stride;

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let threads = CUBE_DIM_X * CUBE_DIM_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);
    let window_stride_len = comptime!(window_stride(blksize, search_radius));
    let window_rows = comptime!(blksize + 2 * search_radius);
    let window_area = comptime!(window_stride_len * window_rows);
    let mut window = SharedMemory::<f32>::new(window_area as usize);
    let mut plane_sads = SharedMemory::<f32>::new(ARGMIN_MAX_PLANES as usize);
    let mut plane_indices = SharedMemory::<u32>::new(ARGMIN_MAX_PLANES as usize);

    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let clamped_x = clamp_i32(block_origin_x + pixel_x as i32, level_width as i32);
            let clamped_y = clamp_i32(block_origin_y + pixel_y as i32, level_height as i32);
            let pixel_index = (clamped_y * level_width as i32 + clamped_x) as u32;
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                pyramid[(centre_base + pixel_index) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }

    let window_x0 = block_origin_x - search_radius as i32;
    let window_y0 = block_origin_y - search_radius as i32;
    stage_window(
        pyramid,
        &mut window,
        neighbour_base,
        window_x0,
        window_y0,
        thread_id,
        threads,
        level_width,
        level_height,
        blksize,
        search_radius,
    );

    sync_cube();

    blocked_window_sads(
        &centre_smem,
        &window,
        &mut sad_scratch,
        thread_id,
        threads,
        blksize,
        search_radius,
    );

    sync_cube();

    let fills_planes = threads.is_multiple_of(PLANE_DIM);
    let best_index = parallel_argmin(
        &sad_scratch,
        &mut plane_sads,
        &mut plane_indices,
        thread_id,
        threads,
        fills_planes,
        candidates,
    );

    if thread_id != 0 {
        terminate!();
    }

    // A huge start rather than a negative initialiser, which cubecl's macro does not lift cleanly.
    let mut best_sad = 1.0e30f32;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    if fills_planes {
        let last_candidate = comptime!(candidates - 1);
        let found = best_index < candidates;
        let safe_index = u32::min(best_index, last_candidate);
        let found_sad = sad_scratch[safe_index as usize];
        let offset_x = (safe_index % window_side) as i32 - search_radius as i32;
        let offset_y = (safe_index / window_side) as i32 - search_radius as i32;
        best_sad = select(found, found_sad, 1.0e30f32);
        best_dx = select(found, offset_x, 0i32);
        best_dy = select(found, offset_y, 0i32);
    } else {
        for dy in 0..window_side {
            for dx in 0..window_side {
                let candidate_sad = sad_scratch[(dy * window_side + dx) as usize];
                if candidate_sad < best_sad {
                    best_sad = candidate_sad;
                    best_dx = dx as i32 - search_radius as i32;
                    best_dy = dy as i32 - search_radius as i32;
                }
            }
        }
    }

    // A tie resolves to zero motion, so a flat block never seeds the fine pass from a shifted
    // position that no comparison preferred.
    let zero_sad = sad_scratch[(search_radius * window_side + search_radius) as usize];
    if zero_sad <= best_sad {
        best_dx = 0i32;
        best_dy = 0i32;
    }

    // Coarse blocks map to fine blocks by position, `col * step * level_scale / fine_step`. The two
    // block counts can round differently, so the last coarse block on each axis extends to the fine
    // grid's edge and every fine block is seeded exactly once. The count varies per block, so the
    // loops below are runtime `while` loops rather than unrolled.
    let mvx_fine = best_dx * level_scale as i32;
    let mvy_fine = best_dy * level_scale as i32;

    let fine_col_start = (block_col * step * level_scale / fine_step).min(fine_blocks_x);
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let fine_col_end = if block_col == CUBE_COUNT_X - 1 {
        fine_blocks_x.into()
    } else {
        ((block_col + 1) * step * level_scale / fine_step).min(fine_blocks_x)
    };
    let fine_row_start = (block_row * step * level_scale / fine_step).min(fine_blocks_y);
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let fine_row_end = if block_row == CUBE_COUNT_Y - 1 {
        fine_blocks_y.into()
    } else {
        ((block_row + 1) * step * level_scale / fine_step).min(fine_blocks_y)
    };

    let mut fine_row = fine_row_start;
    while fine_row < fine_row_end {
        let mut fine_col = fine_col_start;
        while fine_col < fine_col_end {
            let idx = (mv_base + (fine_row * fine_blocks_x + fine_col) * 2) as usize;
            mv_field[idx] = mvx_fine;
            mv_field[idx + 1] = mvy_fine;
            fine_col += 1;
        }

        fine_row += 1;
    }
}

/// Refines each block's motion at full resolution and optionally scores its confidence.
///
/// When `use_seed` is 1, the search window centres on the vector already in `mv_field`, and the
/// refined vector replaces it. A `search_radius` of 0 with no seed scores only the unshifted block.
///
/// One GPU block handles one image block. Every thread scores a share of the
/// `(2 * search_radius + 1)^2` candidates, and the lowest SAD is picked. A tie picks the first in
/// row-major order, and the seed wins a tie with that pick.
///
/// Every frame is a slot of `pyramid`, `slot_pixels` values apart. The launch runs one cube per
/// block per neighbour, with the neighbour on z. Neighbour `n` matches slot `neighbour_slots[n]`
/// against `centre_slot`, and writes its field `n * mv_stride` values in and its confidence
/// `n * conf_stride` values in.
///
/// When `write_confidence` is set, each block also writes a confidence between 0 and 1 so a poor
/// match can suppress its frame. `sad_noise_floor` is the SAD two noisy copies of the same content
/// show, and `thsad` is how far past it the confidence reaches zero. `thsad` must be positive, or
/// a perfect match divides zero by zero. When it is unset `confidence` is never touched and can be
/// a placeholder.
///
/// The neighbour pixels the search reaches are staged into shared memory once, clamped to the frame.
#[cube(launch_unchecked)]
pub fn nlm_mc_block_match_fine(
    pyramid: &Array<f32>,
    neighbour_slots: &Array<u32>,
    mv_field: &mut Array<i32>,
    confidence: &mut Array<f32>,
    centre_slot: u32,
    #[comptime] write_confidence: bool,
    sad_noise_floor: f32,
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    use_seed: u32,
    #[comptime] blocks_x: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let neighbour_index = CUBE_POS_Z;

    let neighbour_slot = neighbour_slots[neighbour_index as usize];
    let centre_base = centre_slot * slot_pixels;
    let neighbour_base = neighbour_slot * slot_pixels;
    let mv_base = neighbour_index * mv_stride;
    let conf_base = neighbour_index * conf_stride;

    let mv_slot = (mv_base + (block_row * blocks_x + block_col) * 2) as usize;

    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let seed_dx = if use_seed == 1u32 {
        mv_field[mv_slot]
    } else {
        0i32.into()
    };
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let seed_dy = if use_seed == 1u32 {
        mv_field[mv_slot + 1]
    } else {
        0i32.into()
    };

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let threads = CUBE_DIM_X * CUBE_DIM_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);
    let window_stride_len = comptime!(window_stride(blksize, search_radius));
    let window_rows = comptime!(blksize + 2 * search_radius);
    let window_area = comptime!(window_stride_len * window_rows);
    let mut window = SharedMemory::<f32>::new(window_area as usize);
    let mut plane_sads = SharedMemory::<f32>::new(ARGMIN_MAX_PLANES as usize);
    let mut plane_indices = SharedMemory::<u32>::new(ARGMIN_MAX_PLANES as usize);

    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let clamped_x = clamp_i32(block_origin_x + pixel_x as i32, width as i32);
            let clamped_y = clamp_i32(block_origin_y + pixel_y as i32, height as i32);
            let pixel_index = (clamped_y * width as i32 + clamped_x) as u32;
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                pyramid[(centre_base + pixel_index) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }

    let window_x0 = block_origin_x + seed_dx - search_radius as i32;
    let window_y0 = block_origin_y + seed_dy - search_radius as i32;
    stage_window(
        pyramid,
        &mut window,
        neighbour_base,
        window_x0,
        window_y0,
        thread_id,
        threads,
        width,
        height,
        blksize,
        search_radius,
    );

    sync_cube();

    blocked_window_sads(
        &centre_smem,
        &window,
        &mut sad_scratch,
        thread_id,
        threads,
        blksize,
        search_radius,
    );

    sync_cube();

    let fills_planes = threads.is_multiple_of(PLANE_DIM);
    let best_index = parallel_argmin(
        &sad_scratch,
        &mut plane_sads,
        &mut plane_indices,
        thread_id,
        threads,
        fills_planes,
        candidates,
    );

    if thread_id != 0 {
        terminate!();
    }

    let mut best_sad = 1.0e30f32;
    let mut best_dx = seed_dx;
    let mut best_dy = seed_dy;

    if fills_planes {
        let last_candidate = comptime!(candidates - 1);
        let found = best_index < candidates;
        let safe_index = u32::min(best_index, last_candidate);
        let found_sad = sad_scratch[safe_index as usize];
        let offset_x = (safe_index % window_side) as i32 - search_radius as i32;
        let offset_y = (safe_index / window_side) as i32 - search_radius as i32;
        best_sad = select(found, found_sad, 1.0e30f32);
        best_dx = select(found, seed_dx + offset_x, seed_dx);
        best_dy = select(found, seed_dy + offset_y, seed_dy);
    } else {
        for dy in 0..window_side {
            for dx in 0..window_side {
                let candidate_sad = sad_scratch[(dy * window_side + dx) as usize];
                if candidate_sad < best_sad {
                    best_sad = candidate_sad;
                    best_dx = seed_dx + (dx as i32 - search_radius as i32);
                    best_dy = seed_dy + (dy as i32 - search_radius as i32);
                }
            }
        }
    }

    // A tie resolves to the seed. On a flat block every candidate ties at zero, and any other
    // winner would warp in unpreferred pixels with a perfect confidence.
    let seed_sad = sad_scratch[(search_radius * window_side + search_radius) as usize];
    if seed_sad <= best_sad {
        best_sad = seed_sad;
        best_dx = seed_dx;
        best_dy = seed_dy;
    }

    mv_field[mv_slot] = best_dx;
    mv_field[mv_slot + 1] = best_dy;

    if write_confidence {
        let mut excess = best_sad - sad_noise_floor;
        if excess < 0.0f32 {
            excess = 0.0f32;
        }

        let thsad_sq = thsad * thsad;
        let excess_sq = excess * excess;
        let mut confidence_val = (thsad_sq - excess_sq) / (thsad_sq + excess_sq);
        if confidence_val < 0.0f32 {
            confidence_val = 0.0f32;
        }

        confidence[(conf_base + block_row * blocks_x + block_col) as usize] = confidence_val;
    }
}

/// Stages every neighbour pixel the search reaches into `window`, clamped to the frame.
///
/// The frame starts `neighbour_base` values into `pyramid`. Window row `window_y` and column
/// `window_x` hold the pixel at `(window_x0 + window_x, window_y0 + window_y)`, with rows
/// `window_stride` values apart.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the staging reads"
)]
fn stage_window(
    pyramid: &Array<f32>,
    window: &mut SharedMemory<f32>,
    neighbour_base: u32,
    window_x0: i32,
    window_y0: i32,
    thread_id: u32,
    threads: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] search_radius: u32,
) {
    let stride = comptime!(window_stride(blksize, search_radius));
    let rows = comptime!(blksize + 2 * search_radius);
    let area = comptime!(stride * rows);

    let mut index = thread_id;
    while index < area {
        let window_y = index / stride;
        let window_x = index % stride;
        let source_x = clamp_i32(window_x0 + window_x as i32, width as i32);
        let source_y = clamp_i32(window_y0 + window_y as i32, height as i32);
        let pixel_index = (source_y * width as i32 + source_x) as u32;
        window[index as usize] = pyramid[(neighbour_base + pixel_index) as usize];
        index += threads;
    }
}

/// Writes the SAD of every candidate offset into `sad_scratch`, indexed `dy * window_side + dx`.
///
/// Each work item covers one candidate row and `OFFSETS_PER_THREAD` adjacent offsets along it. It
/// loads each window row into registers once and scores all of its offsets from them. Every SAD
/// sums its pixels in row-major order, one block row at a time, so the result does not depend on
/// how candidates are grouped.
#[cube]
fn blocked_window_sads(
    centre_smem: &SharedMemory<f32>,
    window: &SharedMemory<f32>,
    sad_scratch: &mut SharedMemory<f32>,
    thread_id: u32,
    threads: u32,
    #[comptime] blksize: u32,
    #[comptime] search_radius: u32,
) {
    let window_side = comptime!(2 * search_radius + 1);
    let groups_per_row = comptime!(window_side.div_ceil(OFFSETS_PER_THREAD));
    let work_items = comptime!(window_side * groups_per_row);
    let row_span = comptime!(blksize + OFFSETS_PER_THREAD - 1);
    let stride = comptime!(window_stride(blksize, search_radius));

    let mut item = thread_id;
    while item < work_items {
        let dy = item / groups_per_row;
        let first_dx = (item % groups_per_row) * OFFSETS_PER_THREAD;

        let mut sads = Array::<f32>::new(OFFSETS_PER_THREAD as usize);
        #[unroll]
        for k in 0..OFFSETS_PER_THREAD {
            sads[k as usize] = 0.0f32;
        }

        for iy in 0..blksize {
            let row_start = (iy + dy) * stride + first_dx;
            let mut row = Array::<f32>::new(row_span as usize);
            #[unroll]
            for i in 0..row_span {
                row[i as usize] = window[(row_start + i) as usize];
            }

            #[unroll]
            for ix in 0..blksize {
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                #[unroll]
                for k in 0..OFFSETS_PER_THREAD {
                    let diff = centre_val - row[comptime!(ix + k) as usize];
                    sads[k as usize] += f32::abs(diff);
                }
            }
        }

        #[unroll]
        for k in 0..OFFSETS_PER_THREAD {
            let dx = first_dx + k;
            if dx < window_side {
                sad_scratch[(dy * window_side + dx) as usize] = sads[k as usize];
            }
        }

        item += threads;
    }
}

/// The index of the lowest SAD in `sad_scratch`, the lowest index on a tie.
///
/// It returns what a row-major scan with a strict `<` returns. Each lane scans every `threads`-th
/// candidate, each plane keeps its lowest SAD and then the lowest index holding it, and every lane
/// combines the planes' winners in plane order.
///
/// Plane reductions are only defined on whole planes, so when `fills_planes` is false it skips the
/// search and returns [NO_CANDIDATE]. Every thread of the cube must call it with the same
/// `fills_planes`.
#[cube]
fn parallel_argmin(
    sad_scratch: &SharedMemory<f32>,
    plane_sads: &mut SharedMemory<f32>,
    plane_indices: &mut SharedMemory<u32>,
    thread_id: u32,
    threads: u32,
    fills_planes: bool,
    #[comptime] candidates: u32,
) -> u32 {
    if fills_planes {
        let mut lane_sad = 1.0e30f32;
        let mut lane_index = NO_CANDIDATE.runtime();
        let mut index = thread_id;

        while index < candidates {
            let sad = sad_scratch[index as usize];
            if sad < lane_sad {
                lane_sad = sad;
                lane_index = index;
            }

            index += threads;
        }

        let plane_sad = plane_min(lane_sad);
        let tied_index = select(lane_sad == plane_sad, lane_index, NO_CANDIDATE);
        let plane_index = plane_min(tied_index);

        let plane = thread_id / PLANE_DIM;
        if UNIT_POS_PLANE == 0u32 {
            plane_sads[plane as usize] = plane_sad;
            plane_indices[plane as usize] = plane_index;
        }
    }

    sync_cube();

    let mut best_index = NO_CANDIDATE.runtime();

    if fills_planes {
        let planes = threads / PLANE_DIM;
        let mut best_sad = plane_sads[0];
        best_index = plane_indices[0];
        let mut other = 1u32;

        while other < planes {
            let sad = plane_sads[other as usize];
            let candidate_index = plane_indices[other as usize];
            let lower = sad < best_sad;
            let lower_tie = sad == best_sad && candidate_index < best_index;
            if lower || lower_tie {
                best_sad = sad;
                best_index = candidate_index;
            }

            other += 1u32;
        }
    }

    best_index
}

#[cube]
fn clamp_i32(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }
    result
}
