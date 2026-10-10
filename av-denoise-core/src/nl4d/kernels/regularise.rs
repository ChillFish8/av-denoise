use cubecl::prelude::*;

/// How many vectors each block scores, its own, the neighbourhood median, its four adjacent
/// blocks' and zero.
pub const REGULARISE_CANDIDATES: u32 = 7;

/// The most neighbours a block has in its 3x3 neighbourhood.
const NEIGHBOURHOOD: u32 = 8;

#[cube]
fn clamp_coord(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }

    result
}

#[cube]
fn abs_i32(value: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = -value;
    }

    result
}

/// Sorts the first `count` entries of `vals` in place and returns the lower median.
#[cube]
fn median_of(vals: &mut Array<i32>, count: u32) -> i32 {
    let mut i: u32 = 1;
    while i < count {
        let key = vals[i as usize];
        let mut j = i;
        while j > 0u32 && vals[(j - 1u32) as usize] > key {
            vals[j as usize] = vals[(j - 1u32) as usize];
            j -= 1u32;
        }

        vals[j as usize] = key;
        i += 1u32;
    }

    vals[((count - 1u32) / 2u32) as usize]
}

/// Loads the block's pixels from `centre` into `centre_smem`, clamped to the frame.
#[cube]
fn load_centre_tile(
    centre: &Array<f32>,
    centre_smem: &mut SharedMemory<f32>,
    block_origin_x: i32,
    block_origin_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
) {
    let mut pixel_y = UNIT_POS_Y;
    while pixel_y < blksize {
        let mut pixel_x = UNIT_POS_X;
        while pixel_x < blksize {
            let centre_x = clamp_coord(block_origin_x + pixel_x as i32, width as i32);
            let centre_y = clamp_coord(block_origin_y + pixel_y as i32, height as i32);
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                centre[(centre_y * width as i32 + centre_x) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }
}

/// Fills `candidates` with the block's 7 vectors and `median` with its neighbourhood median.
///
/// Only one thread calls it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "the helper binds the kernel's buffers and shape"
)]
fn gather_candidates(
    mv_in: &Array<i32>,
    candidates: &mut SharedMemory<i32>,
    median: &mut SharedMemory<i32>,
    block_col: u32,
    block_row: u32,
    block: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let mut neighbour_xs = Array::<i32>::new(NEIGHBOURHOOD as usize);
    let mut neighbour_ys = Array::<i32>::new(NEIGHBOURHOOD as usize);
    let mut neighbour_count: u32 = 0;
    let mut dy: u32 = 0;
    while dy < 3u32 {
        let mut dx: u32 = 0;
        while dx < 3u32 {
            if dx != 1u32 || dy != 1u32 {
                let neighbour_x = block_col as i32 + dx as i32 - 1i32;
                let neighbour_y = block_row as i32 + dy as i32 - 1i32;
                if neighbour_x >= 0
                    && neighbour_y >= 0
                    && neighbour_x < blocks_x as i32
                    && neighbour_y < blocks_y as i32
                {
                    let mv_index = ((neighbour_y as u32 * blocks_x + neighbour_x as u32) * 2u32) as usize;
                    neighbour_xs[neighbour_count as usize] = mv_in[mv_index];
                    neighbour_ys[neighbour_count as usize] = mv_in[mv_index + 1];
                    neighbour_count += 1u32;
                }
            }

            dx += 1u32;
        }

        dy += 1u32;
    }

    let own_x = mv_in[(block * 2u32) as usize];
    let own_y = mv_in[(block * 2u32 + 1u32) as usize];

    // A block with no neighbours is its own median.
    let mut median_x = own_x;
    let mut median_y = own_y;
    if neighbour_count > 0u32 {
        median_x = median_of(&mut neighbour_xs, neighbour_count);
        median_y = median_of(&mut neighbour_ys, neighbour_count);
    }

    median[0] = median_x;
    median[1] = median_y;

    candidates[0] = own_x;
    candidates[1] = own_y;
    candidates[2] = median_x;
    candidates[3] = median_y;

    // Left, right, up, down. Off-grid neighbours repeat the block's own vector, which the tie
    // rule then discards.
    let mut candidate: u32 = 2;
    let mut side: u32 = 0;
    while side < 4u32 {
        let mut neighbour_x = block_col as i32;
        let mut neighbour_y = block_row as i32;
        if side == 0u32 {
            neighbour_x -= 1;
        } else if side == 1u32 {
            neighbour_x += 1;
        } else if side == 2u32 {
            neighbour_y -= 1;
        } else {
            neighbour_y += 1;
        }

        let mut vector_x = own_x;
        let mut vector_y = own_y;
        if neighbour_x >= 0
            && neighbour_y >= 0
            && neighbour_x < blocks_x as i32
            && neighbour_y < blocks_y as i32
        {
            let mv_index = ((neighbour_y as u32 * blocks_x + neighbour_x as u32) * 2u32) as usize;
            vector_x = mv_in[mv_index];
            vector_y = mv_in[mv_index + 1];
        }

        candidates[(candidate * 2u32) as usize] = vector_x;
        candidates[(candidate * 2u32 + 1u32) as usize] = vector_y;
        candidate += 1u32;
        side += 1u32;
    }

    candidates[(candidate * 2u32) as usize] = 0;
    candidates[(candidate * 2u32 + 1u32) as usize] = 0;
}

/// Writes the cheapest candidate's vector and its confidence for `block`.
///
/// Only one thread calls it. A tie keeps the earlier candidate.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "the helper binds the kernel's buffers and shape"
)]
fn write_winner(
    candidates: &SharedMemory<i32>,
    sad_scratch: &SharedMemory<f32>,
    cost: &SharedMemory<f32>,
    mv_out: &mut Array<i32>,
    confidence_out: &mut Array<f32>,
    block: u32,
    sad_noise_floor: f32,
    thsad: f32,
) {
    let mut best: u32 = 0;
    let mut best_cost = cost[0];
    let mut candidate: u32 = 1;
    while candidate < REGULARISE_CANDIDATES {
        if cost[candidate as usize] < best_cost {
            best_cost = cost[candidate as usize];
            best = candidate;
        }

        candidate += 1u32;
    }

    mv_out[(block * 2u32) as usize] = candidates[(best * 2u32) as usize];
    mv_out[(block * 2u32 + 1u32) as usize] = candidates[(best * 2u32 + 1u32) as usize];

    let mut excess = sad_scratch[best as usize] - sad_noise_floor;
    if excess < 0.0f32 {
        excess = 0.0f32;
    }

    let thsad_sq = thsad * thsad;
    let excess_sq = excess * excess;
    let mut confidence = (thsad_sq - excess_sq) / (thsad_sq + excess_sq);
    if confidence < 0.0f32 {
        confidence = 0.0f32;
    }

    confidence_out[block as usize] = confidence;
}

/// Re-scores each block's motion vector against its 3x3 neighbourhood median.
///
/// Launch one cube per block with at least `REGULARISE_CANDIDATES` threads. `centre` and
/// `neighbour` are full-resolution luma planes. Each candidate costs its SAD over the block plus
/// `lambda_pixel` times its distance in pixels from the component-wise median of the neighbours.
/// The lowest cost wins, and a tie keeps the earlier candidate, so the block's own vector wins
/// every tie.
///
/// The winner goes to `mv_out` and a confidence from its SAD to `confidence_out`. The confidence
/// is 1 up to `sad_noise_floor` and falls to 0 at `thsad` past it.
///
/// `mv_in` and `mv_out` must be separate buffers, so the result does not depend on block order.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn nl4d_mv_regularise(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    mv_in: &Array<i32>,
    mv_out: &mut Array<i32>,
    confidence_out: &mut Array<f32>,
    lambda_pixel: f32,
    sad_noise_floor: f32,
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let block = block_row * blocks_x + block_col;
    let thread_id = UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);
    let mut candidates = SharedMemory::<i32>::new(comptime!(2 * REGULARISE_CANDIDATES) as usize);
    let mut median = SharedMemory::<i32>::new(2usize);
    let mut sad_scratch = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);
    let mut cost = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    load_centre_tile(
        centre,
        &mut centre_smem,
        block_origin_x,
        block_origin_y,
        width,
        height,
        blksize,
    );

    if thread_id == 0u32 {
        gather_candidates(
            mv_in,
            &mut candidates,
            &mut median,
            block_col,
            block_row,
            block,
            blocks_x,
            blocks_y,
        );
    }

    sync_cube();

    if thread_id < REGULARISE_CANDIDATES {
        let mvx = candidates[(thread_id * 2u32) as usize];
        let mvy = candidates[(thread_id * 2u32 + 1u32) as usize];
        let mut sad: f32 = 0.0;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let centre_x = block_origin_x + ix as i32;
                let centre_y = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let neighbour_x = clamp_coord(centre_x + mvx, width as i32);
                let neighbour_y = clamp_coord(centre_y + mvy, height as i32);
                let diff = centre_val - neighbour[(neighbour_y * width as i32 + neighbour_x) as usize];
                let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                sad += abs_diff;
            }
        }

        let deviation = abs_i32(mvx - median[0]) + abs_i32(mvy - median[1]);
        sad_scratch[thread_id as usize] = sad;
        cost[thread_id as usize] = sad + lambda_pixel * deviation as f32;
    }

    sync_cube();

    if thread_id == 0u32 {
        write_winner(
            &candidates,
            &sad_scratch,
            &cost,
            mv_out,
            confidence_out,
            block,
            sad_noise_floor,
            thsad,
        );
    }
}

/// [nl4d_mv_regularise] with every thread scoring every candidate.
///
/// Thread `t` sums pixels `t`, `t + cube_threads` and so on for all 7 candidates, so neighbouring
/// threads read neighbouring pixels. A shared memory tree then reduces each candidate's partial
/// sums. `cube_threads` must be the launched cube volume, a power of two of at least 8.
///
/// The sums run in a different order from the serial kernel, so a near tie between two candidates
/// can resolve differently.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn nl4d_mv_regularise_coop(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    mv_in: &Array<i32>,
    mv_out: &mut Array<i32>,
    confidence_out: &mut Array<f32>,
    lambda_pixel: f32,
    sad_noise_floor: f32,
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] cube_threads: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let block = block_row * blocks_x + block_col;
    let thread_id = UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);
    let mut candidates = SharedMemory::<i32>::new(comptime!(2 * REGULARISE_CANDIDATES) as usize);
    let mut median = SharedMemory::<i32>::new(2usize);
    let mut sad_scratch = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);
    let mut cost = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);
    let mut partials = SharedMemory::<f32>::new(comptime!(REGULARISE_CANDIDATES * cube_threads) as usize);

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    load_centre_tile(
        centre,
        &mut centre_smem,
        block_origin_x,
        block_origin_y,
        width,
        height,
        blksize,
    );

    if thread_id == 0u32 {
        gather_candidates(
            mv_in,
            &mut candidates,
            &mut median,
            block_col,
            block_row,
            block,
            blocks_x,
            blocks_y,
        );
    }

    sync_cube();

    let mut partial = Array::<f32>::new(REGULARISE_CANDIDATES as usize);
    #[unroll]
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

        #[unroll]
        for candidate in 0..REGULARISE_CANDIDATES {
            let mvx = candidates[(candidate * 2u32) as usize];
            let mvy = candidates[(candidate * 2u32 + 1u32) as usize];
            let neighbour_x = clamp_coord(centre_x + mvx, width as i32);
            let neighbour_y = clamp_coord(centre_y + mvy, height as i32);
            let neighbour_index = (neighbour_y * width as i32 + neighbour_x) as usize;
            let diff = centre_val - neighbour[neighbour_index];
            let abs_diff = if diff < 0.0f32 { -diff } else { diff };
            partial[candidate as usize] += abs_diff;
        }

        pixel += cube_threads;
    }

    #[unroll]
    for candidate in 0..REGULARISE_CANDIDATES {
        partials[(candidate * cube_threads + thread_id) as usize] = partial[candidate as usize];
    }

    sync_cube();

    let levels = comptime!(cube_threads.trailing_zeros());
    #[unroll]
    for level in 0..levels {
        let stride = comptime!(cube_threads >> (level + 1));
        if thread_id < stride {
            #[unroll]
            for candidate in 0..REGULARISE_CANDIDATES {
                let low = (candidate * cube_threads + thread_id) as usize;
                let high = low + stride as usize;
                partials[low] += partials[high];
            }
        }

        sync_cube();
    }

    if thread_id < REGULARISE_CANDIDATES {
        let mvx = candidates[(thread_id * 2u32) as usize];
        let mvy = candidates[(thread_id * 2u32 + 1u32) as usize];
        let sad = partials[(thread_id * cube_threads) as usize];
        let deviation = abs_i32(mvx - median[0]) + abs_i32(mvy - median[1]);
        sad_scratch[thread_id as usize] = sad;
        cost[thread_id as usize] = sad + lambda_pixel * deviation as f32;
    }

    sync_cube();

    if thread_id == 0u32 {
        write_winner(
            &candidates,
            &sad_scratch,
            &cost,
            mv_out,
            confidence_out,
            block,
            sad_noise_floor,
            thsad,
        );
    }
}
