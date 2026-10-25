use cubecl::prelude::*;
use cubecl::terminate;

/// The per-candidate block match the production kernel must reproduce.
#[cube(launch_unchecked)]
pub(super) fn reference_block_match_coarse(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    mv_field: &mut Array<i32>,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    #[comptime] level_scale: u32,
    #[comptime] fine_blocks_x: u32,
    #[comptime] fine_blocks_y: u32,
    #[comptime] fine_step: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;

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

    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let clamped_x = clamp_i32(block_origin_x + pixel_x as i32, level_width as i32);
            let clamped_y = clamp_i32(block_origin_y + pixel_y as i32, level_height as i32);
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                centre[(clamped_y * level_width as i32 + clamped_x) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }

    sync_cube();

    let mut candidate_idx = thread_id;
    while candidate_idx < candidates {
        let dy = candidate_idx / window_side;
        let dx = candidate_idx % window_side;
        let mvx = dx as i32 - search_radius as i32;
        let mvy = dy as i32 - search_radius as i32;

        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let centre_x = block_origin_x + ix as i32;
                let centre_y = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let neighbour_x = clamp_i32(centre_x + mvx, level_width as i32);
                let neighbour_y = clamp_i32(centre_y + mvy, level_height as i32);
                let neighbour_val = neighbour[(neighbour_y * level_width as i32 + neighbour_x) as usize];
                let diff = centre_val - neighbour_val;
                let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                sad += abs_diff;
            }
        }

        sad_scratch[candidate_idx as usize] = sad;
        candidate_idx += threads;
    }

    sync_cube();

    if thread_id != 0 {
        terminate!();
    }

    // A huge start rather than a negative initialiser, which cubecl's macro does not lift cleanly.
    let mut best_sad = 1.0e30f32;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

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
            let idx = ((fine_row * fine_blocks_x + fine_col) * 2) as usize;
            mv_field[idx] = mvx_fine;
            mv_field[idx + 1] = mvy_fine;
            fine_col += 1;
        }

        fine_row += 1;
    }
}

/// The per-candidate block match the production kernel must reproduce.
#[cube(launch_unchecked)]
pub(super) fn reference_block_match_fine(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    mv_field: &mut Array<i32>,
    confidence: &mut Array<f32>,
    #[comptime] write_confidence: bool,
    sad_noise_floor: f32,
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    use_seed: u32,
    #[comptime] blocks_x: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;

    let mv_slot = ((block_row * blocks_x + block_col) * 2) as usize;

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

    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let clamped_x = clamp_i32(block_origin_x + pixel_x as i32, width as i32);
            let clamped_y = clamp_i32(block_origin_y + pixel_y as i32, height as i32);
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                centre[(clamped_y * width as i32 + clamped_x) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }

    sync_cube();

    let mut candidate_idx = thread_id;
    while candidate_idx < candidates {
        let dy = candidate_idx / window_side;
        let dx = candidate_idx % window_side;
        let mvx = seed_dx + (dx as i32 - search_radius as i32);
        let mvy = seed_dy + (dy as i32 - search_radius as i32);

        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let centre_x = block_origin_x + ix as i32;
                let centre_y = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let neighbour_x = clamp_i32(centre_x + mvx, width as i32);
                let neighbour_y = clamp_i32(centre_y + mvy, height as i32);
                let neighbour_val = neighbour[(neighbour_y * width as i32 + neighbour_x) as usize];
                let diff = centre_val - neighbour_val;
                let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                sad += abs_diff;
            }
        }

        sad_scratch[candidate_idx as usize] = sad;
        candidate_idx += threads;
    }

    sync_cube();

    if thread_id != 0 {
        terminate!();
    }

    let mut best_sad = 1.0e30f32;
    let mut best_dx = seed_dx;
    let mut best_dy = seed_dy;

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

        confidence[(block_row * blocks_x + block_col) as usize] = confidence_val;
    }
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
