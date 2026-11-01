use cubecl::prelude::*;
use cubecl::terminate;

use crate::motion::global::AFFINE_WORDS;

/// One window's winner, or the running winner across windows.
#[derive(CubeType, Clone, Copy)]
struct ReferenceBest {
    sad: f32,
    dx: i32,
    dy: i32,
}

/// The per-candidate block match the production kernel must reproduce.
///
/// With `model_candidate`, a second window centres on the rounded model displacement at the block
/// centre, and wins only when its SAD plus `model_bias` is below the zero window's. An identity
/// prediction never wins.
#[cube(launch_unchecked)]
pub(super) fn reference_block_match_coarse(
    centre: &[f32],
    neighbour: &[f32],
    mv_field: &mut [i32],
    predictions: &[f32],
    model_bias: f32,
    #[comptime] model_candidate: bool,
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

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = Shared::<[f32]>::new_slice(block_pixels as usize);
    load_centre(
        centre,
        &mut centre_smem,
        block_origin_x,
        block_origin_y,
        level_width,
        level_height,
        blksize,
    );

    let zero_window = reference_window(
        &centre_smem,
        neighbour,
        block_origin_x,
        block_origin_y,
        0i32,
        0i32,
        level_width,
        level_height,
        blksize,
        search_radius,
    );
    let mut best_dx = zero_window.dx;
    let mut best_dy = zero_window.dy;

    if model_candidate {
        let model_dx = reference_model_dx(
            predictions,
            block_origin_x,
            block_origin_y,
            blksize,
            level_scale,
            level_width,
        );
        let model_dy = reference_model_dy(
            predictions,
            block_origin_x,
            block_origin_y,
            blksize,
            level_scale,
            level_height,
        );
        let model_window = reference_window(
            &centre_smem,
            neighbour,
            block_origin_x,
            block_origin_y,
            model_dx,
            model_dy,
            level_width,
            level_height,
            blksize,
            search_radius,
        );

        let moves = reference_predicts_motion(predictions);
        if moves && model_window.sad + model_bias < zero_window.sad {
            best_dx = model_window.dx;
            best_dy = model_window.dy;
        }
    }

    if UNIT_POS != 0 {
        terminate!();
    }

    // Coarse blocks map to fine blocks by position, `col * step * level_scale / fine_step`. The two
    // block counts can round differently, so the last coarse block on each axis extends to the fine
    // grid's edge and every fine block is seeded exactly once. The count varies per block, so the
    // loops below are runtime `while` loops rather than unrolled.
    let mvx_fine = best_dx * level_scale as i32;
    let mvy_fine = best_dy * level_scale as i32;

    let fine_col_start = (block_col * step * level_scale / fine_step).min(fine_blocks_x);
    let fine_col_end = if block_col == CUBE_COUNT_X - 1 {
        fine_blocks_x
    } else {
        ((block_col + 1) * step * level_scale / fine_step).min(fine_blocks_x)
    };
    let fine_row_start = (block_row * step * level_scale / fine_step).min(fine_blocks_y);
    let fine_row_end = if block_row == CUBE_COUNT_Y - 1 {
        fine_blocks_y
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
///
/// With `model_candidate`, a second window centres on the rounded model displacement at the block
/// centre, and wins only when its SAD plus `model_bias` is below the seed window's. An identity
/// prediction never wins.
#[cube(launch_unchecked)]
pub(super) fn reference_block_match_fine(
    centre: &[f32],
    neighbour: &[f32],
    mv_field: &mut [i32],
    confidence: &mut [f32],
    predictions: &[f32],
    model_bias: f32,
    #[comptime] model_candidate: bool,
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

    let seed_dx = if use_seed == 1u32 { mv_field[mv_slot] } else { 0i32 };
    let seed_dy = if use_seed == 1u32 {
        mv_field[mv_slot + 1]
    } else {
        0i32
    };

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = Shared::<[f32]>::new_slice(block_pixels as usize);
    load_centre(
        centre,
        &mut centre_smem,
        block_origin_x,
        block_origin_y,
        width,
        height,
        blksize,
    );

    let seed_window = reference_window(
        &centre_smem,
        neighbour,
        block_origin_x,
        block_origin_y,
        seed_dx,
        seed_dy,
        width,
        height,
        blksize,
        search_radius,
    );
    let mut best_sad = seed_window.sad;
    let mut best_dx = seed_window.dx;
    let mut best_dy = seed_window.dy;

    if model_candidate {
        let model_dx = reference_model_dx(predictions, block_origin_x, block_origin_y, blksize, 1u32, width);
        let model_dy = reference_model_dy(predictions, block_origin_x, block_origin_y, blksize, 1u32, height);
        let model_window = reference_window(
            &centre_smem,
            neighbour,
            block_origin_x,
            block_origin_y,
            model_dx,
            model_dy,
            width,
            height,
            blksize,
            search_radius,
        );

        let moves = reference_predicts_motion(predictions);
        if moves && model_window.sad + model_bias < seed_window.sad {
            best_sad = model_window.sad;
            best_dx = model_window.dx;
            best_dy = model_window.dy;
        }
    }

    if UNIT_POS != 0 {
        terminate!();
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
fn load_centre(
    centre: &[f32],
    centre_smem: &mut Shared<[f32]>,
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
            let clamped_x = clamp_i32(block_origin_x + pixel_x as i32, width as i32);
            let clamped_y = clamp_i32(block_origin_y + pixel_y as i32, height as i32);
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                centre[(clamped_y * width as i32 + clamped_x) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }
}

/// Scores every offset around `(centre_dx, centre_dy)` and returns the lowest on thread 0.
///
/// A tie picks the first in row-major order, and the centre wins a tie with that pick.
#[cube]
#[expect(clippy::too_many_arguments, reason = "each argument is one search parameter")]
fn reference_window(
    centre_smem: &Shared<[f32]>,
    neighbour: &[f32],
    block_origin_x: i32,
    block_origin_y: i32,
    centre_dx: i32,
    centre_dy: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] search_radius: u32,
) -> ReferenceBest {
    let threads = CUBE_DIM_X * CUBE_DIM_Y;
    let thread_id = UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let mut sad_scratch = Shared::<[f32]>::new_slice(candidates as usize);

    // The centre tile is loaded, and an earlier window's scratch is no longer read.
    sync_cube();

    let mut candidate_idx = thread_id;
    while candidate_idx < candidates {
        let dy = candidate_idx / window_side;
        let dx = candidate_idx % window_side;
        let mvx = centre_dx + (dx as i32 - search_radius as i32);
        let mvy = centre_dy + (dy as i32 - search_radius as i32);

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

    let mut best_sad = 1.0e30f32;
    let mut best_dx = centre_dx;
    let mut best_dy = centre_dy;

    if thread_id == 0 {
        for dy in 0..window_side {
            for dx in 0..window_side {
                let candidate_sad = sad_scratch[(dy * window_side + dx) as usize];
                if candidate_sad < best_sad {
                    best_sad = candidate_sad;
                    best_dx = centre_dx + (dx as i32 - search_radius as i32);
                    best_dy = centre_dy + (dy as i32 - search_radius as i32);
                }
            }
        }

        let centre_sad = sad_scratch[(search_radius * window_side + search_radius) as usize];
        if centre_sad <= best_sad {
            best_sad = centre_sad;
            best_dx = centre_dx;
            best_dy = centre_dy;
        }
    }

    ReferenceBest {
        sad: best_sad,
        dx: best_dx,
        dy: best_dy,
    }
}

/// Whether the neighbour's prediction differs from identity.
#[cube]
fn reference_predicts_motion(predictions: &[f32]) -> bool {
    let base = (CUBE_POS_Z * AFFINE_WORDS) as usize;
    let mut moves = false;
    for word in 0..AFFINE_WORDS {
        let identity_word = if word == 0 || word == 3 { 1.0f32 } else { 0.0f32 };
        if predictions[base + word as usize] != identity_word {
            moves = true;
        }
    }

    moves
}

/// The model's x displacement at the block centre on a level `level_scale` times smaller.
#[cube]
fn reference_model_dx(
    predictions: &[f32],
    block_origin_x: i32,
    block_origin_y: i32,
    #[comptime] blksize: u32,
    #[comptime] level_scale: u32,
    #[comptime] level_width: u32,
) -> i32 {
    let base = (CUBE_POS_Z * AFFINE_WORDS) as usize;
    let scale = comptime!(level_scale as f32);
    let half_block = comptime!((blksize / 2) as i32);
    let centre_x = (block_origin_x + half_block) as f32 * scale;
    let centre_y = (block_origin_y + half_block) as f32 * scale;
    let mapped_x = predictions[base] * centre_x + predictions[base + 1] * centre_y + predictions[base + 4];
    reference_round((mapped_x - centre_x) / scale, level_width)
}

/// The model's y displacement at the block centre on a level `level_scale` times smaller.
#[cube]
fn reference_model_dy(
    predictions: &[f32],
    block_origin_x: i32,
    block_origin_y: i32,
    #[comptime] blksize: u32,
    #[comptime] level_scale: u32,
    #[comptime] level_height: u32,
) -> i32 {
    let base = (CUBE_POS_Z * AFFINE_WORDS) as usize;
    let scale = comptime!(level_scale as f32);
    let half_block = comptime!((blksize / 2) as i32);
    let centre_x = (block_origin_x + half_block) as f32 * scale;
    let centre_y = (block_origin_y + half_block) as f32 * scale;
    let mapped_y =
        predictions[base + 2] * centre_x + predictions[base + 3] * centre_y + predictions[base + 5];
    reference_round((mapped_y - centre_y) / scale, level_height)
}

/// Caps `value` to the level's size and rounds halves up.
#[cube]
fn reference_round(value: f32, #[comptime] limit: u32) -> i32 {
    let bound = comptime!(limit as f32);
    let capped = f32::clamp(value, -bound, bound);
    let rounded = f32::floor(capped + 0.5f32);
    rounded as i32
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
