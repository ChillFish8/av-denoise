use cubecl::prelude::*;

use crate::nl4d::kernels::REGULARISE_CANDIDATES;
use crate::noise::DERIVED_SAD_FLOOR;

/// The most neighbours a block has in its 3x3 neighbourhood.
const NEIGHBOURHOOD: u32 = 8;

#[cube]
fn reference_clamp_coord(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }

    result
}

#[cube]
fn reference_abs_i32(value: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = -value;
    }

    result
}

/// Sorts the first `count` entries of `vals` in place and returns the lower median.
#[cube]
fn reference_median_of(vals: &mut [i32], count: u32) -> i32 {
    let mut i = 1u32;
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

/// The serial regularise kernel, kept verbatim as the oracle the shared helpers are checked against.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn nl4d_mv_regularise_reference(
    centre: &[f32],
    neighbour: &[f32],
    mv_in: &[i32],
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    lambda_pixel: f32,
    noise: &[f32],
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
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = Shared::<[f32]>::new_slice(block_pixels as usize);
    let mut candidates = Shared::<[i32]>::new_slice(comptime!(2 * REGULARISE_CANDIDATES) as usize);
    let mut median = Shared::<[i32]>::new_slice(2usize);
    let mut sad_scratch = Shared::<[f32]>::new_slice(REGULARISE_CANDIDATES as usize);
    let mut cost = Shared::<[f32]>::new_slice(REGULARISE_CANDIDATES as usize);

    let block_origin_x = block_col as i32 * step as i32;
    let block_origin_y = block_row as i32 * step as i32;

    // The centre tile, loaded once and shared by every candidate.
    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let centre_x = reference_clamp_coord(block_origin_x + pixel_x as i32, width as i32);
            let centre_y = reference_clamp_coord(block_origin_y + pixel_y as i32, height as i32);
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                centre[(centre_y * width as i32 + centre_x) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }

    if thread_id == 0u32 {
        let mut neighbour_xs = Array::<i32>::new(NEIGHBOURHOOD as usize);
        let mut neighbour_ys = Array::<i32>::new(NEIGHBOURHOOD as usize);
        let mut neighbour_count = 0u32;
        let mut dy = 0u32;
        while dy < 3u32 {
            let mut dx = 0u32;
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
            median_x = reference_median_of(neighbour_xs.as_mut_slice(), neighbour_count);
            median_y = reference_median_of(neighbour_ys.as_mut_slice(), neighbour_count);
        }

        median[0] = median_x;
        median[1] = median_y;

        candidates[0] = own_x;
        candidates[1] = own_y;
        candidates[2] = median_x;
        candidates[3] = median_y;

        // Left, right, up, down. Off-grid neighbours repeat the block's own vector, which the tie
        // rule then discards.
        let mut candidate = 2u32;
        let mut side = 0u32;
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

    sync_cube();

    if thread_id < REGULARISE_CANDIDATES {
        let mvx = candidates[(thread_id * 2u32) as usize];
        let mvy = candidates[(thread_id * 2u32 + 1u32) as usize];
        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let centre_x = block_origin_x + ix as i32;
                let centre_y = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let neighbour_x = reference_clamp_coord(centre_x + mvx, width as i32);
                let neighbour_y = reference_clamp_coord(centre_y + mvy, height as i32);
                let diff = centre_val - neighbour[(neighbour_y * width as i32 + neighbour_x) as usize];
                let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                sad += abs_diff;
            }
        }

        let deviation = reference_abs_i32(mvx - median[0]) + reference_abs_i32(mvy - median[1]);
        sad_scratch[thread_id as usize] = sad;
        cost[thread_id as usize] = sad + lambda_pixel * deviation as f32;
    }

    sync_cube();

    if thread_id == 0u32 {
        let mut best = 0u32;
        let mut best_cost = cost[0];
        let mut candidate = 1u32;
        while candidate < REGULARISE_CANDIDATES {
            if cost[candidate as usize] < best_cost {
                best_cost = cost[candidate as usize];
                best = candidate;
            }

            candidate += 1u32;
        }

        mv_out[(block * 2u32) as usize] = candidates[(best * 2u32) as usize];
        mv_out[(block * 2u32 + 1u32) as usize] = candidates[(best * 2u32 + 1u32) as usize];

        let sad_noise_floor = noise[DERIVED_SAD_FLOOR as usize];
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
}
