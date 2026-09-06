use cubecl::prelude::*;

/// How many vectors a block considers, its own, the neighbourhood
/// median, the four adjacent blocks' and zero.
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

/// Sorts the first `n` entries of `vals` in place and returns the lower
/// median.
#[cube]
fn median_of(vals: &mut Array<i32>, n: u32) -> i32 {
    let mut i: u32 = 1;
    while i < n {
        let key = vals[i as usize];
        let mut j = i;
        while j > 0u32 && vals[(j - 1u32) as usize] > key {
            vals[j as usize] = vals[(j - 1u32) as usize];
            j -= 1u32;
        }
        vals[j as usize] = key;
        i += 1u32;
    }
    vals[((n - 1u32) / 2u32) as usize]
}

/// Re-scores one block's motion vector against its neighbourhood and
/// writes the winner, with a fresh confidence, to the output field.
///
/// One cube handles one block of the field. Thread 0 gathers the 3x3
/// neighbourhood's vectors from `mv_in`, takes their component-wise
/// median, and lays out the candidates in shared memory. The block's
/// own vector is candidate 0. Each of the next threads scores one
/// candidate by SAD over the block on the level-0 luma planes, plus
/// `lambda_pixel` times the candidate's distance from the median in
/// pixels. Thread 0 then picks the lowest cost, and a tie keeps the
/// earlier candidate, so the block's own vector wins every tie.
///
/// The winner's confidence is derived from its SAD exactly as
/// `nlm_mc_block_match_fine` derives it, with the same
/// `sad_noise_floor` and `thsad`.
///
/// `mv_in` and `mv_out` are separate buffers. Every block reads the
/// whole input before any block's output exists, so the result does
/// not depend on block order.
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
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;
    let block = by * blocks_x + bx;
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let block_pixels = comptime!(blksize * blksize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);
    let mut cand = SharedMemory::<i32>::new(comptime!(2 * REGULARISE_CANDIDATES) as usize);
    let mut median = SharedMemory::<i32>::new(2usize);
    let mut sad_scratch = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);
    let mut cost = SharedMemory::<f32>::new(REGULARISE_CANDIDATES as usize);

    let block_origin_x = bx as i32 * step as i32;
    let block_origin_y = by as i32 * step as i32;

    // The centre tile, loaded once and shared by every candidate.
    let mut py = local_y;
    while py < blksize {
        let mut px = local_x;
        while px < blksize {
            let cx = clamp_coord(block_origin_x + px as i32, width as i32);
            let cy = clamp_coord(block_origin_y + py as i32, height as i32);
            centre_smem[(py * blksize + px) as usize] = centre[(cy * width as i32 + cx) as usize];
            px += CUBE_DIM_X;
        }
        py += CUBE_DIM_Y;
    }

    if thread_id == 0u32 {
        let mut xs = Array::<i32>::new(NEIGHBOURHOOD as usize);
        let mut ys = Array::<i32>::new(NEIGHBOURHOOD as usize);
        let mut n: u32 = 0;
        let mut dy: u32 = 0;
        while dy < 3u32 {
            let mut dx: u32 = 0;
            while dx < 3u32 {
                if dx != 1u32 || dy != 1u32 {
                    let nx = bx as i32 + dx as i32 - 1i32;
                    let ny = by as i32 + dy as i32 - 1i32;
                    if nx >= 0 && ny >= 0 && nx < blocks_x as i32 && ny < blocks_y as i32 {
                        let idx = ((ny as u32 * blocks_x + nx as u32) * 2u32) as usize;
                        xs[n as usize] = mv_in[idx];
                        ys[n as usize] = mv_in[idx + 1];
                        n += 1u32;
                    }
                }
                dx += 1u32;
            }
            dy += 1u32;
        }
        let own_x = mv_in[(block * 2u32) as usize];
        let own_y = mv_in[(block * 2u32 + 1u32) as usize];
        // A block with no neighbours is its own median.
        let mut mx = own_x;
        let mut my = own_y;
        if n > 0u32 {
            mx = median_of(&mut xs, n);
            my = median_of(&mut ys, n);
        }
        median[0] = mx;
        median[1] = my;

        cand[0] = own_x;
        cand[1] = own_y;
        cand[2] = mx;
        cand[3] = my;
        // Left, right, up, down. Off-grid neighbours repeat the block's
        // own vector, which the tie rule then discards.
        let mut c: u32 = 2;
        let mut side: u32 = 0;
        while side < 4u32 {
            let mut nx = bx as i32;
            let mut ny = by as i32;
            if side == 0u32 {
                nx -= 1;
            } else if side == 1u32 {
                nx += 1;
            } else if side == 2u32 {
                ny -= 1;
            } else {
                ny += 1;
            }
            let mut vx = own_x;
            let mut vy = own_y;
            if nx >= 0 && ny >= 0 && nx < blocks_x as i32 && ny < blocks_y as i32 {
                let idx = ((ny as u32 * blocks_x + nx as u32) * 2u32) as usize;
                vx = mv_in[idx];
                vy = mv_in[idx + 1];
            }
            cand[(c * 2u32) as usize] = vx;
            cand[(c * 2u32 + 1u32) as usize] = vy;
            c += 1u32;
            side += 1u32;
        }
        cand[(c * 2u32) as usize] = 0;
        cand[(c * 2u32 + 1u32) as usize] = 0;
    }
    sync_cube();

    if thread_id < REGULARISE_CANDIDATES {
        let mvx = cand[(thread_id * 2u32) as usize];
        let mvy = cand[(thread_id * 2u32 + 1u32) as usize];
        let mut sad: f32 = 0.0;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let cx = block_origin_x + ix as i32;
                let cy = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let nx = clamp_coord(cx + mvx, width as i32);
                let ny = clamp_coord(cy + mvy, height as i32);
                let diff = centre_val - neighbour[(ny * width as i32 + nx) as usize];
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
        let mut best: u32 = 0;
        let mut best_cost = cost[0];
        let mut c: u32 = 1;
        while c < REGULARISE_CANDIDATES {
            if cost[c as usize] < best_cost {
                best_cost = cost[c as usize];
                best = c;
            }
            c += 1u32;
        }
        mv_out[(block * 2u32) as usize] = cand[(best * 2u32) as usize];
        mv_out[(block * 2u32 + 1u32) as usize] = cand[(best * 2u32 + 1u32) as usize];

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
