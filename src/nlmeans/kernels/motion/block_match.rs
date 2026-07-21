use cubecl::prelude::*;
use cubecl::terminate;

/// SAD-based block matcher run on a single luma pyramid level. One
/// cube per block; threads of the cube collectively scan the
/// `(2·search_radius + 1)²` candidate offset window and produce a
/// single i32x2 MV (in source-level pixels) at the block's slot.
///
/// The reduction works as follows. The cube first cooperatively loads
/// the block's `blksize × blksize` centre pixels into shared memory
/// once (a row-strided split across threads). Each thread then owns a
/// strided subset of the candidate offsets and accumulates that candidate's
/// full SAD serially, reading the centre value from shared memory and
/// the neighbour value from global memory, and writes its candidate's
/// scratch slot exactly once. Because every scratch slot is written by
/// exactly one thread, no atomics or partial-then-reduce split is
/// needed. Thread 0 then does a serial argmin over the scratch buffer
/// after a `sync_cube`. `blksize` is capped at [`MAX_BLKSIZE`] (32), so
/// the shared centre tile never exceeds 1024 `f32`.
///
/// `level_scale` rescales the produced MV components into "fine-level
/// pixels" (caller passes `2^coarse_level`).
///
/// [`MAX_BLKSIZE`]: crate::nlmeans::motion::MAX_BLKSIZE
#[cube(launch_unchecked)]
pub fn nlm_mc_block_match_coarse(
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
) {
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;

    let block_origin_x = bx as i32 * step as i32;
    let block_origin_y = by as i32 * step as i32;

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let threads = CUBE_DIM_X * CUBE_DIM_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);

    // This is a cooperative load. Each thread claims a row-strided
    // subset of the block's pixels and caches the (clamped) centre
    // value once, so every candidate below reads it from shared memory
    // instead of re-fetching it from global memory once per candidate.
    let mut py = local_y;
    while py < blksize {
        let mut px = local_x;
        while px < blksize {
            let cx_c = clamp_i32(block_origin_x + px as i32, level_width as i32);
            let cy_c = clamp_i32(block_origin_y + py as i32, level_height as i32);
            centre_smem[(py * blksize + px) as usize] = centre[(cy_c * level_width as i32 + cx_c) as usize];
            px += CUBE_DIM_X;
        }
        py += CUBE_DIM_Y;
    }
    sync_cube();

    // Each thread owns a strided subset of candidates and accumulates
    // that candidate's full SAD serially over the cached block, then
    // writes its scratch slot exactly once. Splitting the reduction by
    // candidate (rather than by pixel) means no two threads ever write
    // the same slot, so no atomics or reduce pass are needed.
    let mut candidate_idx = thread_id;
    while candidate_idx < candidates {
        let dy = candidate_idx / window_side;
        let dx = candidate_idx % window_side;
        let mvx = dx as i32 - search_radius as i32;
        let mvy = dy as i32 - search_radius as i32;

        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let cx = block_origin_x + ix as i32;
                let cy = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let nx = clamp_i32(cx + mvx, level_width as i32);
                let ny = clamp_i32(cy + mvy, level_height as i32);
                let neighbour_val = neighbour[(ny * level_width as i32 + nx) as usize];
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

    // Serial argmin over candidates. window_side is comptime so this
    // unrolls cleanly for small search radii. Initialise with a huge
    // sentinel so the first iteration always wins, avoiding a comptime
    // negative-init dance that cubecl's macro doesn't lift cleanly.
    let mut best_sad = 1.0e30f32;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    for dy in 0..window_side {
        for dx in 0..window_side {
            let s = sad_scratch[(dy * window_side + dx) as usize];
            if s < best_sad {
                best_sad = s;
                best_dx = dx as i32 - search_radius as i32;
                best_dy = dy as i32 - search_radius as i32;
            }
        }
    }

    // Project coarse block index into the fine-block index space. The
    // fine grid runs at `level_scale × coarse` density, so a single
    // coarse block seeds a `level_scale × level_scale` patch of fine
    // blocks. Each thread-0 here writes one coarse block; the seeding
    // pattern below covers the corresponding fine patch.
    let mvx_fine = best_dx * level_scale as i32;
    let mvy_fine = best_dy * level_scale as i32;

    let fine_bx_origin = bx * level_scale;
    let fine_by_origin = by * level_scale;
    for fy in 0..level_scale {
        let fby = fine_by_origin + fy;
        if fby >= fine_blocks_y {
            // outside; skip
        } else {
            for fx in 0..level_scale {
                let fbx = fine_bx_origin + fx;
                if fbx < fine_blocks_x {
                    let idx = ((fby * fine_blocks_x + fbx) * 2) as usize;
                    mv_field[idx] = mvx_fine;
                    mv_field[idx + 1] = mvy_fine;
                }
            }
        }
    }
}

/// Fine-resolution refinement pass. Reads a seed MV from `mv_field`
/// when `use_seed != 0`, then searches a small `(2·search_radius + 1)²`
/// window around it. Writes the refined MV back into the same slot.
/// Uses the same shared-memory-cached, candidate-parallel SAD
/// reduction as [`nlm_mc_block_match_coarse`] (see its doc comment).
///
/// When `write_confidence` is `true`, also writes a per-block
/// confidence score to `confidence`, derived from the winning SAD.
/// `sad_noise_floor` is the SAD two noisy copies of otherwise identical
/// content produce by chance, subtracted before thresholding so a
/// clean match isn't penalised for the noise it carries. `thsad` is
/// the excess-SAD value beyond which confidence collapses to zero.
/// Callers must pass a strictly positive `thsad` whenever
/// `write_confidence` is `true`, or the confidence expression divides
/// zero by zero on a perfect match. A `search_radius` of 0 with
/// `use_seed = 0` reduces the match to a single candidate at the
/// block's un-shifted position, useful for a confidence-only pass with
/// no actual motion search.
///
/// When `write_confidence` is `false`, the confidence computation and
/// write are skipped entirely (a comptime branch, so this costs
/// nothing at runtime). Callers that don't need confidence pass a
/// small placeholder buffer for `confidence` in that case. Its size
/// never matters since the kernel never indexes into it.
#[cube(launch_unchecked)]
pub fn nlm_mc_block_match_fine(
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
    #[comptime] _blocks_y: u32,
) {
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;

    let mv_slot = ((by * blocks_x + bx) * 2) as usize;

    // The `.into()` calls clippy's `useless_conversion` lint
    // away from these lines, but are actually required: the `if`
    // branches must produce matching cubecl `NativeExpand<i32>` types
    // and a bare `0i32` literal won't coerce inside the cube macro.
    #[allow(clippy::useless_conversion)]
    let seed_dx = if use_seed == 1u32 {
        mv_field[mv_slot]
    } else {
        0i32.into()
    };
    #[allow(clippy::useless_conversion)]
    let seed_dy = if use_seed == 1u32 {
        mv_field[mv_slot + 1]
    } else {
        0i32.into()
    };

    let block_origin_x = bx as i32 * step as i32;
    let block_origin_y = by as i32 * step as i32;

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let threads = CUBE_DIM_X * CUBE_DIM_Y;
    let thread_id = local_y * CUBE_DIM_X + local_x;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut centre_smem = SharedMemory::<f32>::new(block_pixels as usize);

    // This is a cooperative load. Each thread claims a row-strided
    // subset of the block's pixels and caches the (clamped) centre
    // value once, so every candidate below reads it from shared memory
    // instead of re-fetching it from global memory once per candidate.
    let mut py = local_y;
    while py < blksize {
        let mut px = local_x;
        while px < blksize {
            let cx_c = clamp_i32(block_origin_x + px as i32, width as i32);
            let cy_c = clamp_i32(block_origin_y + py as i32, height as i32);
            centre_smem[(py * blksize + px) as usize] = centre[(cy_c * width as i32 + cx_c) as usize];
            px += CUBE_DIM_X;
        }
        py += CUBE_DIM_Y;
    }
    sync_cube();

    // Each thread owns a strided subset of candidates and accumulates
    // that candidate's full SAD serially over the cached block, then
    // writes its scratch slot exactly once. Splitting the reduction by
    // candidate (rather than by pixel) means no two threads ever write
    // the same slot, so no atomics or reduce pass are needed.
    let mut candidate_idx = thread_id;
    while candidate_idx < candidates {
        let dy = candidate_idx / window_side;
        let dx = candidate_idx % window_side;
        let mvx = seed_dx + (dx as i32 - search_radius as i32);
        let mvy = seed_dy + (dy as i32 - search_radius as i32);

        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let cx = block_origin_x + ix as i32;
                let cy = block_origin_y + iy as i32;
                let centre_val = centre_smem[(iy * blksize + ix) as usize];
                let nx = clamp_i32(cx + mvx, width as i32);
                let ny = clamp_i32(cy + mvy, height as i32);
                let neighbour_val = neighbour[(ny * width as i32 + nx) as usize];
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
            let s = sad_scratch[(dy * window_side + dx) as usize];
            if s < best_sad {
                best_sad = s;
                best_dx = seed_dx + (dx as i32 - search_radius as i32);
                best_dy = seed_dy + (dy as i32 - search_radius as i32);
            }
        }
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
        confidence[(by * blocks_x + bx) as usize] = confidence_val;
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
