use cubecl::prelude::*;
use cubecl::terminate;

/// SAD-based block matcher run on a single luma pyramid level. One
/// cube per block; threads of the cube collectively scan the
/// `(2·search_radius + 1)²` candidate offset window and produce a
/// single i32x2 MV (in source-level pixels) at the block's slot.
///
/// Reduction strategy: each thread sums SAD over a row-strided subset
/// of the `blksize × blksize` block. Per-candidate SAD is then summed
/// into a shared-memory scratch buffer via atomic-free serial reduction
/// inside thread 0 after a `sync_cube`. This is not maximally fast but
/// is straightforward and correct, and the per-frame analyse cost is
/// already dominated by the warp kernel.
///
/// `level_scale` rescales the produced MV components into "fine-level
/// pixels" (caller passes `2^coarse_level`).
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
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);

    // Each thread initialises a slice of the scratch.
    let mut init_idx = thread_id;
    while init_idx < candidates {
        sad_scratch[init_idx as usize] = 0.0f32;
        init_idx += threads;
    }
    sync_cube();

    // Each thread scans a row-strided subset of pixels in the block,
    // accumulating partial SAD contributions for every candidate offset.
    let mut py = local_y;
    while py < blksize {
        let mut px = local_x;
        while px < blksize {
            let cx = block_origin_x + px as i32;
            let cy = block_origin_y + py as i32;
            let cx_c = clamp_i32(cx, level_width as i32);
            let cy_c = clamp_i32(cy, level_height as i32);
            let centre_val = centre[(cy_c * level_width as i32 + cx_c) as usize];

            for dy in 0..window_side {
                for dx in 0..window_side {
                    let mvx = dx as i32 - search_radius as i32;
                    let mvy = dy as i32 - search_radius as i32;
                    let nx = clamp_i32(cx + mvx, level_width as i32);
                    let ny = clamp_i32(cy + mvy, level_height as i32);
                    let neighbour_val = neighbour[(ny * level_width as i32 + nx) as usize];
                    let diff = centre_val - neighbour_val;
                    let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                    let scratch_idx = (dy * window_side + dx) as usize;
                    sad_scratch[scratch_idx] += abs_diff;
                }
            }
            px += CUBE_DIM_X;
        }
        py += CUBE_DIM_Y;
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
#[cube(launch_unchecked)]
pub fn nlm_mc_block_match_fine(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    mv_field: &mut Array<i32>,
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
    let mut sad_scratch = SharedMemory::<f32>::new(candidates as usize);

    let mut init_idx = thread_id;
    while init_idx < candidates {
        sad_scratch[init_idx as usize] = 0.0f32;
        init_idx += threads;
    }
    sync_cube();

    let mut py = local_y;
    while py < blksize {
        let mut px = local_x;
        while px < blksize {
            let cx = block_origin_x + px as i32;
            let cy = block_origin_y + py as i32;
            let cx_c = clamp_i32(cx, width as i32);
            let cy_c = clamp_i32(cy, height as i32);
            let centre_val = centre[(cy_c * width as i32 + cx_c) as usize];

            for dy in 0..window_side {
                for dx in 0..window_side {
                    let mvx = seed_dx + (dx as i32 - search_radius as i32);
                    let mvy = seed_dy + (dy as i32 - search_radius as i32);
                    let nx = clamp_i32(cx + mvx, width as i32);
                    let ny = clamp_i32(cy + mvy, height as i32);
                    let neighbour_val = neighbour[(ny * width as i32 + nx) as usize];
                    let diff = centre_val - neighbour_val;
                    let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                    let scratch_idx = (dy * window_side + dx) as usize;
                    sad_scratch[scratch_idx] += abs_diff;
                }
            }
            px += CUBE_DIM_X;
        }
        py += CUBE_DIM_Y;
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
