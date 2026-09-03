use cubecl::prelude::*;
use cubecl::terminate;

/// Finds where each block of pixels moved to, by searching one level of
/// the luma pyramid for the offset with the smallest sum of absolute
/// differences.
///
/// One GPU block handles one image block. Its threads scan the
/// `(2 * search_radius + 1)^2` candidate offsets between them and
/// produce a single motion vector, in this level's pixels, at the
/// block's slot.
///
/// # How the search is split up
///
/// The block first loads its own `blksize x blksize` centre pixels into
/// shared memory once, splitting the rows across threads.
///
/// Each thread then takes a strided share of the candidate offsets and
/// works out each one's full score on its own, reading the centre from
/// shared memory and the neighbour from global memory. It writes each of
/// its candidates' scores exactly once.
///
/// Because every scratch slot has exactly one writer, no atomics and no
/// second reduction pass are needed. After a `sync_cube`, thread 0 walks
/// the scratch buffer and picks the winner.
///
/// `blksize` is capped at [`MAX_BLKSIZE`], which is 32, so the shared
/// centre tile never exceeds 1024 values.
///
/// # Seeding the fine pass
///
/// `level_scale` rescales the motion vector into fine-level pixels. The
/// caller passes 2 raised to the coarse level.
///
/// After finding its own winner, the block seeds every fine block whose
/// position falls inside its own source region. `step` and `fine_step`
/// are the coarse and fine grids' block spacings, which is what converts
/// between the two index spaces.
///
/// The seeding code below spells out the exact formula.
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
    #[comptime] fine_step: u32,
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

    // A cooperative load. Each thread claims a row-strided share of the
    // block's pixels and caches the clamped centre value once, so every
    // candidate below reads it from shared memory instead of fetching
    // it again from global memory.
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

    // Each thread takes a strided share of the candidates, works out
    // each one's full score over the cached block, and writes its
    // scratch slot exactly once. Splitting the work by candidate rather
    // than by pixel means no two threads ever write the same slot, so
    // no atomics and no reduce pass are needed.
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

    // Walk the candidates and keep the lowest score. `window_side` is
    // known at compile time, so this unrolls cleanly for small search
    // radii.
    //
    // The starting value is deliberately huge so the first iteration
    // always wins, which avoids a negative-initialiser pattern cubecl's
    // macro does not lift cleanly.
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

    // An exact tie resolves to the zero-motion candidate, rather than
    // whichever candidate the scan above happened to reach first, which
    // is the window corner.
    //
    // A block lying entirely inside a flat region scores the same at
    // every candidate. Reporting motion there would seed the fine pass,
    // and every block it covers, from a shifted position that no pixel
    // comparison ever preferred.
    let zero_sad = sad_scratch[(search_radius * window_side + search_radius) as usize];
    if zero_sad <= best_sad {
        best_dx = 0i32;
        best_dy = 0i32;
    }

    // Map the coarse block index into the fine block index space by
    // position, rather than by simply doubling the index.
    //
    // Coarse block `bx` covers source pixels from `bx * step` up to
    // `(bx + 1) * step` at the coarse level, which at the fine level
    // means those bounds multiplied by `level_scale`. Dividing that
    // span by `fine_step` gives the range of fine blocks this coarse
    // block seeds.
    //
    // Floor-division tiling leaves every interior boundary touching,
    // with no gap and no overlap. The two block counts are each their
    // own ceil-division over a different width and a different step
    // though, so they can round differently, and the coarse grid's
    // nominal reach can fall just short of the fine grid's true edge.
    //
    // The last coarse block on each axis therefore extends its end to
    // the fine grid's own edge, absorbing that remainder so every fine
    // block still gets seeded exactly once.
    //
    // One formula covers all three cases, a genuinely coarser grid, two
    // equal grids, and the ragged last block on either geometry, with
    // no separate code path.
    //
    // How many fine blocks a coarse block seeds can vary from block to
    // block, so this is a runtime `while` loop rather than an unrolled
    // `for` loop.
    let mvx_fine = best_dx * level_scale as i32;
    let mvy_fine = best_dy * level_scale as i32;

    let fbx_start = (bx * step * level_scale / fine_step).min(fine_blocks_x);
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let fbx_end = if bx == CUBE_COUNT_X - 1 {
        fine_blocks_x.into()
    } else {
        ((bx + 1) * step * level_scale / fine_step).min(fine_blocks_x)
    };
    let fby_start = (by * step * level_scale / fine_step).min(fine_blocks_y);
    #[expect(
        clippy::useless_conversion,
        reason = "both branches have to expand to the same cubecl native type, which the \
                  conversion supplies"
    )]
    let fby_end = if by == CUBE_COUNT_Y - 1 {
        fine_blocks_y.into()
    } else {
        ((by + 1) * step * level_scale / fine_step).min(fine_blocks_y)
    };

    let mut fby = fby_start;
    while fby < fby_end {
        let mut fbx = fbx_start;
        while fbx < fbx_end {
            let idx = ((fby * fine_blocks_x + fbx) * 2) as usize;
            mv_field[idx] = mvx_fine;
            mv_field[idx + 1] = mvy_fine;
            fbx += 1;
        }
        fby += 1;
    }
}

/// Refines a motion estimate at full resolution.
///
/// When `use_seed` is set, the block starts from the vector the coarse
/// pass left in `mv_field` and searches a small window around it. The
/// refined vector goes back into the same slot.
///
/// The search itself works exactly like `nlm_mc_block_match_coarse`,
/// with the same cached centre tile and the same split by candidate.
///
/// # Confidence
///
/// When `write_confidence` is true, the block also writes a confidence
/// score derived from the winning score. That is what lets a poor match
/// suppress its own frame's contribution later on, instead of blurring
/// in content that does not belong.
///
/// `sad_noise_floor` is the score two noisy copies of the same content
/// produce by chance. Subtracting it first stops a clean match being
/// penalised for the noise it carries.
///
/// `thsad` is how far past that floor a block can go before its
/// confidence reaches zero. It has to be strictly positive whenever
/// `write_confidence` is true, or a perfect match divides zero by zero.
///
/// A `search_radius` of 0 with no seed reduces the match to the single
/// unshifted candidate, which is useful for a confidence-only pass with
/// no motion search at all.
///
/// When `write_confidence` is false, the whole confidence step is
/// dropped at compile time and costs nothing. Callers pass a small
/// placeholder buffer in that case, and its size never matters because
/// the kernel never reads it.
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
) {
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;

    let mv_slot = ((by * blocks_x + bx) * 2) as usize;

    // Clippy flags these `.into()` calls as useless, but they are
    // required. Both `if` branches have to produce the same cubecl
    // `NativeExpand<i32>` type, and a bare `0i32` literal does not
    // coerce inside the cube macro.
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

    // A cooperative load. Each thread claims a row-strided share of the
    // block's pixels and caches the clamped centre value once, so every
    // candidate below reads it from shared memory instead of fetching
    // it again from global memory.
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

    // Each thread takes a strided share of the candidates, works out
    // each one's full score over the cached block, and writes its
    // scratch slot exactly once. Splitting the work by candidate rather
    // than by pixel means no two threads ever write the same slot, so
    // no atomics and no reduce pass are needed.
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

    // An exact tie resolves to the seed itself, adding no motion beyond
    // whatever the coarse pass found, rather than to whichever
    // candidate the scan above happened to reach first, which is the
    // window corner.
    //
    // A block lying entirely inside a flat region scores the same at
    // every candidate. Reporting motion there would warp in pixels no
    // comparison ever preferred, and since the winning score is zero in
    // that case, it would also write a perfect confidence for what may
    // be a genuinely occluded block.
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
