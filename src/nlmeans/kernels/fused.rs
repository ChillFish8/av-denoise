use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{channel_scale, line_sum_sq, read_clamped_line, read_line, welsch_weight};

/// Measures the distance, box-filters it over the patch, and turns the
/// result into a Welsch weight, all in one kernel.
///
/// The block first loads a `(block + 2 * patch_radius)^2` tile of
/// per-pixel scaled distances into shared memory. Each thread then sums
/// its own `(2 * patch_radius + 1)^2` patch and applies the Welsch
/// kernel.
///
/// An `interior` flag picks unclamped reads when the whole tile, and its
/// shifted twin, lie inside the image. Blocks near the border take the
/// clamped path instead. The flag is the same for every thread in the
/// block, so the branch costs nothing in divergence.
#[cube(launch_unchecked)]
pub fn nlm_dist_2d_weight<N: Size>(
    input: &Array<Vector<f32, N>>,
    output: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
    noise_offset: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_height = comptime!(block_y + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let mut smem = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let tile_end_x = tile_start_x + tile_width as i32;
    let tile_end_y = tile_start_y + tile_height as i32;

    let scale = channel_scale(channels);

    let interior = tile_start_x >= 0
        && tile_end_x <= width as i32
        && tile_start_y >= 0
        && tile_end_y <= height as i32
        && (tile_start_x + q_x) >= 0
        && (tile_end_x + q_x) <= width as i32
        && (tile_start_y + q_y) >= 0
        && (tile_end_y + q_y) <= height as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;

    if interior {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = (tile_start_x + tile_x as i32) as u32;
            let src_y = (tile_start_y + tile_y as i32) as u32;
            let center = read_line(input, src_x, src_y, frame_t, width, height);
            let neighbor = read_line(
                input,
                (src_x as i32 + q_x) as u32,
                (src_y as i32 + q_y) as u32,
                frame_q,
                width,
                height,
            );
            smem[idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
            idx += threads;
        }
    } else {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = tile_start_x + tile_x as i32;
            let src_y = tile_start_y + tile_y as i32;
            let center = read_clamped_line(input, src_x, src_y, frame_t, width, height);
            let neighbor = read_clamped_line(input, src_x + q_x, src_y + q_y, frame_q, width, height);
            smem[idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
            idx += threads;
        }
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let center_tile_x = local_x + patch_radius;
    let center_tile_y = local_y + patch_radius;
    let patch_size = 2 * patch_radius + 1;
    let mut patch_sum = 0.0f32;
    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width + center_tile_x
                - patch_radius
                + offset_x) as usize;
            patch_sum += smem[smem_idx];
        }
    }

    output[(global_y * width + global_x) as usize] = welsch_weight(patch_sum, h2_inv_norm, noise_offset);
}

/// The reference-image version of `nlm_dist_2d_weight`.
///
/// Distances are read from `reference`, a prefiltered or externally
/// supplied image with the same layout as `input`. The weight output is
/// unchanged.
///
/// This runs when a prefilter is active, so the weights are computed
/// from a cleaner image than the noisy input.
#[cube(launch_unchecked)]
pub fn nlm_dist_2d_weight_ref<N: Size>(
    reference: &Array<Vector<f32, N>>,
    output: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
    noise_offset: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_height = comptime!(block_y + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let mut smem = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let tile_end_x = tile_start_x + tile_width as i32;
    let tile_end_y = tile_start_y + tile_height as i32;

    let scale = channel_scale(channels);

    let interior = tile_start_x >= 0
        && tile_end_x <= width as i32
        && tile_start_y >= 0
        && tile_end_y <= height as i32
        && (tile_start_x + q_x) >= 0
        && (tile_end_x + q_x) <= width as i32
        && (tile_start_y + q_y) >= 0
        && (tile_end_y + q_y) <= height as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;

    if interior {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = (tile_start_x + tile_x as i32) as u32;
            let src_y = (tile_start_y + tile_y as i32) as u32;
            let center = read_line(reference, src_x, src_y, frame_t, width, height);
            let neighbor = read_line(
                reference,
                (src_x as i32 + q_x) as u32,
                (src_y as i32 + q_y) as u32,
                frame_q,
                width,
                height,
            );
            smem[idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
            idx += threads;
        }
    } else {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = tile_start_x + tile_x as i32;
            let src_y = tile_start_y + tile_y as i32;
            let center = read_clamped_line(reference, src_x, src_y, frame_t, width, height);
            let neighbor = read_clamped_line(reference, src_x + q_x, src_y + q_y, frame_q, width, height);
            smem[idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
            idx += threads;
        }
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let center_tile_x = local_x + patch_radius;
    let center_tile_y = local_y + patch_radius;
    let patch_size = 2 * patch_radius + 1;
    let mut patch_sum = 0.0f32;
    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width + center_tile_x
                - patch_radius
                + offset_x) as usize;
            patch_sum += smem[smem_idx];
        }
    }

    output[(global_y * width + global_x) as usize] = welsch_weight(patch_sum, h2_inv_norm, noise_offset);
}

/// Compares the centre frame against one pair of temporal neighbours,
/// covering the whole search window in a single launch.
///
/// The kernel loops over every offset in the search window for one
/// temporal distance, keeping the running accumulator, weight sum, and
/// max weight in registers. Those are written to global memory once at
/// the end, which collapses `(2 * search_radius + 1)^2` launches into
/// one.
///
/// # Caching the centre frame
///
/// The centre frame is read once into a shared-memory tile of
/// `(block + 2 * patch_radius + 2 * search_radius)^2` pixels, big enough
/// to cover every neighbour offset the window can reach.
///
/// The forward and backward comparisons both centre on that same patch,
/// so one cached tile serves both. Only the shifted neighbour pixels
/// come from global memory each iteration.
///
/// That roughly halves the global read traffic compared with re-reading
/// the centre frame for every offset.
///
/// The two distance tiles are reused across iterations, with a
/// `sync_cube` between them.
///
/// # Confidence weighting
///
/// When `use_confidence` is true, each weight is multiplied by its
/// block's confidence before it folds into the accumulators, using the
/// same pixel-to-block mapping `nlm_mc_warp` uses.
///
/// The block index depends only on the pixel position, so it is the same
/// for every offset in the window.
///
/// When `use_confidence` is false the lookup and the multiply are
/// dropped at compile time, and the confidence buffers are never read.
#[cube(launch_unchecked)]
pub fn nlm_fused_pair_accumulate_window<N: Size>(
    input: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    conf_fwd: &Array<f32>,
    conf_bwd: &Array<f32>,
    #[comptime] use_confidence: bool,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    h2_inv_norm: f32,
    noise_offset: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] search_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_fwd = SharedMemory::<f32>::new(tile_elems as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let scale = channel_scale(channels);

    let fwd_tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = fwd_tile_x0 - search_radius as i32;
    let expanded_y0 = fwd_tile_y0 - search_radius as i32;

    // Cache `frame_t` once across the expanded tile that covers every
    // forward and shifted-backward center position.
    let mut idx = thread_id;
    while idx < expanded_elems {
        let ex = idx % expanded_width;
        let ey = idx / expanded_width;
        let src_x = expanded_x0 + ex as i32;
        let src_y = expanded_y0 + ey as i32;
        smem_center[idx as usize] = read_clamped_line(input, src_x, src_y, frame_t, width, height);
        idx += threads;
    }
    sync_cube();

    let mut accum_reg = Vector::<f32, N>::empty();
    let mut weight_sum_reg = 0.0f32;
    let mut max_weight_reg = 0.0f32;

    let window_side = comptime!(2 * search_radius + 1);

    #[unroll]
    for q_yi in 0..window_side {
        #[unroll]
        for q_xi in 0..window_side {
            let q_x = q_xi as i32 - search_radius as i32;
            let q_y = q_yi as i32 - search_radius as i32;

            let mut idx = thread_id;
            while idx < tile_elems {
                let tile_x = idx % tile_width;
                let tile_y = idx / tile_width;

                // Both the forward and backward centres sit at
                // (tile_x + search_radius, tile_y + search_radius) in
                // expanded-tile coordinates. The backward comparison is
                // centred on the same output pixel as the forward one,
                // mirroring it against `frame_bwd` at `-q` instead of
                // `frame_fwd` at `+q`.
                let center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let center = smem_center[center_idx];
                let fwd_neighbor = read_clamped_line(
                    input,
                    fwd_tile_x0 + tile_x as i32 + q_x,
                    fwd_tile_y0 + tile_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );
                smem_fwd[idx as usize] = line_sum_sq(center - fwd_neighbor, channels) * scale;

                let bwd_neighbor = read_clamped_line(
                    input,
                    fwd_tile_x0 + tile_x as i32 - q_x,
                    fwd_tile_y0 + tile_y as i32 - q_y,
                    frame_bwd,
                    width,
                    height,
                );
                smem_bwd[idx as usize] = line_sum_sq(center - bwd_neighbor, channels) * scale;

                idx += threads;
            }

            sync_cube();

            if in_image {
                let center_tile_x = local_x + patch_radius;
                let center_tile_y = local_y + patch_radius;
                let patch_size = 2 * patch_radius + 1;
                let mut sum_fwd = 0.0f32;
                let mut sum_bwd = 0.0f32;
                for offset_y in 0..patch_size {
                    for offset_x in 0..patch_size {
                        let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width
                            + center_tile_x
                            - patch_radius
                            + offset_x) as usize;
                        sum_fwd += smem_fwd[smem_idx];
                        sum_bwd += smem_bwd[smem_idx];
                    }
                }

                let mut weight_fwd = welsch_weight(sum_fwd, h2_inv_norm, noise_offset);
                let mut weight_bwd = welsch_weight(sum_bwd, h2_inv_norm, noise_offset);

                if use_confidence {
                    let bx = (global_x / step).min(blocks_x - 1);
                    let by = (global_y / step).min(blocks_y - 1);
                    let block_idx = (by * blocks_x + bx) as usize;
                    weight_fwd *= conf_fwd[block_idx];
                    weight_bwd *= conf_bwd[block_idx];
                }

                let fwd_pixel = read_clamped_line(
                    input,
                    global_x as i32 + q_x,
                    global_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );

                let bwd_pixel = read_clamped_line(
                    input,
                    global_x as i32 - q_x,
                    global_y as i32 - q_y,
                    frame_bwd,
                    width,
                    height,
                );

                let line_w_fwd = Vector::<f32, N>::empty().fill(weight_fwd);
                let line_w_bwd = Vector::<f32, N>::empty().fill(weight_bwd);
                accum_reg = accum_reg + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;
                weight_sum_reg += weight_fwd + weight_bwd;
                max_weight_reg = f32::max(max_weight_reg, f32::max(weight_fwd, weight_bwd));
            }

            // Wait for every thread to finish reading the tiles before
            // the next q overwrites them.
            sync_cube();
        }
    }

    if in_image {
        let pixel_idx = (global_y * width + global_x) as usize;
        let cur_accum = accum[pixel_idx];
        accum[pixel_idx] = cur_accum + accum_reg;
        weight_sum[pixel_idx] += weight_sum_reg;
        let cur_max = max_weight[pixel_idx];
        max_weight[pixel_idx] = f32::max(cur_max, max_weight_reg);
    }
}

/// Compares a frame against itself across the search window, which is
/// the spatial-only case.
///
/// The structure matches `nlm_fused_pair_accumulate_window`, but this
/// kernel takes advantage of the weight map's symmetry. Patch distance
/// reads the same in either direction, so walking the full window one
/// way gives the same accumulator as the paired half-window version,
/// with one distance tile and one neighbour read per offset.
///
/// The centre frame is cached in the expanded shared-memory tile, so
/// each offset only touches global memory for its shifted neighbour
/// pixel.
///
/// The zero offset is skipped at compile time. `nlm_finish` folds that
/// self-contribution back in through its `wref * max_weight` term.
///
/// # Spatial offset table
///
/// `spatial_offset_lut` holds one noise-floor offset per candidate, laid
/// out row-major over the window.
///
/// Nearby candidates share more of the grain's spatial correlation, so
/// their offset is reduced relative to distant ones. See
/// `noise::build_spatial_offset_lut`.
#[cube(launch_unchecked)]
pub fn nlm_fused_single_window<N: Size>(
    input: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    h2_inv_norm: f32,
    spatial_offset_lut: &Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] search_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_dist = SharedMemory::<f32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let scale = channel_scale(channels);

    let fwd_tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = fwd_tile_x0 - search_radius as i32;
    let expanded_y0 = fwd_tile_y0 - search_radius as i32;

    let mut idx = thread_id;
    while idx < expanded_elems {
        let ex = idx % expanded_width;
        let ey = idx / expanded_width;
        let src_x = expanded_x0 + ex as i32;
        let src_y = expanded_y0 + ey as i32;
        smem_center[idx as usize] = read_clamped_line(input, src_x, src_y, frame_t, width, height);
        idx += threads;
    }
    sync_cube();

    let mut accum_reg = Vector::<f32, N>::empty();
    let mut weight_sum_reg = 0.0f32;
    let mut max_weight_reg = 0.0f32;

    let window_side = comptime!(2 * search_radius + 1);

    #[unroll]
    for q_yi in 0..window_side {
        #[unroll]
        for q_xi in 0..window_side {
            let q_x = q_xi as i32 - search_radius as i32;
            let q_y = q_yi as i32 - search_radius as i32;
            if comptime!(q_x == 0 && q_y == 0) {
                // Skip the zero offset. `nlm_finish` puts that
                // contribution back through `wref * max_weight`.
                //
                // CubeCL has no `continue` yet. This does not become a
                // branch in the kernel, because it is optimised out at
                // compile time.
            } else {
                let mut tidx = thread_id;
                while tidx < tile_elems {
                    let tile_x = tidx % tile_width;
                    let tile_y = tidx / tile_width;
                    let center_idx =
                        ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                    let center = smem_center[center_idx];
                    let neighbor = read_clamped_line(
                        input,
                        fwd_tile_x0 + tile_x as i32 + q_x,
                        fwd_tile_y0 + tile_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    smem_dist[tidx as usize] = line_sum_sq(center - neighbor, channels) * scale;
                    tidx += threads;
                }
                sync_cube();

                if in_image {
                    let center_tile_x = local_x + patch_radius;
                    let center_tile_y = local_y + patch_radius;
                    let patch_size = 2 * patch_radius + 1;
                    let mut patch_sum = 0.0f32;
                    for offset_y in 0..patch_size {
                        for offset_x in 0..patch_size {
                            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width
                                + center_tile_x
                                - patch_radius
                                + offset_x) as usize;
                            patch_sum += smem_dist[smem_idx];
                        }
                    }
                    let lut_idx = (q_yi * window_side + q_xi) as usize;
                    let offset = spatial_offset_lut[lut_idx];
                    let weight = welsch_weight(patch_sum, h2_inv_norm, offset);

                    let neighbor_pixel = read_clamped_line(
                        input,
                        global_x as i32 + q_x,
                        global_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    let line_w = Vector::<f32, N>::empty().fill(weight);
                    accum_reg += neighbor_pixel * line_w;
                    weight_sum_reg += weight;
                    max_weight_reg = f32::max(max_weight_reg, weight);
                }

                sync_cube();
            }
        }
    }

    if in_image {
        let pixel_idx = (global_y * width + global_x) as usize;
        let cur_accum = accum[pixel_idx];
        accum[pixel_idx] = cur_accum + accum_reg;
        weight_sum[pixel_idx] += weight_sum_reg;
        let cur_max = max_weight[pixel_idx];
        max_weight[pixel_idx] = f32::max(cur_max, max_weight_reg);
    }
}

/// The reference-image version of `nlm_fused_pair_accumulate_window`.
///
/// Distances, both the cached centre tile and the per-offset
/// neighbours, are read from `reference`. The pixels being accumulated
/// still come from `input`, so the original values reach `accum` while
/// the weights come from the cleaner reference frames.
///
/// Confidence weighting works the same way as in the plain version.
#[cube(launch_unchecked)]
pub fn nlm_fused_pair_accumulate_window_ref<N: Size>(
    input: &Array<Vector<f32, N>>,
    reference: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    conf_fwd: &Array<f32>,
    conf_bwd: &Array<f32>,
    #[comptime] use_confidence: bool,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    h2_inv_norm: f32,
    noise_offset: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] search_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_fwd = SharedMemory::<f32>::new(tile_elems as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let scale = channel_scale(channels);

    let fwd_tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = fwd_tile_x0 - search_radius as i32;
    let expanded_y0 = fwd_tile_y0 - search_radius as i32;

    // Cache `reference[frame_t]` once.
    let mut idx = thread_id;
    while idx < expanded_elems {
        let ex = idx % expanded_width;
        let ey = idx / expanded_width;
        let src_x = expanded_x0 + ex as i32;
        let src_y = expanded_y0 + ey as i32;
        smem_center[idx as usize] = read_clamped_line(reference, src_x, src_y, frame_t, width, height);
        idx += threads;
    }
    sync_cube();

    let mut accum_reg = Vector::<f32, N>::empty();
    let mut weight_sum_reg = 0.0f32;
    let mut max_weight_reg = 0.0f32;

    let window_side = comptime!(2 * search_radius + 1);

    #[unroll]
    for q_yi in 0..window_side {
        #[unroll]
        for q_xi in 0..window_side {
            let q_x = q_xi as i32 - search_radius as i32;
            let q_y = q_yi as i32 - search_radius as i32;

            let mut idx = thread_id;
            while idx < tile_elems {
                let tile_x = idx % tile_width;
                let tile_y = idx / tile_width;

                // Both the forward and backward comparisons centre on
                // the same patch of the centre frame. The plain
                // variant's doc comment explains why.
                let center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let center = smem_center[center_idx];
                let fwd_neighbor = read_clamped_line(
                    reference,
                    fwd_tile_x0 + tile_x as i32 + q_x,
                    fwd_tile_y0 + tile_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );
                smem_fwd[idx as usize] = line_sum_sq(center - fwd_neighbor, channels) * scale;

                let bwd_neighbor = read_clamped_line(
                    reference,
                    fwd_tile_x0 + tile_x as i32 - q_x,
                    fwd_tile_y0 + tile_y as i32 - q_y,
                    frame_bwd,
                    width,
                    height,
                );
                smem_bwd[idx as usize] = line_sum_sq(center - bwd_neighbor, channels) * scale;

                idx += threads;
            }

            sync_cube();

            if in_image {
                let center_tile_x = local_x + patch_radius;
                let center_tile_y = local_y + patch_radius;
                let patch_size = 2 * patch_radius + 1;
                let mut sum_fwd = 0.0f32;
                let mut sum_bwd = 0.0f32;
                for offset_y in 0..patch_size {
                    for offset_x in 0..patch_size {
                        let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width
                            + center_tile_x
                            - patch_radius
                            + offset_x) as usize;
                        sum_fwd += smem_fwd[smem_idx];
                        sum_bwd += smem_bwd[smem_idx];
                    }
                }

                let mut weight_fwd = welsch_weight(sum_fwd, h2_inv_norm, noise_offset);
                let mut weight_bwd = welsch_weight(sum_bwd, h2_inv_norm, noise_offset);

                if use_confidence {
                    let bx = (global_x / step).min(blocks_x - 1);
                    let by = (global_y / step).min(blocks_y - 1);
                    let block_idx = (by * blocks_x + bx) as usize;
                    weight_fwd *= conf_fwd[block_idx];
                    weight_bwd *= conf_bwd[block_idx];
                }

                // Pixel accumulation reads from `input`, not `reference`.
                let fwd_pixel = read_clamped_line(
                    input,
                    global_x as i32 + q_x,
                    global_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );
                let bwd_pixel = read_clamped_line(
                    input,
                    global_x as i32 - q_x,
                    global_y as i32 - q_y,
                    frame_bwd,
                    width,
                    height,
                );
                let line_w_fwd = Vector::<f32, N>::empty().fill(weight_fwd);
                let line_w_bwd = Vector::<f32, N>::empty().fill(weight_bwd);
                accum_reg = accum_reg + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;
                weight_sum_reg += weight_fwd + weight_bwd;
                max_weight_reg = f32::max(max_weight_reg, f32::max(weight_fwd, weight_bwd));
            }

            sync_cube();
        }
    }

    if in_image {
        let pixel_idx = (global_y * width + global_x) as usize;
        let cur_accum = accum[pixel_idx];
        accum[pixel_idx] = cur_accum + accum_reg;
        weight_sum[pixel_idx] += weight_sum_reg;
        let cur_max = max_weight[pixel_idx];
        max_weight[pixel_idx] = f32::max(cur_max, max_weight_reg);
    }
}

/// The reference-image version of `nlm_fused_single_window`.
///
/// Distances come from the reference image, both the cached centre and
/// the per-offset neighbours, while the pixels being accumulated come
/// from the input.
///
/// `spatial_offset_lut` has the same layout as in
/// `nlm_fused_single_window`.
#[cube(launch_unchecked)]
pub fn nlm_fused_single_window_ref<N: Size>(
    input: &Array<Vector<f32, N>>,
    reference: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    h2_inv_norm: f32,
    spatial_offset_lut: &Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] patch_radius: u32,
    #[comptime] search_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_dist = SharedMemory::<f32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let scale = channel_scale(channels);

    let fwd_tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = fwd_tile_x0 - search_radius as i32;
    let expanded_y0 = fwd_tile_y0 - search_radius as i32;

    let mut idx = thread_id;
    while idx < expanded_elems {
        let ex = idx % expanded_width;
        let ey = idx / expanded_width;
        let src_x = expanded_x0 + ex as i32;
        let src_y = expanded_y0 + ey as i32;
        smem_center[idx as usize] = read_clamped_line(reference, src_x, src_y, frame_t, width, height);
        idx += threads;
    }
    sync_cube();

    let mut accum_reg = Vector::<f32, N>::empty();
    let mut weight_sum_reg = 0.0f32;
    let mut max_weight_reg = 0.0f32;

    let window_side = comptime!(2 * search_radius + 1);

    #[unroll]
    for q_yi in 0..window_side {
        #[unroll]
        for q_xi in 0..window_side {
            let q_x = q_xi as i32 - search_radius as i32;
            let q_y = q_yi as i32 - search_radius as i32;
            if comptime!(q_x == 0 && q_y == 0) {
                // CubeCL has no `continue` yet. This does not become a
                // branch in the kernel, because it is optimised out at
                // compile time.
            } else {
                let mut tidx = thread_id;
                while tidx < tile_elems {
                    let tile_x = tidx % tile_width;
                    let tile_y = tidx / tile_width;
                    let center_idx =
                        ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                    let center = smem_center[center_idx];
                    let neighbor = read_clamped_line(
                        reference,
                        fwd_tile_x0 + tile_x as i32 + q_x,
                        fwd_tile_y0 + tile_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    smem_dist[tidx as usize] = line_sum_sq(center - neighbor, channels) * scale;
                    tidx += threads;
                }
                sync_cube();

                if in_image {
                    let center_tile_x = local_x + patch_radius;
                    let center_tile_y = local_y + patch_radius;
                    let patch_size = 2 * patch_radius + 1;
                    let mut patch_sum = 0.0f32;
                    for offset_y in 0..patch_size {
                        for offset_x in 0..patch_size {
                            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width
                                + center_tile_x
                                - patch_radius
                                + offset_x) as usize;
                            patch_sum += smem_dist[smem_idx];
                        }
                    }
                    let lut_idx = (q_yi * window_side + q_xi) as usize;
                    let offset = spatial_offset_lut[lut_idx];
                    let weight = welsch_weight(patch_sum, h2_inv_norm, offset);

                    let neighbor_pixel = read_clamped_line(
                        input,
                        global_x as i32 + q_x,
                        global_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    let line_w = Vector::<f32, N>::empty().fill(weight);
                    accum_reg += neighbor_pixel * line_w;
                    weight_sum_reg += weight;
                    max_weight_reg = f32::max(max_weight_reg, weight);
                }

                sync_cube();
            }
        }
    }

    if in_image {
        let pixel_idx = (global_y * width + global_x) as usize;
        let cur_accum = accum[pixel_idx];
        accum[pixel_idx] = cur_accum + accum_reg;
        weight_sum[pixel_idx] += weight_sum_reg;
        let cur_max = max_weight[pixel_idx];
        max_weight[pixel_idx] = f32::max(cur_max, max_weight_reg);
    }
}
