use cubecl::prelude::*;
use cubecl::terminate;

use super::fill::{expanded_interior, fill_center_tile, halo_slot_index};
use super::helpers::{channel_scale, line_sum_sq, read_clamped_line, read_line, welsch_weight};

/// Writes the Welsch weight of the patch distance between each pixel and its neighbour at `q`.
///
/// The block caches a `(block + 2 * patch_radius)^2` tile of per-pixel distances in shared memory.
/// Blocks whose tile and shifted tile lie inside the image skip clamping, and the branch never
/// diverges because the whole block takes the same side.
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
/// Distances are read from `reference`, a cleaner image with the same layout as the input.
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

/// Elements of one row-sum buffer, `block_y + 2 * patch_radius` rows of `block_x` sums.
///
/// Without separable sums the buffer is a one-element placeholder, so both forms of a window kernel
/// share one body.
pub(crate) fn hsum_elements(separable: bool, block_x: u32, block_y: u32, patch_radius: u32) -> u32 {
    if !separable {
        return 1;
    }

    (block_y + 2 * patch_radius) * block_x
}

/// Sums one pixel's `(2 * patch_radius + 1)^2` patch of distances straight from `tile`, row by row.
#[cube]
fn direct_patch_sum(
    tile: &SharedMemory<f32>,
    local_x: u32,
    local_y: u32,
    #[comptime] tile_width: u32,
    #[comptime] patch_radius: u32,
) -> f32 {
    let center_tile_x = local_x + patch_radius;
    let center_tile_y = local_y + patch_radius;
    let patch_size = 2 * patch_radius + 1;
    let mut patch_sum = 0.0f32;

    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width + center_tile_x
                - patch_radius
                + offset_x) as usize;
            patch_sum += tile[smem_idx];
        }
    }

    patch_sum
}

/// Writes each tile row's sums of `2 * patch_radius + 1` neighbouring distances into `hsum`.
///
/// `hsum` holds `block_y + 2 * patch_radius` rows of `block_x` sums, one per pixel column of the
/// cube. Every thread of the cube must call it, and the cube syncs before any thread reads `hsum`.
#[cube]
fn row_sums(
    tile: &SharedMemory<f32>,
    hsum: &mut SharedMemory<f32>,
    thread_id: u32,
    threads: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
    #[comptime] patch_radius: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let count = comptime!((block_y + 2 * patch_radius) * block_x);
    let taps = comptime!(2 * patch_radius + 1);
    let mut index = thread_id;

    while index < count {
        let row = index / block_x;
        let column = index % block_x;
        let start = row * tile_width + column;
        let mut sum = 0.0f32;

        #[unroll]
        for tap in 0..taps {
            sum += tile[(start + tap) as usize];
        }

        hsum[index as usize] = sum;
        index += threads;
    }
}

/// Sums `2 * patch_radius + 1` row sums down the pixel's column, which completes its patch sum.
#[cube]
fn column_sum(
    hsum: &SharedMemory<f32>,
    local_x: u32,
    local_y: u32,
    #[comptime] block_x: u32,
    #[comptime] patch_radius: u32,
) -> f32 {
    let taps = comptime!(2 * patch_radius + 1);
    let mut sum = 0.0f32;

    #[unroll]
    for tap in 0..taps {
        sum += hsum[((local_y + tap) * block_x + local_x) as usize];
    }

    sum
}

/// One pixel's patch sum, from the row sums in `hsum` when `separable` is set, otherwise from `tile`.
#[cube]
fn patch_sum(
    tile: &SharedMemory<f32>,
    hsum: &SharedMemory<f32>,
    local_x: u32,
    local_y: u32,
    #[comptime] tile_width: u32,
    #[comptime] patch_radius: u32,
    #[comptime] separable: bool,
) -> f32 {
    let block_x = comptime!(tile_width - 2 * patch_radius);

    if comptime!(separable) {
        column_sum(hsum, local_x, local_y, block_x, patch_radius)
    } else {
        direct_patch_sum(tile, local_x, local_y, tile_width, patch_radius)
    }
}

/// Accumulates one pair of temporal neighbours over the whole search window in a single launch.
///
/// The running sums stay in registers and are added to `accum`, `weight_sum` and `max_weight`
/// once at the end. The centre frame is cached once in a
/// `(block + 2 * patch_radius + 2 * search_radius)^2` shared tile that both directions share.
///
/// When `use_confidence` is set, each weight is scaled by its block's confidence from `conf_fwd`
/// and `conf_bwd`. `step`, `blocks_x` and `blocks_y` describe the motion block grid, which pixels
/// map onto the same way as in `nlm_mc_warp`. When it is unset the confidence buffers are never read.
///
/// With `separable` set, each distance tile is first summed along its rows into a row-sum buffer,
/// and each pixel then sums `2 * patch_radius + 1` of those down its column. That reads shared
/// memory about a third as often per pixel, at the cost of one barrier per offset, and changes the
/// order of the additions.
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
    #[comptime] separable: bool,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let halo_elems =
        comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius) - block_x * block_y);
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_fwd = SharedMemory::<f32>::new(tile_elems as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(tile_elems as usize);
    let hsum_len = comptime!(hsum_elements(separable, block_x, block_y, patch_radius));
    let mut hsum_fwd = SharedMemory::<f32>::new(hsum_len as usize);
    let mut hsum_bwd = SharedMemory::<f32>::new(hsum_len as usize);

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
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_center_tile(
        input,
        &mut smem_center,
        expanded_x0,
        expanded_y0,
        interior,
        frame_t,
        thread_id,
        threads,
        expanded_width,
        expanded_elems,
        width,
        height,
    );

    sync_cube();

    let own_x = local_x + patch_radius;
    let own_y = local_y + patch_radius;
    let own_idx = (own_y * tile_width + own_x) as usize;
    let own_center_idx = ((own_y + search_radius) * expanded_width + (own_x + search_radius)) as usize;

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

            // Each thread fills its own pixel's element first and keeps both neighbours for the
            // accumulate. The backward comparison mirrors the forward one against `frame_bwd` at
            // `-q`, so both centre on the same cached patch.
            let own_center = smem_center[own_center_idx];
            let fwd_x = global_x as i32 + q_x;
            let fwd_y = global_y as i32 + q_y;
            let bwd_x = global_x as i32 - q_x;
            let bwd_y = global_y as i32 - q_y;
            let fwd_pixel = read_clamped_line(input, fwd_x, fwd_y, frame_fwd, width, height);
            let bwd_pixel = read_clamped_line(input, bwd_x, bwd_y, frame_bwd, width, height);
            smem_fwd[own_idx] = line_sum_sq(own_center - fwd_pixel, channels) * scale;
            smem_bwd[own_idx] = line_sum_sq(own_center - bwd_pixel, channels) * scale;

            let mut halo = thread_id;
            while halo < halo_elems {
                let idx = halo_slot_index(halo, block_x, block_y, patch_radius);
                let tile_x = idx % tile_width;
                let tile_y = idx / tile_width;
                let center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let center = smem_center[center_idx];
                let src_x = fwd_tile_x0 + tile_x as i32;
                let src_y = fwd_tile_y0 + tile_y as i32;
                let fwd_neighbor =
                    read_clamped_line(input, src_x + q_x, src_y + q_y, frame_fwd, width, height);
                let bwd_neighbor =
                    read_clamped_line(input, src_x - q_x, src_y - q_y, frame_bwd, width, height);
                smem_fwd[idx as usize] = line_sum_sq(center - fwd_neighbor, channels) * scale;
                smem_bwd[idx as usize] = line_sum_sq(center - bwd_neighbor, channels) * scale;
                halo += threads;
            }

            sync_cube();

            if comptime!(separable) {
                row_sums(
                    &smem_fwd,
                    &mut hsum_fwd,
                    thread_id,
                    threads,
                    block_x,
                    block_y,
                    patch_radius,
                );
                row_sums(
                    &smem_bwd,
                    &mut hsum_bwd,
                    thread_id,
                    threads,
                    block_x,
                    block_y,
                    patch_radius,
                );
                sync_cube();
            }

            if in_image {
                let sum_fwd = patch_sum(
                    &smem_fwd,
                    &hsum_fwd,
                    local_x,
                    local_y,
                    tile_width,
                    patch_radius,
                    separable,
                );
                let sum_bwd = patch_sum(
                    &smem_bwd,
                    &hsum_bwd,
                    local_x,
                    local_y,
                    tile_width,
                    patch_radius,
                    separable,
                );

                let mut weight_fwd = welsch_weight(sum_fwd, h2_inv_norm, noise_offset);
                let mut weight_bwd = welsch_weight(sum_bwd, h2_inv_norm, noise_offset);

                if use_confidence {
                    let block_col = (global_x / step).min(blocks_x - 1);
                    let block_row = (global_y / step).min(blocks_y - 1);
                    let block_idx = (block_row * blocks_x + block_col) as usize;
                    weight_fwd *= conf_fwd[block_idx];
                    weight_bwd *= conf_bwd[block_idx];
                }

                let line_w_fwd = Vector::<f32, N>::empty().fill(weight_fwd);
                let line_w_bwd = Vector::<f32, N>::empty().fill(weight_bwd);
                accum_reg = accum_reg + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;
                weight_sum_reg += weight_fwd + weight_bwd;
                max_weight_reg = f32::max(max_weight_reg, f32::max(weight_fwd, weight_bwd));
            }

            // The next offset overwrites the distance tiles.
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

/// Accumulates a frame against itself over the whole search window, the spatial-only case.
///
/// Patch distance is symmetric, so walking the full window one way gives the same result as the
/// paired half-window, with one distance tile and one neighbour read per offset. The zero offset is
/// skipped because `nlm_finish` adds it back through `wref * max_weight`.
///
/// `spatial_offset_lut` holds one noise-floor offset per candidate, row-major over the window.
/// Nearby candidates share more of the grain's spatial correlation, so their offsets are smaller.
///
/// With `separable` set, each distance tile is first summed along its rows into a row-sum buffer,
/// and each pixel then sums `2 * patch_radius + 1` of those down its column. That reads shared
/// memory about a third as often per pixel, at the cost of one barrier per offset, and changes the
/// order of the additions.
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
    #[comptime] separable: bool,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let halo_elems =
        comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius) - block_x * block_y);
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_dist = SharedMemory::<f32>::new(tile_elems as usize);
    let hsum_len = comptime!(hsum_elements(separable, block_x, block_y, patch_radius));
    let mut hsum = SharedMemory::<f32>::new(hsum_len as usize);

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
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_center_tile(
        input,
        &mut smem_center,
        expanded_x0,
        expanded_y0,
        interior,
        frame_t,
        thread_id,
        threads,
        expanded_width,
        expanded_elems,
        width,
        height,
    );

    sync_cube();

    let own_x = local_x + patch_radius;
    let own_y = local_y + patch_radius;
    let own_idx = (own_y * tile_width + own_x) as usize;
    let own_center_idx = ((own_y + search_radius) * expanded_width + (own_x + search_radius)) as usize;

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
                // An empty comptime branch stands in for `continue`, which cubecl lacks.
            } else {
                // Each thread fills its own pixel's element first and keeps the neighbour for the
                // accumulate.
                let own_center = smem_center[own_center_idx];
                let pixel_x = global_x as i32 + q_x;
                let pixel_y = global_y as i32 + q_y;
                let neighbor_pixel = read_clamped_line(input, pixel_x, pixel_y, frame_t, width, height);
                smem_dist[own_idx] = line_sum_sq(own_center - neighbor_pixel, channels) * scale;

                let mut halo = thread_id;
                while halo < halo_elems {
                    let tile_idx = halo_slot_index(halo, block_x, block_y, patch_radius);
                    let tile_x = tile_idx % tile_width;
                    let tile_y = tile_idx / tile_width;
                    let center_idx =
                        ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                    let center = smem_center[center_idx];
                    let neighbor_x = fwd_tile_x0 + tile_x as i32 + q_x;
                    let neighbor_y = fwd_tile_y0 + tile_y as i32 + q_y;
                    let neighbor = read_clamped_line(input, neighbor_x, neighbor_y, frame_t, width, height);
                    smem_dist[tile_idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
                    halo += threads;
                }

                sync_cube();

                if comptime!(separable) {
                    row_sums(
                        &smem_dist,
                        &mut hsum,
                        thread_id,
                        threads,
                        block_x,
                        block_y,
                        patch_radius,
                    );
                    sync_cube();
                }

                if in_image {
                    let patch_distance = patch_sum(
                        &smem_dist,
                        &hsum,
                        local_x,
                        local_y,
                        tile_width,
                        patch_radius,
                        separable,
                    );

                    let lut_idx = (q_yi * window_side + q_xi) as usize;
                    let offset = spatial_offset_lut[lut_idx];
                    let weight = welsch_weight(patch_distance, h2_inv_norm, offset);

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
/// Distances are read from `reference` while the accumulated pixels still come from `input`.
/// `separable` works as there.
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
    #[comptime] separable: bool,
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
    let hsum_len = comptime!(hsum_elements(separable, block_x, block_y, patch_radius));
    let mut hsum_fwd = SharedMemory::<f32>::new(hsum_len as usize);
    let mut hsum_bwd = SharedMemory::<f32>::new(hsum_len as usize);

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
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_center_tile(
        reference,
        &mut smem_center,
        expanded_x0,
        expanded_y0,
        interior,
        frame_t,
        thread_id,
        threads,
        expanded_width,
        expanded_elems,
        width,
        height,
    );

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
                let center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let center = smem_center[center_idx];
                let src_x = fwd_tile_x0 + tile_x as i32;
                let src_y = fwd_tile_y0 + tile_y as i32;
                let fwd_neighbor =
                    read_clamped_line(reference, src_x + q_x, src_y + q_y, frame_fwd, width, height);
                let bwd_neighbor =
                    read_clamped_line(reference, src_x - q_x, src_y - q_y, frame_bwd, width, height);
                smem_fwd[idx as usize] = line_sum_sq(center - fwd_neighbor, channels) * scale;
                smem_bwd[idx as usize] = line_sum_sq(center - bwd_neighbor, channels) * scale;
                idx += threads;
            }

            sync_cube();

            if comptime!(separable) {
                row_sums(
                    &smem_fwd,
                    &mut hsum_fwd,
                    thread_id,
                    threads,
                    block_x,
                    block_y,
                    patch_radius,
                );
                row_sums(
                    &smem_bwd,
                    &mut hsum_bwd,
                    thread_id,
                    threads,
                    block_x,
                    block_y,
                    patch_radius,
                );
                sync_cube();
            }

            if in_image {
                let sum_fwd = patch_sum(
                    &smem_fwd,
                    &hsum_fwd,
                    local_x,
                    local_y,
                    tile_width,
                    patch_radius,
                    separable,
                );
                let sum_bwd = patch_sum(
                    &smem_bwd,
                    &hsum_bwd,
                    local_x,
                    local_y,
                    tile_width,
                    patch_radius,
                    separable,
                );

                let mut weight_fwd = welsch_weight(sum_fwd, h2_inv_norm, noise_offset);
                let mut weight_bwd = welsch_weight(sum_bwd, h2_inv_norm, noise_offset);

                if use_confidence {
                    let block_col = (global_x / step).min(blocks_x - 1);
                    let block_row = (global_y / step).min(blocks_y - 1);
                    let block_idx = (block_row * blocks_x + block_col) as usize;
                    weight_fwd *= conf_fwd[block_idx];
                    weight_bwd *= conf_bwd[block_idx];
                }

                let fwd_x = global_x as i32 + q_x;
                let fwd_y = global_y as i32 + q_y;
                let bwd_x = global_x as i32 - q_x;
                let bwd_y = global_y as i32 - q_y;
                let fwd_pixel = read_clamped_line(input, fwd_x, fwd_y, frame_fwd, width, height);
                let bwd_pixel = read_clamped_line(input, bwd_x, bwd_y, frame_bwd, width, height);
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
/// Distances are read from `reference` while the accumulated pixels still come from `input`.
/// `separable` works as there.
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
    #[comptime] separable: bool,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<f32, N>>::new(expanded_elems as usize);
    let mut smem_dist = SharedMemory::<f32>::new(tile_elems as usize);
    let hsum_len = comptime!(hsum_elements(separable, block_x, block_y, patch_radius));
    let mut hsum = SharedMemory::<f32>::new(hsum_len as usize);

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
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_center_tile(
        reference,
        &mut smem_center,
        expanded_x0,
        expanded_y0,
        interior,
        frame_t,
        thread_id,
        threads,
        expanded_width,
        expanded_elems,
        width,
        height,
    );

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
                // An empty comptime branch stands in for `continue`, which cubecl lacks.
            } else {
                let mut tile_idx = thread_id;
                while tile_idx < tile_elems {
                    let tile_x = tile_idx % tile_width;
                    let tile_y = tile_idx / tile_width;
                    let center_idx =
                        ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                    let center = smem_center[center_idx];
                    let neighbor_x = fwd_tile_x0 + tile_x as i32 + q_x;
                    let neighbor_y = fwd_tile_y0 + tile_y as i32 + q_y;
                    let neighbor =
                        read_clamped_line(reference, neighbor_x, neighbor_y, frame_t, width, height);
                    smem_dist[tile_idx as usize] = line_sum_sq(center - neighbor, channels) * scale;
                    tile_idx += threads;
                }

                sync_cube();

                if comptime!(separable) {
                    row_sums(
                        &smem_dist,
                        &mut hsum,
                        thread_id,
                        threads,
                        block_x,
                        block_y,
                        patch_radius,
                    );
                    sync_cube();
                }

                if in_image {
                    let patch_distance = patch_sum(
                        &smem_dist,
                        &hsum,
                        local_x,
                        local_y,
                        tile_width,
                        patch_radius,
                        separable,
                    );

                    let lut_idx = (q_yi * window_side + q_xi) as usize;
                    let offset = spatial_offset_lut[lut_idx];
                    let weight = welsch_weight(patch_distance, h2_inv_norm, offset);

                    let pixel_x = global_x as i32 + q_x;
                    let pixel_y = global_y as i32 + q_y;
                    let neighbor_pixel = read_clamped_line(input, pixel_x, pixel_y, frame_t, width, height);
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
