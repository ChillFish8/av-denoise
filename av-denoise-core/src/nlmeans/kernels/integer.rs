use cubecl::prelude::*;

use super::fill::{expanded_interior, halo_slot_index};
use super::helpers::{clamp_coord, read_clamped_line, welsch_weight};

/// Reads the codes at `(x, y)` in `frame`, clamped to the image edges.
#[cube]
fn read_clamped_codes<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    x: i32,
    y: i32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<C, N> {
    let clamped_x = clamp_coord(x, width);
    let clamped_y = clamp_coord(y, height);
    let idx = (frame * height + clamped_y) * width + clamped_x;
    codes[idx as usize]
}

/// Sums the squared code differences across the first `channels` lanes.
#[cube]
fn code_sum_sq<C: Int, N: Size>(
    centre: Vector<C, N>,
    neighbour: Vector<C, N>,
    #[comptime] channels: u32,
) -> i32 {
    let mut sum = 0i32;

    #[unroll]
    for channel in 0..channels {
        let centre_code = i32::cast_from(centre[channel as usize]);
        let neighbour_code = i32::cast_from(neighbour[channel as usize]);
        let diff = centre_code - neighbour_code;
        sum += diff * diff;
    }

    sum
}

/// Caches the expanded centre region of `frame`'s codes in `tile`, row-major over `expanded_width`
/// columns.
///
/// When `interior` holds the reads skip clamping. Every thread of the cube must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, a block coordinate or a comptime shape the kernel binds"
)]
fn fill_code_tile<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &mut SharedMemory<Vector<C, N>>,
    expanded_x0: i32,
    expanded_y0: i32,
    interior: bool,
    frame: u32,
    thread_id: u32,
    threads: u32,
    #[comptime] expanded_width: u32,
    #[comptime] expanded_elems: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    if interior {
        let mut idx = thread_id;
        while idx < expanded_elems {
            let src_x = expanded_x0 as u32 + idx % expanded_width;
            let src_y = expanded_y0 as u32 + idx / expanded_width;
            let src_idx = (frame * height + src_y) * width + src_x;
            tile[idx as usize] = codes[src_idx as usize];
            idx += threads;
        }
    } else {
        let mut idx = thread_id;
        while idx < expanded_elems {
            let src_x = expanded_x0 + (idx % expanded_width) as i32;
            let src_y = expanded_y0 + (idx / expanded_width) as i32;
            tile[idx as usize] = read_clamped_codes(codes, src_x, src_y, frame, width, height);
            idx += threads;
        }
    }
}

/// Sums one pixel's `(2 * patch_radius + 1)^2` patch of squared code differences from `tile`.
///
/// Integer sums are exact, so the order of the additions never changes the result.
#[cube]
fn int_patch_sum(
    tile: &SharedMemory<i32>,
    local_x: u32,
    local_y: u32,
    #[comptime] tile_width: u32,
    #[comptime] patch_radius: u32,
) -> i32 {
    let patch_size = 2 * patch_radius + 1;
    let mut patch_sum = 0i32;

    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((local_y + offset_y) * tile_width + local_x + offset_x) as usize;
            patch_sum += tile[smem_idx];
        }
    }

    patch_sum
}

/// The Welsch weight of an exact integer patch SSD.
///
/// `inv_scale2` turns the SSD into the distance the float kernels sum, and is computed on the host.
#[cube]
pub(crate) fn int_window_weight(ssd: i32, inv_scale2: f32, h2_inv_norm: f32, offset: f32) -> f32 {
    let distance = f32::cast_from(ssd) * inv_scale2;
    welsch_weight(distance, h2_inv_norm, offset)
}

/// Adds one weighted neighbour to a pixel's running sums.
#[cube]
pub(crate) fn accumulate_single_weight<N: Size>(
    accum_reg: &mut Vector<f32, N>,
    weight_sum_reg: &mut f32,
    max_weight_reg: &mut f32,
    pixel: Vector<f32, N>,
    weight: f32,
) {
    let line_w = Vector::<f32, N>::empty().fill(weight);
    *accum_reg += pixel * line_w;
    *weight_sum_reg += weight;
    *max_weight_reg = f32::max(*max_weight_reg, weight);
}

/// Adds a forward and a backward neighbour to a pixel's running sums.
///
/// When `use_confidence` is set, each weight is first scaled by its motion block's confidence from
/// `conf_fwd` and `conf_bwd`.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a running sum, a neighbour or the confidence shape the weights read"
)]
pub(crate) fn accumulate_pair_weights<N: Size>(
    accum_reg: &mut Vector<f32, N>,
    weight_sum_reg: &mut f32,
    max_weight_reg: &mut f32,
    fwd_pixel: Vector<f32, N>,
    bwd_pixel: Vector<f32, N>,
    fwd_weight: f32,
    bwd_weight: f32,
    conf_fwd: &Array<f32>,
    conf_bwd: &Array<f32>,
    global_x: u32,
    global_y: u32,
    #[comptime] use_confidence: bool,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let mut weight_fwd = fwd_weight;
    let mut weight_bwd = bwd_weight;

    if use_confidence {
        let block_col = (global_x / step).min(blocks_x - 1);
        let block_row = (global_y / step).min(blocks_y - 1);
        let block_idx = (block_row * blocks_x + block_col) as usize;
        weight_fwd *= conf_fwd[block_idx];
        weight_bwd *= conf_bwd[block_idx];
    }

    let line_w_fwd = Vector::<f32, N>::empty().fill(weight_fwd);
    let line_w_bwd = Vector::<f32, N>::empty().fill(weight_bwd);
    *accum_reg = *accum_reg + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;
    *weight_sum_reg += weight_fwd + weight_bwd;
    let pair_max = f32::max(weight_fwd, weight_bwd);
    *max_weight_reg = f32::max(*max_weight_reg, pair_max);
}

/// Adds a pixel's running sums into the frame accumulators.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is an accumulator, a running sum or the pixel it belongs to"
)]
pub(crate) fn store_window_sums<N: Size>(
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    accum_reg: Vector<f32, N>,
    weight_sum_reg: f32,
    max_weight_reg: f32,
    global_x: u32,
    global_y: u32,
    #[comptime] width: u32,
) {
    let pixel_idx = (global_y * width + global_x) as usize;
    let cur_accum = accum[pixel_idx];
    accum[pixel_idx] = cur_accum + accum_reg;
    weight_sum[pixel_idx] += weight_sum_reg;
    let cur_max = max_weight[pixel_idx];
    max_weight[pixel_idx] = f32::max(cur_max, max_weight_reg);
}

/// The integer twin of [nlm_fused_single_window](crate::nlmeans::kernels::nlm_fused_single_window).
///
/// Distances are exact i32 SSDs over `codes`, an integer copy of `input` with the same layout.
/// Each SSD becomes a float distance through `inv_scale2`, and the accumulated pixels still come
/// from `input`.
#[cube(launch_unchecked)]
pub fn nlm_int_single_window<C: Int, N: Size>(
    input: &Array<Vector<f32, N>>,
    codes: &Array<Vector<C, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    h2_inv_norm: f32,
    inv_scale2: f32,
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
    let halo_elems =
        comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius) - block_x * block_y);
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<C, N>>::new(expanded_elems as usize);
    let mut smem_dist = SharedMemory::<i32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;

    let tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = tile_x0 - search_radius as i32;
    let expanded_y0 = tile_y0 - search_radius as i32;
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_code_tile(
        codes,
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
                let own_center = smem_center[own_center_idx];
                let pixel_x = global_x as i32 + q_x;
                let pixel_y = global_y as i32 + q_y;
                let own_neighbour = read_clamped_codes(codes, pixel_x, pixel_y, frame_t, width, height);
                smem_dist[own_idx] = code_sum_sq(own_center, own_neighbour, channels);

                let mut halo = thread_id;
                while halo < halo_elems {
                    let tile_idx = halo_slot_index(halo, block_x, block_y, patch_radius);
                    let tile_x = tile_idx % tile_width;
                    let tile_y = tile_idx / tile_width;
                    let center_idx =
                        ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                    let center = smem_center[center_idx];
                    let neighbour_x = tile_x0 + tile_x as i32 + q_x;
                    let neighbour_y = tile_y0 + tile_y as i32 + q_y;
                    let neighbour =
                        read_clamped_codes(codes, neighbour_x, neighbour_y, frame_t, width, height);
                    smem_dist[tile_idx as usize] = code_sum_sq(center, neighbour, channels);
                    halo += threads;
                }

                sync_cube();

                if in_image {
                    let ssd = int_patch_sum(&smem_dist, local_x, local_y, tile_width, patch_radius);
                    let lut_idx = (q_yi * window_side + q_xi) as usize;
                    let offset = spatial_offset_lut[lut_idx];
                    let weight = int_window_weight(ssd, inv_scale2, h2_inv_norm, offset);
                    let neighbour_pixel = read_clamped_line(input, pixel_x, pixel_y, frame_t, width, height);

                    accumulate_single_weight(
                        &mut accum_reg,
                        &mut weight_sum_reg,
                        &mut max_weight_reg,
                        neighbour_pixel,
                        weight,
                    );
                }

                // The next offset overwrites the distance tile.
                sync_cube();
            }
        }
    }

    if in_image {
        store_window_sums(
            accum,
            weight_sum,
            max_weight,
            accum_reg,
            weight_sum_reg,
            max_weight_reg,
            global_x,
            global_y,
            width,
        );
    }
}

/// The integer twin of
/// [nlm_fused_pair_accumulate_window](crate::nlmeans::kernels::nlm_fused_pair_accumulate_window).
///
/// Distances are exact i32 SSDs over `codes`, an integer copy of `input` with the same layout.
/// Each SSD becomes a float distance through `inv_scale2`, and the accumulated pixels still come
/// from `input`. With `use_confidence` set, each weight is scaled by its motion block's confidence
/// from `conf_fwd` and `conf_bwd`.
#[cube(launch_unchecked)]
pub fn nlm_int_pair_window<C: Int, N: Size>(
    input: &Array<Vector<f32, N>>,
    codes: &Array<Vector<C, N>>,
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
    inv_scale2: f32,
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
    let halo_elems =
        comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius) - block_x * block_y);
    let expanded_width = comptime!(block_x + 2 * patch_radius + 2 * search_radius);
    let expanded_height = comptime!(block_y + 2 * patch_radius + 2 * search_radius);
    let expanded_elems = comptime!(
        (block_x + 2 * patch_radius + 2 * search_radius) * (block_y + 2 * patch_radius + 2 * search_radius)
    );
    let mut smem_center = SharedMemory::<Vector<C, N>>::new(expanded_elems as usize);
    let mut smem_fwd = SharedMemory::<i32>::new(tile_elems as usize);
    let mut smem_bwd = SharedMemory::<i32>::new(tile_elems as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let in_image = global_x < width && global_y < height;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;

    let tile_x0 = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let tile_y0 = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let expanded_x0 = tile_x0 - search_radius as i32;
    let expanded_y0 = tile_y0 - search_radius as i32;
    let interior = expanded_interior(
        expanded_x0,
        expanded_y0,
        expanded_width,
        expanded_height,
        width,
        height,
    );

    fill_code_tile(
        codes,
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

            // The backward comparison mirrors the forward one against `frame_bwd` at `-q`, so both
            // centre on the same cached patch.
            let own_center = smem_center[own_center_idx];
            let fwd_x = global_x as i32 + q_x;
            let fwd_y = global_y as i32 + q_y;
            let bwd_x = global_x as i32 - q_x;
            let bwd_y = global_y as i32 - q_y;
            let own_fwd = read_clamped_codes(codes, fwd_x, fwd_y, frame_fwd, width, height);
            let own_bwd = read_clamped_codes(codes, bwd_x, bwd_y, frame_bwd, width, height);
            smem_fwd[own_idx] = code_sum_sq(own_center, own_fwd, channels);
            smem_bwd[own_idx] = code_sum_sq(own_center, own_bwd, channels);

            let mut halo = thread_id;
            while halo < halo_elems {
                let idx = halo_slot_index(halo, block_x, block_y, patch_radius);
                let tile_x = idx % tile_width;
                let tile_y = idx / tile_width;
                let center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let center = smem_center[center_idx];
                let src_x = tile_x0 + tile_x as i32;
                let src_y = tile_y0 + tile_y as i32;
                let fwd_neighbour =
                    read_clamped_codes(codes, src_x + q_x, src_y + q_y, frame_fwd, width, height);
                let bwd_neighbour =
                    read_clamped_codes(codes, src_x - q_x, src_y - q_y, frame_bwd, width, height);
                smem_fwd[idx as usize] = code_sum_sq(center, fwd_neighbour, channels);
                smem_bwd[idx as usize] = code_sum_sq(center, bwd_neighbour, channels);
                halo += threads;
            }

            sync_cube();

            if in_image {
                let ssd_fwd = int_patch_sum(&smem_fwd, local_x, local_y, tile_width, patch_radius);
                let ssd_bwd = int_patch_sum(&smem_bwd, local_x, local_y, tile_width, patch_radius);
                let weight_fwd = int_window_weight(ssd_fwd, inv_scale2, h2_inv_norm, noise_offset);
                let weight_bwd = int_window_weight(ssd_bwd, inv_scale2, h2_inv_norm, noise_offset);
                let fwd_pixel = read_clamped_line(input, fwd_x, fwd_y, frame_fwd, width, height);
                let bwd_pixel = read_clamped_line(input, bwd_x, bwd_y, frame_bwd, width, height);

                accumulate_pair_weights(
                    &mut accum_reg,
                    &mut weight_sum_reg,
                    &mut max_weight_reg,
                    fwd_pixel,
                    bwd_pixel,
                    weight_fwd,
                    weight_bwd,
                    conf_fwd,
                    conf_bwd,
                    global_x,
                    global_y,
                    use_confidence,
                    step,
                    blocks_x,
                    blocks_y,
                );
            }

            // The next offset overwrites the distance tiles.
            sync_cube();
        }
    }

    if in_image {
        store_window_sums(
            accum,
            weight_sum,
            max_weight,
            accum_reg,
            weight_sum_reg,
            max_weight_reg,
            global_x,
            global_y,
            width,
        );
    }
}
