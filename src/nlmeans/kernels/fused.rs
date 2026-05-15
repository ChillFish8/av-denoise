use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{accumulate_pair, channel_scale, line_sum_sq, read_clamped_line, read_line};

/// Distance + 2D box filter + Welsch weight, written to `output[gx, gy]`.
///
/// The cube cooperatively loads a `(block + 2·patch_radius)²` tile of
/// per-pixel scaled distances into shared memory, then each thread
/// sums its `(2·patch_radius + 1)²` patch and applies the Welsch
/// kernel. A cube-uniform `interior` flag picks unclamped reads when
/// the whole tile (and its q-shifted twin) lies inside the image; warps
/// near the border fall back to the clamped path. The flag is uniform
/// across the cube, so the branch causes no warp divergence.
#[cube(launch)]
pub fn nlm_dist_2d_weight(
    input: &Array<Line<f32>>,
    output: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
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

    output[(global_y * width + global_x) as usize] = f32::exp(-patch_sum * h2_inv_norm);
}

/// Fully fused distance + 2D box filter + Welsch weight + accumulate.
///
/// Each thread accumulates two contributions at its output pixel
/// `(global_x, global_y)`:
/// * the forward neighbour at `(global + q, frame_fwd)` weighted by the
///   patch similarity at `(global, frame_t)` vs the shifted neighbour;
/// * the backward neighbour at `(global − q, frame_bwd)` weighted by
///   the patch similarity centred at `(global − q, frame_t)`.
///
/// Both weights live in registers, computed from two SMEM tiles. The
/// forward tile is centred on the cube so its tile-local centre maps
/// to `(global_x, global_y)`. The backward tile is centred at the cube
/// shifted by `(−q_x, −q_y)` so its tile-local centre maps to
/// `(global_x − q_x, global_y − q_y)`.
///
/// `bwd_shift_(x|y)` controls which neighbour the backward distance
/// reads against: `+q` for `k == 0` (the patch comparison degenerates
/// to a symmetric self-pair), `−q` for `k != 0` (true temporal pair).
#[cube(launch)]
pub fn nlm_fused_pair_accumulate(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    bwd_shift_x: i32,
    bwd_shift_y: i32,
    h2_inv_norm: f32,
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
    let mut smem_fwd = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let fwd_tile_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let bwd_tile_x = fwd_tile_x - q_x;
    let bwd_tile_y = fwd_tile_y - q_y;

    let scale = channel_scale(channels);

    // The four read regions are (frame_t, fwd_tile), (frame_fwd, fwd_tile+q),
    // (frame_t, bwd_tile), and (frame_bwd, bwd_tile − q). Since
    // bwd_tile = fwd_tile − q, the third region is fwd_tile − q and the
    // fourth is fwd_tile − 2q.
    let fwd_end_x = fwd_tile_x + tile_width as i32;
    let fwd_end_y = fwd_tile_y + tile_height as i32;
    let interior = fwd_tile_x >= 0
        && fwd_end_x <= width as i32
        && fwd_tile_y >= 0
        && fwd_end_y <= height as i32
        && (fwd_tile_x + q_x) >= 0
        && (fwd_end_x + q_x) <= width as i32
        && (fwd_tile_y + q_y) >= 0
        && (fwd_end_y + q_y) <= height as i32
        && (fwd_tile_x - q_x) >= 0
        && (fwd_end_x - q_x) <= width as i32
        && (fwd_tile_y - q_y) >= 0
        && (fwd_end_y - q_y) <= height as i32
        && (fwd_tile_x - 2 * q_x) >= 0
        && (fwd_end_x - 2 * q_x) <= width as i32
        && (fwd_tile_y - 2 * q_y) >= 0
        && (fwd_end_y - 2 * q_y) <= height as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;

    if interior {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let fwd_src_x = (fwd_tile_x + tile_x as i32) as u32;
            let fwd_src_y = (fwd_tile_y + tile_y as i32) as u32;
            let bwd_src_x = (bwd_tile_x + tile_x as i32) as u32;
            let bwd_src_y = (bwd_tile_y + tile_y as i32) as u32;

            let fwd_center = read_line(input, fwd_src_x, fwd_src_y, frame_t, width, height);
            let fwd_neighbor = read_line(
                input,
                (fwd_src_x as i32 + q_x) as u32,
                (fwd_src_y as i32 + q_y) as u32,
                frame_fwd,
                width,
                height,
            );
            smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

            let bwd_center = read_line(input, bwd_src_x, bwd_src_y, frame_t, width, height);
            let bwd_neighbor = read_line(
                input,
                (bwd_src_x as i32 + bwd_shift_x) as u32,
                (bwd_src_y as i32 + bwd_shift_y) as u32,
                frame_bwd,
                width,
                height,
            );
            smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

            idx += threads;
        }
    } else {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let fwd_src_x = fwd_tile_x + tile_x as i32;
            let fwd_src_y = fwd_tile_y + tile_y as i32;
            let bwd_src_x = bwd_tile_x + tile_x as i32;
            let bwd_src_y = bwd_tile_y + tile_y as i32;

            let fwd_center = read_clamped_line(input, fwd_src_x, fwd_src_y, frame_t, width, height);
            let fwd_neighbor =
                read_clamped_line(input, fwd_src_x + q_x, fwd_src_y + q_y, frame_fwd, width, height);
            smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

            let bwd_center = read_clamped_line(input, bwd_src_x, bwd_src_y, frame_t, width, height);
            let bwd_neighbor = read_clamped_line(
                input,
                bwd_src_x + bwd_shift_x,
                bwd_src_y + bwd_shift_y,
                frame_bwd,
                width,
                height,
            );
            smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

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
    let mut sum_fwd = 0.0f32;
    let mut sum_bwd = 0.0f32;
    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width + center_tile_x
                - patch_radius
                + offset_x) as usize;
            sum_fwd += smem_fwd[smem_idx];
            sum_bwd += smem_bwd[smem_idx];
        }
    }

    let weight_fwd = f32::exp(-sum_fwd * h2_inv_norm);
    let weight_bwd = f32::exp(-sum_bwd * h2_inv_norm);

    accumulate_pair(
        input, accum, weight_sum, max_weight, global_x, global_y, q_x, q_y, frame_fwd, frame_bwd, weight_fwd,
        weight_bwd, width, height,
    );
}

/// `_ref` variant of [`nlm_dist_2d_weight`]. Distance reads come from
/// `reference` (a prefiltered or externally-supplied clip with the same
/// layout as `input`); the weight output is unchanged. Used when an
/// rclip is active so weight calculation sees a cleaner image than the
/// noisy input.
#[cube(launch)]
pub fn nlm_dist_2d_weight_ref(
    reference: &Array<Line<f32>>,
    output: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
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

    output[(global_y * width + global_x) as usize] = f32::exp(-patch_sum * h2_inv_norm);
}

/// `_ref` variant of [`nlm_fused_pair_accumulate`]. Patch distances are
/// computed from `reference` (prefiltered clip); the pixel values
/// folded into `accum` still come from `input`. Same SMEM footprint and
/// dispatch shape as the non-`_ref` variant.
#[cube(launch)]
pub fn nlm_fused_pair_accumulate_ref(
    input: &Array<Line<f32>>,
    reference: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    bwd_shift_x: i32,
    bwd_shift_y: i32,
    h2_inv_norm: f32,
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
    let mut smem_fwd = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(comptime!(
        (block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)
    ) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let fwd_tile_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;
    let fwd_tile_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let bwd_tile_x = fwd_tile_x - q_x;
    let bwd_tile_y = fwd_tile_y - q_y;

    let scale = channel_scale(channels);

    let fwd_end_x = fwd_tile_x + tile_width as i32;
    let fwd_end_y = fwd_tile_y + tile_height as i32;
    let interior = fwd_tile_x >= 0
        && fwd_end_x <= width as i32
        && fwd_tile_y >= 0
        && fwd_end_y <= height as i32
        && (fwd_tile_x + q_x) >= 0
        && (fwd_end_x + q_x) <= width as i32
        && (fwd_tile_y + q_y) >= 0
        && (fwd_end_y + q_y) <= height as i32
        && (fwd_tile_x - q_x) >= 0
        && (fwd_end_x - q_x) <= width as i32
        && (fwd_tile_y - q_y) >= 0
        && (fwd_end_y - q_y) <= height as i32
        && (fwd_tile_x - 2 * q_x) >= 0
        && (fwd_end_x - 2 * q_x) <= width as i32
        && (fwd_tile_y - 2 * q_y) >= 0
        && (fwd_end_y - 2 * q_y) <= height as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;

    if interior {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let fwd_src_x = (fwd_tile_x + tile_x as i32) as u32;
            let fwd_src_y = (fwd_tile_y + tile_y as i32) as u32;
            let bwd_src_x = (bwd_tile_x + tile_x as i32) as u32;
            let bwd_src_y = (bwd_tile_y + tile_y as i32) as u32;

            let fwd_center = read_line(reference, fwd_src_x, fwd_src_y, frame_t, width, height);
            let fwd_neighbor = read_line(
                reference,
                (fwd_src_x as i32 + q_x) as u32,
                (fwd_src_y as i32 + q_y) as u32,
                frame_fwd,
                width,
                height,
            );
            smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

            let bwd_center = read_line(reference, bwd_src_x, bwd_src_y, frame_t, width, height);
            let bwd_neighbor = read_line(
                reference,
                (bwd_src_x as i32 + bwd_shift_x) as u32,
                (bwd_src_y as i32 + bwd_shift_y) as u32,
                frame_bwd,
                width,
                height,
            );
            smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

            idx += threads;
        }
    } else {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let fwd_src_x = fwd_tile_x + tile_x as i32;
            let fwd_src_y = fwd_tile_y + tile_y as i32;
            let bwd_src_x = bwd_tile_x + tile_x as i32;
            let bwd_src_y = bwd_tile_y + tile_y as i32;

            let fwd_center = read_clamped_line(reference, fwd_src_x, fwd_src_y, frame_t, width, height);
            let fwd_neighbor = read_clamped_line(
                reference,
                fwd_src_x + q_x,
                fwd_src_y + q_y,
                frame_fwd,
                width,
                height,
            );
            smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

            let bwd_center = read_clamped_line(reference, bwd_src_x, bwd_src_y, frame_t, width, height);
            let bwd_neighbor = read_clamped_line(
                reference,
                bwd_src_x + bwd_shift_x,
                bwd_src_y + bwd_shift_y,
                frame_bwd,
                width,
                height,
            );
            smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

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
    let mut sum_fwd = 0.0f32;
    let mut sum_bwd = 0.0f32;
    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let smem_idx = ((center_tile_y - patch_radius + offset_y) * tile_width + center_tile_x
                - patch_radius
                + offset_x) as usize;
            sum_fwd += smem_fwd[smem_idx];
            sum_bwd += smem_bwd[smem_idx];
        }
    }

    let weight_fwd = f32::exp(-sum_fwd * h2_inv_norm);
    let weight_bwd = f32::exp(-sum_bwd * h2_inv_norm);

    accumulate_pair(
        input, accum, weight_sum, max_weight, global_x, global_y, q_x, q_y, frame_fwd, frame_bwd, weight_fwd,
        weight_bwd, width, height,
    );
}
