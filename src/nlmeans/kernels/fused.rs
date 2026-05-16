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

/// Windowed temporal pair kernel: loops over every `(q_x, q_y)` in the
/// search window for one `q_k != 0` inside a single launch, keeping the
/// running accumulator / weight sum / max weight in registers. The final
/// values are added to global once at the end, collapsing what used to be
/// `(2·search_radius + 1)²` launches into one.
///
/// `frame_t` is read once into an expanded SMEM tile of size
/// `(block + 2·patch_radius + 2·search_radius)²`, large enough to cover
/// both the forward tile (centred on the cube) and every shifted backward
/// tile (centred at `cube − q` for q in the window). Per q, only the
/// shifted neighbour pixels in `frame_fwd` / `frame_bwd` come from
/// global; both center reads hit the cache. This roughly halves the
/// global read traffic vs the naive windowed version that re-reads
/// `frame_t` for every q.
///
/// The two distance tiles (`smem_fwd`, `smem_bwd`) are reused across q
/// iterations and invalidated by `sync_cube` between iterations.
#[cube(launch)]
pub fn nlm_fused_pair_accumulate_window(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored: u32,
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
    let mut smem_center = SharedMemory::<f32>::new_lined(expanded_elems as usize, stored as usize);
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

    let mut accum_reg = Line::<f32>::empty(input.line_size());
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

                // fwd center sits at (tile_x + search_radius, tile_y + search_radius)
                // in expanded-tile coordinates.
                let fwd_center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let fwd_center = smem_center[fwd_center_idx];
                let fwd_neighbor = read_clamped_line(
                    input,
                    fwd_tile_x0 + tile_x as i32 + q_x,
                    fwd_tile_y0 + tile_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );
                smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

                // bwd center sits at (tile_x − q_x + search_radius, tile_y − q_y + search_radius).
                // q ∈ [−search_radius, +search_radius] keeps the result in [0, expanded_width).
                let bwd_ex = (tile_x as i32 - q_x + search_radius as i32) as u32;
                let bwd_ey = (tile_y as i32 - q_y + search_radius as i32) as u32;
                let bwd_center = smem_center[(bwd_ey * expanded_width + bwd_ex) as usize];
                let bwd_neighbor = read_clamped_line(
                    input,
                    fwd_tile_x0 + tile_x as i32 - 2 * q_x,
                    fwd_tile_y0 + tile_y as i32 - 2 * q_y,
                    frame_bwd,
                    width,
                    height,
                );
                smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

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

                let weight_fwd = f32::exp(-sum_fwd * h2_inv_norm);
                let weight_bwd = f32::exp(-sum_bwd * h2_inv_norm);

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
                let line_w_fwd = Line::<f32>::empty(input.line_size()).fill(weight_fwd);
                let line_w_bwd = Line::<f32>::empty(input.line_size()).fill(weight_bwd);
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

/// Windowed spatial (q_k == 0) kernel: same structure as
/// [`nlm_fused_pair_accumulate_window`] but exploits the q-symmetry of
/// the weight map. Patch distance is symmetric (`w(x, −q) = w(x−q, q)`),
/// so iterating the full search window in a single direction at pixel x
/// produces the same accumulator as the original half-window paired
/// dispatch. One distance tile, one neighbour read per q.
///
/// `frame_t` is cached in the expanded SMEM tile so per-q work touches
/// global only for the shifted neighbour pixel. `q = (0, 0)` is skipped
/// at comptime; the self-pair contribution is folded back in by `nlm_finish`
/// via the `wref · max_weight` term.
#[cube(launch)]
pub fn nlm_fused_single_window(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored: u32,
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
    let mut smem_center = SharedMemory::<f32>::new_lined(expanded_elems as usize, stored as usize);
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

    let mut accum_reg = Line::<f32>::empty(input.line_size());
    let mut weight_sum_reg = 0.0f32;
    let mut max_weight_reg = 0.0f32;

    let window_side = comptime!(2 * search_radius + 1);

    #[unroll]
    for q_yi in 0..window_side {
        #[unroll]
        for q_xi in 0..window_side {
            let q_x = q_xi as i32 - search_radius as i32;
            let q_y = q_yi as i32 - search_radius as i32;
            // Skip the self-pair; its contribution is reintroduced by
            // `nlm_finish` via `wref · max_weight`.
            if comptime!(q_x == 0 && q_y == 0) {
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
                    let weight = f32::exp(-patch_sum * h2_inv_norm);

                    let neighbor_pixel = read_clamped_line(
                        input,
                        global_x as i32 + q_x,
                        global_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    let line_w = Line::<f32>::empty(input.line_size()).fill(weight);
                    accum_reg = accum_reg + neighbor_pixel * line_w;
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

/// `_ref` variant of [`nlm_fused_pair_accumulate_window`]. Distance reads
/// (the cached center tile and per-q neighbours) come from `reference`;
/// pixel accumulation reads from `input` so the original-clip values
/// flow into `accum` while weights are derived from the cleaner
/// reference frames.
#[cube(launch)]
pub fn nlm_fused_pair_accumulate_window_ref(
    input: &Array<Line<f32>>,
    reference: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored: u32,
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
    let mut smem_center = SharedMemory::<f32>::new_lined(expanded_elems as usize, stored as usize);
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

    let mut accum_reg = Line::<f32>::empty(input.line_size());
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

                let fwd_center_idx =
                    ((tile_y + search_radius) * expanded_width + (tile_x + search_radius)) as usize;
                let fwd_center = smem_center[fwd_center_idx];
                let fwd_neighbor = read_clamped_line(
                    reference,
                    fwd_tile_x0 + tile_x as i32 + q_x,
                    fwd_tile_y0 + tile_y as i32 + q_y,
                    frame_fwd,
                    width,
                    height,
                );
                smem_fwd[idx as usize] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;

                let bwd_ex = (tile_x as i32 - q_x + search_radius as i32) as u32;
                let bwd_ey = (tile_y as i32 - q_y + search_radius as i32) as u32;
                let bwd_center = smem_center[(bwd_ey * expanded_width + bwd_ex) as usize];
                let bwd_neighbor = read_clamped_line(
                    reference,
                    fwd_tile_x0 + tile_x as i32 - 2 * q_x,
                    fwd_tile_y0 + tile_y as i32 - 2 * q_y,
                    frame_bwd,
                    width,
                    height,
                );
                smem_bwd[idx as usize] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;

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

                let weight_fwd = f32::exp(-sum_fwd * h2_inv_norm);
                let weight_bwd = f32::exp(-sum_bwd * h2_inv_norm);

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
                let line_w_fwd = Line::<f32>::empty(input.line_size()).fill(weight_fwd);
                let line_w_bwd = Line::<f32>::empty(input.line_size()).fill(weight_bwd);
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

/// `_ref` variant of [`nlm_fused_single_window`]. Distance reads from
/// `reference[frame_t]` (cached center + per-q neighbours, all at q_k=0);
/// pixel accumulation reads from `input[frame_t]`.
#[cube(launch)]
pub fn nlm_fused_single_window_ref(
    input: &Array<Line<f32>>,
    reference: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_t: u32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored: u32,
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
    let mut smem_center = SharedMemory::<f32>::new_lined(expanded_elems as usize, stored as usize);
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

    let mut accum_reg = Line::<f32>::empty(input.line_size());
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
                    let weight = f32::exp(-patch_sum * h2_inv_norm);

                    let neighbor_pixel = read_clamped_line(
                        input,
                        global_x as i32 + q_x,
                        global_y as i32 + q_y,
                        frame_t,
                        width,
                        height,
                    );
                    let line_w = Line::<f32>::empty(input.line_size()).fill(weight);
                    accum_reg = accum_reg + neighbor_pixel * line_w;
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
