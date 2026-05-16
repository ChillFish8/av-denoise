use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{
    accumulate_pair,
    channel_scale,
    clamp_coord,
    line_sum_sq,
    read_clamped_line,
    read_line,
};

/// Per-pixel squared distance for the `+q` / `−q` pair. Writes both
/// raw distance buffers in a single pass; downstream the separable
/// box filter and the fused vweight+accumulate consume them.
#[cube(launch_unchecked)]
pub fn nlm_distance_pair<N: Size>(
    input: &Array<Vector<f32, N>>,
    dist_fwd: &mut Array<f32>,
    dist_bwd: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let scale = channel_scale(channels);

    let fwd_center = read_line(input, x, y, frame_t, width, height);
    let bwd_center = read_line(input, x, y, frame_bwd, width, height);

    let neighbor_x = x as i32 + q_x;
    let neighbor_y = y as i32 + q_y;
    let interior =
        neighbor_x >= 0 && neighbor_x < width as i32 && neighbor_y >= 0 && neighbor_y < height as i32;

    let fwd_neighbor = if interior {
        read_line(
            input,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_fwd,
            width,
            height,
        )
    } else {
        read_clamped_line(input, neighbor_x, neighbor_y, frame_fwd, width, height)
    };
    let bwd_neighbor = if interior {
        read_line(
            input,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_t,
            width,
            height,
        )
    } else {
        read_clamped_line(input, neighbor_x, neighbor_y, frame_t, width, height)
    };

    let pixel_idx = (y * width + x) as usize;
    dist_fwd[pixel_idx] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;
    dist_bwd[pixel_idx] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;
}

/// Per-pixel squared distance between `(frame_t, x, y)` and
/// `(frame_q, x + q_x, y + q_y)`, scaled to the channel convention.
#[cube(launch_unchecked)]
pub fn nlm_distance<N: Size>(
    input: &Array<Vector<f32, N>>,
    dist: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let scale = channel_scale(channels);
    let center = read_line(input, x, y, frame_t, width, height);

    let neighbor_x = x as i32 + q_x;
    let neighbor_y = y as i32 + q_y;
    let interior =
        neighbor_x >= 0 && neighbor_x < width as i32 && neighbor_y >= 0 && neighbor_y < height as i32;
    let neighbor = if interior {
        read_line(
            input,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_q,
            width,
            height,
        )
    } else {
        read_clamped_line(input, neighbor_x, neighbor_y, frame_q, width, height)
    };

    dist[(y * width + x) as usize] = line_sum_sq(center - neighbor, channels) * scale;
}

/// Horizontal 1D box filter (width = 2·patch_radius + 1) via shared
/// memory. Loads a `(block_x + 2·patch_radius) × block_y` tile cooperatively
/// then writes the per-row patch sum at each `(global_x, global_y)`.
#[cube(launch_unchecked)]
pub fn nlm_horizontal_sum(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * block_y);
    let mut smem = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * block_y) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;
    while idx < tile_elems {
        let tile_x = idx % tile_width;
        let tile_y = idx / tile_width;
        let src_x = tile_start_x + tile_x as i32;
        let src_y = CUBE_POS_Y * block_y + tile_y;
        let clamped_x = clamp_coord(src_x, width);
        let clamped_y = clamp_coord(src_y as i32, height);
        smem[idx as usize] = input[(clamped_y * width + clamped_x) as usize];
        idx += threads;
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let patch_size = 2 * patch_radius + 1;
    let smem_base = local_y * tile_width + local_x;
    let mut sum = 0.0f32;
    for offset_x in 0..patch_size {
        sum += smem[(smem_base + offset_x) as usize];
    }
    output[(global_y * width + global_x) as usize] = sum;
}

/// Vertical 1D box filter (height = 2·patch_radius + 1) over the
/// hsum buffer, followed by the Welsch weight `exp(−sum · h2_inv_norm)`.
#[cube(launch_unchecked)]
pub fn nlm_vertical_weight(
    input: &Array<f32>,
    output: &mut Array<f32>,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_elems = comptime!(block_x * (block_y + 2 * patch_radius));
    let mut smem = SharedMemory::<f32>::new(comptime!(block_x * (block_y + 2 * patch_radius)) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;
    while idx < tile_elems {
        let tile_x = idx % block_x;
        let tile_y = idx / block_x;
        let src_x = CUBE_POS_X * block_x + tile_x;
        let src_y = tile_start_y + tile_y as i32;
        let clamped_x = clamp_coord(src_x as i32, width);
        let clamped_y = clamp_coord(src_y, height);
        smem[idx as usize] = input[(clamped_y * width + clamped_x) as usize];
        idx += threads;
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let patch_size = 2 * patch_radius + 1;
    let mut sum = 0.0f32;
    for offset_y in 0..patch_size {
        sum += smem[((local_y + offset_y) * block_x + local_x) as usize];
    }
    output[(global_y * width + global_x) as usize] = f32::exp(-sum * h2_inv_norm);
}

/// Paired horizontal 1D box filter — the forward and backward hsum
/// passes share one cooperative tile load and one `sync_cube`.
#[cube(launch_unchecked)]
pub fn nlm_horizontal_sum_pair(
    input_fwd: &Array<f32>,
    input_bwd: &Array<f32>,
    output_fwd: &mut Array<f32>,
    output_bwd: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let tile_elems = comptime!((block_x + 2 * patch_radius) * block_y);
    let mut smem_fwd = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * block_y) as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * block_y) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;
    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;
    while idx < tile_elems {
        let tile_x = idx % tile_width;
        let tile_y = idx / tile_width;
        let src_x = tile_start_x + tile_x as i32;
        let src_y = CUBE_POS_Y * block_y + tile_y;
        let clamped_x = clamp_coord(src_x, width);
        let clamped_y = clamp_coord(src_y as i32, height);
        let src_idx = (clamped_y * width + clamped_x) as usize;
        smem_fwd[idx as usize] = input_fwd[src_idx];
        smem_bwd[idx as usize] = input_bwd[src_idx];
        idx += threads;
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let patch_size = 2 * patch_radius + 1;
    let smem_base = local_y * tile_width + local_x;
    let mut sum_fwd = 0.0f32;
    let mut sum_bwd = 0.0f32;
    for offset_x in 0..patch_size {
        sum_fwd += smem_fwd[(smem_base + offset_x) as usize];
        sum_bwd += smem_bwd[(smem_base + offset_x) as usize];
    }

    let out_idx = (global_y * width + global_x) as usize;
    output_fwd[out_idx] = sum_fwd;
    output_bwd[out_idx] = sum_bwd;
}

/// Fused vertical 1D box filter + Welsch weight + accumulate for the
/// paired-distance separable path. The backward tile is loaded from
/// `hsum_bwd` at the cube position shifted by `(−q_x, −q_y)` so the
/// vsum at each thread directly produces the neighbour backward weight.
#[cube(launch_unchecked)]
pub fn nlm_vweight_pair_accumulate<N: Size>(
    hsum_fwd: &Array<f32>,
    hsum_bwd: &Array<f32>,
    input: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_elems = comptime!(block_x * (block_y + 2 * patch_radius));
    let mut smem_fwd = SharedMemory::<f32>::new(comptime!(block_x * (block_y + 2 * patch_radius)) as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(comptime!(block_x * (block_y + 2 * patch_radius)) as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let fwd_tile_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;
    let bwd_tile_y = fwd_tile_y - q_y;
    let bwd_tile_x_origin = CUBE_POS_X as i32 * block_x as i32 - q_x;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;
    while idx < tile_elems {
        let tile_x = idx % block_x;
        let tile_y = idx / block_x;

        let fwd_src_x = CUBE_POS_X * block_x + tile_x;
        let fwd_src_y = fwd_tile_y + tile_y as i32;
        let fwd_clamped_x = clamp_coord(fwd_src_x as i32, width);
        let fwd_clamped_y = clamp_coord(fwd_src_y, height);
        smem_fwd[idx as usize] = hsum_fwd[(fwd_clamped_y * width + fwd_clamped_x) as usize];

        let bwd_src_x = bwd_tile_x_origin + tile_x as i32;
        let bwd_src_y = bwd_tile_y + tile_y as i32;
        let bwd_clamped_x = clamp_coord(bwd_src_x, width);
        let bwd_clamped_y = clamp_coord(bwd_src_y, height);
        smem_bwd[idx as usize] = hsum_bwd[(bwd_clamped_y * width + bwd_clamped_x) as usize];

        idx += threads;
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let patch_size = 2 * patch_radius + 1;
    let mut sum_fwd = 0.0f32;
    let mut sum_bwd = 0.0f32;
    for offset_y in 0..patch_size {
        let smem_idx = ((local_y + offset_y) * block_x + local_x) as usize;
        sum_fwd += smem_fwd[smem_idx];
        sum_bwd += smem_bwd[smem_idx];
    }

    let weight_fwd = f32::exp(-sum_fwd * h2_inv_norm);
    let weight_bwd = f32::exp(-sum_bwd * h2_inv_norm);

    accumulate_pair(
        input, accum, weight_sum, max_weight, global_x, global_y, q_x, q_y, frame_fwd, frame_bwd, weight_fwd,
        weight_bwd, width, height,
    );
}

/// `_ref` variant of [`nlm_distance`]. Reads `reference` for both
/// center and neighbour; downstream separable kernels consume the
/// distance buffer unchanged.
#[cube(launch_unchecked)]
pub fn nlm_distance_ref<N: Size>(
    reference: &Array<Vector<f32, N>>,
    dist: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let scale = channel_scale(channels);
    let center = read_line(reference, x, y, frame_t, width, height);

    let neighbor_x = x as i32 + q_x;
    let neighbor_y = y as i32 + q_y;
    let interior =
        neighbor_x >= 0 && neighbor_x < width as i32 && neighbor_y >= 0 && neighbor_y < height as i32;
    let neighbor = if interior {
        read_line(
            reference,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_q,
            width,
            height,
        )
    } else {
        read_clamped_line(reference, neighbor_x, neighbor_y, frame_q, width, height)
    };

    dist[(y * width + x) as usize] = line_sum_sq(center - neighbor, channels) * scale;
}

/// `_ref` variant of [`nlm_distance_pair`]. Both forward and backward
/// distance signals read from `reference`.
#[cube(launch_unchecked)]
pub fn nlm_distance_pair_ref<N: Size>(
    reference: &Array<Vector<f32, N>>,
    dist_fwd: &mut Array<f32>,
    dist_bwd: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let scale = channel_scale(channels);

    let fwd_center = read_line(reference, x, y, frame_t, width, height);
    let bwd_center = read_line(reference, x, y, frame_bwd, width, height);

    let neighbor_x = x as i32 + q_x;
    let neighbor_y = y as i32 + q_y;
    let interior =
        neighbor_x >= 0 && neighbor_x < width as i32 && neighbor_y >= 0 && neighbor_y < height as i32;

    let fwd_neighbor = if interior {
        read_line(
            reference,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_fwd,
            width,
            height,
        )
    } else {
        read_clamped_line(reference, neighbor_x, neighbor_y, frame_fwd, width, height)
    };

    let bwd_neighbor = if interior {
        read_line(
            reference,
            neighbor_x as u32,
            neighbor_y as u32,
            frame_t,
            width,
            height,
        )
    } else {
        read_clamped_line(reference, neighbor_x, neighbor_y, frame_t, width, height)
    };

    let pixel_idx = (y * width + x) as usize;
    dist_fwd[pixel_idx] = line_sum_sq(fwd_center - fwd_neighbor, channels) * scale;
    dist_bwd[pixel_idx] = line_sum_sq(bwd_center - bwd_neighbor, channels) * scale;
}
