use cubecl::prelude::*;
use cubecl::terminate;

#[cube]
fn clamp_coord(value: i32, #[comptime] limit: u32) -> u32 {
    let mut result = value as u32;
    if value < 0 {
        result = 0u32;
    } else if value >= limit as i32 {
        result = limit - 1;
    }
    result
}

/// Vectorised pixel read at `(x, y)` in `frame`, clamped to the image
/// edges on both axes. The frame index is trusted; callers always pass
/// a physical slot that references loaded data.
#[cube]
fn read_clamped_line(
    buf: &Array<Line<f32>>,
    x: i32,
    y: i32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Line<f32> {
    let clamped_x = clamp_coord(x, width);
    let clamped_y = clamp_coord(y, height);
    let idx = (frame * height + clamped_y) * width + clamped_x;
    buf[idx as usize]
}

/// Unchecked variant of `read_clamped_line`. The caller guarantees
/// `x ∈ [0, width)` and `y ∈ [0, height)`.
#[cube]
fn read_line(
    buf: &Array<Line<f32>>,
    x: u32,
    y: u32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Line<f32> {
    let idx = (frame * height + y) * width + x;
    buf[idx as usize]
}

/// Sum of squared lane differences over a `Line`. The loop is fully
/// unrolled at compile time because `channels` is comptime.
#[cube]
fn line_sum_sq(diff: Line<f32>, #[comptime] channels: u32) -> f32 {
    let mut sum = 0.0f32;
    #[unroll]
    for c in 0..channels {
        sum += diff[c as usize] * diff[c as usize];
    }
    sum
}

/// Per-channel distance scale (luma×3, chroma×1.5, full YUV×1) so the
/// three channel modes share one `h2_inv_norm`.
#[cube]
fn channel_scale(#[comptime] channels: u32) -> f32 {
    let mut scale = 1.0f32;
    if channels == 1 {
        scale = 3.0f32;
    } else if channels == 2 {
        scale = 1.5f32;
    }
    scale
}

/// GPU→GPU buffer copy. Uses a strided loop so the grid can be capped
/// under the 65 535 1D dispatch limit.
#[cube(launch)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = src[idx as usize];
        idx += total_threads;
    }
}

/// Zero `accum`, `weight_sum`, `max_weight` in one dispatch. The hot
/// loop covers all three up to `weight_len`; a tail loop finishes the
/// channel-padded remainder of `accum` (which is always at least as
/// long as `weight_sum` and `max_weight`).
#[cube(launch)]
pub fn gpu_zero_buffers(
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    #[comptime] accum_len: u32,
    #[comptime] weight_len: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < weight_len {
        accum[idx as usize] = 0.0f32;
        weight_sum[idx as usize] = 0.0f32;
        max_weight[idx as usize] = 0.0f32;
        idx += total_threads;
    }
    while idx < accum_len {
        accum[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

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

/// Per-pixel squared distance for the `+q` / `−q` pair. Writes both
/// raw distance buffers in a single pass; downstream the separable
/// box filter and the fused vweight+accumulate consume them.
#[cube(launch)]
pub fn nlm_distance_pair(
    input: &Array<Line<f32>>,
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
#[cube(launch)]
pub fn nlm_distance(
    input: &Array<Line<f32>>,
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
#[cube(launch)]
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
#[cube(launch)]
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
#[cube(launch)]
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
#[cube(launch)]
pub fn nlm_vweight_pair_accumulate(
    hsum_fwd: &Array<f32>,
    hsum_bwd: &Array<f32>,
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
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

/// Add the `+q` and `−q` contributions at thread `(global_x, global_y)`.
/// The forward neighbour lives at `(global + q, frame_fwd)` weighted by
/// `weight_fwd`; the backward neighbour at `(global − q, frame_bwd)`
/// weighted by `weight_bwd`. A single per-thread interior check covers
/// both reads, with a clamped fallback for the border.
#[cube]
fn accumulate_pair(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    global_x: u32,
    global_y: u32,
    q_x: i32,
    q_y: i32,
    frame_fwd: u32,
    frame_bwd: u32,
    weight_fwd: f32,
    weight_bwd: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let fwd_nx = global_x as i32 + q_x;
    let fwd_ny = global_y as i32 + q_y;
    let bwd_nx = global_x as i32 - q_x;
    let bwd_ny = global_y as i32 - q_y;
    let interior = fwd_nx >= 0
        && fwd_nx < width as i32
        && fwd_ny >= 0
        && fwd_ny < height as i32
        && bwd_nx >= 0
        && bwd_nx < width as i32
        && bwd_ny >= 0
        && bwd_ny < height as i32;

    let fwd_pixel = if interior {
        read_line(input, fwd_nx as u32, fwd_ny as u32, frame_fwd, width, height)
    } else {
        read_clamped_line(input, fwd_nx, fwd_ny, frame_fwd, width, height)
    };
    let bwd_pixel = if interior {
        read_line(input, bwd_nx as u32, bwd_ny as u32, frame_bwd, width, height)
    } else {
        read_clamped_line(input, bwd_nx, bwd_ny, frame_bwd, width, height)
    };

    let pixel_idx = (global_y * width + global_x) as usize;
    let cur_max = max_weight[pixel_idx];
    max_weight[pixel_idx] = f32::max(f32::max(weight_fwd, weight_bwd), cur_max);

    let line_w_fwd = Line::<f32>::empty(input.line_size()).fill(weight_fwd);
    let line_w_bwd = Line::<f32>::empty(input.line_size()).fill(weight_bwd);
    let cur = accum[pixel_idx];
    accum[pixel_idx] = cur + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;

    weight_sum[pixel_idx] += weight_fwd + weight_bwd;
}

/// Apply the `+q` and `−q` contributions at every pixel using a single
/// weight map. `weights_fwd` and `weights_bwd` may point to the same
/// buffer for the symmetric (k=0) case. The backward lookup uses the
/// clamped neighbour index so border pixels read a valid weight.
#[cube(launch)]
pub fn nlm_accumulate(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    weights_fwd: &Array<f32>,
    weights_bwd: &Array<f32>,
    max_weight: &mut Array<f32>,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let pixel_idx = (y * width + x) as usize;
    let weight_fwd = weights_fwd[pixel_idx];

    let clamped_bwd_x = clamp_coord(x as i32 - q_x, width);
    let clamped_bwd_y = clamp_coord(y as i32 - q_y, height);
    let weight_bwd = weights_bwd[(clamped_bwd_y * width + clamped_bwd_x) as usize];

    accumulate_pair(
        input, accum, weight_sum, max_weight, x, y, q_x, q_y, frame_fwd, frame_bwd, weight_fwd, weight_bwd,
        width, height,
    );
}

/// Normalise the accumulated sums into the denoised output:
///     `out = (original × m + acc) / (m + weight_sum)`  where  `m = wref × max_weight`.
/// When the denominator is near zero (no usable matches across the
/// search window) the original pixel value is preserved unchanged.
#[cube(launch)]
pub fn nlm_finish(
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    accum: &Array<Line<f32>>,
    weight_sum: &Array<f32>,
    max_weight: &Array<f32>,
    center_frame: u32,
    wref: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let pixel_idx = (y * width + x) as usize;
    let frame_idx = ((center_frame * height + y) * width + x) as usize;

    let m = wref * max_weight[pixel_idx];
    let denominator = m + weight_sum[pixel_idx];

    let original = input[frame_idx];
    let accumulated = accum[pixel_idx];

    // `Line::empty` zero-initialises so any padding lanes (vec3 → vec4)
    // stay 0 regardless of which branch runs below.
    let mut out = Line::empty(input.line_size());

    if denominator > 1e-30f32 {
        let inv_denominator = 1.0f32 / denominator;
        #[unroll]
        for c in 0..channels {
            out[c as usize] = (original[c as usize] * m + accumulated[c as usize]) * inv_denominator;
        }
    } else {
        #[unroll]
        for c in 0..channels {
            out[c as usize] = original[c as usize];
        }
    }

    output[pixel_idx] = out;
}
