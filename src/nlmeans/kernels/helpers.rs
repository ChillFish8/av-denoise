use cubecl::prelude::*;

#[cube]
pub(super) fn clamp_coord(value: i32, #[comptime] limit: u32) -> u32 {
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
pub(super) fn read_clamped_line<N: Size>(
    buf: &Array<Vector<f32, N>>,
    x: i32,
    y: i32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<f32, N> {
    let clamped_x = clamp_coord(x, width);
    let clamped_y = clamp_coord(y, height);
    let idx = (frame * height + clamped_y) * width + clamped_x;
    buf[idx as usize]
}

/// Unchecked variant of `read_clamped_line`. The caller guarantees
/// `x ∈ [0, width)` and `y ∈ [0, height)`.
#[cube]
pub(super) fn read_line<N: Size>(
    buf: &Array<Vector<f32, N>>,
    x: u32,
    y: u32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<f32, N> {
    let idx = (frame * height + y) * width + x;
    buf[idx as usize]
}

/// Sum of squared lane differences over a vector. The loop is fully
/// unrolled at compile time because `channels` is comptime.
#[cube]
pub(super) fn line_sum_sq<N: Size>(diff: Vector<f32, N>, #[comptime] channels: u32) -> f32 {
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
pub(super) fn channel_scale(#[comptime] channels: u32) -> f32 {
    let mut scale = 1.0f32;
    if channels == 1 {
        scale = 3.0f32;
    } else if channels == 2 {
        scale = 1.5f32;
    }
    scale
}

/// Welsch weight for a box-summed patch distance. `noise_offset` is
/// the expected distance between two noisy copies of identical
/// content. Subtracting it stops matches being penalised for the
/// noise they carry. An offset of `0.0` reproduces the plain weight
/// exactly because the box sum is never negative.
#[cube]
pub(super) fn welsch_weight(sum: f32, h2_inv_norm: f32, noise_offset: f32) -> f32 {
    f32::exp(-f32::max(sum - noise_offset, 0.0) * h2_inv_norm)
}

/// Add the `+q` and `−q` contributions at thread `(global_x, global_y)`.
/// The forward neighbour lives at `(global + q, frame_fwd)` weighted by
/// `weight_fwd`; the backward neighbour at `(global − q, frame_bwd)`
/// weighted by `weight_bwd`. A single per-thread interior check covers
/// both reads, with a clamped fallback for the border.
#[cube]
pub(super) fn accumulate_pair<N: Size>(
    input: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
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

    let line_w_fwd = Vector::<f32, N>::empty().fill(weight_fwd);
    let line_w_bwd = Vector::<f32, N>::empty().fill(weight_bwd);
    let cur = accum[pixel_idx];
    accum[pixel_idx] = cur + fwd_pixel * line_w_fwd + bwd_pixel * line_w_bwd;

    weight_sum[pixel_idx] += weight_fwd + weight_bwd;
}
