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

/// Reads the pixel at `(x, y)` in `frame`, clamped to the image edges on
/// both axes.
///
/// The frame index is taken on trust, because callers always pass a
/// physical slot that holds loaded data.
#[cube]
pub(crate) fn read_clamped_line<N: Size>(
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

/// The unchecked version of `read_clamped_line`.
///
/// The caller promises that `x` is inside `[0, width)` and `y` is inside
/// `[0, height)`.
#[cube]
pub(crate) fn read_line<N: Size>(
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

/// Sums the squared differences across a vector's lanes.
///
/// The loop unrolls fully at compile time, because `channels` is known
/// then.
#[cube]
pub(crate) fn line_sum_sq<N: Size>(diff: Vector<f32, N>, #[comptime] channels: u32) -> f32 {
    let mut sum = 0.0f32;
    #[unroll]
    for c in 0..channels {
        sum += diff[c as usize] * diff[c as usize];
    }
    sum
}

/// The per-channel distance scale, which is 3 for luma, 1.5 for chroma,
/// and 1 for full YUV.
///
/// Scaling this way lets all three channel modes share one
/// `h2_inv_norm`.
#[cube]
pub(crate) fn channel_scale(#[comptime] channels: u32) -> f32 {
    let mut scale = 1.0f32;
    if channels == 1 {
        scale = 3.0f32;
    } else if channels == 2 {
        scale = 1.5f32;
    }
    scale
}

/// The host mirror of [`channel_scale`].
pub fn channel_scale_host(channels: u32) -> f32 {
    match channels {
        1 => 3.0,
        2 => 1.5,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    use super::{channel_scale, channel_scale_host};

    type R = WgpuRuntime;

    fn make_client() -> ComputeClient<R> {
        let device = <R as Runtime>::Device::default();
        R::client(&device)
    }

    /// Runs [`channel_scale`] on the GPU for all three channel counts it
    /// ever runs with, so the host mirror is checked against the code
    /// the filter actually calls rather than against itself.
    #[cube(launch_unchecked)]
    fn channel_scale_probe(out: &mut Array<f32>) {
        out[0] = channel_scale(1u32);
        out[1] = channel_scale(2u32);
        out[2] = channel_scale(3u32);
    }

    #[test]
    fn channel_scale_host_matches_the_kernel_mirror() {
        let client = make_client();
        let out_buf = client.empty(3 * size_of::<f32>());

        unsafe {
            channel_scale_probe::launch_unchecked::<R>(
                &client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(out_buf.clone(), 3),
            );
        }

        let bytes = client.read_one(out_buf).expect("channel_scale readback failed");
        let got = f32::from_bytes(&bytes)[..3].to_vec();

        for (idx, channels) in [1u32, 2, 3].into_iter().enumerate() {
            assert_eq!(
                got[idx],
                channel_scale_host(channels),
                "channels={channels}: host mirror disagrees with the GPU kernel"
            );
        }
    }
}

/// The Welsch weight for a box-summed patch distance.
///
/// `noise_offset` is the distance two noisy copies of the same content
/// are expected to show. Subtracting it stops a good match being
/// penalised for the noise it carries.
///
/// An offset of 0.0 gives exactly the plain weight, because the box sum
/// is never negative.
#[cube]
pub(super) fn welsch_weight(sum: f32, h2_inv_norm: f32, noise_offset: f32) -> f32 {
    f32::exp(-f32::max(sum - noise_offset, 0.0) * h2_inv_norm)
}

/// Adds the forward and backward neighbour contributions at the thread's
/// pixel.
///
/// The forward neighbour sits at `(global + q, frame_fwd)` with
/// `weight_fwd`, and the backward one at `(global - q, frame_bwd)` with
/// `weight_bwd`.
///
/// One interior check per thread covers both reads, falling back to
/// clamped reads at the border.
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
