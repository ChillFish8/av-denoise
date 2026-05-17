use cubecl::prelude::*;
use cubecl::terminate;

/// 2x2 box downsample, luma-only. Reads from a single source pyramid
/// level (full resolution or a previously-downscaled level) and writes
/// a half-resolution copy into the destination level slot.
///
/// `src_frame` and `dst_frame` are slot indices inside the per-level
/// frame rings; `src_*` / `dst_*` dimensions are the half-resolution
/// destination dimensions plus the source full-resolution dimensions.
#[cube(launch_unchecked)]
pub fn nlm_mc_downscale(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_frame: u32,
    dst_frame: u32,
    #[comptime] src_width: u32,
    #[comptime] src_height: u32,
    #[comptime] dst_width: u32,
    #[comptime] dst_height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= dst_width || y >= dst_height {
        terminate!();
    }

    let sx = x * 2;
    let sy = y * 2;
    let sx1 = if sx + 1 < src_width { sx + 1 } else { sx };
    let sy1 = if sy + 1 < src_height { sy + 1 } else { sy };

    let src_base = src_frame * src_width * src_height;
    let s00 = src[(src_base + sy * src_width + sx) as usize];
    let s10 = src[(src_base + sy * src_width + sx1) as usize];
    let s01 = src[(src_base + sy1 * src_width + sx) as usize];
    let s11 = src[(src_base + sy1 * src_width + sx1) as usize];

    let avg = (s00 + s10 + s01 + s11) * 0.25f32;
    dst[(dst_frame * dst_width * dst_height + y * dst_width + x) as usize] = avg;
}

/// Extract the luma (channel 0) plane from a packed `Vector<f32, N>`
/// input frame into a flat `Array<f32>` pyramid level-0 slot. Used so
/// the analyse / pyramid kernels can work on luma-only data without
/// caring about the full ChannelMode layout.
#[cube(launch_unchecked)]
pub fn nlm_mc_extract_luma<N: Size>(
    src: &Array<Vector<f32, N>>,
    dst: &mut Array<f32>,
    src_frame: u32,
    dst_frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let src_idx = (src_frame * height + y) * width + x;
    let pixel = src[src_idx as usize];

    let dst_idx = (dst_frame * height + y) * width + x;
    dst[dst_idx as usize] = pixel[0usize];
}
