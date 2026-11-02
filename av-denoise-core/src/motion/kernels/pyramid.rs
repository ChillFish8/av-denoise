use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::clamp_i32;

/// Builds the next pyramid level by averaging each 2x2 group of luma pixels.
///
/// `src_frame` and `dst_frame` are slots in the per-level frame rings. An odd last row or column
/// repeats its edge pixel.
#[cube(launch_unchecked)]
pub fn mc_downscale(
    src: &[f32],
    dst: &mut [f32],
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

    let src_x = x * 2;
    let src_y = y * 2;
    let src_x1 = if src_x + 1 < src_width { src_x + 1 } else { src_x };
    let src_y1 = if src_y + 1 < src_height { src_y + 1 } else { src_y };

    let src_base = src_frame * src_width * src_height;
    let top_left = src[(src_base + src_y * src_width + src_x) as usize];
    let top_right = src[(src_base + src_y * src_width + src_x1) as usize];
    let bottom_left = src[(src_base + src_y1 * src_width + src_x) as usize];
    let bottom_right = src[(src_base + src_y1 * src_width + src_x1) as usize];

    let avg = (top_left + top_right + bottom_left + bottom_right) * 0.25f32;
    dst[(dst_frame * dst_width * dst_height + y * dst_width + x) as usize] = avg;
}

/// Copies lane 0 of a packed frame into a flat luma array, which becomes level 0 of the pyramid.
#[cube(launch_unchecked)]
pub fn mc_extract_luma<N: Size>(
    src: &[Vector<f32, N>],
    dst: &mut [f32],
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
    dst[dst_idx as usize] = pixel.extract(0usize);
}

/// Averages each pixel with its row neighbours up to `radius` either side, clamped to the frame.
///
/// The taps are summed left to right, so the result is the same on every run.
#[cube(launch_unchecked)]
pub fn mc_box_rows(
    src: &[f32],
    dst: &mut [f32],
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] radius: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let taps = comptime!(2 * radius + 1);
    let inverse_taps = comptime!(1.0f32 / taps as f32);
    let row_base = y * width;
    let mut total = 0.0f32;

    for tap in 0..taps {
        let source_x = x as i32 + tap as i32 - radius as i32;
        let clamped_x = clamp_i32(source_x, width as i32) as u32;
        total += src[(row_base + clamped_x) as usize];
    }

    dst[(row_base + x) as usize] = total * inverse_taps;
}

/// Writes `src` minus its local mean, the box `radius` either side on both axes, to `dst`.
///
/// `rows` holds `src` already averaged along each row by [mc_box_rows]. Taps are summed top to
/// bottom and clamped to the frame.
#[cube(launch_unchecked)]
pub fn mc_high_pass_cols(
    src: &[f32],
    rows: &[f32],
    dst: &mut [f32],
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] radius: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let taps = comptime!(2 * radius + 1);
    let inverse_taps = comptime!(1.0f32 / taps as f32);
    let mut total = 0.0f32;

    for tap in 0..taps {
        let source_y = y as i32 + tap as i32 - radius as i32;
        let clamped_y = clamp_i32(source_y, height as i32) as u32;
        total += rows[(clamped_y * width + x) as usize];
    }

    let index = (y * width + x) as usize;
    let local_mean = total * inverse_taps;
    dst[index] = src[index] - local_mean;
}
