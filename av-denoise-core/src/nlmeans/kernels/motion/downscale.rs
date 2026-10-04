use cubecl::prelude::*;
use cubecl::terminate;

/// Builds the next pyramid level by averaging each 2x2 group of luma pixels.
///
/// `src_frame` and `dst_frame` are slots in the per-level frame rings. An odd last row or column
/// repeats its edge pixel.
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
