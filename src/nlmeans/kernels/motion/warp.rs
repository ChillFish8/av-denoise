use cubecl::prelude::*;
use cubecl::terminate;

/// Shifts a neighbour frame into line with the centre frame, using the
/// per-block motion field the analyse pass produced.
///
/// Each output pixel works out which block it belongs to, reads that
/// block's motion vector, and takes its source pixel from the offset
/// position, clamped at the borders. Padding lanes are copied straight
/// through.
///
/// # Overlapping blocks
///
/// A pixel inside a block's interior takes that block's vector
/// directly.
///
/// A pixel in the band where two adjacent blocks overlap, which happens
/// when the step is smaller than the block size, takes the vector of
/// whichever block is closest. Nothing is blended between them.
///
/// That winner-takes-all rule is a simplification of MVTools'
/// raised-cosine blend.
#[cube(launch_unchecked)]
pub fn nlm_mc_warp<N: Size>(
    src: &Array<Vector<f32, N>>,
    dst: &mut Array<Vector<f32, N>>,
    mv_field: &Array<i32>,
    src_frame: u32,
    dst_frame: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let bx = (x / step).min(blocks_x - 1);
    let by = (y / step).min(blocks_y - 1);

    let mv_idx = ((by * blocks_x + bx) * 2) as usize;
    let mvx = mv_field[mv_idx];
    let mvy = mv_field[mv_idx + 1];

    let sx = clamp_pos(x as i32 + mvx, width as i32);
    let sy = clamp_pos(y as i32 + mvy, height as i32);

    let src_idx = (src_frame * height + sy as u32) * width + sx as u32;
    let dst_idx = (dst_frame * height + y) * width + x;
    dst[dst_idx as usize] = src[src_idx as usize];
}

#[cube]
fn clamp_pos(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }
    result
}
