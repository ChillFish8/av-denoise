use cubecl::prelude::*;
use cubecl::terminate;

/// Warp a neighbour frame into alignment with the centre frame using
/// the per-block MV field produced by the analyse pass. Each output
/// pixel resolves which block it belongs to, looks up that block's MV,
/// and reads the source pixel from the offset position (clamped at
/// borders). Padding lanes are copied through unchanged.
///
/// Overlap handling: pixels inside a block's interior take the block's
/// MV directly. Pixels in the overlap band of two adjacent blocks
/// (relevant when `step < blksize`) use the closest block's MV (no
/// blending in v1). This "winner-block" rule is a v1 simplification
/// of MVTools' raised-cosine blend.
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

/// Trivial centre-frame copy used to populate the `compensated_*` slot
/// for the centre so temporal kernels can read uniformly.
#[cube(launch_unchecked)]
pub fn nlm_mc_copy_frame<N: Size>(
    src: &Array<Vector<f32, N>>,
    dst: &mut Array<Vector<f32, N>>,
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
