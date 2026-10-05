use cubecl::prelude::*;
use cubecl::terminate;

/// Shifts a neighbour frame into line with the centre frame using a per-block motion field.
///
/// Aligning the neighbours stops temporal averaging from blurring moving edges. Each pixel takes
/// the vector of block `pixel / step`, clamped to the grid, and reads its source pixel clamped at
/// the borders. Overlapping blocks are not blended.
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

    let block_col = (x / step).min(blocks_x - 1);
    let block_row = (y / step).min(blocks_y - 1);

    let mv_idx = ((block_row * blocks_x + block_col) * 2) as usize;
    let mvx = mv_field[mv_idx];
    let mvy = mv_field[mv_idx + 1];

    let src_x = clamp_pos(x as i32 + mvx, width as i32);
    let src_y = clamp_pos(y as i32 + mvy, height as i32);

    let src_idx = (src_frame * height + src_y as u32) * width + src_x as u32;
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
