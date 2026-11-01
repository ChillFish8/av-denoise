use cubecl::prelude::*;
use cubecl::terminate;

/// Divides a vector component by `1 << shift`, rounding half away from zero.
#[cube]
fn scale_down(value: i32, #[comptime] shift: u32) -> i32 {
    let mut scaled = value;
    if comptime!(shift > 0) {
        let factor = comptime!(1i32 << shift);
        let half = comptime!(1i32 << (shift - 1));
        let magnitude = if value < 0 { -value } else { value };
        let rounded = (magnitude + half) / factor;
        scaled = if value < 0 { -rounded } else { rounded };
    }
    scaled
}

/// Picks the lower median of the `live` entries of `values`.
///
/// Each entry is ranked by counting how many live entries sit below it, so no sorting is needed.
#[cube]
fn lower_median(values: &Array<i32>, live: &Array<u32>, count: u32, #[comptime] length: u32) -> i32 {
    let middle = (count - 1u32) / 2u32;
    let mut median = values[0];

    #[unroll]
    for i in 0..length {
        let mut below = 0u32;
        let mut equal = 0u32;
        #[unroll]
        for j in 0..length {
            let is_below = u32::cast_from(values[j as usize] < values[i as usize]);
            let is_equal = u32::cast_from(values[j as usize] == values[i as usize]);
            below += live[j as usize] * is_below;
            equal += live[j as usize] * is_equal;
        }

        let is_median = live[i as usize] == 1u32 && below <= middle && middle < below + equal;
        if is_median {
            median = values[i as usize];
        }
    }

    median
}

/// Maps luma motion fields onto the chroma block grid.
///
/// One thread builds one chroma block. It takes the luma blocks inside the area the chroma block
/// covers on the luma plane, clamped to the luma grid, and keeps the per-component lower median of
/// their vectors. The median is then divided by the subsampling, rounding half away from zero.
/// `covers_x` and `covers_y` count the luma blocks along each axis, so 4:4:4 reads one block.
///
/// Fields are `(x, y)` pairs of `i32`, laid out block by block. The launch has one cube layer per
/// field, and `luma_stride` and `chroma_stride` are the `i32` counts between fields.
#[cube(launch_unchecked)]
pub fn mc_map_luma_fields(
    luma_field: &[i32],
    chroma_field: &mut [i32],
    #[comptime] luma_blocks_x: u32,
    #[comptime] luma_blocks_y: u32,
    #[comptime] luma_step: u32,
    #[comptime] luma_stride: u32,
    #[comptime] chroma_blocks_x: u32,
    #[comptime] chroma_blocks_y: u32,
    #[comptime] chroma_step: u32,
    #[comptime] chroma_stride: u32,
    #[comptime] shift_x: u32,
    #[comptime] shift_y: u32,
    #[comptime] covers_x: u32,
    #[comptime] covers_y: u32,
) {
    let block_col = ABSOLUTE_POS_X;
    let block_row = ABSOLUTE_POS_Y;
    let field = ABSOLUTE_POS_Z;

    if block_col >= chroma_blocks_x || block_row >= chroma_blocks_y {
        terminate!();
    }

    let candidates = comptime!(covers_x * covers_y);
    let origin_x = (block_col * chroma_step) << shift_x;
    let origin_y = (block_row * chroma_step) << shift_y;
    let reach_x = comptime!(covers_x - 1);
    let reach_y = comptime!(covers_y - 1);
    let start_col = origin_x / luma_step;
    let start_row = origin_y / luma_step;
    let last_col = u32::min(start_col + reach_x, luma_blocks_x - 1);
    let last_row = u32::min(start_row + reach_y, luma_blocks_y - 1);
    let first_col = u32::min(start_col, last_col);
    let first_row = u32::min(start_row, last_row);

    let mut vectors_x = Array::<i32>::new(candidates as usize);
    let mut vectors_y = Array::<i32>::new(candidates as usize);
    let mut live = Array::<u32>::new(candidates as usize);
    let mut count = 0u32;

    #[unroll]
    for row_offset in 0..covers_y {
        #[unroll]
        for col_offset in 0..covers_x {
            let index = comptime!(row_offset * covers_x + col_offset) as usize;
            let col = first_col + col_offset;
            let row = first_row + row_offset;
            let inside = col <= last_col && row <= last_row;
            let block = u32::min(row, last_row) * luma_blocks_x + u32::min(col, last_col);
            let source = (field * luma_stride + block * 2) as usize;

            vectors_x[index] = luma_field[source];
            vectors_y[index] = luma_field[source + 1];
            live[index] = u32::cast_from(inside);
            count += u32::cast_from(inside);
        }
    }

    let median_x = lower_median(&vectors_x, &live, count, candidates);
    let median_y = lower_median(&vectors_y, &live, count, candidates);

    let block = block_row * chroma_blocks_x + block_col;
    let target = (field * chroma_stride + block * 2) as usize;
    chroma_field[target] = scale_down(median_x, shift_x);
    chroma_field[target + 1] = scale_down(median_y, shift_y);
}
