use cubecl::prelude::*;

/// Every threshold is left as it is.
pub const STRENGTH_MAP_OFF: u32 = 0;
/// The map scales channel 0's noise-curve ratio.
pub const STRENGTH_MAP_LUMA: u32 = 1;
/// The map scales every channel's threshold, for a denoiser with no curve.
pub const STRENGTH_MAP_ALL: u32 = 2;

/// The side of one map quarter, in pixels.
const MAP_QUARTER: u32 = 8;

/// The mean of the four map quarters a reference patch at `(rx, ry)` overlaps.
///
/// A patch on the 8-pixel grid reads one quarter four times. The sum always runs in the same order,
/// so a host copy of it reproduces the result exactly.
#[cube]
pub(crate) fn strength_map_scale(
    map: &Array<f32>,
    rx: u32,
    ry: u32,
    #[comptime] map_cols: u32,
    #[comptime] map_rows: u32,
) -> f32 {
    let col_lo = rx / MAP_QUARTER;
    let col_ceil = rx.div_ceil(MAP_QUARTER);
    let col_hi = u32::min(col_ceil, comptime!(map_cols - 1));
    let row_lo = ry / MAP_QUARTER;
    let row_ceil = ry.div_ceil(MAP_QUARTER);
    let row_hi = u32::min(row_ceil, comptime!(map_rows - 1));

    let top_left = map[(row_lo * map_cols + col_lo) as usize];
    let top_right = map[(row_lo * map_cols + col_hi) as usize];
    let bottom_left = map[(row_hi * map_cols + col_lo) as usize];
    let bottom_right = map[(row_hi * map_cols + col_hi) as usize];
    let sum = top_left + top_right + bottom_left + bottom_right;
    sum / 4.0f32
}
