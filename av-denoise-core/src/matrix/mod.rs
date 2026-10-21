#[cfg(test)]
mod tests;

use cubecl::cmma;
use cubecl::prelude::*;
use half::f16;

use crate::tune::Int8Layout;

/// The side of the square tile every matrix product runs at.
pub(crate) const MATRIX_TILE: u32 = 16;

/// The values one matrix tile holds.
pub(crate) const TILE_VALUES: u32 = MATRIX_TILE * MATRIX_TILE;

/// A 16x16x16 cooperative-matrix fragment, filled before it is read.
#[cube]
pub(crate) fn fragment<E: Scalar>(
    #[comptime] ident: cmma::MatrixIdent,
    layout: cmma::MatrixLayout,
) -> cmma::Matrix<E> {
    let size = comptime!(MATRIX_TILE as usize);

    // Safety: every fragment is loaded or filled before it is read.
    unsafe { cmma::Matrix::<E>::uninitialized(ident, size, size, size, layout) }
}

/// Stores `accumulator` into `tile` from the first subgroup of the calling logical plane.
///
/// A logical plane can run as several subgroups that each computed the same accumulator, so only
/// the first one writes.
#[cube]
pub(crate) fn store_guarded<A: CubePrimitive>(
    tile: &mut SharedMemory<A>,
    start: u32,
    accumulator: &cmma::Matrix<A>,
    #[comptime] plane_width: u32,
) {
    let lane_in_plane = UNIT_POS_X % plane_width;

    if lane_in_plane < PLANE_DIM {
        let end = start + TILE_VALUES;
        let mut slice = tile.slice_mut(start as usize, end as usize);
        cmma::store(&mut slice, accumulator, MATRIX_TILE, cmma::MatrixLayout::RowMajor);
    }
}

/// Splits `code` into its high part `code >> shift` and its low part, both as f16.
#[cube]
pub(crate) fn split_code(code: u32, #[comptime] shift: u32) -> (f16, f16) {
    let low_mask = comptime!((1u32 << shift) - 1);
    let high = code >> shift;
    let low = code & low_mask;
    let high_value = f16::cast_from(high);
    let low_value = f16::cast_from(low);

    (high_value, low_value)
}

/// The exact dot product of two code patches from the f32 sums of their split parts.
///
/// A code is `(high << shift) + low`, so the dot is the high product shifted twice, the two cross
/// products shifted once, and the low product. Each part sums below 2^24, so each converts exactly.
#[cube]
pub(crate) fn combine_split_dot(high_high: f32, cross: f32, low_low: f32, #[comptime] shift: u32) -> i32 {
    let high = i32::cast_from(high_high);
    let mixed = i32::cast_from(cross);
    let low = i32::cast_from(low_low);

    combine_split_dot_i32(high, mixed, low, shift)
}

/// The exact dot product of two code patches from the i32 sums of their split parts.
///
/// A code is `(high << low_bits) + low`, so the dot is the high product shifted twice, the two
/// cross products shifted once, and the low product.
#[cube]
pub(crate) fn combine_split_dot_i32(
    high_high: i32,
    cross: i32,
    low_low: i32,
    #[comptime] low_bits: u32,
) -> i32 {
    let high_shift = comptime!((2 * low_bits) as i32);
    let cross_shift = comptime!(low_bits as i32);

    (high_high << high_shift) + (cross << cross_shift) + low_low
}

/// The part of `code` an int8 operand holds under `layout`.
///
/// The whole code when the layout does not split, otherwise its high part `code >> low_bits` or
/// its low part.
#[cube]
pub(crate) fn int8_part<E: Int>(code: u32, #[comptime] layout: Int8Layout, #[comptime] high: bool) -> E {
    let low_bits = comptime!(layout.low_bits);
    let low_mask = comptime!((1u32 << low_bits) - 1);
    let mut part = code;

    if comptime!(low_bits > 0) {
        if comptime!(high) {
            part = code >> low_bits;
        } else {
            part = code & low_mask;
        }
    }

    E::cast_from(part)
}
