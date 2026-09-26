use cubecl::prelude::*;
use cubecl::terminate;

use crate::nlmeans::kernels::helpers::{read_clamped_line, read_line};

/// Writes one ring slot's four phase planes.
///
/// Plane 0 copies the frame. Plane 1 holds each pixel's sample half a
/// pixel to the right, plane 2 half a pixel down, and plane 3 both. The
/// half-pel filter reads the 8 pixels around the gap, clamped at the
/// frame's edges. Plane 3 filters plane 1's values vertically without
/// rounding them first.
///
/// Plane `p` of `slot` is frame `slot * 4 + p` of `phase_ring`.
#[cube(launch_unchecked)]
pub fn nl4d_phase_planes<N: Size>(
    ring: &Array<Vector<f32, N>>,
    phase_ring: &mut Array<Vector<f32, N>>,
    taps: &Array<f32>,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let column = x as i32;
    let row = y as i32;

    let whole = read_line(ring, x, y, slot, width, height);
    let horizontal = half_row(ring, taps, column, row, slot, width, height);

    let mut vertical = Vector::<f32, N>::empty().fill(0.0f32);
    let mut diagonal = Vector::<f32, N>::empty().fill(0.0f32);
    #[unroll]
    for k in 0..8u32 {
        let tap = Vector::<f32, N>::empty().fill(taps[k as usize]);
        let tap_row = row + k as i32 - 3;
        let above_or_below = read_clamped_line(ring, column, tap_row, slot, width, height);
        let row_half = half_row(ring, taps, column, tap_row, slot, width, height);
        vertical += above_or_below * tap;
        diagonal += row_half * tap;
    }

    let first_plane = slot * 4u32;
    let pixel = y * width + x;
    let plane_len = width * height;
    phase_ring[(first_plane * plane_len + pixel) as usize] = whole;
    phase_ring[((first_plane + 1u32) * plane_len + pixel) as usize] = horizontal;
    phase_ring[((first_plane + 2u32) * plane_len + pixel) as usize] = vertical;
    phase_ring[((first_plane + 3u32) * plane_len + pixel) as usize] = diagonal;
}

/// The half-pel sample between `x` and `x + 1` on row `y`, with both
/// axes clamped to the frame.
#[cube]
fn half_row<N: Size>(
    ring: &Array<Vector<f32, N>>,
    taps: &Array<f32>,
    x: i32,
    y: i32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<f32, N> {
    let mut sum = Vector::<f32, N>::empty().fill(0.0f32);
    #[unroll]
    for k in 0..8u32 {
        let tap = Vector::<f32, N>::empty().fill(taps[k as usize]);
        let pixel = read_clamped_line(ring, x + k as i32 - 3, y, slot, width, height);
        sum += pixel * tap;
    }
    sum
}
