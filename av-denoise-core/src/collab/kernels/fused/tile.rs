use cubecl::prelude::*;

use crate::collab::{PATCH_SIZE, STEP};
use crate::nlmeans::kernels::helpers::read_line;

/// Columns in the cube's centre search tile, the union of every group's search window.
pub(crate) fn tile_width(groups: u32, spatial_radius: u32) -> u32 {
    (groups - 1) * STEP + 2 * spatial_radius + PATCH_SIZE
}

/// Rows in the cube's centre search tile.
pub(crate) fn tile_height(spatial_radius: u32) -> u32 {
    2 * spatial_radius + PATCH_SIZE
}

/// Copies the centre frame's search window for every group of the cube into `tile`.
///
/// The tile holds `stored_ch` values per pixel, row-major from `(tile_x, tile_y)`. Pixels past the
/// frame edge clamp to it, and no candidate ever reads them. It ends in a `sync_cube()` barrier, so
/// every lane of the cube must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the copy needs"
)]
pub(crate) fn load_search_tile<S: Float, N: Size>(
    search_ring: &Array<Vector<S, N>>,
    tile: &mut SharedMemory<S>,
    tile_x: u32,
    tile_y: u32,
    slot: u32,
    #[comptime] tile_w: u32,
    #[comptime] tile_h: u32,
    #[comptime] cube_threads: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
) {
    let count = comptime!(tile_w * tile_h);
    let last_x = comptime!(width - 1);
    let last_y = comptime!(height - 1);

    let mut index = UNIT_POS_X;
    while index < count {
        let column = index % tile_w;
        let row = index / tile_w;
        let x = u32::min(tile_x + column, last_x);
        let y = u32::min(tile_y + row, last_y);
        let pixel = read_line(search_ring, x, y, slot, width, height);

        #[unroll]
        for c in 0..stored_ch {
            tile[(index * stored_ch + c) as usize] = pixel[c as usize];
        }

        index += cube_threads;
    }

    sync_cube();
}

/// One lane's squared difference against the candidate column at tile position `(column, row)`.
///
/// It repeats the arithmetic of the global-memory search in the same order, so both give the same
/// distance bit for bit.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn tile_partial<S: Float>(
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    column: u32,
    row: u32,
    #[comptime] f16_search: bool,
    #[comptime] tile_w: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let mut partial = 0.0f32;

    if comptime!(f16_search && stored_ch == 1) {
        let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));

        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let top = (row + r) * tile_w + column;
            let bottom = (row + r + comptime!(PATCH_SIZE / 2)) * tile_w + column;
            let mut pixel = Vector::<S, Const<2>>::empty();
            pixel[0] = tile[top as usize];
            pixel[1] = tile[bottom as usize];
            let diff = reference[r as usize] - pixel;
            sum += diff * diff;
        }

        let low = f32::cast_from(sum[0]);
        let high = f32::cast_from(sum[1]);
        partial = low + high;
    } else if comptime!(f16_search) {
        let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));
        let pairs_per_row = comptime!(stored_ch / 2);

        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = ((row + r) * tile_w + column) * stored_ch;

            #[unroll]
            for p in 0..pairs_per_row {
                let mut pixel = Vector::<S, Const<2>>::empty();
                pixel[0] = tile[(line + comptime!(2 * p)) as usize];
                pixel[1] = tile[(line + comptime!(2 * p + 1)) as usize];
                let diff = reference[comptime!(r * pairs_per_row + p) as usize] - pixel;
                sum += diff * diff;
            }
        }

        let low = f32::cast_from(sum[0]);
        let high = f32::cast_from(sum[1]);
        partial = low + high;
    } else {
        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = ((row + r) * tile_w + column) * stored_ch;

            #[unroll]
            for c in 0..channels {
                let pixel = f32::cast_from(tile[(line + c) as usize]);
                let diff = current[(r * channels + c) as usize] - pixel;
                partial += diff * diff;
            }
        }
    }

    partial
}
