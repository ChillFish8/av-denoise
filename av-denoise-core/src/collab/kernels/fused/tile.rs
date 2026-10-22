use cubecl::prelude::*;

use super::distance::{pair_squared_difference, pair_sum_total};
use super::shape::SearchShape;
use crate::collab::PATCH_SIZE;
use crate::engine::kernels::read_line;

/// Loads the cube's centre tile, `tile_width` by `tile_height` pixels from `(origin_x, origin_y)`.
///
/// Pixels are row-major with every stored lane interleaved, each read clamped to the frame. With
/// `f16_search` the tile copies `search_ring`, otherwise it copies `ring`. Every lane of the cube
/// loads a strided share, and the caller syncs the cube before any lane reads the tile.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the load needs"
)]
pub(crate) fn load_centre_tile<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &mut SharedMemory<S>,
    origin_x: u32,
    origin_y: u32,
    centre_slot: u32,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] lanes: u32,
    #[comptime] shape: SearchShape,
) {
    let f16_search = comptime!(shape.f16_search);
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let stored_ch = comptime!(shape.stored_ch);
    let pixels = comptime!(tile_width * tile_height);
    let last_x = comptime!(width - 1);
    let last_y = comptime!(height - 1);
    let mut index = UNIT_POS_X;

    while index < pixels {
        let tile_x = index % tile_width;
        let tile_y = index / tile_width;
        let source_x = u32::min(origin_x + tile_x, last_x);
        let source_y = u32::min(origin_y + tile_y, last_y);
        let first_lane = index * stored_ch;

        if comptime!(f16_search) {
            let line = read_line(search_ring, source_x, source_y, centre_slot, width, height);
            #[unroll]
            for c in 0..stored_ch {
                tile[(first_lane + c) as usize] = line[c as usize];
            }
        } else {
            let line = read_line(ring, source_x, source_y, centre_slot, width, height);
            #[unroll]
            for c in 0..stored_ch {
                tile[(first_lane + c) as usize] = S::cast_from(line[c as usize]);
            }
        }

        index += lanes;
    }
}

/// One lane's share of the distance to the candidate whose column `column` and top row `row` sit
/// in the tile.
///
/// It reads the same values and runs the same operations in the same order as
/// [candidate_partial](crate::collab::kernels::fused::distance::candidate_partial), so the two give
/// bit identical partials.
#[cube]
pub(crate) fn tile_partial<S: Float>(
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    column: u32,
    row: u32,
    #[comptime] tile_width: u32,
    #[comptime] shape: SearchShape,
) -> f32 {
    let f16_search = comptime!(shape.f16_search);
    let channels = comptime!(shape.channels);
    let stored_ch = comptime!(shape.stored_ch);
    let mut partial = 0.0f32;

    if comptime!(f16_search && stored_ch == 1) {
        let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));

        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let top = tile[((row + r) * tile_width + column) as usize];
            let bottom_row = row + r + comptime!(PATCH_SIZE / 2);
            let bottom = tile[(bottom_row * tile_width + column) as usize];
            let squared = pair_squared_difference(reference, top, bottom, r);
            sum += squared;
        }

        partial = pair_sum_total(sum);
    } else if comptime!(f16_search) {
        let pairs_per_row = comptime!(stored_ch / 2);
        let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));

        #[unroll]
        for r in 0..PATCH_SIZE {
            let first_lane = ((row + r) * tile_width + column) * stored_ch;
            #[unroll]
            for p in 0..pairs_per_row {
                let first = tile[(first_lane + comptime!(2 * p)) as usize];
                let second = tile[(first_lane + comptime!(2 * p + 1)) as usize];
                let pair = comptime!(r * pairs_per_row + p);
                let squared = pair_squared_difference(reference, first, second, pair);
                sum += squared;
            }
        }

        partial = pair_sum_total(sum);
    } else {
        #[unroll]
        for r in 0..PATCH_SIZE {
            let first_lane = ((row + r) * tile_width + column) * stored_ch;
            #[unroll]
            for c in 0..channels {
                let pixel = f32::cast_from(tile[(first_lane + c) as usize]);
                let diff = current[(r * channels + c) as usize] - pixel;
                partial += diff * diff;
            }
        }
    }

    partial
}
