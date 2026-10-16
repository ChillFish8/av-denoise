use cubecl::prelude::*;

use super::helpers::{read_clamped_line, read_line};

/// The distance tile element that halo slot `halo` writes.
///
/// The halo is the tile outside the block's own pixels. Its slots run in row-major order, so the
/// top rows come first, then the left and right columns beside the block, then the bottom rows.
///
/// The mapping is branch free, because its result indexes shared memory inside unrolled loops.
#[cube]
pub(crate) fn halo_slot_index(
    halo: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
    #[comptime] patch_radius: u32,
) -> u32 {
    let tile_width = comptime!(block_x + 2 * patch_radius);
    let top_count = comptime!(patch_radius * (block_x + 2 * patch_radius));
    // A zero patch radius has no halo, and the floor of 1 keeps its divisions defined.
    let side_width = comptime!(u32::max(2 * patch_radius, 1));
    let side_count = comptime!(block_y * 2 * patch_radius);
    let bottom_start = comptime!((block_y + patch_radius) * (block_x + 2 * patch_radius));

    let side = u32::max(halo, top_count) - top_count;
    let side_row = side / side_width + patch_radius;
    let side_column = side % side_width;
    let shifted_column = select(side_column < patch_radius, side_column, side_column + block_x);
    let side_index = side_row * tile_width + shifted_column;
    let bottom = u32::max(side, side_count) - side_count;
    let bottom_index = bottom_start + bottom;

    let in_top = halo < top_count;
    let in_side = side < side_count;
    let below_top_index = select(in_side, side_index, bottom_index);

    select(in_top, halo, below_top_index)
}

/// The host version of [halo_slot_index](crate::nlmeans::kernels::fill::halo_slot_index).
#[cfg(test)]
pub(crate) fn halo_slot_index_host(halo: u32, block_x: u32, block_y: u32, patch_radius: u32) -> u32 {
    let tile_width = block_x + 2 * patch_radius;
    let top_count = patch_radius * tile_width;
    let side_width = 2 * patch_radius;
    let side_count = block_y * side_width;

    if halo < top_count {
        return halo;
    }

    let side = halo - top_count;
    if side < side_count {
        let side_row = side / side_width + patch_radius;
        let side_column = side % side_width;
        let column = if side_column < patch_radius {
            side_column
        } else {
            side_column + block_x
        };
        return side_row * tile_width + column;
    }

    let bottom = side - side_count;
    (block_y + patch_radius) * tile_width + bottom
}

/// Whether the expanded centre region, and so every read of the block, lies inside the frame.
#[cube]
pub(crate) fn expanded_interior(
    expanded_x0: i32,
    expanded_y0: i32,
    #[comptime] expanded_width: u32,
    #[comptime] expanded_height: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> bool {
    let left_inside = expanded_x0 >= 0;
    let top_inside = expanded_y0 >= 0;
    let right_inside = expanded_x0 + expanded_width as i32 <= width as i32;
    let bottom_inside = expanded_y0 + expanded_height as i32 <= height as i32;

    left_inside && top_inside && right_inside && bottom_inside
}

/// Caches the expanded centre region of `frame` in `tile`, row-major over `expanded_width` columns.
///
/// When `interior` holds the reads skip clamping. Every thread of the cube must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, a block coordinate or a comptime shape the kernel binds"
)]
pub(crate) fn fill_center_tile<N: Size>(
    frame_data: &Array<Vector<f32, N>>,
    tile: &mut SharedMemory<Vector<f32, N>>,
    expanded_x0: i32,
    expanded_y0: i32,
    interior: bool,
    frame: u32,
    thread_id: u32,
    threads: u32,
    #[comptime] expanded_width: u32,
    #[comptime] expanded_elems: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    if interior {
        let mut idx = thread_id;
        while idx < expanded_elems {
            let src_x = expanded_x0 + (idx % expanded_width) as i32;
            let src_y = expanded_y0 + (idx / expanded_width) as i32;
            tile[idx as usize] = read_line(frame_data, src_x as u32, src_y as u32, frame, width, height);
            idx += threads;
        }
    } else {
        let mut idx = thread_id;
        while idx < expanded_elems {
            let src_x = expanded_x0 + (idx % expanded_width) as i32;
            let src_y = expanded_y0 + (idx / expanded_width) as i32;
            tile[idx as usize] = read_clamped_line(frame_data, src_x, src_y, frame, width, height);
            idx += threads;
        }
    }
}
