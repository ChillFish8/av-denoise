use cubecl::cmma;
use cubecl::prelude::*;

use super::scalar::pin_self_match_int;
use super::{IntCursor, IntShape, IntegerVariant};
use crate::collab::geometry::refs_along;
use crate::collab::kernels::fused::centre::SearchRect;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{UNFILLED_SSD, ranks_before_int, shift_insert8_ordered_int};
use crate::collab::{PATCH_SIZE, STEP};
use crate::matrix::{
    MATRIX_TILE,
    TILE_VALUES,
    combine_split_dot,
    combine_split_dot_i32,
    fragment,
    int8_part,
    split_code,
    store_guarded,
};
use crate::tune::{Int8Layout, Int8Matrix};

/// How the matrix centre feeds codes to the matrix units, as a comptime kernel argument.
///
/// The f16 route holds codes or their split parts as f16 and sums in f32. The int8 route holds the
/// parts `layout` gives as u8 or i8 and sums in i32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MatrixRoute {
    /// The logical plane width the products run at.
    pub(crate) plane_width: u32,
    /// Bits in a code's low part, or 0 when codes go in whole.
    pub(crate) low_bits: u32,
    /// Whether the operands are int8 parts.
    pub(crate) int8: bool,
    /// The int8 split, a placeholder on the f16 route.
    pub(crate) layout: Int8Layout,
}

impl MatrixRoute {
    /// The int8 layout the f16 route and the non-matrix variants carry, so they compile once.
    pub(crate) const PLACEHOLDER_LAYOUT: Int8Layout = Int8Layout {
        element: Int8Matrix::Unsigned,
        low_bits: 0,
    };

    /// The f16 route at `plane_width`, splitting codes at `split_shift`.
    pub(crate) fn f16(plane_width: u32, split_shift: u32) -> MatrixRoute {
        MatrixRoute {
            plane_width,
            low_bits: split_shift,
            int8: false,
            layout: MatrixRoute::PLACEHOLDER_LAYOUT,
        }
    }

    /// The int8 route at `plane_width`, splitting codes as `layout` does.
    pub(crate) fn int8(plane_width: u32, layout: Int8Layout) -> MatrixRoute {
        MatrixRoute {
            plane_width,
            low_bits: layout.low_bits,
            int8: true,
            layout,
        }
    }

    /// The parts each code splits into, 1 or 2.
    pub(crate) fn parts(self) -> u32 {
        if self.low_bits == 0 { 1 } else { 2 }
    }

    /// The accumulators one patch pair takes, 1 or 3.
    pub(crate) fn accumulators(self) -> u32 {
        if self.low_bits == 0 { 1 } else { 3 }
    }
}

/// The part of `code` an operand holds on `route`, the high part when `high` is set.
#[cube]
fn code_part<E: CubePrimitive>(code: u32, #[comptime] route: MatrixRoute, #[comptime] high: bool) -> E {
    let mut part = E::cast_from(code);

    if comptime!(route.int8) {
        part = int8_part::<E>(code, route.layout, high);
    } else if comptime!(route.low_bits > 0) {
        let (high_part, low_part) = split_code(code, route.low_bits);

        if comptime!(high) {
            part = E::cast_from(high_part);
        } else {
            part = E::cast_from(low_part);
        }
    }

    part
}

/// The exact dot product from the accumulator sums of one patch pair on `route`.
#[cube]
fn exact_dot<A: CubePrimitive>(high_high: A, cross: A, low_low: A, #[comptime] route: MatrixRoute) -> i32 {
    let low_bits = comptime!(route.low_bits);
    let mut dot = i32::cast_from(high_high);

    if comptime!(low_bits > 0) {
        if comptime!(route.int8) {
            let high_sum = i32::cast_from(high_high);
            let cross_sum = i32::cast_from(cross);
            let low_sum = i32::cast_from(low_low);
            dot = combine_split_dot_i32(high_sum, cross_sum, low_sum, low_bits);
        } else {
            let high_sum = f32::cast_from(high_high);
            let cross_sum = f32::cast_from(cross);
            let low_sum = f32::cast_from(low_low);
            dot = combine_split_dot(high_sum, cross_sum, low_sum, low_bits);
        }
    }

    dot
}

/// The top-left of the cube's reference `reference`, clamped to the last real reference as
/// `collab_search_int` clamps its groups.
#[cube]
fn cube_reference_corner(
    reference: u32,
    #[comptime] frame_refs_x: u32,
    #[comptime] variant: IntegerVariant,
    #[comptime] shape: IntShape,
) -> (u32, u32) {
    let cube_refs_x = comptime!(variant.cube_refs_x);
    let ref_rows = comptime!(variant.ref_rows);
    let frame_refs_y = comptime!(refs_along(shape.height));
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);

    let column = reference % cube_refs_x;
    let row = reference / cube_refs_x;
    let x_index = CUBE_POS_X * cube_refs_x + column;
    let y_index = CUBE_POS_Y * ref_rows + row;
    let x_clamped = u32::min(x_index, frame_refs_x - 1u32);
    let y_clamped = u32::min(y_index, frame_refs_y - 1u32);
    let ref_x = u32::min(x_clamped * STEP, max_x);
    let ref_y = u32::min(y_clamped * STEP, max_y);

    (ref_x, ref_y)
}

/// Loads the cube's union tile, `tile_width` by `tile_height` pixels from `(origin_x, origin_y)`,
/// as operand parts.
///
/// Pixels are row-major with the summed channels interleaved, and every high part comes before
/// every low part. Each read is clamped to the frame, and the clamped pixels feed only positions
/// no search reads.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the load needs"
)]
fn load_union_tile<C: Int, N: Size, E: CubePrimitive>(
    codes: &Array<Vector<C, N>>,
    tile: &mut SharedMemory<E>,
    origin_x: u32,
    origin_y: u32,
    centre_slot: u32,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] route: MatrixRoute,
    #[comptime] shape: IntShape,
) {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let pixels = comptime!(tile_width * tile_height);
    let part_values = comptime!(pixels * channels);
    let last_x = comptime!(width - 1);
    let last_y = comptime!(height - 1);
    let mut index = UNIT_POS_X;

    while index < pixels {
        let tile_x = index % tile_width;
        let tile_y = index / tile_width;
        let source_x = u32::min(origin_x + tile_x, last_x);
        let source_y = u32::min(origin_y + tile_y, last_y);
        let source = (centre_slot * height + source_y) * width + source_x;
        let pixel = codes[source as usize];
        let first_lane = index * channels;

        #[unroll]
        for c in 0..channels {
            let code = u32::cast_from(pixel[c as usize]);
            let lane = first_lane + c;
            tile[lane as usize] = code_part::<E>(code, route, true);

            if comptime!(route.low_bits > 0) {
                tile[(part_values + lane) as usize] = code_part::<E>(code, route, false);
            }
        }

        index += CUBE_DIM_X;
    }
}

/// The sum of squared codes along one patch row, 8 pixels from `(x, y)` in frame `slot`.
#[cube]
fn row_square_sum<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    x: u32,
    y: u32,
    slot: u32,
    #[comptime] shape: IntShape,
) -> i32 {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let first = (slot * height + y) * width + x;
    let mut sum = 0i32;

    #[unroll]
    for column in 0..PATCH_SIZE {
        let pixel = codes[(first + column) as usize];

        #[unroll]
        for c in 0..channels {
            let code = i32::cast_from(pixel[c as usize]);
            sum += code * code;
        }
    }

    sum
}

/// Writes the sum of squared codes of every live 8x8 patch whose corner lies in the union to
/// `norms`.
///
/// A union position is live when its patch lies inside the frame. Each lane owns union columns and
/// slides a patch-tall window down them, adding the entering row and removing the leaving one, so
/// each row sum is computed twice.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the sums need"
)]
fn union_norms<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    norms: &mut SharedMemory<i32>,
    tile_x0: u32,
    tile_y0: u32,
    centre_slot: u32,
    #[comptime] union_width: u32,
    #[comptime] union_height: u32,
    #[comptime] shape: IntShape,
) {
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);
    let live_width = u32::min(union_width, max_x - tile_x0 + 1u32);
    let live_height = u32::min(union_height, max_y - tile_y0 + 1u32);
    let live_rows = live_height + PATCH_SIZE - 1u32;
    let mut column = UNIT_POS_X;

    while column < live_width {
        let x = tile_x0 + column;
        let mut running = 0i32;
        let mut row = 0u32;

        while row < live_rows {
            let entering_y = tile_y0 + row;
            let entering = row_square_sum(codes, x, entering_y, centre_slot, shape);
            let has_leaving = row >= PATCH_SIZE;
            let leaving_y = tile_y0 + u32::max(row, PATCH_SIZE) - PATCH_SIZE;
            let leaving_sum = row_square_sum(codes, x, leaving_y, centre_slot, shape);
            let leaving = select(has_leaving, leaving_sum, 0i32);
            running += entering - leaving;

            let complete = row + 1u32 >= PATCH_SIZE;
            let norm_row = u32::max(row + 1u32, PATCH_SIZE) - PATCH_SIZE;
            let norm_index = norm_row * union_width + column;
            if complete {
                norms[norm_index as usize] = running;
            }

            row += 1u32;
        }

        column += CUBE_DIM_X;
    }
}

/// The tile index of value `k` of the patch whose corner is at tile column `column` and row `row`.
///
/// A patch's values run row by row, then column by column, then channel by channel.
#[cube]
fn patch_value_index(
    column: u32,
    row: u32,
    k: u32,
    #[comptime] tile_width: u32,
    #[comptime] channels: u32,
) -> u32 {
    let row_values = comptime!(PATCH_SIZE * channels);
    let patch_row = k / row_values;
    let in_row = k % row_values;
    let patch_column = in_row / channels;
    let channel = in_row % channels;

    ((row + patch_row) * tile_width + column + patch_column) * channels + channel
}

/// Scores the centre frame's `spatial_radius` rectangle against the reference patch on the matrix
/// units and keeps the best eight positions, one per lane, ranked by SSD and then position.
///
/// Every lane of the cube must call it. The A operand holds the cube's 16 reference patches and
/// each round's B operand holds 16 union positions per logical plane, so each C entry is the dot
/// of one reference with one position. A group scores an entry inside its clipped rectangle as
/// `norm_ref + norm_candidate - 2 * dot` and inserts it as the uniform walk does. The self-match
/// holds slot 0. Returns the number of positions the clipped rectangle holds.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, slot or comptime shape the search reads"
)]
pub(crate) fn matrix_centre_search<C: Int, N: Size, E: Numeric, A: Numeric>(
    codes: &Array<Vector<C, N>>,
    cursor: &IntCursor,
    best_ssd: &mut i32,
    best_pos: &mut u32,
    #[comptime] frame_refs_x: u32,
    #[comptime] variant: IntegerVariant,
    #[comptime] route: MatrixRoute,
    #[comptime] shape: IntShape,
) -> u32 {
    let spatial_radius = comptime!(shape.spatial_radius);
    let channels = comptime!(shape.channels);
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);
    let tile_width = comptime!(variant.matrix_tile_width(spatial_radius));
    let tile_height = comptime!(variant.matrix_tile_height(spatial_radius));
    let union_width = comptime!(variant.union_dims(spatial_radius).0);
    let union_height = comptime!(variant.union_dims(spatial_radius).1);
    let union_len = comptime!(union_width * union_height);
    let plane_width = comptime!(route.plane_width);
    let planes = comptime!(variant.lanes() / route.plane_width);
    let parts = comptime!(route.parts());
    let accumulators = comptime!(route.accumulators());
    let split = comptime!(route.low_bits > 0);
    let part_values = comptime!(tile_width * tile_height * channels);
    let k_steps = comptime!((PATCH_SIZE * PATCH_SIZE * channels / MATRIX_TILE) as usize);
    let slice_rounds = comptime!(TILE_VALUES / route.plane_width);
    let chunk_positions = comptime!(planes * MATRIX_TILE);
    let rounds = comptime!(union_len.div_ceil(chunk_positions));

    let mut tile = SharedMemory::<E>::new(comptime!((part_values * parts) as usize));
    let mut norms = SharedMemory::<i32>::new(union_len as usize);
    let mut slices = SharedMemory::<E>::new(comptime!((planes * TILE_VALUES * parts) as usize));
    let mut c_tiles = SharedMemory::<A>::new(comptime!((planes * TILE_VALUES * accumulators) as usize));

    let plane = UNIT_POS_X / plane_width;
    let lane = UNIT_POS_X % plane_width;
    let group = UNIT_POS_X / 8u32;
    let slice_start = plane * comptime!(TILE_VALUES * parts);
    let c_start = plane * comptime!(TILE_VALUES * accumulators);
    let zero = E::from_int(0);
    let zero_sum = A::from_int(0);

    load_union_tile(
        codes,
        &mut tile,
        cursor.tile_x0,
        cursor.tile_y0,
        cursor.centre_slot,
        tile_width,
        tile_height,
        route,
        shape,
    );
    union_norms(
        codes,
        &mut norms,
        cursor.tile_x0,
        cursor.tile_y0,
        cursor.centre_slot,
        union_width,
        union_height,
        shape,
    );
    sync_cube();

    // Each logical plane stages every reference slice itself and keeps the A fragments.
    let mut a_highs = Sequence::<cmma::Matrix<E>>::new();
    let mut a_lows = Sequence::<cmma::Matrix<E>>::new();

    #[unroll]
    for step in 0..k_steps {
        let first_k = comptime!(step as u32 * MATRIX_TILE);

        for slice_round in 0..slice_rounds {
            let index = slice_round * plane_width + lane;
            let reference = index / MATRIX_TILE;
            let k = first_k + index % MATRIX_TILE;
            let (ref_x, ref_y) = cube_reference_corner(reference, frame_refs_x, variant, shape);
            let column = ref_x - cursor.tile_x0;
            let row = ref_y - cursor.tile_y0;
            let at = patch_value_index(column, row, k, tile_width, channels);
            let slot = slice_start + index;

            slices[slot as usize] = tile[at as usize];

            if comptime!(split) {
                slices[(slot + TILE_VALUES) as usize] = tile[(part_values + at) as usize];
            }
        }

        sync_cube();

        let high_end = slice_start + TILE_VALUES;
        let high_slice = slices.slice(slice_start as usize, high_end as usize);
        let a_high = fragment::<E>(cmma::MatrixIdent::A, cmma::MatrixLayout::RowMajor);
        cmma::load(&a_high, &high_slice, MATRIX_TILE);
        a_highs.push(a_high);

        if comptime!(split) {
            let low_end = high_end + TILE_VALUES;
            let low_slice = slices.slice(high_end as usize, low_end as usize);
            let a_low = fragment::<E>(cmma::MatrixIdent::A, cmma::MatrixLayout::RowMajor);
            cmma::load(&a_low, &low_slice, MATRIX_TILE);
            a_lows.push(a_low);
        }

        sync_cube();
    }

    let left = clamp_top_left(cursor.rx as i32 - spatial_radius as i32, max_x);
    let right = clamp_top_left(cursor.rx as i32 + spatial_radius as i32, max_x);
    let top = clamp_top_left(cursor.ry as i32 - spatial_radius as i32, max_y);
    let bottom = clamp_top_left(cursor.ry as i32 + spatial_radius as i32, max_y);
    let rect = SearchRect {
        left,
        right,
        top,
        bottom,
    };
    let ref_norm_index = (cursor.ry - cursor.tile_y0) * union_width + cursor.rx - cursor.tile_x0;
    let ref_norm = norms[ref_norm_index as usize];

    let high_high = fragment::<A>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let cross = fragment::<A>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
    let low_low = fragment::<A>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);

    let mut worst_ssd = UNFILLED_SSD.runtime();
    let mut worst_pos = 0u32;

    for round in 0..rounds {
        let first_position = (round * planes + plane) * MATRIX_TILE;

        cmma::fill(&high_high, zero_sum);

        if comptime!(split) {
            cmma::fill(&cross, zero_sum);
            cmma::fill(&low_low, zero_sum);
        }

        #[unroll]
        for step in 0..k_steps {
            let first_k = comptime!(step as u32 * MATRIX_TILE);

            for slice_round in 0..slice_rounds {
                let index = slice_round * plane_width + lane;
                let position = first_position + index / MATRIX_TILE;
                let k = first_k + index % MATRIX_TILE;
                let inside = position < union_len;
                let clamped = u32::min(position, union_len - 1u32);
                let column = clamped % union_width;
                let row = clamped / union_width;
                let at = patch_value_index(column, row, k, tile_width, channels);
                let slot = slice_start + index;
                let high_value = tile[at as usize];

                slices[slot as usize] = select(inside, high_value, zero);

                if comptime!(split) {
                    let low_value = tile[(part_values + at) as usize];
                    slices[(slot + TILE_VALUES) as usize] = select(inside, low_value, zero);
                }
            }

            sync_cube();

            // Each slice holds its 16 positions' values position by position, so B loads column-major.
            let high_end = slice_start + TILE_VALUES;
            let high_slice = slices.slice(slice_start as usize, high_end as usize);
            let b_high = fragment::<E>(cmma::MatrixIdent::B, cmma::MatrixLayout::ColMajor);
            let a_high = a_highs.index(step);
            cmma::load(&b_high, &high_slice, MATRIX_TILE);
            cmma::execute::<E, E, A, A>(a_high, &b_high, &high_high, &high_high);

            if comptime!(split) {
                let low_end = high_end + TILE_VALUES;
                let low_slice = slices.slice(high_end as usize, low_end as usize);
                let b_low = fragment::<E>(cmma::MatrixIdent::B, cmma::MatrixLayout::ColMajor);
                let a_low = a_lows.index(step);
                cmma::load(&b_low, &low_slice, MATRIX_TILE);
                cmma::execute::<E, E, A, A>(a_high, &b_low, &cross, &cross);
                cmma::execute::<E, E, A, A>(a_low, &b_high, &cross, &cross);
                cmma::execute::<E, E, A, A>(a_low, &b_low, &low_low, &low_low);
            }

            // The last step's slices are next overwritten after the store's barrier.
            if comptime!(step + 1 < k_steps) {
                sync_cube();
            }
        }

        store_guarded(&mut c_tiles, c_start, &high_high, plane_width);

        if comptime!(split) {
            store_guarded(&mut c_tiles, c_start + TILE_VALUES, &cross, plane_width);
            store_guarded(&mut c_tiles, c_start + 2u32 * TILE_VALUES, &low_low, plane_width);
        }

        sync_cube();

        // Every group reads its own row of each plane's C tiles, in plane order. No barrier follows,
        // since the next round stores its C tiles only after its first gather barrier.
        for entry in 0..chunk_positions {
            let entry_plane = entry / MATRIX_TILE;
            let position = round * chunk_positions + entry;
            let inside = position < union_len;
            let clamped = u32::min(position, union_len - 1u32);
            let candidate_x = cursor.tile_x0 + clamped % union_width;
            let candidate_y = cursor.tile_y0 + clamped / union_width;
            let in_columns = candidate_x >= rect.left && candidate_x <= rect.right;
            let in_rows = candidate_y >= rect.top && candidate_y <= rect.bottom;
            let live_pos = inside && in_columns && in_rows;

            if plane_any(live_pos) {
                let first_c = entry_plane * comptime!(TILE_VALUES * accumulators);
                let c_index = first_c + group * MATRIX_TILE + entry % MATRIX_TILE;
                let high_sum = c_tiles[c_index as usize];
                let mut cross_sum = high_sum;
                let mut low_sum = high_sum;

                if comptime!(split) {
                    cross_sum = c_tiles[(c_index + TILE_VALUES) as usize];
                    low_sum = c_tiles[(c_index + 2u32 * TILE_VALUES) as usize];
                }

                let dot = exact_dot::<A>(high_sum, cross_sum, low_sum, route);
                let candidate_norm = norms[clamped as usize];
                let scored = ref_norm + candidate_norm - 2i32 * dot;
                let masked = select(live_pos, scored, UNFILLED_SSD);
                let ssd =
                    pin_self_match_int(masked, live_pos, candidate_x, candidate_y, cursor.rx, cursor.ry);
                let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
                let places = ranks_before_int(ssd, packed, worst_ssd, worst_pos);

                if plane_any(places) {
                    shift_insert8_ordered_int(best_ssd, best_pos, ssd, packed, cursor.sub);
                    worst_ssd = plane_shuffle(*best_ssd, cursor.base + 7u32);
                    worst_pos = plane_shuffle(*best_pos, cursor.base + 7u32);
                }
            }
        }
    }

    (right - left + 1u32) * (bottom - top + 1u32)
}
