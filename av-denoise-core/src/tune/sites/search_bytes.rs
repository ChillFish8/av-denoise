use half::f16;

use super::collab::CollabShape;
use super::collab_split::SearchVariant;
use crate::collab::kernels::fused::{CollabVariant, GROUP_LANES};
use crate::collab::kernels::integer::{IntCentre, IntegerVariant};
use crate::matrix::TILE_VALUES;
use crate::tune::{DeviceLimits, Int8Layout, Legality, int8_matrix_plane_width, matrix_plane_width};

/// Whether `candidate` may run on a search of `shape` under `legality`.
///
/// Without codes only the float variants run, as the walk allows. With codes only the float
/// baseline and the built integer variants run, each on a walk the device allows. An f16 matrix
/// variant also needs the device's matrix plane to be the shape's. An int8 one needs the device to
/// run the shape's int8 product, and the device's int8 plane to be the shape's.
pub fn search_candidate_legal(
    candidate: SearchVariant,
    shape: &CollabShape,
    limits: &DeviceLimits,
    legality: &Legality,
) -> bool {
    let uniform_only = shape.uniform_walk_only || legality.uniform_walk_only;
    let has_codes = shape.codes.is_some();

    match candidate {
        SearchVariant::Float(variant) => {
            if has_codes {
                let baseline = CollabVariant::baseline_for(uniform_only);
                return variant == baseline;
            }

            variant.runs_where(uniform_only)
        },
        SearchVariant::Integer(variant) => {
            let walk_runs = variant.runs_where(uniform_only);
            let plane_width = matrix_plane_width(limits);
            let matrix_fits = !variant.is_matrix() || plane_width == Some(shape.matrix_plane);
            let int8_width = int8_matrix_plane_width(limits);
            let product_runs = shape.int8.is_some_and(|element| limits.runs_int8_matrix(element));
            let int8_offered = product_runs && int8_width == Some(shape.int8_plane);
            let int8_fits = !variant.is_int8() || int8_offered;

            has_codes && variant.is_built() && walk_runs && matrix_fits && int8_fits
        },
    }
}

/// The bytes of every shared buffer `candidate` allocates, with float tiles of `search_bytes`
/// bytes per value.
///
/// A float variant allocates its centre tile. A scalar or tiled integer variant allocates its tile
/// of f32 codes, one per summed channel, or a one-code placeholder. A matrix centre allocates the
/// buffers [matrix_centre_bytes] counts. A sliding centre allocates two i32 sums per block, and
/// its neighbour pass allocates none.
pub fn search_shared_bytes(candidate: SearchVariant, shape: &CollabShape, search_bytes: usize) -> usize {
    match candidate {
        SearchVariant::Float(variant) => {
            let tile_elements = variant.tile_elements(shape.spatial_radius, shape.stored_ch);
            tile_elements as usize * search_bytes
        },
        SearchVariant::Integer(variant) if variant.is_sliding() => {
            let sums = 2 * variant.sliding_blocks() as usize;
            sums * size_of::<i32>()
        },
        SearchVariant::Integer(variant) => {
            let matrix_bytes = matrix_centre_bytes(variant, shape);
            let tile_elements = variant.tile_elements(shape.spatial_radius, shape.channels);
            let tile_bytes = tile_elements as usize * size_of::<f32>();

            matrix_bytes.unwrap_or(tile_bytes)
        },
    }
}

/// The operand layout of a matrix centre, as the shared buffers count it.
struct MatrixOperands {
    /// Bytes of one operand value.
    value_bytes: usize,
    /// Parts each code splits into.
    parts: usize,
    /// Accumulators, and so C tiles, each logical plane keeps.
    accumulators: usize,
    plane_width: u32,
}

/// The operands of `variant`'s matrix centre on `shape`, or `None` when it has none there.
fn matrix_operands(variant: IntegerVariant, shape: &CollabShape) -> Option<MatrixOperands> {
    let codes = shape.codes?;

    match variant.centre {
        IntCentre::Matrix if shape.matrix_plane > 0 => {
            let parts = if codes.wide() { 2 } else { 1 };
            let accumulators = if codes.wide() { 3 } else { 1 };

            Some(MatrixOperands {
                value_bytes: size_of::<f16>(),
                parts,
                accumulators,
                plane_width: shape.matrix_plane,
            })
        },
        IntCentre::MatrixI8 if shape.int8_plane > 0 => {
            let element = shape.int8?;
            let layout = Int8Layout::for_codes(element, codes);

            Some(MatrixOperands {
                value_bytes: size_of::<u8>(),
                parts: layout.parts() as usize,
                accumulators: layout.accumulators() as usize,
                plane_width: shape.int8_plane,
            })
        },
        _ => None,
    }
}

/// The shared bytes of `variant`'s matrix centre on `shape`, or `None` when it has none there.
///
/// It counts the tile of operand parts, one i32 norm per union position, and each logical plane's
/// slice of operand parts and its 4-byte C tiles.
fn matrix_centre_bytes(variant: IntegerVariant, shape: &CollabShape) -> Option<usize> {
    let operands = matrix_operands(variant, shape)?;
    let spatial_radius = shape.spatial_radius;
    let tile_width = variant.matrix_tile_width(spatial_radius) as usize;
    let tile_height = variant.matrix_tile_height(spatial_radius) as usize;
    let (union_width, union_height) = variant.union_dims(spatial_radius);
    let planes = (variant.lanes() / operands.plane_width) as usize;
    let tile_values = TILE_VALUES as usize;

    let part_bytes = operands.parts * operands.value_bytes;
    let tile_bytes = tile_width * tile_height * shape.channels as usize * part_bytes;
    let norm_bytes = (union_width * union_height) as usize * size_of::<i32>();
    let slice_bytes = tile_values * part_bytes;
    let c_tile_bytes = tile_values * size_of::<i32>() * operands.accumulators;
    let plane_bytes = planes * (slice_bytes + c_tile_bytes);

    Some(tile_bytes + norm_bytes + plane_bytes)
}

/// Legal, and its shared memory, threads, cube shape and plane width fit `limits`, as
/// `VariantSite::fits` checks them (the baseline skips the plane width check).
pub fn search_candidate_fits(
    candidate: SearchVariant,
    shape: &CollabShape,
    limits: &DeviceLimits,
    legality: &Legality,
    search_bytes: usize,
) -> bool {
    let lanes = candidate.lanes();
    let shared_bytes = search_shared_bytes(candidate, shape, search_bytes);
    let (max_x, _, _) = limits.max_cube_dim;
    let baseline_variant = CollabVariant::baseline_for(shape.uniform_walk_only);
    let is_baseline = candidate == SearchVariant::Float(baseline_variant);

    let legal = search_candidate_legal(candidate, shape, limits, legality);
    let shared_fits = shared_bytes <= limits.max_shared_bytes;
    let units_fit = lanes <= limits.max_units_per_cube;
    let dims_fit = lanes <= max_x;
    let plane_fits = is_baseline || limits.plane_size_min >= GROUP_LANES;

    legal && shared_fits && units_fit && dims_fit && plane_fits
}
