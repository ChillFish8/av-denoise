use super::collab::CollabShape;
use super::collab_split::SearchVariant;
use crate::collab::kernels::fused::{CollabVariant, GROUP_LANES};
use crate::tune::{DeviceLimits, Legality};

/// Whether `candidate` may run on a search of `shape` under `legality`.
///
/// Without codes only the float variants run, as the walk allows. With codes only the float
/// baseline and the built integer variants run, each on a walk the device allows.
pub fn search_candidate_legal(candidate: SearchVariant, shape: &CollabShape, legality: &Legality) -> bool {
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

            has_codes && variant.is_built() && walk_runs
        },
    }
}

/// The bytes of every shared buffer `candidate` allocates, with float tiles of `search_bytes`
/// bytes per value.
///
/// A float variant allocates its centre tile. A scalar or tiled integer variant allocates its tile
/// of f32 codes, one per summed channel, or a one-code placeholder. A sliding centre allocates two
/// i32 sums per block, and its neighbour pass allocates none.
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
            let tile_elements = variant.tile_elements(shape.spatial_radius, shape.channels);
            tile_elements as usize * size_of::<f32>()
        },
    }
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

    let legal = search_candidate_legal(candidate, shape, legality);
    let shared_fits = shared_bytes <= limits.max_shared_bytes;
    let units_fit = lanes <= limits.max_units_per_cube;
    let dims_fit = lanes <= max_x;
    let plane_fits = is_baseline || limits.plane_size_min >= GROUP_LANES;

    legal && shared_fits && units_fit && dims_fit && plane_fits
}
