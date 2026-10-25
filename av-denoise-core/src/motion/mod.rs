mod analyse;
mod chain;
mod compensate;
mod confidence;
mod context;
pub(crate) mod kernels;
mod pyramid;
mod search;

#[cfg(test)]
mod tests;

pub(crate) use self::analyse::{
    MatchTargets,
    confidence_byte_offset,
    mv_field_byte_offset,
    run_analyse,
    run_seeded_refine,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::chain::pair_byte_offset;
pub(crate) use self::chain::{neighbour_idx_for_k, run_pair_analyse, zero_pair_slot};
pub(crate) use self::compensate::run_compensate;
pub(crate) use self::confidence::{THSAD_PIXEL, run_confidence, sad_noise_floor, thsad};
pub(crate) use self::context::{MotionCtx, pair_ring_slot_count};
pub(crate) use self::pyramid::{
    level_dims,
    pyramid_pixels_per_frame,
    pyramid_slot_pixels,
    run_pyramid_build,
};
pub use self::search::{
    CHAINED_RADIUS_THRESHOLD,
    DEFAULT_BLKSIZE,
    DEFAULT_OVERLAP,
    DEFAULT_PYRAMID_LEVELS,
    DEFAULT_REFINE_RADIUS,
    DEFAULT_SEARCH_RADIUS,
    MAX_BLKSIZE,
    MAX_PYRAMID_LEVELS,
    MAX_SEARCH_RADIUS,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
};
