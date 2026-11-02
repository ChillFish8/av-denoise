mod analyse;
mod chain;
mod chroma_map;
mod compensate;
mod confidence;
mod context;
mod escalate;
mod global;
pub(crate) mod kernels;
mod luma_ring;
mod pyramid;
mod search;
pub(crate) mod truemotion;

#[cfg(test)]
pub(crate) mod tests;

pub(crate) use self::analyse::{
    Escalation,
    MatchTargets,
    confidence_byte_offset,
    escalation_state_len,
    mv_field_byte_offset,
    run_analyse,
    run_seeded_refine,
};
pub(crate) use self::chain::{neighbour_idx_for_k, pair_byte_offset, run_pair_analyse, zero_pair_slot};
pub(crate) use self::chroma_map::{FieldLayout, Subsampling, map_luma_fields};
pub(crate) use self::compensate::run_compensate;
pub(crate) use self::confidence::{THSAD_PIXEL, run_confidence, sad_noise_floor, thsad};
pub(crate) use self::context::{MotionCtx, pair_ring_slot_count};
pub use self::global::{
    AFFINE_WORDS,
    FIT_BAND_WORDS,
    FIT_INLIER_CAP,
    FIT_LANES,
    FIT_SCALES,
    FIT_STAGE_GATE,
    FIT_STAGE_POLISH,
    FIT_STAGE_SCORE,
    FIT_STAGE_SOLVE,
    FIT_STAGE_WEIGH,
    FIT_STATE_WORDS,
    FIT_THREADS,
    FitScratch,
    MAX_ROLL_TAN,
    MODEL_WORDS,
    ModelRing,
    NO_PAIR,
    PairModelProducer,
    compose_gaps,
    run_compose_models,
};
pub(crate) use self::global::{SharedMotion, producer_buffer_sizes};
pub(crate) use self::luma_ring::{LumaMotionRing, RingRole};
pub(crate) use self::pyramid::{
    level_dims,
    pyramid_pixels_per_frame,
    pyramid_slot_pixels,
    run_pyramid_build,
};
pub use self::search::{
    CHAINED_RADIUS_THRESHOLD,
    DEFAULT_BLKSIZE,
    DEFAULT_MAX_SEARCH_ESCALATION,
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
