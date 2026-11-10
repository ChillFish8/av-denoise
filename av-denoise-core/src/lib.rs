#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod bench_api;
mod collab;
mod engine;
mod error;
mod kernel_cache;
#[cfg(test)]
mod kernel_hash;
pub mod lockfile;
mod motion;
pub mod nl4d;
pub mod nlmeans;
mod noise;
mod options;
#[doc(hidden)]
pub mod testing;
#[cfg(test)]
mod tests;
pub mod tune;

pub use self::engine::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    EdgePadding,
    Engine,
    EngineSpec,
    EngineTiming,
    Geometry,
    PassTimings,
    PlaneSource,
    SampleFormat,
    WindowSpan,
};
pub use self::error::Error;
pub use self::kernel_cache::prepare_kernel_cache;
pub use self::nl4d::Nl4dOptions;
pub use self::nl4d::grain::{FrameGrain, GrainChunk, GrainChunks, SceneGrain, build_table, frame_rows};
pub use self::nlmeans::{
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    PrefilterMode,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
};
pub use self::options::{Preset, search_escalation_for};
pub use self::tune::{
    BudgetRequest,
    SearchMode,
    TuneOutcome,
    TuneReport,
    TuneRequest,
    TunedGroup,
    TuningConfig,
    TuningFrame,
    TuningMode,
    build_engine,
    is_tuned,
    tune_frames,
};

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature needs an Apple target, such as `aarch64-apple-darwin`");

/// A hash of this crate's sources and the CubeCL version, which names a build's directory for its
/// compiled kernels and tuning results.
pub const KERNEL_HASH: &str = env!("AV_DENOISE_KERNEL_HASH");
