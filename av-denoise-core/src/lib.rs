#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod bench_api;
mod collab;
mod engine;
mod error;
#[cfg(test)]
mod kernel_hash;
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

pub use self::engine::{DevicePlane, EdgePadding, Engine, EngineTiming, Geometry, SampleFormat, WindowSpan};
pub use self::error::Error;
pub use self::nl4d::grain::{GrainChunk, SceneGrain, build_table};
pub use self::nl4d::{
    Nl4d,
    Nl4dOptions,
    PsyParams,
    nl4d_default_lambda_ht,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
};
pub use self::nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    NlmTuning,
    Nlmeans,
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
pub use self::options::Preset;

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature needs an Apple target, such as `aarch64-apple-darwin`");

/// A hash of this crate's sources and the CubeCL version, which names a build's directory for its
/// compiled kernels and tuning results.
pub const KERNEL_HASH: &str = env!("AV_DENOISE_KERNEL_HASH");
