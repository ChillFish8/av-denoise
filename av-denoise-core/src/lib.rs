#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod bench_api;
mod collab;
mod engine;
mod error;
pub mod nl4d;
pub mod nlmeans;
mod options;
pub(crate) mod tune;

pub use self::engine::{DevicePlane, EdgePadding, Engine, Geometry, SampleFormat, WindowSpan};
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

/// A hash of this crate's sources, which names a build's directory in the compiled kernel cache.
pub const KERNEL_HASH: &str = env!("AV_DENOISE_KERNEL_HASH");
