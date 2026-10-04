#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

pub mod accelerate;
pub mod cache;
#[doc(hidden)]
pub mod collab;
mod denoiser;
pub mod device;
mod engine;
#[doc(hidden)]
pub mod engine_kernels {
    pub use crate::engine::kernels::*;
}
pub mod enumerate;
mod error;
pub mod frame;
#[doc(hidden)]
pub mod nl4d;
#[doc(hidden)]
pub mod nlmeans;
mod options;
mod probe;
pub mod sniff;
pub mod stack;
pub mod warmup;

pub use cache::{
    COMPILATION_CACHE_ENV,
    CacheError,
    compilation_cache_dir,
    default_cache_dir,
    install_compilation_cache,
    install_compilation_cache_at,
    install_compilation_cache_once,
};
pub use denoiser::{
    Algorithm,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    FrameOutput,
    MAX_PENDING,
    Nl4dOptions,
    OutputFormat,
    nl4d_default_lambda_ht,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
};
pub use device::Device;
pub use engine::{DevicePlane, EdgePadding, Engine, Geometry, SampleFormat, WindowSpan};
pub use error::Error;
pub use frame::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    ReseedWindow,
    Subsampling,
    push_needs_retry,
};
pub use nl4d::grain::{GrainChunk, SceneGrain, build_table};
pub use nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    Depth,
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
    UnsupportedDepthError,
    WirePack,
    denormalize,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    normalize,
    parse_prefilter,
};
pub use options::Preset;
pub use stack::{CODEGEN_STACK_BYTES, codegen_stack_is_sufficient, raise_codegen_stack_limit};
pub use warmup::{WarmUp, kernel_key};
