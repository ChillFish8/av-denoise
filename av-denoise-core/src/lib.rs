#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

pub mod accelerate;
pub mod cache;
#[doc(hidden)]
pub mod collab;
mod denoiser;
pub mod device;
pub mod enumerate;
pub mod frame;
#[doc(hidden)]
pub mod nl4d;
#[doc(hidden)]
pub mod nlmeans;
mod probe;
pub mod sniff;
pub mod warmup;

pub use cache::{
    COMPILATION_CACHE_ENV,
    CacheAlreadyInitialisedError,
    compilation_cache_dir,
    install_compilation_cache,
    install_compilation_cache_once,
};
pub use denoiser::{
    Algorithm,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    MAX_PENDING,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    Preset,
    WindowSpan,
    nl4d_default_lambda_ht,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
};
pub use device::Device;
pub use frame::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    push_needs_retry,
};
pub use nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    Depth,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    PrefilterMode,
    UnsupportedDepthError,
    denormalize,
    normalize,
    parse_prefilter,
};
pub use warmup::{WarmUp, kernel_key};
