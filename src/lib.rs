#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

pub mod accelerate;
pub mod cache;
#[doc(hidden)]
pub mod collab;
mod denoiser;
pub mod device;
#[doc(hidden)]
pub mod nl4d;
#[doc(hidden)]
pub mod nlmeans;
pub mod sniff;

pub use cache::{COMPILATION_CACHE_ENV, CacheAlreadyInitialisedError, install_compilation_cache};
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
    nl4d_default_lambda_ht,
};
pub use device::Device;
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
};
