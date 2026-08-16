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
pub mod nl3d;
#[doc(hidden)]
pub mod nlmeans;
pub mod sniff;

pub use cache::{COMPILATION_CACHE_ENV, CacheAlreadyInitialisedError, apply_compilation_cache_env};
pub use denoiser::{
    Algorithm,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    MAX_PENDING,
    Nl3dOptions,
    NlmTuning,
    nl3d_default_residual_sigma_scale,
};
pub use device::Device;
pub use nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    Depth,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    PrefilterMode,
    UnsupportedDepthError,
    denormalize,
    normalize,
};
