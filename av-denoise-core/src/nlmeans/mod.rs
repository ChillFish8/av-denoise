//! Non-local means denoising on the GPU
//!
//! A pixel is averaged with pixels whose surrounding patches look alike, within one frame or across
//! a temporal window. The module provides:
//!
//! - [Nlmeans], the engine
//! - the options and tuning parameters it is built from
//! - motion compensation and prefilters

pub(crate) mod denoiser;
pub(crate) mod kernels;
pub(crate) mod mirror;
pub(crate) mod motion;
pub(crate) mod params;
pub(crate) mod prefilter;

mod dispatch;
mod edges;
mod engine;
mod options;

// The tests run against a real GPU runtime, so they need a wgpu-backed feature.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) mod tests;

pub(crate) use self::denoiser::{NlmDenoiser, RingView};
pub use self::engine::Nlmeans;
pub use self::motion::{MotionCompensationMode, MotionEstimation, MotionSearch};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::options::resolve_params;
pub use self::options::{
    DenoisingMode,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
};
pub use self::params::{ChannelMode, HqParams};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::params::{MAX_PATCH_RADIUS, MAX_SEARCH_RADIUS, MAX_TEMPORAL_RADIUS};
pub(crate) use self::params::{NlmParams, hq_default_strength};
pub use self::prefilter::{DEFAULT_PILOT_STRENGTH_SCALE, PrefilterMode, parse_prefilter};
pub(crate) use crate::engine::MAX_GRID_1D;

/// Cube X dimension for tile-heavy fused/separable kernels.
pub(crate) const BLOCK_X: u32 = 32;
/// Cube Y dimension for tile-heavy fused/separable kernels.
pub(crate) const BLOCK_Y: u32 = 8;

/// Block size for the 1D copy and zero kernels.
pub(crate) const BLOCK_1D: u32 = 256;
