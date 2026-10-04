//! Non-local means denoising on the GPU
//!
//! A pixel is averaged with pixels whose surrounding patches look alike, within one frame or across
//! a temporal window. The module provides:
//!
//! - [Nlmeans], the engine
//! - the options and tuning parameters it is built from
//! - noise estimation, motion compensation and prefilters

pub(crate) mod denoiser;
pub(crate) mod kernels;
pub(crate) mod motion;
pub(crate) mod params;
pub(crate) mod prefilter;

mod align;
mod dispatch;
mod edges;
mod engine;
mod noise;
mod options;

// The tests run against a real GPU runtime, so they need a wgpu-backed feature.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) mod tests;

pub(crate) use self::denoiser::{NlmDenoiser, RingView};
pub use self::engine::Nlmeans;
pub use self::motion::{MotionCompensationMode, MotionEstimation, MotionSearch};
#[cfg(test)]
pub(crate) use self::noise::QuarterClass;
pub(crate) use self::noise::{NOISE_CURVE_BINS, QuarterClasses, StrengthMapParams};
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

/// Cube X dimension for tile-heavy fused/separable kernels.
pub(crate) const BLOCK_X: u32 = 32;
/// Cube Y dimension for tile-heavy fused/separable kernels.
pub(crate) const BLOCK_Y: u32 = 8;

/// Cube shape for the per-pixel `nlm_accumulate` kernel, which has no shared-memory tile.
///
/// On RDNA-class GPUs it benchmarks 10 to 25% faster than the tile-heavy default, because the
/// kernel waits on memory and the extra threads hide the load latency.
pub(crate) const BLOCK_X_THIN: u32 = 32;
pub(crate) const BLOCK_Y_THIN: u32 = 16;

/// Largest 1D grid a dispatch may ask for, set by the WebGPU and Vulkan limits.
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for the 1D copy and zero kernels.
pub(crate) const BLOCK_1D: u32 = 256;
