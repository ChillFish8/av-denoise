//! The non-local means denoiser that sits behind [Nlmeans].
//!
//! Non-local means cleans a pixel by finding patches elsewhere that look
//! like the patch around it, then averaging them. Similar patches get a
//! large weight and dissimilar ones get almost none, so flat areas
//! smooth out while edges survive.
//!
//! The search can reach across neighbouring frames as well as within one
//! frame, which is what the temporal radius controls.
//!
//! # Layout
//!
//! `params` holds the tuning values and the calibrated defaults, and
//! `NlmParams` is the single struct everything else is built from.
//!
//! `NlmDenoiser` owns the GPU buffers and the frame ring, and
//! `dispatch` turns one set of parameters into the sequence of kernel
//! launches that produces a frame.
//!
//! `kernels` holds the GPU code itself. `noise` measures how noisy a
//! frame is, `motion` tracks movement between frames, and
//! `prefilter` builds the cleaner reference image that patches are
//! compared against.

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

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
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

/// Cube shape for the per-pixel `nlm_accumulate` kernel, which has no
/// shared-memory tile.
///
/// On RDNA-class GPUs this shape benchmarks 10 to 25% faster than the
/// tile-heavy default. The kernel waits on memory rather than compute,
/// so the extra threads hide the load latency.
pub(crate) const BLOCK_X_THIN: u32 = 32;
pub(crate) const BLOCK_Y_THIN: u32 = 16;

/// Largest 1D grid a dispatch may ask for, set by the WebGPU and Vulkan
/// limits.
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for 1D utility kernels (copy, zero).
pub(crate) const BLOCK_1D: u32 = 256;
