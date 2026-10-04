//! nl4d groups patches across several noisy frames rather than within a
//! single one.
//!
//! The collaborative filter groups similar 8x8 patches within one frame and
//! denoises each group jointly. This module extends that search across a
//! motion-compensated window of frames.
//!
//! Each group is a few centre-frame patches, each followed through its
//! best-matching neighbour frames along its own motion vector. Patches
//! followed through time carry independent grain, so the transform along
//! time separates it from the texture they share.

pub(crate) mod denoiser;
mod engine;
pub mod grain;
pub(crate) mod harness;
pub(crate) mod kernels;
mod options;
pub(crate) mod params;
mod regularise;
pub(crate) mod snapshot;

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) mod tests;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::denoiser::Nl4dDenoiser;
pub use self::engine::Nl4d;
pub(crate) use self::options::nl4d_pool_ratio;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::options::resolve_params;
pub use self::options::{
    Nl4dOptions,
    nl4d_default_lambda_ht,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
};
#[cfg(test)]
pub(crate) use self::params::MAX_KAISER_BETA;
pub(crate) use self::params::Nl4dParams;
pub(crate) use self::snapshot::MotionSnapshot;
