//! nl4d groups patches across several noisy frames rather than within a
//! single one.
//!
//! [`crate::collab`] groups similar 8x8 patches within one frame and
//! denoises each group jointly. This module extends that search across a
//! motion-compensated window of frames.
//!
//! Each group is a few centre-frame patches, each followed through its
//! best-matching neighbour frames along its own motion vector. Patches
//! followed through time carry independent grain, so the transform along
//! time separates it from the texture they share.

mod denoiser;
pub mod harness;
pub mod kernels;
mod params;
mod regularise;
mod snapshot;

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) mod tests;

pub use self::denoiser::Nl4dDenoiser;
pub use self::params::{MAX_KAISER_BETA, Nl4dParams};
pub use self::snapshot::MotionSnapshot;
