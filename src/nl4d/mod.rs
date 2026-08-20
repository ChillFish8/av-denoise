//! nl4d groups patches across several noisy frames rather than within a
//! single one.
//!
//! [`crate::collab`] groups similar 8x8 patches within one frame and
//! denoises each group jointly. This module extends that search across a
//! motion-compensated window of frames. For each reference patch it
//! searches the centre frame spatially, and also searches each neighbour
//! frame in a small window around where the motion field predicts that
//! patch moved.
//!
//! Patches matched in different frames carry independent grain, so
//! grouping them lets the collaborative transform cancel more of it than
//! a single-frame search can.

mod denoiser;
mod params;

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

pub use denoiser::Nl4dDenoiser;
pub use params::Nl4dParams;
