//! nl3d chains non-local means and a BM3D-style collaborative filter into
//! one denoiser.
//!
//! Non-local means cleans a frame by averaging matching patches from a
//! temporal window around it. Averaging trades detail for noise removal,
//! and even a strong pass stops well short of removing every trace of
//! noise, so its output still carries a residual. A second pass then
//! cleans that residual up.
//!
//! The second pass models the residual differently. Rather than
//! averaging matching patches together, it groups them into a stack,
//! runs the stack through a joint transform, and shrinks the transform
//! coefficients that noise is most likely to have produced. That reaches
//! noise the first pass's averaging could not remove without smoothing
//! away real structure along with it.
//!
//! The frame itself never leaves the GPU between passes. The first
//! stage's finished frame feeds straight into the second stage's input,
//! and only the second stage's output makes the trip back to the host.
//!
//! One small scalar does cross back to the host every frame, the
//! residual noise ratio the second stage shrinks by, measured from the
//! first stage's own accumulators. That readback is deliberately lagged
//! by one frame, reusing the ratio measured on the previous frame rather
//! than waiting on the current one, so it never stalls the GPU queue
//! behind a value that is not ready yet. See
//! [`Nl3dDenoiser::run_collab_stage`] for the reasoning.
//!
//! # Layout
//!
//! [`Nl3dDenoiser`] owns both stages and drives one frame through both of
//! them per call. [`Nl3dParams`] holds the tuning values for the whole
//! cascade, wrapping [`crate::nlmeans::NlmParams`] for the front end and
//! [`crate::collab::CollabParams`] for the collaborative stage that runs
//! second.

mod denoiser;
mod rho;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

pub use denoiser::{Nl3dDenoiser, Nl3dParams};
