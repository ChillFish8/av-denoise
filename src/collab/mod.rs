//! A BM3D-style collaborative filter core, shared by the `nl3d` and
//! `nl4d` denoisers.
//!
//! Collaborative filtering cleans a frame by grouping similar patches
//! together and denoising the whole group at once, rather than pixel by
//! pixel. A reference patch collects a stack of the patches that match it
//! best, the stack is filtered as a unit, and the filtered results are
//! aggregated back onto the frame.
//!
//! # Layout
//!
//! `geometry` lays out the grid of reference patches a frame is covered
//! with, and works out how large the buffers driven by that grid need to
//! be.
//!
//! `params` holds the tuning values the filter is built from, in
//! [`CollabParams`].
//!
//! [`kernels`] holds the GPU code, starting with the DCT and Haar
//! transforms the filter runs patches and patch stacks through.
//!
//! `pipeline` chains the grouping, filtering, and aggregation kernels
//! into the two-stage filter itself, in [`CollabPipeline`].

pub mod geometry;
pub mod kernels;
mod params;
mod pipeline;

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

pub use params::CollabParams;
pub use pipeline::CollabPipeline;

/// Side length of a collaborative patch in pixels.
pub const PATCH_SIZE: u32 = 8;
/// Pixels in one patch.
pub const PATCH_AREA: u32 = PATCH_SIZE * PATCH_SIZE;
/// Stride of the reference-patch grid.
pub const STEP: u32 = 4;
/// Hard ceiling on the group size. Power of two, sized so a stack of K
/// 8x8 f32 patches stays small in shared memory.
pub const MAX_K: u32 = 8;
