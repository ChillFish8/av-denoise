//! A BM3D-style collaborative filter core, used by the `nl4d` denoiser.
//!
//! Collaborative filtering cleans a frame by grouping similar patches
//! together and denoising the whole group at once, rather than pixel by
//! pixel. A reference patch collects a stack of the patches that match it
//! best, the stack is filtered as a unit, and the filtered results are
//! blended back onto the frame.
//!
//! [`geometry`] lays out the grid of reference patches a frame is covered
//! with, and sizes the buffers driven by that grid. [`kernels`] holds the
//! GPU code that does the grouping, filtering, and blending.

pub mod geometry;
pub mod kernels;

use cubecl::prelude::*;

/// Whether `R` needs [`kernels::fused::collab_fused`]'s warp-uniform
/// search.
///
/// That kernel's searches are group-scoped: eight lanes share one
/// reference patch and complete every distance with a shuffle across
/// just those eight. The CUDA backend still lowers each shuffle to a
/// `__shfl_*_sync` over the whole 32-lane warp, which on Volta and later
/// waits for every lane it names. Groups that leave the search early
/// never come back to release the ones still in it, so the warp
/// deadlocks and the launch never retires a frame.
///
/// The wgpu backends emit subgroup operations that reconverge on their
/// own, so they run the cheaper path that walks only the clipped
/// rectangles. See the `warp_uniform` argument for what the other path
/// changes.
pub fn needs_warp_uniform_search<R: Runtime>(client: &ComputeClient<R>) -> bool {
    R::name(client) == "cuda"
}

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

/// Side length of a collaborative patch in pixels.
pub const PATCH_SIZE: u32 = 8;
/// Pixels in one patch.
pub const PATCH_AREA: u32 = PATCH_SIZE * PATCH_SIZE;
/// Stride of the reference-patch grid.
pub const STEP: u32 = 4;
/// Hard ceiling on the group size. Power of two, sized so a stack of K
/// 8x8 f32 patches stays small in shared memory.
pub const MAX_K: u32 = 8;
/// Hard ceiling on a cross-frame denoiser's temporal radius.
///
/// [`crate::nl4d::Nl4dParams::validate`] rejects a `temporal_radius`
/// outside `1..=MAX_TEMPORAL_RADIUS`, so this is also the largest radius
/// [`kernels::aggregate::cross_frame_accum_scale`] has to derive a scale
/// for. The widest cross-frame accumulator holds
/// `2 * MAX_TEMPORAL_RADIUS + 1` frames, which bounds how many passes can
/// write into one pixel before it is read back.
pub const MAX_TEMPORAL_RADIUS: u32 = 8;
