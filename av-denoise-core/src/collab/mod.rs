pub mod geometry;
pub mod kernels;

// The tests run against a real GPU runtime, so they need a wgpu-backed feature.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

use cubecl::prelude::*;

/// Side length of a collaborative patch in pixels.
pub const PATCH_SIZE: u32 = 8;
pub const PATCH_AREA: u32 = PATCH_SIZE * PATCH_SIZE;
/// Stride of the reference-patch grid.
pub const STEP: u32 = 4;
/// Hard ceiling on the group size.
///
/// A power of two, sized so a stack of 8x8 f32 patches stays small in shared memory.
pub const MAX_K: u32 = 8;
/// Hard ceiling on a cross-frame denoiser's temporal radius.
///
/// The widest cross-frame accumulator holds `2 * MAX_TEMPORAL_RADIUS + 1` frames, which bounds how
/// many passes can write into one pixel before it is read back.
pub const MAX_TEMPORAL_RADIUS: u32 = 8;

/// Whether `R` needs the fused kernel's warp-uniform search.
///
/// The fused kernel's searches are group-scoped. Eight lanes share one reference patch and finish
/// each distance with a shuffle across just those eight. CUDA lowers each shuffle to a
/// `__shfl_*_sync` over the whole 32-lane warp, which on Volta and later waits for every lane it
/// names. A group that leaves the search early never releases the ones still in it, so the warp
/// deadlocks. The wgpu backends reconverge on their own and run the cheaper clipped search.
pub fn needs_warp_uniform_search<R: Runtime>(client: &ComputeClient<R>) -> bool {
    R::name(client) == "cuda"
}

/// Frames per volume in a cross-frame group at `temporal_radius`.
///
/// A group holds `MAX_K` patches as `MAX_K / grid_frames` volumes. Radius 0 has no neighbour to
/// follow, so every group is a single-frame one.
pub fn grid_frames(temporal_radius: u32) -> u32 {
    match temporal_radius {
        0 => 1,
        1 => 2,
        _ => 4,
    }
}
