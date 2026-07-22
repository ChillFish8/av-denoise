pub mod kernels;
pub mod motion;
pub mod prefilter;

mod denoiser;
mod dispatch;
mod noise;
mod params;
mod pending;

#[cfg(test)]
mod tests;

pub use denoiser::NlmDenoiser;
pub use motion::{MotionCompensationMode, MotionEstimation};
pub use params::{
    ChannelMode,
    HqParams,
    MAX_PATCH_RADIUS,
    MAX_SEARCH_RADIUS,
    MAX_TEMPORAL_RADIUS,
    NlmParams,
    hq_default_strength,
};
pub use pending::Pending;
pub use prefilter::{DEFAULT_PILOT_STRENGTH_SCALE, PrefilterMode};

/// Cube X dimension for tile-heavy fused/separable kernels.
pub const BLOCK_X: u32 = 32;
/// Cube Y dimension for tile-heavy fused/separable kernels.
pub const BLOCK_Y: u32 = 8;

/// Cube shape for the per-pixel `nlm_accumulate` kernel, which has no
/// SMEM tile. On RDNA-class GPUs it benchmarks 10 to 25% faster at
/// (32, 16) than at the tile-heavy default, because it's
/// memory-latency-bound and the extra threads hide load latency.
pub const BLOCK_X_THIN: u32 = 32;
pub const BLOCK_Y_THIN: u32 = 16;

/// Maximum 1D grid size for GPU dispatch (WebGPU/Vulkan limit).
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for 1D utility kernels (copy, zero).
pub(crate) const BLOCK_1D: u32 = 256;

pub fn normalize_u8(input: &[u8]) -> Vec<f32> {
    input.iter().map(|&v| v as f32 / 255.0).collect()
}

pub fn denormalize_u8(input: &[f32]) -> Vec<u8> {
    input
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

pub fn normalize_u16(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&v| v as f32 / 65535.0).collect()
}

pub fn denormalize_u16(input: &[f32]) -> Vec<u16> {
    input
        .iter()
        .map(|&v| (v * 65535.0).round().clamp(0.0, 65535.0) as u16)
        .collect()
}
