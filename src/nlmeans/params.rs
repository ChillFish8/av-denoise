//! NLM parameters and channel layout. Pure data + validation; no
//! kernel dispatch lives here, so `NlmDenoiser` can keep its
//! construction-time checks in one cheap module.

use super::{MotionCompensationMode, PrefilterMode};

/// SSD normalisation reference, matching FFmpeg's nlmeans (255² for
/// 8-bit normalisation). Distances are computed in `[0, 1]` units so
/// this folds in the implied scale-up.
pub(super) const NLM_NORM: f32 = 255.0 * 255.0;
/// Legacy scaling factor inherited from FFmpeg's nlmeans; preserved so
/// our `strength` parameter has equivalent meaning.
pub(super) const NLM_LEGACY: f32 = 3.0;

/// Patch radius threshold: above this the dispatcher switches to the
/// separable path so the per-pixel cost stays linear in `patch_radius`.
pub(super) const SEPARABLE_THRESHOLD: u32 = 8;

/// Hard ceiling on `patch_radius`. The fused kernels load a
/// `(block + 2·patch_radius)²` SMEM tile; values above this run out of
/// SMEM on RDNA-class GPUs.
pub const MAX_PATCH_RADIUS: u32 = 16;

/// Hard ceiling on `search_radius`. The windowed kernel SMEM tile is
/// `(block + 2·patch_radius + 2·search_radius)² × stored_ch × 4` bytes;
/// the per-q dispatch path is also gated on this so launch counts stay
/// sane (`(2·a+1)²` launches per frame).
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// Hard ceiling on `temporal_radius`. The ring buffer is sized for
/// `2·t + 1` frames; values above this consume excessive device memory
/// (e.g. 1080p YUV at `t = 16` ≈ 540 MB just for input).
pub const MAX_TEMPORAL_RADIUS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// How to apply denoising to the input frame channels.
pub enum ChannelMode {
    /// Single luminance channel. Distance scaled by 3.0.
    Luma,
    /// Two chroma channels (U, V). Distance scaled by 1.5.
    Chroma,
    /// Three channels (Y, U, V). Unscaled sum of squared differences.
    Yuv,
}

impl ChannelMode {
    /// Number of meaningful channels participating in distance/output.
    pub fn count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 3,
        }
    }

    /// Channels-per-pixel in GPU storage. Padded up to the next supported
    /// vectorization factor so kernels can use coalesced `Line<f32>` reads
    /// (backends only support power-of-two line sizes; YUV pads to 4).
    pub fn storage_count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NlmParams {
    /// Temporal radius. 0 = spatial only, d > 0 uses 2*d+1 frames.
    pub temporal_radius: u32,
    /// Search window half-size. Search window is (2*a+1)^2. Default: 2.
    pub search_radius: u32,
    /// Patch comparison half-size. Patch is (2*s+1)^2. Default: 4, range [0, 8].
    pub patch_radius: u32,
    /// Filtering strength. Higher = more smoothing. Default: 1.2.
    pub strength: f32,
    /// Self-weight multiplier. Default: 1.0. Set to 0 for pure NLM.
    pub self_weight: f32,
    /// Which channels to process.
    pub channels: ChannelMode,
    /// Reference clip source used for patch-distance / weight
    /// computation. Default: `None`. When set, weights are derived
    /// from a prefiltered or externally-supplied clip while pixel
    /// accumulation continues to read the original input.
    pub prefilter: PrefilterMode,
    /// Motion-compensation mode. Default: `None`. When set to
    /// `Mvtools`, each `denoise_submit` warps the temporal neighbours
    /// into spatial alignment with the centre before NLM weighting.
    /// Only takes effect when `temporal_radius > 0`.
    pub motion_compensation: MotionCompensationMode,
}

impl Default for NlmParams {
    fn default() -> Self {
        Self {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Yuv,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
        }
    }
}

impl NlmParams {
    pub fn h2_inv_norm(&self) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        NLM_NORM / (NLM_LEGACY * self.strength * self.strength * s_size as f32)
    }

    pub(super) fn total_frames(&self) -> u32 {
        1 + 2 * self.temporal_radius
    }

    /// Reject parameter combinations that would either fail to launch
    /// (kernels hitting SMEM/register limits) or produce numerically
    /// degenerate output. Called automatically by `NlmDenoiser::new` —
    /// callers building params manually can invoke it directly to
    /// surface errors before construction.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.patch_radius > MAX_PATCH_RADIUS {
            anyhow::bail!(
                "patch_radius={} exceeds the supported maximum ({}); larger patches \
                 exhaust on-chip SMEM in the fused/windowed kernels",
                self.patch_radius,
                MAX_PATCH_RADIUS,
            );
        }

        if self.search_radius > MAX_SEARCH_RADIUS {
            anyhow::bail!(
                "search_radius={} exceeds the supported maximum ({}); the windowed \
                 kernel allocates `(block + 2·patch_radius + 2·search_radius)²` of SMEM",
                self.search_radius,
                MAX_SEARCH_RADIUS,
            );
        }

        if self.temporal_radius > MAX_TEMPORAL_RADIUS {
            anyhow::bail!(
                "temporal_radius={} exceeds the supported maximum ({}); the ring \
                 buffer grows linearly with the window size",
                self.temporal_radius,
                MAX_TEMPORAL_RADIUS,
            );
        }

        if !(self.strength.is_finite() && self.strength > 0.0) {
            anyhow::bail!(
                "strength must be finite and > 0 (got {}); strength = 0 produces an \
                 infinite Welsch normalisation factor",
                self.strength,
            );
        }

        if !self.self_weight.is_finite() || self.self_weight < 0.0 {
            anyhow::bail!("self_weight must be finite and >= 0 (got {})", self.self_weight,);
        }

        self.motion_compensation.validate()?;

        Ok(())
    }
}
