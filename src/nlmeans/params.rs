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

/// Parameters for the quality-focused `nlmeans-hq` variant. The noise
/// level drives both the effective strength and the distance floor,
/// so weighting adapts to how noisy the source actually is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HqParams {
    /// Interpret `strength` as a multiplier on the noise level. The
    /// effective FFmpeg-style strength becomes
    /// `strength × sigma × 255`. Default: true.
    pub auto_strength: bool,
    /// Subtract the expected noise floor from patch distances before
    /// weighting, so matches are not penalised for the noise they
    /// carry. Default: true.
    pub noise_floor: bool,
    /// Noise standard deviation in `[0, 1]` units. Required. The CLI
    /// takes this in 8-bit units via `--hq-sigma` and divides by 255.
    pub sigma: f32,
}

impl HqParams {
    /// HQ defaults for a known noise level in `[0, 1]` units.
    pub fn with_sigma(sigma: f32) -> Self {
        Self {
            auto_strength: true,
            noise_floor: true,
            sigma,
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
    /// Quality-mode parameters. `None` runs the fast path unchanged.
    pub hq: Option<HqParams>,
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
            hq: None,
        }
    }
}

impl NlmParams {
    /// FFmpeg-style strength actually used for weighting. With HQ
    /// auto strength the user value multiplies the noise level, so
    /// one setting tracks differently noisy sources.
    pub(super) fn effective_strength(&self) -> f32 {
        match self.hq {
            Some(hq) if hq.auto_strength => self.strength * hq.sigma * 255.0,
            _ => self.strength,
        }
    }

    pub fn h2_inv_norm(&self) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        let s = self.effective_strength();
        NLM_NORM / (NLM_LEGACY * s * s * s_size as f32)
    }

    /// Expected box-summed patch distance between two noisy copies of
    /// identical content. Each summed term contributes
    /// `channel_scale × channels × 2σ²`, which is `6σ²` in every
    /// channel mode, over `(2s+1)²` taps. Zero without HQ noise floor.
    pub(super) fn noise_offset(&self) -> f32 {
        match self.hq {
            Some(hq) if hq.noise_floor => {
                let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
                6.0 * hq.sigma * hq.sigma * s_size as f32
            },
            _ => 0.0,
        }
    }

    pub(super) fn total_frames(&self) -> u32 {
        1 + 2 * self.temporal_radius
    }

    /// Reject parameter combinations that would either fail to launch
    /// (kernels hitting SMEM/register limits) or produce numerically
    /// degenerate output. Called automatically by `NlmDenoiser::new`;
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

        if let Some(hq) = self.hq {
            if !hq.sigma.is_finite() || hq.sigma <= 0.0 || hq.sigma > 1.0 {
                anyhow::bail!(
                    "hq sigma must be finite and in (0, 1] in normalised units (got {})",
                    hq.sigma,
                );
            }
        }

        self.motion_compensation.validate()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_offset_scales_with_sigma_and_patch_size() {
        let sigma = 4.0 / 255.0;
        let params = NlmParams {
            patch_radius: 4,
            hq: Some(HqParams::with_sigma(sigma)),
            ..NlmParams::default()
        };

        let expected = 6.0 * sigma * sigma * 81.0;
        assert!(
            (params.noise_offset() - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            params.noise_offset()
        );
    }

    #[test]
    fn noise_offset_zero_without_noise_floor() {
        let params = NlmParams {
            hq: Some(HqParams {
                auto_strength: true,
                noise_floor: false,
                sigma: 4.0 / 255.0,
            }),
            ..NlmParams::default()
        };

        assert_eq!(params.noise_offset(), 0.0);
    }

    #[test]
    fn noise_offset_zero_without_hq() {
        let params = NlmParams::default();
        assert_eq!(params.noise_offset(), 0.0);
    }

    #[test]
    fn h2_inv_norm_with_auto_strength_matches_hand_computed() {
        let sigma = 8.0 / 255.0;
        let params = NlmParams {
            strength: 1.0,
            hq: Some(HqParams::with_sigma(sigma)),
            ..NlmParams::default()
        };

        let s_size = (2 * params.patch_radius + 1) * (2 * params.patch_radius + 1);
        let effective_strength = 1.0 * sigma * 255.0;
        let expected = NLM_NORM / (NLM_LEGACY * effective_strength * effective_strength * s_size as f32);

        assert!(
            (params.h2_inv_norm() - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            params.h2_inv_norm()
        );
    }

    #[test]
    fn validate_rejects_zero_hq_sigma() {
        let params = NlmParams {
            hq: Some(HqParams::with_sigma(0.0)),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_hq_sigma_above_one() {
        let params = NlmParams {
            hq: Some(HqParams::with_sigma(1.5)),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_hq_sigma() {
        let params = NlmParams {
            hq: Some(HqParams::with_sigma(f32::NAN)),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }
}
