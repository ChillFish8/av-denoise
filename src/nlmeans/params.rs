use super::{MotionCompensationMode, PrefilterMode};

/// SSD normalisation reference, matching FFmpeg's nlmeans (255² for
/// 8-bit normalisation). Distances are computed in `[0, 1]` units so
/// this folds in the implied scale-up.
pub(super) const NLM_NORM: f32 = 255.0 * 255.0;
/// Legacy scaling factor inherited from FFmpeg's nlmeans; preserved so
/// our `strength` parameter has equivalent meaning.
pub(super) const NLM_LEGACY: f32 = 3.0;

/// Calibrated default `strength` multiplier for `nlmeans-hq`'s
/// auto-strength mode. A sweep across noise levels found a flat XPSNR
/// plateau with the luma optimum near 0.42 and the chroma optimum near
/// 0.5. This value sits on that measured plateau between the two.
pub const HQ_DEFAULT_STRENGTH: f32 = 0.45;

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
    /// `strength × sigma_eff × 255`. Default: true.
    pub auto_strength: bool,
    /// Subtract the expected noise floor from patch distances before
    /// weighting, so matches are not penalised for the noise they
    /// carry. Default: true.
    pub noise_floor: bool,
    /// Fixed noise standard deviation in `[0, 1]` units, overriding
    /// automatic per-frame estimation. `None` (the default) measures
    /// noise from each pushed frame and smooths it over time. `Some`
    /// applies one fixed value to every frame instead. The CLI takes
    /// this in 8-bit units via `--hq-sigma` and divides by 255.
    pub sigma_override: Option<f32>,
    /// Weight temporal neighbours by how well they block-match the
    /// centre frame, so occlusion or content change collapses their
    /// contribution instead of blurring it in. Only takes effect when
    /// `temporal_radius > 0`. Default: true.
    pub temporal_confidence: bool,
    /// Multiplier on the per-pixel mismatch threshold that decides how
    /// much excess SAD a block tolerates before its confidence starts
    /// dropping. Higher values tolerate larger mismatches. Default: 1.0.
    pub thsad_scale: f32,
}

impl Default for HqParams {
    fn default() -> Self {
        Self {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
        }
    }
}

impl HqParams {
    /// HQ defaults with a fixed noise level in `[0, 1]` units, skipping
    /// automatic estimation.
    pub fn with_sigma(sigma: f32) -> Self {
        Self {
            sigma_override: Some(sigma),
            ..Self::default()
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
    /// one setting tracks differently noisy sources. `sigma_eff` is
    /// the scale-weighted RMS of the per-channel noise estimates.
    pub(super) fn effective_strength_with(&self, sigma_eff: Option<f32>) -> f32 {
        match (self.hq, sigma_eff) {
            (Some(hq), Some(sigma)) if hq.auto_strength => self.strength * sigma * 255.0,
            _ => self.strength,
        }
    }

    /// `h2_inv_norm` for an explicit noise estimate, bypassing whatever
    /// `self.hq.sigma_override` holds. Used by the denoiser to refresh
    /// the derived value from a freshly measured sigma each submit.
    pub fn h2_inv_norm_with(&self, sigma_eff: Option<f32>) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        let s = self.effective_strength_with(sigma_eff);
        NLM_NORM / (NLM_LEGACY * s * s * s_size as f32)
    }

    /// `h2_inv_norm` using `self.hq.sigma_override` as the noise
    /// estimate (or none, for the fast path). Auto-estimating HQ
    /// denoisers recompute with [`Self::h2_inv_norm_with`] each submit
    /// instead of calling this.
    pub fn h2_inv_norm(&self) -> f32 {
        self.h2_inv_norm_with(self.hq.and_then(|hq| hq.sigma_override))
    }

    /// Expected box-summed patch distance between two noisy copies of
    /// identical content, for an explicit set of per-channel sigma
    /// estimates. Each active channel contributes `2 × channel_scale ×
    /// σ_c²`, summed over `(2s+1)²` taps. Zero without HQ noise floor
    /// or without an estimate to apply.
    pub(super) fn noise_offset_with(&self, sigmas: Option<&[f32]>) -> f32 {
        match (self.hq, sigmas) {
            (Some(hq), Some(sigmas)) if hq.noise_floor => {
                let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
                let scale = channel_scale(self.channels);
                let count = self.channels.count() as usize;
                let sum_sq: f32 = sigmas.iter().take(count).map(|&s| s * s).sum();
                2.0 * scale * sum_sq * s_size as f32
            },
            _ => 0.0,
        }
    }

    /// `noise_offset` using `self.hq.sigma_override` applied to every
    /// active channel (or none, for the fast path). Auto-estimating HQ
    /// denoisers recompute with [`Self::noise_offset_with`] each submit
    /// instead of calling this.
    pub(super) fn noise_offset(&self) -> f32 {
        match self.hq.and_then(|hq| hq.sigma_override) {
            Some(sigma) => {
                let sigmas = [sigma; 3];
                self.noise_offset_with(Some(&sigmas[..self.channels.count() as usize]))
            },
            None => 0.0,
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

        if let Some(hq) = self.hq
            && let Some(sigma) = hq.sigma_override
            && (!sigma.is_finite() || sigma <= 0.0 || sigma > 1.0)
        {
            anyhow::bail!(
                "hq sigma_override must be finite and in (0, 1] in normalised units (got {})",
                sigma,
            );
        }

        if let Some(hq) = self.hq
            && !(hq.thsad_scale.is_finite() && hq.thsad_scale > 0.0)
        {
            anyhow::bail!(
                "hq thsad_scale must be finite and > 0 (got {}); thsad_scale = 0 collapses \
                 every block's confidence to zero regardless of match quality",
                hq.thsad_scale,
            );
        }

        if let PrefilterMode::NlmSpatial { strength_scale } = self.prefilter {
            if !strength_scale.is_finite() || strength_scale <= 0.0 {
                anyhow::bail!(
                    "nlm pilot strength_scale must be finite and > 0 (got {})",
                    strength_scale,
                );
            }
            if self.patch_radius > SEPARABLE_THRESHOLD {
                anyhow::bail!(
                    "the nlm pilot uses the windowed spatial kernel, which supports \
                     patch_radius up to {} (got {})",
                    SEPARABLE_THRESHOLD,
                    self.patch_radius,
                );
            }
        }

        self.motion_compensation.validate()?;

        Ok(())
    }
}

/// Per-channel distance scale for a channel mode (luma×3, chroma×1.5,
/// full YUV×1), matching the GPU-side `channel_scale` used inside the
/// weighting kernels. Uniform across every channel in a given mode.
pub(super) fn channel_scale(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma => 3.0,
        ChannelMode::Chroma => 1.5,
        ChannelMode::Yuv => 1.0,
    }
}

/// Scale-weighted RMS of the per-channel noise estimates over a
/// channel mode's active channel count. Because `channel_scale` is
/// uniform within a mode, the weighting cancels and this reduces to a
/// plain RMS. Extra elements in `sigmas` beyond the mode's channel
/// count are ignored.
pub(super) fn sigma_eff(sigmas: &[f32], channels: ChannelMode) -> f32 {
    let count = channels.count() as usize;
    let sum_sq: f32 = sigmas.iter().take(count).map(|&s| s * s).sum();
    (sum_sq / count as f32).sqrt()
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
            (params.noise_offset() - expected).abs() < 1e-6,
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
                sigma_override: Some(4.0 / 255.0),
                temporal_confidence: true,
                thsad_scale: 1.0,
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

    #[test]
    fn validate_rejects_zero_thsad_scale() {
        let params = NlmParams {
            hq: Some(HqParams {
                thsad_scale: 0.0,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_thsad_scale() {
        let params = NlmParams {
            hq: Some(HqParams {
                thsad_scale: -1.0,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_thsad_scale() {
        let params = NlmParams {
            hq: Some(HqParams {
                thsad_scale: f32::NAN,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_default_thsad_scale() {
        let params = NlmParams {
            hq: Some(HqParams::default()),
            ..NlmParams::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn noise_offset_with_handles_distinct_per_channel_sigmas() {
        let sigma_u = 4.0 / 255.0;
        let sigma_v = 10.0 / 255.0;
        let params = NlmParams {
            patch_radius: 4,
            channels: ChannelMode::Chroma,
            hq: Some(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
            }),
            ..NlmParams::default()
        };

        let s_size = (2 * params.patch_radius + 1) * (2 * params.patch_radius + 1);
        // Chroma scale is 1.5, applied per channel; each channel keeps
        // its own sigma instead of a shared value.
        let expected = 2.0 * 1.5 * (sigma_u * sigma_u + sigma_v * sigma_v) * s_size as f32;

        let got = params.noise_offset_with(Some(&[sigma_u, sigma_v]));
        assert!((got - expected).abs() < 1e-9, "expected {expected}, got {got}");
    }

    #[test]
    fn sigma_eff_is_rms_over_active_channels() {
        let sigmas = [3.0 / 255.0, 4.0 / 255.0];
        let got = sigma_eff(&sigmas, ChannelMode::Chroma);
        let expected = ((sigmas[0] * sigmas[0] + sigmas[1] * sigmas[1]) / 2.0).sqrt();
        assert!((got - expected).abs() < 1e-9, "expected {expected}, got {got}");
    }

    #[test]
    fn validate_rejects_non_positive_pilot_strength_scale() {
        let zero = NlmParams {
            prefilter: PrefilterMode::NlmSpatial { strength_scale: 0.0 },
            ..NlmParams::default()
        };
        assert!(zero.validate().is_err());

        let nan = NlmParams {
            prefilter: PrefilterMode::NlmSpatial {
                strength_scale: f32::NAN,
            },
            ..NlmParams::default()
        };
        assert!(nan.validate().is_err());
    }

    #[test]
    fn validate_rejects_pilot_with_patch_radius_above_separable_threshold() {
        let params = NlmParams {
            prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
            patch_radius: SEPARABLE_THRESHOLD + 1,
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_pilot_within_limits() {
        let params = NlmParams {
            prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
            patch_radius: SEPARABLE_THRESHOLD,
            ..NlmParams::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn sigma_eff_ignores_channels_past_the_mode_count() {
        // Luma only looks at the first element even when given extra
        // (chroma) samples.
        let sigmas = [6.0 / 255.0, 100.0 / 255.0, 200.0 / 255.0];
        let got = sigma_eff(&sigmas, ChannelMode::Luma);
        assert!(
            (got - sigmas[0]).abs() < 1e-9,
            "expected {}, got {got}",
            sigmas[0]
        );
    }
}
