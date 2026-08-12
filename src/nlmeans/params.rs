use super::{MotionCompensationMode, PrefilterMode, prefilter};

/// SSD normalisation reference, matching FFmpeg's nlmeans (255² for
/// 8-bit normalisation). Distances are computed in `[0, 1]` units so
/// this folds in the implied scale-up.
pub(super) const NLM_NORM: f32 = 255.0 * 255.0;
/// Legacy scaling factor inherited from FFmpeg's nlmeans; preserved so
/// our `strength` parameter has equivalent meaning.
pub(super) const NLM_LEGACY: f32 = 3.0;

/// Measured HQ default `strength` per temporal radius for the luma
/// table, `hq_default_strength`'s `ChannelMode::Luma`/`ChannelMode::Yuv`
/// source. Index is `temporal_radius.min(8)`.
const HQ_DEFAULT_STRENGTH_LUMA: [f32; 9] = [0.45, 0.45, 0.42, 0.42, 0.35, 0.35, 0.35, 0.30, 0.30];

/// Measured HQ default `strength` per temporal radius for the chroma
/// table, `hq_default_strength`'s `ChannelMode::Chroma` source. Index is
/// `temporal_radius.min(8)`.
const HQ_DEFAULT_STRENGTH_CHROMA: [f32; 9] = [1.00, 0.85, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70];

/// Calibrated default `strength` multiplier for `nlmeans-hq`'s
/// auto-strength mode, as a function of which plane is denoised and how
/// far the temporal window reaches.
///
/// Each entry is a measured value, not a fitted curve. Both tables
/// come from quality-harness sweeps that score a strength grid at
/// three noise levels per radius.
///
/// The luma sweep covers every radius 0..=8 directly.
/// The chroma sweep pins luma at each radius's chosen
/// value, so the chroma metrics stay uncontaminated, and covers radii
/// 0, 1, 2, 4, and 8 with a bracketed peak at each.
///
/// Radius 0 peaks at 1.00, radius 1 at 0.85, and radii 2, 4,
/// and 8 all land at 0.70, a plateau that does not keep falling
/// with radius the way luma does.
///
/// Radii 3, 5, 6, and 7 sit on that measured flat plateau between the
/// r2/r4/r8 anchors rather than being swept directly. For both tables
/// the value picked at each measured radius is the one whose smallest
/// XPSNR gain across the tested noise levels (max-min) is largest, so
/// the choice holds up at whichever noise level is hardest to serve.
///
/// Luma stays monotonically non-increasing with radius, since wider
/// temporal windows already gather more samples to average over.
///
/// Chroma drops the same way out to r2 and then holds flat, rather than
/// continuing to decrease.
///
/// `ChannelMode::Yuv` reads the luma table on the assumption that a
/// fused pass is luma-dominant. That mode wasn't part of the sweep, so
/// this is an assumption rather than a separate measurement.
///
/// `temporal_radius` is clamped to the table's last index
/// (`temporal_radius.min(8)`, matching [`MAX_TEMPORAL_RADIUS`]) as a
/// belt-and-braces guard. Callers should already reject radii above the
/// maximum in [`NlmParams::validate`].
pub fn hq_default_strength(channels: ChannelMode, temporal_radius: u32) -> f32 {
    let idx = temporal_radius.min(MAX_TEMPORAL_RADIUS) as usize;
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => HQ_DEFAULT_STRENGTH_LUMA[idx],
        ChannelMode::Chroma => HQ_DEFAULT_STRENGTH_CHROMA[idx],
    }
}

/// Smallest supported frame side length. The Immerkær noise estimate
/// only measures interior pixels (the 3×3 mask cannot reach the
/// one-pixel border), so a frame narrower or shorter than 3 pixels
/// has no interior at all and the estimate is undefined.
pub const MIN_FRAME_DIM: u32 = 3;

/// Reject frame dimensions the kernels cannot handle. Called by both
/// denoiser constructors before any buffer is allocated.
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), anyhow::Error> {
    if width < MIN_FRAME_DIM || height < MIN_FRAME_DIM {
        anyhow::bail!(
            "frame dimensions {width}x{height} are below the supported minimum \
             ({MIN_FRAME_DIM}x{MIN_FRAME_DIM}); the noise estimate needs at least \
             one interior pixel"
        );
    }
    Ok(())
}

/// Patch radius threshold: above this the dispatcher switches to the
/// separable path so the per-pixel cost stays linear in `patch_radius`.
pub(super) const SEPARABLE_THRESHOLD: u32 = 8;

/// Hard ceiling on `patch_radius`. The fused kernels load a
/// `(block + 2·patch_radius)²` SMEM tile; values above this run out of
/// SMEM on RDNA-class GPUs.
pub const MAX_PATCH_RADIUS: u32 = 16;

/// Hard ceiling on `search_radius`. The windowed kernel's SMEM tile is
/// `(block + 2·patch_radius + 2·search_radius)² × stored_ch × 4` bytes,
/// well within hardware SMEM limits at every supported size. The real
/// cost that grows with `search_radius` is the kernel's `#[unroll]`ed
/// `(2·search_radius + 1)²` window loop, whose compiled size and
/// codegen time both scale with it (see the stack-size note in
/// `.cargo/config.toml`). The per-q dispatch path (used when
/// `patch_radius` forces the separable fallback) is gated on this too,
/// so launch counts stay sane (`(2·a+1)²` launches per temporal
/// offset).
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
    /// Multiplier applied to each channel's automatically measured
    /// sigma before it folds into the running estimate. `1.0` keeps
    /// the measurement as-is. Has no effect when `sigma_override` is
    /// set, since the estimation path never runs in that case.
    /// Default: 1.0. The CLI takes this as `--hq-sigma-scale`.
    pub sigma_scale: f32,
}

impl Default for HqParams {
    fn default() -> Self {
        Self {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
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
                "search_radius={} exceeds the supported maximum ({}). The windowed \
                 kernel's search window loop is fully unrolled, so its compiled size \
                 and codegen time both grow with search_radius",
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

        if let Some(hq) = self.hq
            && !(hq.sigma_scale.is_finite() && (0.1..=10.0).contains(&hq.sigma_scale))
        {
            anyhow::bail!(
                "--hq-sigma-scale must be finite and in [0.1, 10.0] (got {})",
                hq.sigma_scale,
            );
        }

        if let PrefilterMode::Bilateral { sigma_s, sigma_r } = self.prefilter {
            if !sigma_s.is_finite() || sigma_s <= 0.0 {
                anyhow::bail!(
                    "bilateral prefilter sigma_s must be finite and > 0 (got {}); \
                     sigma_s = 0 produces an infinite spatial-weight normalisation factor",
                    sigma_s,
                );
            }
            if !sigma_r.is_finite() || sigma_r <= 0.0 {
                anyhow::bail!(
                    "bilateral prefilter sigma_r must be finite and > 0 (got {}); \
                     sigma_r = 0 produces an infinite range-weight normalisation factor \
                     that turns the centre tap into NaN",
                    sigma_r,
                );
            }
            // A `sigma` can be finite and positive yet still be small
            // enough that `sigma * sigma` underflows to `0.0` in f32
            // (true for any positive value below roughly 3.8e-20 here),
            // which makes the reciprocal normalisation factor the
            // kernel actually uses infinite. Checking the same derived
            // factor `run_bilateral` computes for the kernel launch
            // catches that regardless of exactly where the underflow
            // threshold falls, rather than picking a sigma-space cutoff
            // by hand or duplicating the expression here.
            if !prefilter::inv_two_sigma_sq(sigma_s).is_finite() {
                anyhow::bail!(
                    "bilateral prefilter sigma_s is too small (got {}); sigma_s * sigma_s \
                     underflows to 0 in f32, making the spatial-weight normalisation factor \
                     infinite",
                    sigma_s,
                );
            }
            if !prefilter::inv_two_sigma_sq(sigma_r).is_finite() {
                anyhow::bail!(
                    "bilateral prefilter sigma_r is too small (got {}); sigma_r * sigma_r \
                     underflows to 0 in f32, making the range-weight normalisation factor \
                     infinite and the centre tap NaN",
                    sigma_r,
                );
            }
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
                sigma_scale: 1.0,
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
    fn hq_params_default_sigma_scale_is_one() {
        assert_eq!(HqParams::default().sigma_scale, 1.0);
    }

    #[test]
    fn validate_rejects_sigma_scale_below_the_minimum() {
        let params = NlmParams {
            hq: Some(HqParams {
                sigma_scale: 0.05,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        let err = params.validate().expect_err("0.05 is below the 0.1 minimum");
        assert!(
            err.to_string().contains("--hq-sigma-scale"),
            "error should name the flag, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_sigma_scale_above_the_maximum() {
        let params = NlmParams {
            hq: Some(HqParams {
                sigma_scale: 10.5,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_sigma_scale() {
        let params = NlmParams {
            hq: Some(HqParams {
                sigma_scale: f32::NAN,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_sigma_scale_at_the_bounds() {
        let low = NlmParams {
            hq: Some(HqParams {
                sigma_scale: 0.1,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(low.validate().is_ok());

        let high = NlmParams {
            hq: Some(HqParams {
                sigma_scale: 10.0,
                ..HqParams::default()
            }),
            ..NlmParams::default()
        };
        assert!(high.validate().is_ok());
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
                sigma_scale: 1.0,
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
    fn validate_rejects_non_positive_bilateral_sigma_r() {
        // sigma_r = 0 makes inv_two_sigma_r_sq = 1/0 = inf. The centre
        // tap's range_sq is 0, so 0 * inf = NaN, which poisons every
        // pixel of the reference clip.
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: 0.0,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());

        let negative = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: -0.02,
            },
            ..NlmParams::default()
        };
        assert!(negative.validate().is_err());

        let nan = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: f32::NAN,
            },
            ..NlmParams::default()
        };
        assert!(nan.validate().is_err());

        let inf = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: f32::INFINITY,
            },
            ..NlmParams::default()
        };
        assert!(inf.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_positive_bilateral_sigma_s() {
        // sigma_s = 0 makes inv_two_sigma_s_sq = 1/0 = inf; the centre
        // tap's spatial_dist_sq is 0, so the same 0 * inf = NaN poisoning
        // applies to the spatial term.
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 0.0,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());

        let negative = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: -3.0,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(negative.validate().is_err());

        let nan = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: f32::NAN,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(nan.validate().is_err());

        let inf = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: f32::INFINITY,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(inf.validate().is_err());
    }

    #[test]
    fn validate_accepts_positive_finite_bilateral_sigmas() {
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_accepts_a_small_positive_bilateral_sigma_at_the_boundary() {
        // Pins the guard to `<= 0.0` rather than `< 0.0`: a value that is
        // small but strictly positive, and far enough from the f32
        // underflow cliff that `sigma * sigma` stays a normal (non-zero)
        // float, must still be accepted for both fields, independently
        // of the other one. 1e-6 squares to 1e-12, nowhere near the
        // smallest normal f32 (~1.18e-38), so `inv_two_sigma_sq` stays
        // finite here.
        let safe_small = 1e-6_f32;
        assert!(
            (safe_small * safe_small).is_normal(),
            "test value itself must not underflow"
        );

        let small_sigma_s = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: safe_small,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(small_sigma_s.validate().is_ok());

        let small_sigma_r = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: safe_small,
            },
            ..NlmParams::default()
        };
        assert!(small_sigma_r.validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_subnormal_bilateral_sigma_that_underflows_on_squaring() {
        // `f32::MIN_POSITIVE` (the smallest *normal* positive f32,
        // ~1.1754944e-38) is finite and > 0, so the old `is_finite() &&
        // > 0.0` guard on the raw value let it through. But squaring it
        // underflows to exactly `0.0` in f32 (its true square,
        // ~1.38e-76, is far below the smallest subnormal, ~1.4e-45), so
        // `inv_two_sigma_sq` — `1.0 / (2.0 * sigma * sigma)` — divides
        // by zero and produces `inf`. That's the exact NaN-poisoning
        // failure mode this validation exists to prevent, just reached
        // through a different sigma than exactly `0.0`.
        let sq = f32::MIN_POSITIVE * f32::MIN_POSITIVE;
        assert_eq!(
            sq, 0.0,
            "test assumption: MIN_POSITIVE must underflow on squaring"
        );
        let inv = prefilter::inv_two_sigma_sq(f32::MIN_POSITIVE);
        assert!(
            !inv.is_finite(),
            "test assumption: the derived factor must be infinite here"
        );

        let sigma_s = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: f32::MIN_POSITIVE,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(
            sigma_s.validate().is_err(),
            "a subnormal sigma_s that underflows to an infinite normalisation factor must be rejected"
        );

        let sigma_r = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: f32::MIN_POSITIVE,
            },
            ..NlmParams::default()
        };
        assert!(
            sigma_r.validate().is_err(),
            "a subnormal sigma_r that underflows to an infinite normalisation factor must be rejected"
        );
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

    #[test]
    fn hq_default_strength_matches_the_measured_luma_table() {
        const EXPECTED: [f32; 9] = [0.45, 0.45, 0.42, 0.42, 0.35, 0.35, 0.35, 0.30, 0.30];
        for (radius, &expected) in EXPECTED.iter().enumerate() {
            let got = hq_default_strength(ChannelMode::Luma, radius as u32);
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "radius={radius}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn hq_default_strength_matches_the_measured_chroma_table() {
        const EXPECTED: [f32; 9] = [1.00, 0.85, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70];
        for (radius, &expected) in EXPECTED.iter().enumerate() {
            let got = hq_default_strength(ChannelMode::Chroma, radius as u32);
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "radius={radius}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn hq_default_strength_yuv_reads_the_luma_table() {
        for radius in 0..=8u32 {
            let yuv = hq_default_strength(ChannelMode::Yuv, radius);
            let luma = hq_default_strength(ChannelMode::Luma, radius);
            assert!(
                (yuv - luma).abs() < f32::EPSILON,
                "radius={radius}: yuv={yuv}, luma={luma}"
            );
        }
    }

    #[test]
    fn validate_dimensions_rejects_frames_below_the_minimum() {
        assert!(validate_dimensions(2, 64).is_err());
        assert!(validate_dimensions(64, 2).is_err());
        assert!(validate_dimensions(0, 0).is_err());
    }

    #[test]
    fn validate_dimensions_accepts_the_minimum() {
        assert!(validate_dimensions(MIN_FRAME_DIM, MIN_FRAME_DIM).is_ok());
        assert!(validate_dimensions(1920, 1080).is_ok());
    }

    #[test]
    fn hq_default_strength_clamps_radius_above_the_table() {
        let at_max = hq_default_strength(ChannelMode::Luma, MAX_TEMPORAL_RADIUS);
        let above_max = hq_default_strength(ChannelMode::Luma, MAX_TEMPORAL_RADIUS + 5);
        assert!(
            (at_max - above_max).abs() < f32::EPSILON,
            "expected clamping to hold the last table entry, got {at_max} vs {above_max}"
        );
    }
}
