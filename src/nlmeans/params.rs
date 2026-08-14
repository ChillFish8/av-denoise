use super::{MotionCompensationMode, PrefilterMode, prefilter};

/// The reference value patch distances are normalised against, matching
/// FFmpeg's nlmeans at 255 squared.
///
/// Distances here are measured in `[0, 1]` units, so this constant folds
/// in the scale-up back to 8-bit terms.
pub(super) const NLM_NORM: f32 = 255.0 * 255.0;

/// A scaling factor inherited from FFmpeg's nlmeans, kept so our
/// `strength` parameter means the same thing theirs does.
pub(super) const NLM_LEGACY: f32 = 3.0;

/// The measured HQ default `strength` for luma at each temporal radius,
/// indexed by `temporal_radius.min(8)`.
///
/// `hq_default_strength` reads this for `ChannelMode::Luma` and
/// `ChannelMode::Yuv`.
const HQ_DEFAULT_STRENGTH_LUMA: [f32; 9] = [0.45, 0.45, 0.42, 0.42, 0.35, 0.35, 0.35, 0.30, 0.30];

/// The measured HQ default `strength` for chroma at each temporal
/// radius, indexed by `temporal_radius.min(8)`.
///
/// `hq_default_strength` reads this for `ChannelMode::Chroma`.
const HQ_DEFAULT_STRENGTH_CHROMA: [f32; 9] = [1.00, 0.85, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70];

/// The calibrated default `strength` multiplier for `nlmeans-hq`'s
/// auto-strength mode.
///
/// The answer depends on which plane is being denoised and how far the
/// temporal window reaches.
///
/// # Where the numbers come from
///
/// Every entry is a measured value rather than a fitted curve. Both
/// tables come from quality-harness sweeps that score a grid of
/// strengths at three noise levels per radius.
///
/// The luma sweep covers each radius from 0 to 8 directly.
///
/// The chroma sweep pins luma at the value already chosen for that
/// radius, so the chroma numbers stay clean, and covers radii 0, 1, 2,
/// 4, and 8 with a bracketed peak at each.
///
/// At each measured radius the chosen value is the one whose worst XPSNR
/// gain across the tested noise levels is highest, so it holds up at
/// whichever noise level is hardest to serve.
///
/// # Shape of the tables
///
/// Luma never rises with radius, because a wider temporal window already
/// gathers more samples to average over.
///
/// Chroma falls the same way out to radius 2 and then holds flat at
/// 0.70. Radii 3, 5, 6, and 7 sit on that measured plateau rather than
/// being swept directly.
///
/// `ChannelMode::Yuv` reads the luma table, on the assumption that a
/// fused pass is dominated by luma. That mode was not part of the sweep,
/// so this is an assumption rather than a measurement.
///
/// # Clamping
///
/// `temporal_radius` is clamped to the last table index, which matches
/// [`MAX_TEMPORAL_RADIUS`]. That is only a safety net, because
/// [`NlmParams::validate`] already rejects anything larger.
pub fn hq_default_strength(channels: ChannelMode, temporal_radius: u32) -> f32 {
    let idx = temporal_radius.min(MAX_TEMPORAL_RADIUS) as usize;
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => HQ_DEFAULT_STRENGTH_LUMA[idx],
        ChannelMode::Chroma => HQ_DEFAULT_STRENGTH_CHROMA[idx],
    }
}

/// The smallest frame side length the denoiser supports.
///
/// The Immerkær noise estimate only reads interior pixels, because its
/// 3x3 mask cannot reach the one-pixel border. A frame under 3 pixels
/// across has no interior at all, which leaves the estimate undefined.
pub const MIN_FRAME_DIM: u32 = 3;

/// Rejects frame dimensions the kernels cannot handle.
///
/// Both denoiser constructors call this before allocating any buffer.
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), anyhow::Error> {
    if width < MIN_FRAME_DIM || height < MIN_FRAME_DIM {
        anyhow::bail!(
            "frame dimensions {width}x{height} are below the supported minimum of \
             {MIN_FRAME_DIM}x{MIN_FRAME_DIM}, because the noise estimate needs at \
             least one interior pixel"
        );
    }
    Ok(())
}

/// The patch radius above which the dispatcher switches to the separable
/// path, so per-pixel cost stays linear in `patch_radius`.
pub(super) const SEPARABLE_THRESHOLD: u32 = 8;

/// The hard ceiling on `patch_radius`.
///
/// The fused kernels load a `(block + 2 * patch_radius)^2` tile into
/// shared memory, and anything larger runs out of it on RDNA-class GPUs.
pub const MAX_PATCH_RADIUS: u32 = 16;

/// The hard ceiling on `search_radius`.
///
/// Shared memory is not the limit here. The windowed kernel's tile is
/// `(block + 2 * patch_radius + 2 * search_radius)^2 * stored_ch * 4`
/// bytes, which stays comfortably inside hardware limits at every
/// supported size.
///
/// The real cost is the kernel's fully unrolled
/// `(2 * search_radius + 1)^2` window loop. Both its compiled size and
/// how long it takes to generate grow with the radius. See the
/// stack-size note in `.cargo/config.toml`.
///
/// The per-offset dispatch path, used when `patch_radius` forces the
/// separable fallback, is limited by the same constant, which keeps its
/// `(2 * search_radius + 1)^2` launches per temporal offset reasonable.
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// The hard ceiling on `temporal_radius`.
///
/// The ring buffer holds `2 * radius + 1` frames, so device memory grows
/// with it. At radius 16, 1080p YUV would need roughly 540 MB for the
/// input alone.
pub const MAX_TEMPORAL_RADIUS: u32 = 8;

/// The hard ceiling on the radius the bilateral prefilter derives from
/// `sigma_s`, where `radius = ceil(2 * sigma_s).max(1)`. See
/// `prefilter::bilateral_radius`.
///
/// `nlm_bilateral` loads a `(32 + 2r) x (8 + 2r)` tile of
/// `Vector<f32, N>` into shared memory. `N` reaches 4 for YUV storage at
/// 4 bytes per `f32`, so the tile costs `16 * (32 + 2r) * (8 + 2r)`
/// bytes.
///
/// RDNA-class hardware gives 64 KiB of shared memory to work with. At
/// `r = 22` the tile is 63,232 bytes, or 61.75 KiB, which fits. At
/// `r = 23` it is 67,392 bytes, or 65.8 KiB, which does not.
///
/// So 22 is the largest radius that fits, and `bilateral_radius` reaches
/// it at `sigma_s = 11.0`.
pub const MAX_BILATERAL_RADIUS: u32 = 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Which channels of a frame the denoiser works on.
pub enum ChannelMode {
    /// The single brightness channel, with distances scaled by 3.0.
    Luma,
    /// The two colour channels, U and V, with distances scaled by 1.5.
    Chroma,
    /// All three channels together, with distances left unscaled.
    Yuv,
}

impl ChannelMode {
    /// How many channels take part in the distance and the output.
    pub fn count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 3,
        }
    }

    /// How many channels each pixel occupies in GPU storage.
    ///
    /// This is padded up to the next supported vector width so kernels
    /// can read whole `Line<f32>` values at once. Backends only support
    /// power-of-two widths, so YUV pads from 3 up to 4.
    pub fn storage_count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 4,
        }
    }
}

/// Parameters for the quality-focused `nlmeans-hq` variant.
///
/// The measured noise level drives both the effective strength and the
/// distance floor, so the weighting adapts to how noisy the source
/// really is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HqParams {
    /// Reads `strength` as a multiplier on the noise level, making the
    /// effective FFmpeg-style strength `strength * sigma_eff * 255`.
    ///
    /// Defaults to true.
    pub auto_strength: bool,
    /// Subtracts the expected noise floor from patch distances before
    /// weighting, so a match is not penalised for the noise it carries.
    ///
    /// Defaults to true.
    pub noise_floor: bool,
    /// A fixed noise standard deviation in `[0, 1]` units, replacing the
    /// automatic per-frame estimate.
    ///
    /// `None`, the default, measures the noise in each pushed frame and
    /// smooths it over time. `Some` applies one fixed value to every
    /// frame instead.
    ///
    /// The CLI takes this in 8-bit units through `--hq-sigma` and
    /// divides by 255.
    pub sigma_override: Option<f32>,
    /// Weights each temporal neighbour by how well it block-matches the
    /// centre frame, so occlusion or a change of content collapses its
    /// contribution rather than blurring it in.
    ///
    /// Only has an effect when `temporal_radius` is above 0. Defaults to
    /// true.
    pub temporal_confidence: bool,
    /// A multiplier on the per-pixel mismatch threshold, which sets how
    /// much extra SAD a block tolerates before its confidence starts to
    /// fall.
    ///
    /// Higher values tolerate larger mismatches. Defaults to 1.0.
    pub thsad_scale: f32,
    /// A multiplier applied to each channel's measured sigma before it
    /// folds into the running estimate.
    ///
    /// `1.0`, the default, keeps the measurement as it is. This does
    /// nothing when `sigma_override` is set, because the estimator never
    /// runs in that case.
    ///
    /// The CLI takes this as `--hq-sigma-scale`.
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
    /// The HQ defaults with a fixed noise level in `[0, 1]` units, which
    /// skips automatic estimation.
    pub fn with_sigma(sigma: f32) -> Self {
        Self {
            sigma_override: Some(sigma),
            ..Self::default()
        }
    }
}

/// The low-level parameters a denoiser is built from.
#[derive(Debug, Clone)]
pub struct NlmParams {
    /// How many frames on each side of the current one to look at.
    ///
    /// 0 means each frame is cleaned on its own. Anything higher uses a
    /// window of `2 * radius + 1` frames.
    pub temporal_radius: u32,
    /// Half the width of the search window, which covers
    /// `(2 * radius + 1)^2` pixels. Defaults to 2.
    pub search_radius: u32,
    /// Half the width of a compared patch, which covers
    /// `(2 * radius + 1)^2` pixels. Defaults to 4.
    pub patch_radius: u32,
    /// How hard to filter. Higher values smooth more. Defaults to 1.2.
    pub strength: f32,
    /// How much weight the centre pixel gets in the average.
    ///
    /// Defaults to 1.0. Set it to 0 for pure NLM, where the centre pixel
    /// only counts through the patches that match it.
    pub self_weight: f32,
    /// Which channels to process.
    pub channels: ChannelMode,
    /// Which image the patch distances are measured against.
    ///
    /// Defaults to `None`. When set, the weights come from a prefiltered
    /// or externally supplied image while the pixels being averaged
    /// still come from the original input.
    pub prefilter: PrefilterMode,
    /// Whether temporal denoising follows motion between frames.
    ///
    /// Defaults to `None`. Set to `Mvtools`, each submit warps the
    /// temporal neighbours into line with the centre frame before the
    /// NLM weighting runs.
    ///
    /// Only has an effect when `temporal_radius` is above 0.
    pub motion_compensation: MotionCompensationMode,
    /// The quality-mode parameters. `None` runs the fast path unchanged.
    pub hq: Option<HqParams>,
    /// Whether the accumulators also track each pixel's sum of squared
    /// neighbour weights, alongside the sum of weights they already
    /// keep.
    ///
    /// Defaults to false. A second-stage filter that shrinks its
    /// coefficients by how much noise this pass left behind needs that
    /// second sum, read back through
    /// [`crate::nlmeans::NlmDenoiser::residual_ratio_sqrt`]. Nothing else
    /// reads it, so every kernel drops the extra write at compile time
    /// while this stays false.
    pub track_weight_sq: bool,
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
            track_weight_sq: false,
        }
    }
}

impl NlmParams {
    /// The FFmpeg-style strength the weighting actually uses.
    ///
    /// With HQ auto-strength the user's value multiplies the noise
    /// level, so one setting follows sources of different noisiness.
    ///
    /// `sigma_eff` is the scale-weighted RMS of the per-channel noise
    /// estimates.
    pub(super) fn effective_strength_with(&self, sigma_eff: Option<f32>) -> f32 {
        match (self.hq, sigma_eff) {
            (Some(hq), Some(sigma)) if hq.auto_strength => self.strength * sigma * 255.0,
            _ => self.strength,
        }
    }

    /// `h2_inv_norm` for a noise estimate given here, ignoring whatever
    /// `self.hq.sigma_override` holds.
    ///
    /// The denoiser calls this each submit to refresh the value from a
    /// freshly measured sigma.
    pub fn h2_inv_norm_with(&self, sigma_eff: Option<f32>) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        let s = self.effective_strength_with(sigma_eff);
        NLM_NORM / (NLM_LEGACY * s * s * s_size as f32)
    }

    /// `h2_inv_norm` using `self.hq.sigma_override` as the noise
    /// estimate, or no estimate at all on the fast path.
    ///
    /// HQ denoisers that estimate noise automatically call
    /// [`Self::h2_inv_norm_with`] each submit instead.
    pub fn h2_inv_norm(&self) -> f32 {
        self.h2_inv_norm_with(self.hq.and_then(|hq| hq.sigma_override))
    }

    /// The patch distance two noisy copies of the same content are
    /// expected to show, for a given set of per-channel sigmas.
    ///
    /// Each active channel contributes `2 * channel_scale * sigma^2`,
    /// summed over all `(2 * patch_radius + 1)^2` taps.
    ///
    /// Returns 0 when the HQ noise floor is off, or when there is no
    /// estimate to apply.
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

    /// `noise_offset` with `self.hq.sigma_override` applied to every
    /// active channel, or no estimate at all on the fast path.
    ///
    /// HQ denoisers that estimate noise automatically call
    /// [`Self::noise_offset_with`] each submit instead.
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

    /// Rejects parameter combinations that would fail to launch, by
    /// running the kernels past their shared-memory or register limits,
    /// or that would produce meaningless output.
    ///
    /// `NlmDenoiser::new` calls this for you. Callers building params by
    /// hand can call it directly to see errors before construction.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.patch_radius > MAX_PATCH_RADIUS {
            anyhow::bail!(
                "patch_radius={} exceeds the supported maximum of {}, because larger \
                 patches exhaust on-chip shared memory in the fused and windowed kernels",
                self.patch_radius,
                MAX_PATCH_RADIUS,
            );
        }

        if self.search_radius > MAX_SEARCH_RADIUS {
            anyhow::bail!(
                "search_radius={} exceeds the supported maximum of {}. The windowed \
                 kernel's search window loop is fully unrolled, so both its compiled \
                 size and its build time grow with search_radius",
                self.search_radius,
                MAX_SEARCH_RADIUS,
            );
        }

        if self.temporal_radius > MAX_TEMPORAL_RADIUS {
            anyhow::bail!(
                "temporal_radius={} exceeds the supported maximum of {}, because the \
                 ring buffer grows in step with the window size",
                self.temporal_radius,
                MAX_TEMPORAL_RADIUS,
            );
        }

        if !(self.strength.is_finite() && self.strength > 0.0) {
            anyhow::bail!(
                "strength must be finite and greater than 0, got {}. A strength of 0 \
                 produces an infinite Welsch normalisation factor",
                self.strength,
            );
        }

        if !self.self_weight.is_finite() || self.self_weight < 0.0 {
            anyhow::bail!(
                "self_weight must be finite and 0 or greater, got {}",
                self.self_weight,
            );
        }

        if let Some(hq) = self.hq
            && let Some(sigma) = hq.sigma_override
            && (!sigma.is_finite() || sigma <= 0.0 || sigma > 1.0)
        {
            anyhow::bail!(
                "hq sigma_override must be finite and in (0, 1] in normalised units, got {}",
                sigma,
            );
        }

        if let Some(hq) = self.hq
            && !(hq.thsad_scale.is_finite() && hq.thsad_scale > 0.0)
        {
            anyhow::bail!(
                "hq thsad_scale must be finite and greater than 0, got {}. A thsad_scale \
                 of 0 collapses every block's confidence to zero no matter how well it \
                 matches",
                hq.thsad_scale,
            );
        }

        if let Some(hq) = self.hq
            && !(hq.sigma_scale.is_finite() && (0.1..=10.0).contains(&hq.sigma_scale))
        {
            anyhow::bail!(
                "hq sigma_scale must be finite and in [0.1, 10.0], got {}",
                hq.sigma_scale,
            );
        }

        if let PrefilterMode::Bilateral { sigma_s, sigma_r } = self.prefilter {
            if !sigma_s.is_finite() || sigma_s <= 0.0 {
                anyhow::bail!(
                    "bilateral prefilter sigma_s must be finite and greater than 0, got \
                     {}. A sigma_s of 0 produces an infinite spatial-weight \
                     normalisation factor",
                    sigma_s,
                );
            }
            if !sigma_r.is_finite() || sigma_r <= 0.0 {
                anyhow::bail!(
                    "bilateral prefilter sigma_r must be finite and greater than 0, got \
                     {}. A sigma_r of 0 produces an infinite range-weight normalisation \
                     factor, which turns the centre tap into NaN",
                    sigma_r,
                );
            }
            // sigma_s decides the shared-memory tile radius through
            // `prefilter::bilateral_radius`. Checking that derived
            // radius, rather than working out an equivalent sigma_s
            // threshold here, keeps this in step if the formula ever
            // changes.
            //
            // A very large sigma_s can also overflow the radius inside
            // the tile-size expression. This check catches that too,
            // because an overflowed radius always lands far past the
            // maximum.
            let bilateral_radius = prefilter::bilateral_radius(sigma_s);
            if bilateral_radius > MAX_BILATERAL_RADIUS {
                anyhow::bail!(
                    "bilateral prefilter sigma_s={} implies a shared-memory tile radius \
                     of {}, from radius = ceil(2 * sigma_s) with a minimum of 1. That \
                     is past the supported maximum of {}, and larger radii exhaust \
                     on-chip shared memory in the bilateral kernel",
                    sigma_s,
                    bilateral_radius,
                    MAX_BILATERAL_RADIUS,
                );
            }
            // A sigma can be finite and positive yet small enough that
            // `sigma * sigma` underflows to 0.0 in f32, which happens
            // below roughly 3.8e-20. That makes the reciprocal
            // normalisation factor the kernel uses infinite.
            //
            // Checking the same derived factor `run_bilateral` computes
            // for the launch catches this wherever the underflow
            // threshold actually falls, without picking a sigma cutoff
            // by hand or repeating the expression here.
            if !prefilter::inv_two_sigma_sq(sigma_s).is_finite() {
                anyhow::bail!(
                    "bilateral prefilter sigma_s is too small, got {}. Squaring it \
                     underflows to 0 in f32, which makes the spatial-weight \
                     normalisation factor infinite",
                    sigma_s,
                );
            }
            if !prefilter::inv_two_sigma_sq(sigma_r).is_finite() {
                anyhow::bail!(
                    "bilateral prefilter sigma_r is too small, got {}. Squaring it \
                     underflows to 0 in f32, which makes the range-weight normalisation \
                     factor infinite and the centre tap NaN",
                    sigma_r,
                );
            }
        }

        if let PrefilterMode::NlmSpatial { strength_scale } = self.prefilter {
            if !strength_scale.is_finite() || strength_scale <= 0.0 {
                anyhow::bail!(
                    "nlm pilot strength_scale must be finite and greater than 0, got {}",
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

/// The per-channel distance scale for a channel mode, which is 3 for
/// luma, 1.5 for chroma, and 1 for full YUV.
///
/// This matches the `channel_scale` the weighting kernels use on the
/// GPU, and it is the same for every channel within a given mode.
pub(super) fn channel_scale(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma => 3.0,
        ChannelMode::Chroma => 1.5,
        ChannelMode::Yuv => 1.0,
    }
}

/// The scale-weighted RMS of the per-channel noise estimates, over the
/// channels a mode actually uses.
///
/// Because `channel_scale` is the same for every channel in a mode, the
/// weighting cancels out and this is really just a plain RMS.
///
/// Anything in `sigmas` past the mode's channel count is ignored.
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
            err.to_string().contains("hq sigma_scale"),
            "error should name the field, got {err}"
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
        // The chroma scale of 1.5 applies per channel, and each channel
        // keeps its own sigma rather than sharing one.
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
        // A sigma_r of 0 makes inv_two_sigma_r_sq infinite. The centre
        // tap's range_sq is 0, and 0 times infinity is NaN, which
        // poisons every pixel of the reference image.
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
        // A sigma_s of 0 makes inv_two_sigma_s_sq infinite. The centre
        // tap's spatial_dist_sq is 0, so the spatial term is poisoned by
        // the same 0 times infinity NaN.
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
        // Pins the guard to `<= 0.0` rather than `< 0.0`. A value that
        // is small but strictly positive, and far enough from the f32
        // underflow cliff that squaring it stays a normal float, has to
        // be accepted for either field on its own.
        //
        // 1e-6 squares to 1e-12, nowhere near the smallest normal f32 of
        // about 1.18e-38, so `inv_two_sigma_sq` stays finite here.
        let safe_small = 1e-6_f32;
        assert!(
            (safe_small * safe_small).is_normal(),
            "the test value itself must not underflow"
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
        // `f32::MIN_POSITIVE`, the smallest normal positive f32 at about
        // 1.1754944e-38, is finite and above 0, so a guard that only
        // checked the raw value let it through.
        //
        // Squaring it underflows to exactly 0.0 in f32, because its true
        // square of about 1.38e-76 is far below the smallest subnormal
        // of about 1.4e-45. `inv_two_sigma_sq` then divides by zero and
        // returns infinity.
        //
        // That is the same NaN poisoning this validation exists to
        // prevent, reached through a sigma other than exactly 0.0.
        let sq = f32::MIN_POSITIVE * f32::MIN_POSITIVE;
        assert_eq!(sq, 0.0, "this test assumes MIN_POSITIVE underflows on squaring");
        let inv = prefilter::inv_two_sigma_sq(f32::MIN_POSITIVE);
        assert!(
            !inv.is_finite(),
            "this test assumes the derived factor is infinite here"
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

    /// A `sigma_s` of 16.0 gives a bilateral radius of 32, which is the
    /// worked example in `prefilter.rs` and well past the maximum of 22.
    #[test]
    fn validate_rejects_bilateral_sigma_s_above_the_smem_ceiling() {
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 16.0,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        let err = params.validate().expect_err("radius 32 exceeds the 22 ceiling");
        assert!(
            err.to_string().contains("sigma_s"),
            "error should name the field, got {err}"
        );
    }

    /// A `sigma_s` of 1e9 overflows the tile-size arithmetic if it ever
    /// reaches the kernel launch.
    ///
    /// Validation has to reject it long before that, through the same
    /// radius check any other oversized `sigma_s` hits.
    #[test]
    fn validate_rejects_extreme_bilateral_sigma_s() {
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 1e9,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    /// The boundary pair for [`MAX_BILATERAL_RADIUS`], written as
    /// literal `sigma_s` values rather than derived from the constant.
    ///
    /// A `sigma_s` of 11.0 gives a radius of 22, right at the ceiling,
    /// so it is accepted. A `sigma_s` of 11.01 gives 23, one past it, so
    /// it is rejected.
    #[test]
    fn validate_accepts_bilateral_sigma_s_at_the_smem_ceiling() {
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 11.0,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_bilateral_sigma_s_just_above_the_smem_ceiling() {
        let params = NlmParams {
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 11.01,
                sigma_r: 0.02,
            },
            ..NlmParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn sigma_eff_ignores_channels_past_the_mode_count() {
        // Luma only reads the first element, even when handed extra
        // chroma samples.
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
                "at radius {radius} expected {expected}, got {got}"
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
                "at radius {radius} expected {expected}, got {got}"
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
                "at radius {radius} yuv is {yuv} but luma is {luma}"
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
