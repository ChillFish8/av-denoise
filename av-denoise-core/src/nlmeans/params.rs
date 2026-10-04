use super::{MotionCompensationMode, PrefilterMode, prefilter};

/// The value patch distances are normalised against, 255 squared as in FFmpeg's nlmeans.
///
/// Distances are measured between 0 and 1, so this folds the scale back up to 8-bit terms.
pub(super) const NLM_NORM: f32 = 255.0 * 255.0;

/// A scale factor from FFmpeg's nlmeans, kept so `strength` means the same as theirs.
pub(super) const NLM_LEGACY: f32 = 3.0;

/// The measured HQ default luma `strength`, indexed by `temporal_radius.min(8)`.
const HQ_DEFAULT_STRENGTH_LUMA: [f32; 9] = [0.45, 0.45, 0.42, 0.42, 0.35, 0.35, 0.35, 0.30, 0.30];

/// The measured HQ default chroma `strength`, indexed by `temporal_radius.min(8)`.
const HQ_DEFAULT_STRENGTH_CHROMA: [f32; 9] = [1.00, 0.85, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70];

/// The calibrated default `strength` multiplier for HQ auto-strength.
///
/// Every entry is measured, from quality sweeps over three noise levels per radius that keep the
/// strength whose worst XPSNR gain is highest. Luma is swept at every radius from 0 to 8 and never
/// rises with radius, because a wider window already gathers more samples. Chroma is swept at
/// radii 0, 1, 2, 4 and 8 with luma pinned, and holds flat at 0.70 from radius 2, so radii 3, 5, 6
/// and 7 sit on that plateau.
///
/// `ChannelMode::Yuv` reads the luma table, which is an assumption because the sweeps do not
/// cover that mode. Clamping the radius to [MAX_TEMPORAL_RADIUS] is a safety net, since
/// [NlmParams::validate] already rejects anything larger.
pub fn hq_default_strength(channels: ChannelMode, temporal_radius: u32) -> f32 {
    let index = temporal_radius.min(MAX_TEMPORAL_RADIUS) as usize;
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => HQ_DEFAULT_STRENGTH_LUMA[index],
        ChannelMode::Chroma => HQ_DEFAULT_STRENGTH_CHROMA[index],
    }
}

/// The smallest supported frame side.
///
/// The Immerkær estimate's 3x3 mask only reads interior pixels, and a frame under 3 pixels across
/// has none.
pub const MIN_FRAME_DIM: u32 = 3;

/// Rejects frame dimensions the kernels cannot handle.
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

/// The patch radius above which the separable path runs, keeping per-pixel cost linear in
/// `patch_radius`.
pub(super) const SEPARABLE_THRESHOLD: u32 = 8;

/// The hard ceiling on `patch_radius`.
///
/// The fused kernels load a `(block + 2 * patch_radius)^2` tile into shared memory, and anything
/// larger runs out of it on RDNA-class GPUs.
pub const MAX_PATCH_RADIUS: u32 = 16;

/// The hard ceiling on `search_radius`.
///
/// The windowed kernel's tile fits in shared memory at every supported size. The limit is its fully
/// unrolled `(2 * search_radius + 1)^2` window loop, whose compiled size and build time grow with
/// the radius. It also bounds the separable path's `(2 * search_radius + 1)^2` launches per
/// temporal offset.
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// The hard ceiling on `temporal_radius`.
///
/// The ring holds `2 * radius + 1` frames, so device memory grows with it. At radius 16, 1080p YUV
/// would need roughly 540 MB for the input alone.
pub const MAX_TEMPORAL_RADIUS: u32 = 8;

/// The hard ceiling on the bilateral prefilter's radius, `ceil(2 * sigma_s).max(1)`.
///
/// `nlm_bilateral` loads a `(32 + 2r) x (8 + 2r)` tile of up to four `f32` lanes into shared
/// memory, which costs `16 * (32 + 2r) * (8 + 2r)` bytes. Against the 64 KiB RDNA-class hardware
/// offers, `r = 22` fits at 63,232 bytes and `r = 23` does not at 67,392. `bilateral_radius`
/// reaches 22 at `sigma_s = 11.0`.
pub const MAX_BILATERAL_RADIUS: u32 = 22;

/// Which channels of a frame the denoiser works on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Kernels read whole `Line<f32>` values and backends only support power-of-two widths, so YUV
    /// pads from 3 to 4.
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
/// The measured noise level drives both the effective strength and the distance floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HqParams {
    /// Reads `strength` as a multiplier on the noise level. Defaults to true.
    ///
    /// The FFmpeg-style strength becomes `strength * sigma_eff * 255`.
    pub auto_strength: bool,
    /// Subtracts the expected noise floor from patch distances before weighting. Defaults to true.
    ///
    /// A match is then not penalised for the noise it carries.
    pub noise_floor: bool,
    /// A fixed noise sigma between 0 and 1 in place of the automatic estimate.
    ///
    /// `None`, the default, measures each pushed frame and smooths the result over time.
    pub sigma_override: Option<f32>,
    /// Weights each temporal neighbour by how well it block-matches the centre frame. Defaults to
    /// true.
    ///
    /// Occlusion or a change of content then collapses a neighbour's contribution rather than
    /// blurring it in. It only has an effect when `temporal_radius` is above 0.
    pub temporal_confidence: bool,
    /// A multiplier on the per-pixel mismatch threshold. Defaults to 1.0.
    ///
    /// Higher values let a block carry more extra SAD before its confidence starts to fall.
    pub thsad_scale: f32,
    /// A multiplier on each channel's measured sigma before it joins the running estimate. Defaults
    /// to 1.0.
    ///
    /// It has no effect with `sigma_override` set, because the estimator never runs.
    pub sigma_scale: f32,
    /// Estimates noise from the current window alone rather than an EMA over every earlier frame.
    ///
    /// `false`, the default, keeps the EMA every calibrated preset assumes. `true` makes a reseed
    /// at frame `n` and a continuous stream reaching frame `n` compute the same sigma. It has no
    /// effect with `sigma_override` set.
    pub windowed_noise_estimation: bool,
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
            windowed_noise_estimation: false,
        }
    }
}

impl HqParams {
    /// The HQ defaults with a fixed noise sigma between 0 and 1, which skips automatic estimation.
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
    /// 0 cleans each frame on its own.
    pub temporal_radius: u32,
    /// Half the width of the search window. Defaults to 2.
    pub search_radius: u32,
    /// Half the width of a compared patch. Defaults to 4.
    pub patch_radius: u32,
    /// How hard to filter. Higher values smooth more. Defaults to 1.2.
    pub strength: f32,
    /// How much weight the centre pixel gets in the average. Defaults to 1.0.
    ///
    /// 0 gives pure NLM, where the centre pixel only counts through the patches that match it.
    pub self_weight: f32,
    pub channels: ChannelMode,
    /// The image patch distances are measured on. Defaults to `None`.
    ///
    /// The averaged pixels always come from the original input.
    pub prefilter: PrefilterMode,
    /// Whether temporal denoising follows motion between frames. Defaults to `None`.
    ///
    /// It only has an effect when `temporal_radius` is above 0.
    pub motion_compensation: MotionCompensationMode,
    /// The quality-mode parameters, or `None` for the fast path.
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
    /// The FFmpeg-style strength the weighting uses.
    ///
    /// With HQ auto-strength, `strength` multiplies `sigma_eff`, so one setting follows sources of
    /// different noisiness.
    pub(super) fn effective_strength_with(&self, sigma_eff: Option<f32>) -> f32 {
        match (self.hq, sigma_eff) {
            (Some(hq), Some(sigma)) if hq.auto_strength => self.strength * sigma * 255.0,
            _ => self.strength,
        }
    }

    /// `h2_inv_norm` for the given noise estimate, ignoring `sigma_override`.
    pub fn h2_inv_norm_with(&self, sigma_eff: Option<f32>) -> f32 {
        let patch_area = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        let strength = self.effective_strength_with(sigma_eff);
        NLM_NORM / (NLM_LEGACY * strength * strength * patch_area as f32)
    }

    /// `h2_inv_norm` from `sigma_override`, or with no estimate on the fast path.
    pub fn h2_inv_norm(&self) -> f32 {
        let sigma_override = self.hq.and_then(|hq| hq.sigma_override);
        self.h2_inv_norm_with(sigma_override)
    }

    /// The patch distance two noisy copies of the same content are expected to show.
    ///
    /// Each active channel adds `2 * channel_scale * sigma^2` per tap over all
    /// `(2 * patch_radius + 1)^2` taps. It is 0 with the HQ noise floor off or no estimate.
    pub(super) fn noise_offset_with(&self, sigmas: Option<&[f32]>) -> f32 {
        match (self.hq, sigmas) {
            (Some(hq), Some(sigmas)) if hq.noise_floor => {
                let patch_area = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
                let scale = channel_scale(self.channels);
                let count = self.channels.count() as usize;
                let sum_sq: f32 = sigmas.iter().take(count).map(|&sigma| sigma * sigma).sum();
                2.0 * scale * sum_sq * patch_area as f32
            },
            _ => 0.0,
        }
    }

    /// `noise_offset` with `sigma_override` on every active channel, or 0 without one.
    pub(super) fn noise_offset(&self) -> f32 {
        match self.hq.and_then(|hq| hq.sigma_override) {
            Some(sigma) => {
                let sigmas = [sigma; 3];
                let active = &sigmas[..self.channels.count() as usize];
                self.noise_offset_with(Some(active))
            },
            None => 0.0,
        }
    }

    pub(super) fn total_frames(&self) -> u32 {
        1 + 2 * self.temporal_radius
    }

    /// Rejects parameters that would push a kernel past its shared-memory or register limits, or
    /// produce meaningless output.
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

            // Checking the derived radius keeps this in step with its formula, and also catches a
            // sigma_s large enough to overflow the tile-size expression.
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

            // A positive sigma below roughly 3.8e-20 squares to 0.0 in f32, so the launch's own
            // factor is checked rather than a hand-picked sigma cutoff.
            let inv_two_sigma_s_sq = prefilter::inv_two_sigma_sq(sigma_s);
            if !inv_two_sigma_s_sq.is_finite() {
                anyhow::bail!(
                    "bilateral prefilter sigma_s is too small, got {}. Squaring it \
                     underflows to 0 in f32, which makes the spatial-weight \
                     normalisation factor infinite",
                    sigma_s,
                );
            }

            let inv_two_sigma_r_sq = prefilter::inv_two_sigma_sq(sigma_r);
            if !inv_two_sigma_r_sq.is_finite() {
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

/// The per-channel distance scale, matching the `channel_scale` the weighting kernels use.
pub(super) fn channel_scale(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma => 3.0,
        ChannelMode::Chroma => 1.5,
        ChannelMode::Yuv => 1.0,
    }
}

/// The scale-weighted RMS of the active channels' noise estimates.
///
/// The scale is the same for every channel in a mode, so this is a plain RMS. Entries past the
/// mode's channel count are ignored.
pub(super) fn sigma_eff(sigmas: &[f32], channels: ChannelMode) -> f32 {
    let count = channels.count() as usize;
    let sum_sq: f32 = sigmas.iter().take(count).map(|&sigma| sigma * sigma).sum();
    (sum_sq / count as f32).sqrt()
}
