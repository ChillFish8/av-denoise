use super::Nl4dParams;
use crate::error::Error;
use crate::nlmeans::{ChannelMode, HqParams, MotionSearch, NlmParams};
use crate::options::Preset;

/// Settings for [Nl4d](crate::Nl4d).
///
/// nl4d runs the HQ front end only for its machinery, the frame ring,
/// the motion field, and the noise estimate. Nothing weights or averages
/// patches the NLM way, so the NLM knobs are absent here and the fields
/// below are the whole surface.
///
/// The temporal radius comes from `temporal_radius`, which has to be at
/// least 1. Motion tracking is always on, because the
/// grouping kernel reads the motion field and confidence scores it
/// produces.
///
/// `lambda_ht` has a per-plane default. `None` resolves through
/// [`nl4d_default_lambda_ht`] once the plane being denoised is known.
/// `lambda_ht_scale` then multiplies whichever value that resolves to.
///
/// Every other default comes from [`Nl4dParams::default`](super::Nl4dParams::default).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl4dOptions {
    /// How motion between frames is tracked.
    pub motion: MotionSearch,
    /// A fixed noise standard deviation in `[0, 1]` units, replacing the
    /// automatic per-frame estimate.
    ///
    /// `None`, the default, measures the noise in each pushed frame and
    /// smooths it over time.
    pub sigma: Option<f32>,
    /// A multiplier applied to the measured noise level before anything
    /// reads it. Defaults to 1.0.
    ///
    /// This does nothing when `sigma` pins the noise level, because the
    /// estimator never runs in that case.
    pub sigma_scale: f32,
    /// A multiplier on the per-block mismatch threshold, which sets how
    /// much extra SAD a block tolerates before its confidence starts to
    /// fall. Defaults to 1.0.
    ///
    /// Higher values tolerate larger mismatches.
    pub thsad_scale: f32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position, in `1..=4`. Defaults to 2.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the centre
    /// frame, in `1..=16`. Defaults to 9.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    /// Higher removes more noise and more fine detail.
    ///
    /// `None` resolves through [`nl4d_default_lambda_ht`], which returns
    /// a different value for luma than for chroma.
    pub lambda_ht: Option<f32>,
    /// A multiplier applied to the resolved `lambda_ht`. Defaults to
    /// 1.0.
    ///
    /// It scales an explicit `lambda_ht` and the calibrated per-plane
    /// default alike, so one value moves both planes together. Has to
    /// be finite and in `[0.1, 10.0]`.
    pub lambda_ht_scale: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Defaults to 0.05. A
    /// block below the floor is never scored, and a volume left short
    /// of frames by the skip makes its group filter from the centre
    /// frame alone.
    pub c_min: f32,
    /// The `beta` of the Kaiser window each filtered patch is tapered
    /// with as it is aggregated. Defaults to 2.0. `0.0` is uniform
    /// aggregation. See [`crate::nl4d::Nl4dParams::kaiser_beta`].
    pub kaiser_beta: f32,
    /// Estimates noise fresh from each frame's own window instead of
    /// smoothing it across the whole stream's history. Defaults to
    /// `false`, matching every calibrated preset.
    ///
    /// `av-denoise-vs` turns this on unconditionally, because a
    /// VapourSynth filter has to return the same pixels for a frame no
    /// matter what order frames were requested in, and history-dependent
    /// estimation breaks that guarantee under random access. See
    /// [`HqParams::windowed_noise_estimation`](crate::nlmeans::HqParams::windowed_noise_estimation).
    pub windowed_noise_estimation: bool,
    /// See [`crate::nl4d::Nl4dParams::field_lambda`].
    pub field_lambda: f32,
    /// See [crate::nl4d::Nl4dParams::noise_map].
    pub noise_map: bool,
    /// See [crate::nl4d::Nl4dParams::flat_boost].
    pub flat_boost: f32,
    /// See [crate::nl4d::Nl4dParams::chroma_flat_boost].
    pub chroma_flat_boost: f32,
    /// See [crate::nl4d::Nl4dParams::shadow_soften].
    pub shadow_soften: f32,
    /// See [crate::nl4d::Nl4dParams::flat_texture_cut].
    pub flat_texture_cut: f32,
    /// See [crate::nl4d::Nl4dParams::pooled_threshold].
    pub pooled_threshold: bool,
    /// See [crate::nl4d::Nl4dParams::grain_export].
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub grain_export: bool,
    /// How many frames on each side of the centre frame the temporal
    /// window reaches, in `1..=8`.
    pub temporal_radius: u32,
}

impl Default for Nl4dOptions {
    fn default() -> Self {
        let defaults = Nl4dParams::default();
        let hq = HqParams::default();
        Self {
            motion: MotionSearch::default(),
            sigma: hq.sigma_override,
            sigma_scale: hq.sigma_scale,
            thsad_scale: hq.thsad_scale,
            refine: defaults.refine,
            spatial_radius: defaults.spatial_radius,
            // Resolved per plane by `nl4d_default_lambda_ht` at
            // construction time, once the plane being denoised is
            // known.
            lambda_ht: None,
            lambda_ht_scale: 1.0,
            c_min: defaults.c_min,
            kaiser_beta: defaults.kaiser_beta,
            windowed_noise_estimation: false,
            field_lambda: defaults.field_lambda,
            noise_map: defaults.noise_map,
            flat_boost: defaults.flat_boost,
            chroma_flat_boost: defaults.chroma_flat_boost,
            shadow_soften: defaults.shadow_soften,
            flat_texture_cut: defaults.flat_texture_cut,
            pooled_threshold: defaults.pooled_threshold,
            grain_export: defaults.grain_export,
            temporal_radius: nl4d_temporal_radius_for(Preset::Base),
        }
    }
}

/// The default `lambda_ht` for nl4d's hard-threshold stage, per plane.
///
/// `lambda_ht` is how many standard deviations of estimated noise a
/// transform coefficient has to clear to survive. Raising it removes more
/// noise and more fine detail with it, so the value is a trade rather
/// than an optimum.
///
/// Luma and chroma values are picked by eye from a ladder of renders against real
/// film grain, accepting more lost detail in exchange for less remaining
/// noise on heavy grain. The reason why we're going a bit heavier on high noise is because
/// the encoders end up reducing that detail _more_ than the denoiser does if
/// that extra entropy remains in and overall produces a worse final image.
///
/// `ChannelMode::Yuv` reads the luma value, on the same "a fused pass is
/// dominated by luma" assumption `hq_default_strength` makes for its own
/// Yuv case.
///
/// Luma and the fused Yuv mode use 4.158, and chroma uses 3.234.
pub fn nl4d_default_lambda_ht(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => 4.158,
        ChannelMode::Chroma => 3.234,
    }
}

/// The pooled threshold at each plane's default lambda, calibrated on real grain.
const NL4D_POOLED_THRESHOLD: f32 = 2.42;

/// The ratio of nl4d's pooled threshold to its lambda for one plane.
///
/// At the default lambda this gives the calibrated pooled threshold, and it scales with any
/// other lambda.
pub(crate) fn nl4d_pool_ratio(channels: ChannelMode) -> f32 {
    NL4D_POOLED_THRESHOLD / nl4d_default_lambda_ht(channels)
}

/// Resolves `Nl4dOptions.lambda_ht` for one plane, falling back to
/// [`nl4d_default_lambda_ht`] when the caller left it unset, then
/// applies `lambda_ht_scale`.
///
/// The scale multiplies an explicit value and the calibrated default
/// alike, so it moves both planes together whether or not one of them
/// is pinned.
///
/// The range check lives here rather than in [`Nl4dParams`](super::Nl4dParams),
/// which only ever sees the product. A scale of 0 would surface there as
/// a complaint about `lambda_ht`, naming a knob the caller never set.
fn resolve_lambda_ht(opts: &Nl4dOptions, channels: ChannelMode) -> Result<f32, String> {
    if !(opts.lambda_ht_scale.is_finite() && (0.1..=10.0).contains(&opts.lambda_ht_scale)) {
        return Err(format!(
            "lambda_ht_scale must be finite and in [0.1, 10.0], got {}",
            opts.lambda_ht_scale
        ));
    }

    let lambda_ht = opts.lambda_ht.unwrap_or_else(|| nl4d_default_lambda_ht(channels));

    Ok(lambda_ht * opts.lambda_ht_scale)
}

/// How far the temporal window reaches at each preset, for `nl4d`.
///
/// Unlike `nlmeans`, `veryfast` keeps a 1-frame window rather than
/// dropping to 0, because nl4d has nothing to do without neighbouring
/// frames to group against.
pub fn nl4d_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How wide the centre frame's candidate search is at each preset, for
/// `nl4d`.
///
/// `veryfast` shares its temporal radius with `fast`, so this is what
/// separates them. The window covers `(2 * radius + 1)^2` positions, so
/// 6 searches a little over half the candidates 9 does.
///
/// Every preset from `fast` up uses the library default. Widening it
/// further at the slow end costs quadratically and has not been measured
/// to be worth it.
pub fn nl4d_spatial_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 6,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => {
            Nl4dOptions::default().spatial_radius
        },
    }
}

/// The front end's HQ parameters for these options.
///
/// `temporal_confidence` is always on, because the grouping kernel
/// reads the confidence scores it produces. The two strength-related
/// switches keep their defaults, since nl4d never runs a weighting
/// pass for them to affect.
fn hq_params(options: &Nl4dOptions) -> HqParams {
    HqParams {
        sigma_override: options.sigma,
        sigma_scale: options.sigma_scale,
        thsad_scale: options.thsad_scale,
        temporal_confidence: true,
        windowed_noise_estimation: options.windowed_noise_estimation,
        ..HqParams::default()
    }
}

/// Builds the front end parameters nl4d runs its motion and noise machinery with.
///
/// No weighting pass runs, so `strength`, `search_radius`, `patch_radius`
/// and `self_weight` stay at their library defaults and no prefilter is built.
pub(crate) fn nlm_params(options: &Nl4dOptions, channels: ChannelMode) -> NlmParams {
    NlmParams {
        channels,
        motion_compensation: options.motion.into(),
        temporal_radius: options.temporal_radius,
        hq: Some(hq_params(options)),
        ..NlmParams::default()
    }
}

/// Resolves the options into the parameters an nl4d denoiser runs with.
///
/// Calibrated defaults are filled in for `channels`.
pub(crate) fn resolve_params(options: &Nl4dOptions, channels: ChannelMode) -> Result<Nl4dParams, Error> {
    if options.temporal_radius == 0 {
        return Err(Error::InvalidOptions(
            "nl4d needs a temporal radius of at least 1".to_string(),
        ));
    }

    let lambda_ht = resolve_lambda_ht(options, channels).map_err(Error::InvalidOptions)?;
    let nlm = nlm_params(options, channels);

    Ok(Nl4dParams {
        temporal_radius: options.temporal_radius,
        nlm,
        refine: options.refine,
        spatial_radius: options.spatial_radius,
        lambda_ht,
        c_min: options.c_min,
        kaiser_beta: options.kaiser_beta,
        field_lambda: options.field_lambda,
        noise_map: options.noise_map,
        flat_boost: options.flat_boost,
        chroma_flat_boost: options.chroma_flat_boost,
        shadow_soften: options.shadow_soften,
        flat_texture_cut: options.flat_texture_cut,
        pooled_threshold: options.pooled_threshold,
        grain_export: options.grain_export,
    })
}
