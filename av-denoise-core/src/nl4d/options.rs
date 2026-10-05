use super::Nl4dParams;
use crate::error::Error;
use crate::nlmeans::{ChannelMode, HqParams, MotionSearch, NlmParams};
use crate::options::Preset;

/// Settings for [Nl4d](crate::Nl4d).
///
/// The HQ front end runs only for its frame ring, motion field and noise estimate, so no NLM
/// weighting knobs appear here. Motion tracking is always on, because the grouping kernel reads
/// the motion field and confidence scores it produces.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl4dOptions {
    /// How motion between frames is tracked.
    pub motion: MotionSearch,
    /// A fixed noise standard deviation between 0 and 1, replacing the per-frame estimate.
    ///
    /// `None`, the default, measures the noise in each pushed frame and smooths it over time.
    pub sigma: Option<f32>,
    /// A multiplier on the measured noise level. Defaults to 1.0.
    ///
    /// It does nothing when `sigma` is set, because the estimator never runs.
    pub sigma_scale: f32,
    /// A multiplier on how much extra SAD a block tolerates before its confidence falls. Defaults
    /// to 1.0.
    pub thsad_scale: f32,
    /// Half-width of the refine window around each neighbour frame's motion-predicted position, in
    /// `1..=4`. Defaults to 2.
    pub refine: u32,
    /// Half-width of the spatial candidate window in the centre frame, in `1..=16`. Defaults to 9.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    ///
    /// Higher removes more noise and more fine detail. `None` resolves per plane through
    /// [nl4d_default_lambda_ht].
    pub lambda_ht: Option<f32>,
    /// A multiplier on the resolved `lambda_ht`, in `0.1..=10.0`. Defaults to 1.0.
    ///
    /// It scales an explicit `lambda_ht` and the per-plane default alike, so one value moves both
    /// planes together.
    pub lambda_ht_scale: f32,
    /// The confidence floor below which a whole neighbour block is skipped, in `0.0..1.0`. Defaults
    /// to 0.05.
    ///
    /// A volume left short of frames by the skip makes its group filter from the centre frame
    /// alone.
    pub c_min: f32,
    /// The `beta` of the Kaiser window each filtered patch is tapered with as it is aggregated.
    ///
    /// Defaults to 2.0, and `0.0` is uniform aggregation. See
    /// [Nl4dParams::kaiser_beta](crate::nl4d::Nl4dParams::kaiser_beta).
    pub kaiser_beta: f32,
    /// Estimates noise from each frame's own window instead of the whole stream's history.
    ///
    /// A VapourSynth filter must return the same pixels for a frame in any request order, and
    /// history-dependent estimation breaks that under random access. Defaults to `false`, matching
    /// every calibrated preset.
    pub windowed_noise_estimation: bool,
    /// See [Nl4dParams::field_lambda](crate::nl4d::Nl4dParams::field_lambda).
    pub field_lambda: f32,
    /// See [Nl4dParams::noise_map](crate::nl4d::Nl4dParams::noise_map).
    pub noise_map: bool,
    /// See [Nl4dParams::flat_boost](crate::nl4d::Nl4dParams::flat_boost).
    pub flat_boost: f32,
    /// See [Nl4dParams::chroma_flat_boost](crate::nl4d::Nl4dParams::chroma_flat_boost).
    pub chroma_flat_boost: f32,
    /// See [Nl4dParams::shadow_soften](crate::nl4d::Nl4dParams::shadow_soften).
    pub shadow_soften: f32,
    /// See [Nl4dParams::flat_texture_cut](crate::nl4d::Nl4dParams::flat_texture_cut).
    pub flat_texture_cut: f32,
    /// See [Nl4dParams::pooled_threshold](crate::nl4d::Nl4dParams::pooled_threshold).
    pub pooled_threshold: bool,
    /// Whether the denoiser measures the source's film grain for an AV1 grain table.
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub grain_export: bool,
    /// How many frames on each side of the centre frame the temporal window reaches, in `1..=8`.
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
            // Resolved per plane at construction, once the plane is known.
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
/// `lambda_ht` is how many standard deviations of estimated noise a transform coefficient must
/// clear to survive. Raising it removes more noise and more fine detail, so the value is a trade.
///
/// The values were picked by eye on real film grain, accepting some lost detail for less remaining
/// noise on heavy grain. Encoders lose more detail to leftover grain than the denoiser does.
/// `ChannelMode::Yuv` uses the luma value, since a fused pass is dominated by luma.
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

/// Resolves `lambda_ht` for one plane and applies `lambda_ht_scale`.
///
/// The scale is range-checked here because [Nl4dParams] only sees the product, where a bad scale
/// would surface as a complaint about a `lambda_ht` the caller never set.
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

/// How far the temporal window reaches at each preset.
///
/// `veryfast` keeps a 1-frame window, because nl4d has nothing to group without neighbouring
/// frames.
pub fn nl4d_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How wide the centre frame's candidate search is at each preset.
///
/// `veryfast` shares its temporal radius with `fast`, so this is what separates them. The window
/// covers `(2 * radius + 1)^2` positions, so 6 searches a little under half the candidates 9 does.
/// Wider windows cost quadratically and have not been measured to help, so the slower presets keep
/// the default.
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
/// `temporal_confidence` is always on, because the grouping kernel reads the confidence scores.
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

/// The front end parameters nl4d runs its motion and noise machinery with.
pub(crate) fn nlm_params(options: &Nl4dOptions, channels: ChannelMode) -> NlmParams {
    let hq = hq_params(options);

    NlmParams {
        channels,
        motion_compensation: options.motion.into(),
        temporal_radius: options.temporal_radius,
        hq: Some(hq),
        ..NlmParams::default()
    }
}

/// Resolves the options into an nl4d denoiser's parameters, with calibrated defaults for
/// `channels`.
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
