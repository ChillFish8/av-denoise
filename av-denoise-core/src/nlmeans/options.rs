use super::{ChannelMode, HqParams, MotionCompensationMode, NlmParams, PrefilterMode, hq_default_strength};
use crate::options::Preset;

/// Settings for the fast [NlmeansAlgorithm] path.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansOptions {
    /// Which reference image the NLM weights are computed against.
    ///
    /// `None`, the default, compares patches on the noisy input
    /// directly. Every other mode costs one extra GPU pass per frame.
    pub prefilter: PrefilterMode,
    /// Whether temporal denoising follows motion between frames.
    ///
    /// `None`, the default, turns motion compensation off. `Mvtools`
    /// warps temporal neighbours into line with the centre frame before
    /// the NLM weighting runs.
    ///
    /// Only has an effect when `mode` is `Temporal { .. }`.
    pub motion_compensation: MotionCompensationMode,
    /// Overrides for the NLM search radius, patch radius, strength, and
    /// self-weight.
    pub tuning: NlmTuning,
    /// Whether each frame is cleaned on its own or across a temporal window.
    pub mode: DenoisingMode,
}

/// Settings for the HQ [NlmeansAlgorithm] path.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansHqOptions {
    /// Everything the fast path takes, which HQ takes too.
    pub nlm: NlmeansOptions,
    /// The noise measurement and confidence weighting HQ adds on top.
    pub hq: HqParams,
}

/// Which nlmeans implementation a preset, or an explicit choice, selects.
#[derive(Debug, Copy, Clone, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum NlmeansVariant {
    /// The fast path. Fixed weighting, no noise measurement.
    Fast,
    /// Quality focused. Calibrates its weighting to the noise level,
    /// measured automatically per frame.
    Hq,
}

/// Which nlmeans variant to build, with its settings.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NlmeansAlgorithm {
    /// The fast variant.
    Fast(NlmeansOptions),
    /// The quality variant.
    Hq(NlmeansHqOptions),
}

impl NlmeansAlgorithm {
    pub(crate) fn mode(&self) -> DenoisingMode {
        match self {
            NlmeansAlgorithm::Fast(options) => options.mode,
            NlmeansAlgorithm::Hq(options) => options.nlm.mode,
        }
    }
}

/// Which [`NlmeansVariant`] a preset runs.
pub fn nlmeans_variant_for(preset: Preset) -> NlmeansVariant {
    match preset {
        Preset::Veryfast => NlmeansVariant::Fast,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => NlmeansVariant::Hq,
    }
}

/// How many neighbouring frames on each side `nlmeans` looks at, at a
/// preset.
pub fn nlmeans_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 0,
        Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How far `nlmeans` looks for similar patches inside a frame, at a
/// preset.
pub fn nlmeans_search_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast | Preset::Base => 2,
        Preset::Slow | Preset::Veryslow => 4,
    }
}

/// Whether a frame is cleaned on its own or alongside its neighbours.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub enum DenoisingMode {
    /// Cleans each frame using only its own pixels.
    #[default]
    Spacial,
    /// Cleans each frame using a window of `2 * radius + 1` frames.
    Temporal { radius: u32 },
}

/// NLM tuning knobs.
///
/// Every field is optional. Whatever is left unset falls back to the
/// library default.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmTuning {
    pub search_radius: Option<u32>,
    pub patch_radius: Option<u32>,
    pub strength: Option<f32>,
    pub self_weight: Option<f32>,
}

/// Resolves the options into the parameters a denoiser runs with.
///
/// Calibrated defaults are filled in for `channels`. An explicit `strength` always wins.
pub(crate) fn resolve_params(algorithm: &NlmeansAlgorithm, channels: ChannelMode) -> NlmParams {
    let temporal_radius = match algorithm.mode() {
        DenoisingMode::Spacial => 0,
        DenoisingMode::Temporal { radius } => radius,
    };

    let (options, hq) = match *algorithm {
        NlmeansAlgorithm::Fast(options) => (options, None),
        NlmeansAlgorithm::Hq(options) => (options.nlm, Some(options.hq)),
    };

    // With `auto_strength` on, HQ reads `strength` as a multiplier on the
    // measured noise level, so it needs its own calibrated default. With
    // it off, HQ reads an absolute value like the fast path does.
    let defaults = NlmParams::default();
    let strength = options.tuning.strength.unwrap_or(match hq {
        Some(hq) if hq.auto_strength => hq_default_strength(channels, temporal_radius),
        _ => defaults.strength,
    });

    NlmParams {
        channels,
        prefilter: options.prefilter,
        motion_compensation: options.motion_compensation,
        temporal_radius,
        hq,
        strength,
        search_radius: options.tuning.search_radius.unwrap_or(defaults.search_radius),
        patch_radius: options.tuning.patch_radius.unwrap_or(defaults.patch_radius),
        self_weight: options.tuning.self_weight.unwrap_or(defaults.self_weight),
    }
}
