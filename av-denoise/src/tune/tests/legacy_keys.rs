use av_denoise_core::tune::{bucket_dim, profile_key as spec_profile_key};
use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    HqParams,
    MotionCompensationMode,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    PrefilterMode,
    Preset,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
};

use crate::backend::Device;
use crate::host::{Algorithm, DenoiserOptions, Depth};
use crate::planar::{ChannelIntent, PlaneOptions};
use crate::tune::TuningMode;

const PRESETS: [Preset; 5] = [
    Preset::Veryfast,
    Preset::Fast,
    Preset::Base,
    Preset::Slow,
    Preset::Veryslow,
];

const VARIANTS: [NlmeansVariant; 2] = [NlmeansVariant::Fast, NlmeansVariant::Hq];

const CHANNELS: [ChannelMode; 3] = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

const DEPTHS: [Depth; 3] = [Depth::Eight, Depth::Ten, Depth::Twelve];

const SIZES: [(u32, u32); 2] = [(1920, 1080), (720, 480)];

/// The legacy key for a plane's profile ranking.
///
/// It names the algorithm with its settings, the channels, the temporal mode, the depth and the
/// plane size rounded up to powers of two.
fn profile_key(options: &DenoiserOptions, width: u32, height: u32) -> String {
    let width_bucket = bucket_dim(width);
    let height_bucket = bucket_dim(height);

    format!(
        "{:?}|{:?}|{:?}|{:?}|w{width_bucket}|h{height_bucket}",
        options.algorithm, options.channel_mode, options.mode, options.depth,
    )
}

/// Which caller's option building a case follows.
#[derive(Debug, Clone, Copy)]
enum Caller {
    Cli,
    VapourSynth,
}

fn preset_mode(radius: u32) -> DenoisingMode {
    match radius {
        0 => DenoisingMode::Spacial,
        radius => DenoisingMode::Temporal { radius },
    }
}

fn plane_options(mode: DenoisingMode, algorithm: Algorithm) -> PlaneOptions {
    PlaneOptions {
        accelerators: Vec::new(),
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode,
        algorithm,
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Auto,
        deterministic: false,
    }
}

/// The nlmeans options a caller builds at `preset` with `variant` and no other flags.
fn nlmeans_options(caller: Caller, preset: Preset, variant: NlmeansVariant) -> PlaneOptions {
    let radius = nlmeans_temporal_radius_for(preset);
    let mode = preset_mode(radius);
    let tuning = NlmTuning {
        search_radius: Some(nlmeans_search_radius_for(preset)),
        ..NlmTuning::default()
    };
    let nlm = NlmeansOptions {
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        tuning,
        mode,
    };

    let hq = match caller {
        Caller::Cli => HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
        },
        Caller::VapourSynth => HqParams::default(),
    };

    let algorithm = match variant {
        NlmeansVariant::Fast => Algorithm::Nlmeans(nlm),
        NlmeansVariant::Hq => {
            let options = NlmeansHqOptions { nlm, hq };
            Algorithm::NlmeansHq(options)
        },
    };

    plane_options(mode, algorithm)
}

/// The nl4d options a caller builds with no flags.
///
/// The CLI writes its radius into the algorithm, while the VapourSynth plugin leaves the
/// algorithm's own default in place.
fn nl4d_options(caller: Caller) -> PlaneOptions {
    let defaults = Nl4dOptions::default();
    let radius = defaults.temporal_radius;
    let options = match caller {
        Caller::Cli => Nl4dOptions {
            temporal_radius: radius,
            ..defaults
        },
        Caller::VapourSynth => defaults,
    };
    let mode = DenoisingMode::Temporal { radius };

    plane_options(mode, Algorithm::Nl4d(options))
}

/// The legacy and spec keys for every plane `options` denoises, as `(legacy, spec)` pairs.
fn key_pairs(options: &PlaneOptions) -> Vec<(String, String)> {
    let mut pairs = Vec::new();

    for channels in CHANNELS {
        for depth in DEPTHS {
            for (width, height) in SIZES {
                let plane_options = options.denoiser_options(channels, depth);
                let legacy = profile_key(&plane_options, width, height);
                let spec = plane_options.algorithm.engine_spec(&plane_options, width, height);
                let rendered = spec_profile_key(&spec);
                pairs.push((legacy, rendered));
            }
        }
    }

    pairs
}

#[test]
fn spec_keys_match_legacy_keys_for_cli_and_vs_settings() {
    let mut cases = Vec::new();

    for caller in [Caller::Cli, Caller::VapourSynth] {
        for preset in PRESETS {
            for variant in VARIANTS {
                let label = format!("{caller:?} nlmeans {preset:?} {variant:?}");
                let options = nlmeans_options(caller, preset, variant);
                cases.push((label, options));
            }
        }

        let label = format!("{caller:?} nl4d");
        let options = nl4d_options(caller);
        cases.push((label, options));
    }

    assert_eq!(cases.len(), 22);

    for (label, options) in &cases {
        for (legacy, rendered) in key_pairs(options) {
            assert_eq!(rendered, legacy, "{label}");
        }
    }
}

#[test]
fn an_explicit_vs_nl4d_radius_changes_only_the_options_radius() {
    let defaults = Nl4dOptions::default();
    let mode = DenoisingMode::Temporal { radius: 3 };
    let options = plane_options(mode, Algorithm::Nl4d(defaults));

    let plane_options = options.denoiser_options(ChannelMode::Luma, Depth::Eight);
    let legacy = profile_key(&plane_options, 1920, 1080);
    let spec = plane_options.algorithm.engine_spec(&plane_options, 1920, 1080);
    let rendered = spec_profile_key(&spec);

    let expected = legacy.replace("temporal_radius: 2", "temporal_radius: 3");
    assert_ne!(rendered, legacy);
    assert_eq!(rendered, expected);
}
