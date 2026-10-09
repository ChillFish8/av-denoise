use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use av_denoise::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    HqParams,
    MotionCompensationMode,
    MotionSearch,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    PlaneOptions,
    PrefilterMode,
    Preset,
    PsyParams,
    Subsampling,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
};
use vapoursynth::format::{ColorFamily, SampleType};

/// The format fields [layout_from_format] reads.
///
/// `vapoursynth::format::Format` wraps a pointer only a running VapourSynth core can hand out, so it
/// cannot be built in a unit test.
#[derive(Debug, Clone, Copy)]
pub struct RawFormat {
    pub sample_type: SampleType,
    pub bits_per_sample: u8,
    pub subsampling_w: u8,
    pub subsampling_h: u8,
    pub color_family: ColorFamily,
}

/// Validates a clip's format and turns it into a [FrameLayout].
///
/// Accepts integer YUV420, YUV422 and YUV444 sources at 8, 10 or 12 bits. RGB is rejected because
/// the denoiser's channel distance weights are calibrated for YUV. GRAY is rejected because
/// [Subsampling] has no "no chroma" variant, and representing it as YUV444 would push full-size
/// neutral chroma every frame, four times the data volume of true 4:2:0 chroma.
pub fn layout_from_format(format: RawFormat, width: u32, height: u32) -> Result<FrameLayout, anyhow::Error> {
    match format.color_family {
        ColorFamily::YUV => {},
        ColorFamily::Gray => {
            anyhow::bail!(
                "GRAY clips are not supported, av-denoise-vs only accepts YUV420, YUV422, and YUV444 sources. Convert the input to YUV first, for example with `ffmpeg -pix_fmt yuv420p`"
            );
        },
        other => {
            anyhow::bail!(
                "{other:?} clips are not supported, av-denoise's channel distance weights are calibrated for YUV. Convert the input to YUV first, for example with `ffmpeg -pix_fmt yuv420p`"
            );
        },
    }

    if format.sample_type == SampleType::Float {
        anyhow::bail!(
            "float sample types are not supported, av-denoise expects integer YUV samples. Convert to an integer format first, for example with `ffmpeg -pix_fmt yuv420p`"
        );
    }

    let depth = Depth::from_bits(format.bits_per_sample as usize)?;

    let subsampling = match (format.subsampling_w, format.subsampling_h) {
        (0, 0) => Subsampling::Yuv444,
        (1, 0) => Subsampling::Yuv422,
        (1, 1) => Subsampling::Yuv420,
        (subsampling_w, subsampling_h) => {
            anyhow::bail!(
                "unsupported chroma subsampling (subsampling_w={subsampling_w}, subsampling_h={subsampling_h}), av-denoise-vs accepts YUV420, YUV422, and YUV444"
            );
        },
    };

    Ok(FrameLayout {
        width,
        height,
        subsampling,
        depth,
    })
}

/// Which denoising algorithm a filter function runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Nlmeans,
    Nl4d,
}

fn variant_name(variant: NlmeansVariant) -> &'static str {
    match variant {
        NlmeansVariant::Fast => "fast",
        NlmeansVariant::Hq => "hq",
    }
}

/// Parses a `variant` name with the parser the CLI's `--variant` flag uses, so both accept the same
/// names.
fn parse_variant(raw: &str) -> Result<NlmeansVariant, anyhow::Error> {
    raw.parse::<NlmeansVariant>()
        .map_err(|_| anyhow::anyhow!("unknown variant '{raw}', expected one of fast, hq"))
}

/// Parses a `preset` name with the parser the CLI's `--preset` flag uses, so both accept the same
/// names.
fn parse_preset(raw: &str) -> Result<Preset, anyhow::Error> {
    raw.parse::<Preset>().map_err(|_| {
        anyhow::anyhow!("unknown preset '{raw}', expected one of veryfast, fast, base, slow, veryslow")
    })
}

/// The script arguments a filter function receives, before they are validated.
///
/// An unset field falls back to the library's own default for the algorithm being built.
#[derive(Debug, Clone, Default)]
pub struct RawParams {
    pub strength: Option<f64>,
    pub variant: Option<String>,
    pub preset: Option<String>,
    pub prefilter: Option<String>,
    pub channel_mode: Option<String>,
    pub luma_strength: Option<f64>,
    pub chroma_strength: Option<f64>,
    pub luma_lambda_ht: Option<f64>,
    pub chroma_lambda_ht: Option<f64>,
    pub device: Option<String>,
    pub accelerators: Option<Vec<String>>,
    pub search_radius: Option<i64>,
    pub patch_radius: Option<i64>,
    pub temporal_radius: Option<i64>,
    pub sigma: Option<f64>,
    pub sigma_scale: Option<f64>,
    pub motion_compensation: Option<bool>,
    pub lambda_ht: Option<f64>,
    pub lambda_ht_scale: Option<f64>,
    pub spatial_radius: Option<i64>,
    pub refine: Option<i64>,
    pub noise_map: Option<bool>,
    pub enable_psy: Option<bool>,
    pub psy_flat_boost: Option<f64>,
    pub psy_chroma_flat_boost: Option<f64>,
    pub psy_shadow_soften: Option<f64>,
    pub psy_flat_texture_cut: Option<f64>,
    pub psy_line_ring: Option<i64>,
    pub pooled_threshold: Option<bool>,
}

fn nonnegative(value: i64, field: &str) -> Result<u32, anyhow::Error> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("{field} must not be negative, got {value}"))
}

/// Parses a `channel_mode` name, rejecting a mode the source cannot support.
fn parse_channel_mode(raw: &str, layout: FrameLayout) -> Result<ChannelIntent, anyhow::Error> {
    let intent = match raw.to_ascii_lowercase().as_str() {
        "luma" => ChannelIntent::Luma,
        "chroma" => ChannelIntent::Chroma,
        "lumachroma" => ChannelIntent::LumaChroma,
        "yuv" => ChannelIntent::YuvFused,
        other => {
            anyhow::bail!("unknown channel_mode '{other}', expected one of luma, chroma, lumachroma, yuv");
        },
    };

    intent.validate_for_source(layout)?;
    Ok(intent)
}

/// Rejects a parameter set on an algorithm that never reads it.
///
/// A script parameter dictionary has no way to warn that a parameter silently does nothing, so this
/// is an error instead. The strength, patch and search radius, variant, prefilter and motion
/// parameters only feed the NLM weighting pass, which nl4d never runs. The `lambda_ht` family and the
/// nl4d dials only feed nl4d. `sigma` and `sigma_scale` are rejected only on nlmeans
/// `variant="fast"`, which has no noise estimator to pin or nudge.
///
/// `sigma_scale` alongside `sigma` is accepted even though the estimator it nudges never runs, to
/// match the CLI, which only warns about that combination. `temporal_radius` and `preset` are
/// accepted everywhere because every algorithm reads them.
fn reject_mismatched_params(
    raw: &RawParams,
    algorithm_kind: AlgorithmKind,
    variant: NlmeansVariant,
) -> Result<(), anyhow::Error> {
    let nlm_only_params: &[(&str, bool)] = &[
        ("strength", raw.strength.is_some()),
        ("luma_strength", raw.luma_strength.is_some()),
        ("chroma_strength", raw.chroma_strength.is_some()),
        ("patch_radius", raw.patch_radius.is_some()),
        ("search_radius", raw.search_radius.is_some()),
        ("variant", raw.variant.is_some()),
        ("prefilter", raw.prefilter.is_some()),
        ("motion_compensation", raw.motion_compensation.is_some()),
    ];
    let nl4d_only_params: &[(&str, bool)] = &[
        ("luma_lambda_ht", raw.luma_lambda_ht.is_some()),
        ("chroma_lambda_ht", raw.chroma_lambda_ht.is_some()),
        ("lambda_ht", raw.lambda_ht.is_some()),
        ("lambda_ht_scale", raw.lambda_ht_scale.is_some()),
        ("spatial_radius", raw.spatial_radius.is_some()),
        ("refine", raw.refine.is_some()),
        ("noise_map", raw.noise_map.is_some()),
        ("enable_psy", raw.enable_psy.is_some()),
        ("psy_flat_boost", raw.psy_flat_boost.is_some()),
        ("psy_chroma_flat_boost", raw.psy_chroma_flat_boost.is_some()),
        ("psy_shadow_soften", raw.psy_shadow_soften.is_some()),
        ("psy_flat_texture_cut", raw.psy_flat_texture_cut.is_some()),
        ("psy_line_ring", raw.psy_line_ring.is_some()),
        ("pooled_threshold", raw.pooled_threshold.is_some()),
    ];

    match algorithm_kind {
        AlgorithmKind::Nl4d => {
            for (name, is_set) in nlm_only_params {
                if *is_set {
                    anyhow::bail!(
                        "{name} has no effect on nl4d, which has no NLM weighting pass to configure"
                    );
                }
            }
        },
        AlgorithmKind::Nlmeans => {
            for (name, is_set) in nl4d_only_params {
                if *is_set {
                    anyhow::bail!("{name} has no effect on nlmeans, which only nl4d reads");
                }
            }

            if variant == NlmeansVariant::Fast {
                let variant_label = variant_name(variant);
                if raw.sigma.is_some() {
                    anyhow::bail!(
                        "sigma has no effect on nlmeans variant=\"{}\", which has no noise measurement to pin. Set variant=\"hq\" to use sigma",
                        variant_label
                    );
                }

                if raw.sigma_scale.is_some() {
                    anyhow::bail!(
                        "sigma_scale has no effect on nlmeans variant=\"{}\", which has no noise measurement to nudge. Set variant=\"hq\" to use sigma_scale",
                        variant_label
                    );
                }
            }
        },
    }

    Ok(())
}

/// The psy options these parameters ask for, or `None` unless `enable_psy` is set.
///
/// A `psy_*` parameter without `enable_psy` is an error, so a setting is never silently ignored.
fn psy_options(raw: &RawParams) -> Result<Option<PsyParams>, anyhow::Error> {
    let tuning_params = [
        ("psy_flat_boost", raw.psy_flat_boost.is_some()),
        ("psy_chroma_flat_boost", raw.psy_chroma_flat_boost.is_some()),
        ("psy_shadow_soften", raw.psy_shadow_soften.is_some()),
        ("psy_flat_texture_cut", raw.psy_flat_texture_cut.is_some()),
        ("psy_line_ring", raw.psy_line_ring.is_some()),
    ];
    let enabled = raw.enable_psy.unwrap_or(false);

    if !enabled {
        let set_param = tuning_params.iter().find(|(_, is_set)| *is_set);
        if let Some((name, _)) = set_param {
            anyhow::bail!("{name} needs enable_psy=True, add enable_psy=True to turn the psy options on");
        }

        return Ok(None);
    }

    let defaults = PsyParams::default();
    let line_ring = match raw.psy_line_ring {
        Some(radius) => nonnegative(radius, "psy_line_ring")?,
        None => defaults.line_ring,
    };
    let psy = PsyParams {
        flat_boost: raw
            .psy_flat_boost
            .map(|value| value as f32)
            .unwrap_or(defaults.flat_boost),
        chroma_flat_boost: raw
            .psy_chroma_flat_boost
            .map(|value| value as f32)
            .unwrap_or(defaults.chroma_flat_boost),
        shadow_soften: raw
            .psy_shadow_soften
            .map(|value| value as f32)
            .unwrap_or(defaults.shadow_soften),
        flat_texture_cut: raw
            .psy_flat_texture_cut
            .map(|value| value as f32)
            .unwrap_or(defaults.flat_texture_cut),
        line_ring,
    };

    Ok(Some(psy))
}

/// Validates `raw` against `layout` and builds the denoiser's [PlaneOptions].
///
/// Parameters the algorithm never reads are rejected first, so `search_radius` on nl4d fails as a
/// mismatch rather than tripping the stack check. A `search_radius` above 4 is then rejected when
/// `RUST_MIN_STACK` is unset or too small, since cubecl's kernel codegen overflows the default 2 MiB
/// stack and aborts the process at that radius.
pub fn plane_options_from(
    raw: &RawParams,
    algorithm_kind: AlgorithmKind,
    layout: FrameLayout,
) -> Result<PlaneOptions, anyhow::Error> {
    // An explicit `variant`, `temporal_radius` or `search_radius` overrides what the preset picks,
    // matching the CLI's precedence.
    let preset = match &raw.preset {
        None => Preset::default(),
        Some(name) => parse_preset(name)?,
    };

    // Only nlmeans reads `variant`, so nl4d never parses it and a set value fails as a mismatch below.
    let variant = match algorithm_kind {
        AlgorithmKind::Nlmeans => match raw.variant.as_deref() {
            None => nlmeans_variant_for(preset),
            Some(name) => parse_variant(name)?,
        },
        AlgorithmKind::Nl4d => NlmeansVariant::Hq,
    };

    reject_mismatched_params(raw, algorithm_kind, variant)?;

    if let Some(radius) = raw.search_radius
        && radius > 4
        && !av_denoise::codegen_stack_is_sufficient()
    {
        anyhow::bail!(
            "search_radius {radius} needs a raised stack, but RUST_MIN_STACK is not set. Values above 4 overflow the default 2 MiB stack during kernel codegen"
        );
    }

    let intent = match raw.channel_mode.as_deref() {
        None => ChannelIntent::LumaChroma,
        Some(mode) => parse_channel_mode(mode, layout)?,
    };

    let device = match &raw.device {
        None => Device::default(),
        Some(name) => name
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid device '{name}': {error}"))?,
    };

    let accelerators = match &raw.accelerators {
        None => get_default_accelerators(),
        Some(names) => names
            .iter()
            .map(|name| {
                name.parse::<Accelerator>()
                    .map_err(|error| anyhow::anyhow!("invalid accelerator '{name}': {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    // An explicit `temporal_radius` overrides the preset's, the same way the CLI resolves it. nl4d has
    // no spatial-only mode, and no nl4d preset resolves to 0, so only the nlmeans `veryfast` preset
    // picks the spatial mode by default.
    let preset_temporal_radius = match algorithm_kind {
        AlgorithmKind::Nlmeans => nlmeans_temporal_radius_for(preset),
        AlgorithmKind::Nl4d => nl4d_temporal_radius_for(preset),
    };

    let mode = match raw.temporal_radius {
        None if preset_temporal_radius == 0 => DenoisingMode::Spacial,
        None => DenoisingMode::Temporal {
            radius: preset_temporal_radius,
        },
        Some(0) => DenoisingMode::Spacial,
        Some(radius) => DenoisingMode::Temporal {
            radius: nonnegative(radius, "temporal_radius")?,
        },
    };

    let algorithm = match algorithm_kind {
        AlgorithmKind::Nlmeans => {
            let search_radius = match raw.search_radius {
                None => nlmeans_search_radius_for(preset),
                Some(radius) => nonnegative(radius, "search_radius")?,
            };

            let tuning = NlmTuning {
                search_radius: Some(search_radius),
                patch_radius: raw
                    .patch_radius
                    .map(|radius| nonnegative(radius, "patch_radius"))
                    .transpose()?,
                strength: raw.strength.map(|value| value as f32),
                ..NlmTuning::default()
            };

            let motion_search = MotionSearch::default();
            let motion_compensation = match raw.motion_compensation {
                Some(true) => MotionCompensationMode::from(motion_search),
                Some(false) | None => MotionCompensationMode::None,
            };

            let prefilter = match &raw.prefilter {
                None => PrefilterMode::None,
                Some(spec) => parse_prefilter(spec)?,
            };

            let nlm = NlmeansOptions {
                prefilter,
                motion_compensation,
                tuning,
                mode,
            };

            match variant {
                NlmeansVariant::Fast => Algorithm::Nlmeans(nlm),
                NlmeansVariant::Hq => {
                    let hq = HqParams {
                        sigma_override: raw.sigma.map(|value| value as f32),
                        sigma_scale: raw
                            .sigma_scale
                            .map(|value| value as f32)
                            .unwrap_or_else(|| HqParams::default().sigma_scale),
                        ..HqParams::default()
                    };

                    let options = NlmeansHqOptions { nlm, hq };
                    Algorithm::NlmeansHq(options)
                },
            }
        },
        AlgorithmKind::Nl4d => {
            let psy = psy_options(raw)?;
            let options = Nl4dOptions {
                sigma: raw.sigma.map(|value| value as f32),
                sigma_scale: raw
                    .sigma_scale
                    .map(|value| value as f32)
                    .unwrap_or_else(|| Nl4dOptions::default().sigma_scale),
                lambda_ht: raw.lambda_ht.map(|value| value as f32),
                lambda_ht_scale: raw
                    .lambda_ht_scale
                    .map(|value| value as f32)
                    .unwrap_or_else(|| Nl4dOptions::default().lambda_ht_scale),
                spatial_radius: match raw.spatial_radius {
                    Some(radius) => nonnegative(radius, "spatial_radius")?,
                    None => nl4d_spatial_radius_for(preset),
                },
                refine: match raw.refine {
                    Some(radius) => nonnegative(radius, "refine")?,
                    None => Nl4dOptions::default().refine,
                },
                noise_map: match raw.noise_map {
                    Some(enabled) => enabled,
                    None => Nl4dOptions::default().noise_map,
                },
                psy,
                pooled_threshold: match raw.pooled_threshold {
                    Some(enabled) => enabled,
                    None => Nl4dOptions::default().pooled_threshold,
                },
                ..Nl4dOptions::default()
            };

            Algorithm::Nl4d(options)
        },
    };

    Ok(PlaneOptions {
        accelerators,
        device,
        intent,
        mode,
        algorithm,
        luma_strength: raw.luma_strength.map(|value| value as f32),
        chroma_strength: raw.chroma_strength.map(|value| value as f32),
        luma_lambda_ht: raw.luma_lambda_ht.map(|value| value as f32),
        chroma_lambda_ht: raw.chroma_lambda_ht.map(|value| value as f32),
    })
}
