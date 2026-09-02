//! Turns a VapourSynth clip's format and a filter's script arguments
//! into the option types `av-denoise-core` denoises with.
//!
//! Everything here is a pure function over plain values, with no
//! VapourSynth core and no GPU, so the whole accept/reject matrix is
//! unit-testable. [`Format`](vapoursynth::format::Format) itself cannot
//! be built outside a running core, so [`layout_from_format`] takes a
//! [`RawFormat`] of the plain fields it needs instead. The caller in
//! `filter.rs` does the short extraction from a real `Format`.

use av_denoise_core::accelerate::{Accelerator, get_default_accelerators};
use av_denoise_core::{
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
    Subsampling,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
};
use vapoursynth::format::{ColorFamily, SampleType};

/// The handful of format fields [`layout_from_format`] actually reads.
///
/// The real caller is `vapoursynth::format::Format`, which wraps a
/// pointer only a running VapourSynth core can hand out, so it cannot be
/// built in a unit test. A caller with a real `Format` builds one of
/// these from `format.sample_type()`, `format.bits_per_sample()`,
/// `format.sub_sampling_w()`, `format.sub_sampling_h()`, and
/// `format.color_family()`.
#[derive(Debug, Clone, Copy)]
pub struct RawFormat {
    pub sample_type: SampleType,
    pub bits_per_sample: u8,
    pub subsampling_w: u8,
    pub subsampling_h: u8,
    pub color_family: ColorFamily,
}

/// Validates a clip's format and turns it into a [`FrameLayout`].
///
/// Accepts integer YUV420, YUV422, and YUV444 sources at 8, 10, or
/// 12-bit. Rejects RGB, since the denoiser's channel distance weights
/// are calibrated for YUV. Rejects float sample types and any other
/// chroma subsampling.
///
/// Rejects GRAY too. Core's [`Subsampling`] has no "no chroma" variant,
/// so a GRAY source would have to be represented as YUV444, which makes
/// [`av_denoise_core::frame::FrameLayout::chroma_dims`] report
/// full-resolution chroma planes that do not exist. The filter would
/// then have to fabricate and push full-size neutral chroma every
/// frame, four times the real data volume of true 4:2:0 chroma, purely
/// to work around a gap in the geometry type. GRAY is out of scope
/// until core can represent a source with no chroma planes at all.
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
        (w, h) => {
            anyhow::bail!(
                "unsupported chroma subsampling (subsampling_w={w}, subsampling_h={h}), av-denoise-vs accepts YUV420, YUV422, and YUV444"
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
///
/// `avd.Nlmeans` builds [`AlgorithmKind::Nlmeans`], `avd.Nl4d` builds
/// [`AlgorithmKind::Nl4d`]. This has no `Hq` variant since the
/// VapourSynth plugin does not expose the HQ variant separately, it
/// takes a plain algorithm choice per filter function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Nlmeans,
    Nl4d,
}

/// The name a [`NlmeansVariant`] parses back from, used in error
/// messages.
fn variant_name(variant: NlmeansVariant) -> &'static str {
    match variant {
        NlmeansVariant::Fast => "fast",
        NlmeansVariant::Hq => "hq",
    }
}

/// Resolves an explicit `variant` string into an [`NlmeansVariant`].
///
/// Uses [`av_denoise_core`]'s own parser, the same one the CLI's
/// `--variant` flag resolves through, so a name accepted on the CLI is
/// accepted here too.
fn parse_variant(raw: &str) -> Result<NlmeansVariant, anyhow::Error> {
    raw.parse::<NlmeansVariant>()
        .map_err(|_| anyhow::anyhow!("unknown variant '{raw}', expected one of fast, hq"))
}

/// Resolves an explicit `preset` string into a [`Preset`].
///
/// Uses [`av_denoise_core`]'s own parser, the same one the CLI's
/// `--preset` flag resolves through, so a name accepted on the CLI is
/// accepted here too.
fn parse_preset(raw: &str) -> Result<Preset, anyhow::Error> {
    raw.parse::<Preset>().map_err(|_| {
        anyhow::anyhow!("unknown preset '{raw}', expected one of veryfast, fast, base, slow, veryslow")
    })
}

/// The raw script arguments a filter function receives, before they are
/// validated and folded into a [`PlaneOptions`].
///
/// Every field is optional. An unset field falls back to the library's
/// own default for whichever algorithm is being built.
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
    pub luma_mismatch_scale: Option<f64>,
    pub chroma_mismatch_scale: Option<f64>,
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
}

/// Turns a nonnegative script integer into a `u32`, naming `field` in
/// the error when it is negative.
fn nonnegative(value: i64, field: &str) -> Result<u32, anyhow::Error> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("{field} must not be negative, got {value}"))
}

/// Resolves an explicit `channel_mode` string into a [`ChannelIntent`],
/// rejecting anything the source can't support.
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
/// `strength`, its per-plane overrides, `patch_radius`, `search_radius`,
/// and `variant` only feed the NLM weighting pass, which `NlmTuning`
/// belongs to. Core's own `DenoiserOptions::to_nlm_params` builds the
/// `Nl4d` arm from `NlmParams::default()` for exactly this group of
/// fields, so none of them reach an `Nl4d` run at all, no matter what a
/// caller sets. `lambda_ht` and `mismatch_scale` and their per-plane
/// overrides only feed nl4d's temporal grouping stage. Setting one on
/// the algorithm that ignores it would silently do nothing, which a
/// script parameter dictionary has no way to warn about on its own, so
/// this rejects it instead.
///
/// `sigma` and `sigma_scale` both pin or nudge the noise level an HQ
/// front end would otherwise measure. nl4d always runs that front end,
/// so both are always valid there. Plain `nlmeans` (`variant="fast"`)
/// has no noise estimator at all, so both are rejected only in that one
/// case, checked separately below since it depends on `variant` rather
/// than `algorithm_kind` alone.
///
/// `sigma_scale` alongside `sigma` is not rejected here, even though it
/// does nothing in that combination: the noise estimator that
/// `sigma_scale` would nudge never runs once `sigma` pins the level.
/// This mirrors the CLI, which only warns about that combination rather
/// than erroring (see `--hq-sigma-scale` and `--sigma-scale`), because
/// `sigma_scale` is a parameter every configuration understands, it
/// just has nothing left to scale.
///
/// `temporal_radius` is not here because it genuinely affects every
/// algorithm: it sets `PlaneOptions::mode`, which every algorithm
/// reads. `preset` is not here for the same reason: both algorithms
/// resolve dials from it.
///
/// This runs before the `RUST_MIN_STACK` stack-safety check in
/// [`plane_options_from`], so `search_radius` on an `Nl4d` call fails
/// here first rather than tripping that check, which nl4d can never
/// actually need since it never applies a caller's `search_radius` in
/// the first place.
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
        ("luma_mismatch_scale", raw.luma_mismatch_scale.is_some()),
        ("chroma_mismatch_scale", raw.chroma_mismatch_scale.is_some()),
        ("lambda_ht", raw.lambda_ht.is_some()),
        ("lambda_ht_scale", raw.lambda_ht_scale.is_some()),
        ("spatial_radius", raw.spatial_radius.is_some()),
        ("refine", raw.refine.is_some()),
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
                if raw.sigma.is_some() {
                    anyhow::bail!(
                        "sigma has no effect on nlmeans variant=\"{}\", which has no noise measurement to pin. Set variant=\"hq\" to use sigma",
                        variant_name(variant)
                    );
                }
                if raw.sigma_scale.is_some() {
                    anyhow::bail!(
                        "sigma_scale has no effect on nlmeans variant=\"{}\", which has no noise measurement to nudge. Set variant=\"hq\" to use sigma_scale",
                        variant_name(variant)
                    );
                }
            }
        },
    }

    Ok(())
}

/// Validates `raw` against `layout` and builds the [`PlaneOptions`] a
/// [`PlanarDenoiser`](av_denoise_core::PlanarDenoiser) is created from.
///
/// Rejects any parameter `algorithm_kind` does not read first, such as
/// `strength` on nl4d or `lambda_ht` on nlmeans, rather than accepting
/// and silently ignoring it. See [`reject_mismatched_params`].
///
/// Then rejects `search_radius` above 4 when `RUST_MIN_STACK` is unset or
/// too small, since cubecl's kernel codegen overflows the default 2 MiB
/// stack and aborts the process at that radius. `filter.rs` raises the
/// stack before this runs in the real plugin, so this only ever fires
/// when that step was skipped, and only for nlmeans, since nl4d never
/// reaches this check with a `search_radius` set at all.
pub fn plane_options_from(
    raw: &RawParams,
    algorithm_kind: AlgorithmKind,
    layout: FrameLayout,
) -> Result<PlaneOptions, anyhow::Error> {
    // Resolved once and read by both algorithms below, exactly like the
    // CLI's own `--preset`. An explicit `variant`, `temporal_radius`, or
    // `search_radius` overrides whatever the preset would have picked,
    // matching `NlmeansArgs::resolve_preset`'s precedence.
    let preset = match &raw.preset {
        None => Preset::default(),
        Some(p) => parse_preset(p)?,
    };

    // Only `Nlmeans` reads `variant` at all, so `Nl4d` never parses it,
    // it is rejected as a mismatched parameter below instead if set.
    let variant = match algorithm_kind {
        AlgorithmKind::Nlmeans => match raw.variant.as_deref() {
            None => nlmeans_variant_for(preset),
            Some(v) => parse_variant(v)?,
        },
        AlgorithmKind::Nl4d => NlmeansVariant::Hq,
    };

    reject_mismatched_params(raw, algorithm_kind, variant)?;

    if let Some(radius) = raw.search_radius
        && radius > 4
        && !av_denoise_core::codegen_stack_is_sufficient()
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
        Some(s) => s
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid device '{s}': {e}"))?,
    };

    let accelerators = match &raw.accelerators {
        None => get_default_accelerators(),
        Some(names) => names
            .iter()
            .map(|s| {
                s.parse::<Accelerator>()
                    .map_err(|e| anyhow::anyhow!("invalid accelerator '{s}': {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    // Both algorithms resolve their preset-driven temporal radius the
    // same way the CLI does: an explicit `temporal_radius` overrides
    // whatever the preset picks. nl4d groups patches across a temporal
    // window and has no spatial-only mode, but no preset ever resolves
    // it to 0, so the `radius == 0` arm below only actually triggers
    // for `nlmeans`, whose `veryfast` preset does.
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
            let tuning = NlmTuning {
                search_radius: Some(match raw.search_radius {
                    None => nlmeans_search_radius_for(preset),
                    Some(r) => nonnegative(r, "search_radius")?,
                }),
                patch_radius: raw
                    .patch_radius
                    .map(|r| nonnegative(r, "patch_radius"))
                    .transpose()?,
                strength: raw.strength.map(|v| v as f32),
                ..NlmTuning::default()
            };

            let motion_compensation = match raw.motion_compensation {
                Some(true) => MotionCompensationMode::from(MotionSearch::default()),
                Some(false) | None => MotionCompensationMode::None,
            };

            let prefilter = match &raw.prefilter {
                None => PrefilterMode::None,
                Some(s) => {
                    let mode = parse_prefilter(s)?;
                    // `parse_prefilter`'s string grammar has no form
                    // that produces `External`, but the check stays
                    // here as a boundary guard rather than trusting
                    // that invariant silently: `External` needs a
                    // reference frame supplied through
                    // `push_frame_with_reference`, which this plugin
                    // has no way to call.
                    if matches!(mode, PrefilterMode::External) {
                        anyhow::bail!(
                            "prefilter 'external' is not supported by av-denoise-vs, which has no way to supply a reference frame"
                        );
                    }
                    mode
                },
            };

            match variant {
                NlmeansVariant::Fast => Algorithm::Nlmeans(NlmeansOptions {
                    prefilter,
                    motion_compensation,
                    tuning,
                }),
                NlmeansVariant::Hq => Algorithm::NlmeansHq(NlmeansHqOptions {
                    nlm: NlmeansOptions {
                        prefilter,
                        motion_compensation,
                        tuning,
                    },
                    hq: HqParams {
                        sigma_override: raw.sigma.map(|v| v as f32),
                        sigma_scale: raw
                            .sigma_scale
                            .map(|v| v as f32)
                            .unwrap_or_else(|| HqParams::default().sigma_scale),
                        // A VapourSynth filter has to return the same
                        // pixels for a frame no matter what order
                        // frames were requested in, and history-
                        // dependent estimation breaks that guarantee
                        // under random access. See `Nl4dOptions`'s own
                        // `windowed_noise_estimation` field for the
                        // same reasoning applied to nl4d.
                        windowed_noise_estimation: true,
                        ..HqParams::default()
                    },
                }),
            }
        },
        AlgorithmKind::Nl4d => Algorithm::Nl4d(Nl4dOptions {
            // A VapourSynth filter has to return the same pixels for a
            // frame no matter what order frames were requested in.
            // window-local estimation computes sigma from only the
            // frames in the current window, so the fast path and a
            // `reseed` after random access agree by construction. There
            // is no reason to expose the stream-history-dependent
            // temporal EMA here at all.
            windowed_noise_estimation: true,
            sigma: raw.sigma.map(|v| v as f32),
            sigma_scale: raw
                .sigma_scale
                .map(|v| v as f32)
                .unwrap_or_else(|| Nl4dOptions::default().sigma_scale),
            lambda_ht: raw.lambda_ht.map(|v| v as f32),
            lambda_ht_scale: raw
                .lambda_ht_scale
                .map(|v| v as f32)
                .unwrap_or_else(|| Nl4dOptions::default().lambda_ht_scale),
            spatial_radius: match raw.spatial_radius {
                Some(r) => nonnegative(r, "spatial_radius")?,
                None => nl4d_spatial_radius_for(preset),
            },
            refine: match raw.refine {
                Some(r) => nonnegative(r, "refine")?,
                None => Nl4dOptions::default().refine,
            },
            ..Nl4dOptions::default()
        }),
    };

    Ok(PlaneOptions {
        accelerators,
        device,
        intent,
        mode,
        algorithm,
        luma_strength: raw.luma_strength.map(|v| v as f32),
        chroma_strength: raw.chroma_strength.map(|v| v as f32),
        luma_lambda_ht: raw.luma_lambda_ht.map(|v| v as f32),
        chroma_lambda_ht: raw.chroma_lambda_ht.map(|v| v as f32),
        luma_mismatch_scale: raw.luma_mismatch_scale.map(|v| v as f32),
        chroma_mismatch_scale: raw.chroma_mismatch_scale.map(|v| v as f32),
    })
}
