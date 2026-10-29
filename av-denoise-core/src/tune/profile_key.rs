use super::consts::TUNE_EXTRA_FRAMES;
use super::key::bucket_dim;
use crate::engine::{EngineSpec, SampleFormat};
use crate::nlmeans::{DenoisingMode, NlmeansAlgorithm};

/// The key a plane's profile ranking is stored under.
///
/// It names the algorithm with its settings, the channels, the temporal mode, the depth and the
/// plane size rounded up to powers of two. A plane whose output format differs from its input, or
/// whose input is not 8, 10 or 12-bit, also names both formats.
pub fn profile_key(spec: &EngineSpec) -> String {
    let geometry = spec.geometry();
    let width_bucket = bucket_dim(geometry.width);
    let height_bucket = bucket_dim(geometry.height);

    let (algorithm, mode) = algorithm_and_mode(spec);
    let channels = geometry.channels;
    let depth = depth_name(geometry.input);
    let mut key = format!("{algorithm}|{channels:?}|{mode:?}|{depth}|w{width_bucket}|h{height_bucket}");

    let has_legacy_depth = legacy_depth(geometry.input).is_some();
    if geometry.input != geometry.output || !has_legacy_depth {
        let input = format_name(geometry.input);
        let output = format_name(geometry.output);
        let suffix = format!("|in-{input}|out-{output}");
        key.push_str(&suffix);
    }

    key
}

/// How many frames a tuning pass over `specs` needs.
///
/// It is the widest temporal span in the group plus [TUNE_EXTRA_FRAMES]. Nl4d spans `4 * radius + 1`
/// frames and NLMeans spans `2 * radius + 1`.
pub fn tuning_frame_count(specs: &[EngineSpec]) -> usize {
    let widest = specs.iter().map(temporal_span).max().unwrap_or(1);

    widest + TUNE_EXTRA_FRAMES
}

fn temporal_span(spec: &EngineSpec) -> usize {
    match spec {
        EngineSpec::Nlmeans { algorithm, .. } => {
            let radius = match algorithm.mode() {
                DenoisingMode::Spacial => 0,
                DenoisingMode::Temporal { radius } => radius as usize,
            };

            2 * radius + 1
        },
        EngineSpec::Nl4d { options, .. } => 4 * options.temporal_radius as usize + 1,
    }
}

/// The algorithm's name with its settings, and the temporal mode it runs in.
fn algorithm_and_mode(spec: &EngineSpec) -> (String, DenoisingMode) {
    match spec {
        EngineSpec::Nlmeans {
            algorithm: NlmeansAlgorithm::Fast(options),
            ..
        } => (format!("Nlmeans({options:?})"), options.mode),
        EngineSpec::Nlmeans {
            algorithm: NlmeansAlgorithm::Hq(options),
            ..
        } => (format!("NlmeansHq({options:?})"), options.nlm.mode),
        EngineSpec::Nl4d { options, .. } => {
            let mode = match options.temporal_radius {
                0 => DenoisingMode::Spacial,
                radius => DenoisingMode::Temporal { radius },
            };

            (format!("Nl4d({options:?})"), mode)
        },
    }
}

/// The legacy depth name for the 8, 10 and 12-bit formats.
fn legacy_depth(format: SampleFormat) -> Option<&'static str> {
    match format {
        SampleFormat::U8 => Some("Eight"),
        SampleFormat::U16 { depth: 10 } => Some("Ten"),
        SampleFormat::U16 { depth: 12 } => Some("Twelve"),
        _ => None,
    }
}

fn depth_name(format: SampleFormat) -> String {
    match legacy_depth(format) {
        Some(name) => name.to_string(),
        None => format_name(format),
    }
}

/// The short name of `format`, such as `u8`, `u10` or `f32`.
fn format_name(format: SampleFormat) -> String {
    match format {
        SampleFormat::U8 => "u8".to_string(),
        SampleFormat::U16 { depth } => format!("u{depth}"),
        SampleFormat::F32 => "f32".to_string(),
    }
}
