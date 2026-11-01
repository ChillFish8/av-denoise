use super::consts::TUNE_EXTRA_FRAMES;
use super::key::bucket_dim;
use super::unit::TuneUnit;
use crate::engine::{EngineSpec, PassSpec, SampleFormat};
use crate::nlmeans::DenoisingMode;

/// The key a plane's profile ranking is stored under.
///
/// It names the algorithm with its settings, the channels, the input and output formats and the
/// plane size rounded up to powers of two.
pub fn profile_key(spec: &PassSpec) -> String {
    let geometry = spec.geometry();
    let width_bucket = bucket_dim(geometry.width);
    let height_bucket = bucket_dim(geometry.height);

    let algorithm = algorithm_name(spec);
    let channels = geometry.channels;
    let input = format_name(geometry.input);
    let output = format_name(geometry.output);

    format!("{algorithm}|{channels:?}|in-{input}|out-{output}|w{width_bucket}|h{height_bucket}")
}

/// How many frames a tuning pass over `specs` needs.
///
/// It is the widest temporal span of any pass in the group plus [TUNE_EXTRA_FRAMES]. Nl4d spans
/// `4 * radius + 1` frames and NLMeans spans `2 * radius + 1`.
pub fn tuning_frame_count(specs: &[EngineSpec]) -> usize {
    let units = TuneUnit::frames(specs);
    frame_count_for(&units)
}

/// How many frames a tuning pass over the single passes `specs` needs, as [tuning_frame_count] counts.
pub fn pass_tuning_frame_count(specs: &[PassSpec]) -> usize {
    let units = TuneUnit::passes_of(specs);
    frame_count_for(&units)
}

fn frame_count_for(units: &[TuneUnit]) -> usize {
    let spans = units
        .iter()
        .flat_map(TuneUnit::passes)
        .map(|pass| temporal_span(&pass));
    let widest = spans.max().unwrap_or(1);

    widest + TUNE_EXTRA_FRAMES
}

fn temporal_span(spec: &PassSpec) -> usize {
    match spec {
        PassSpec::Nlmeans { algorithm, .. } => {
            let radius = match algorithm.mode() {
                DenoisingMode::Spacial => 0,
                DenoisingMode::Temporal { radius } => radius as usize,
            };

            2 * radius + 1
        },
        PassSpec::Nl4d { options, .. } => 4 * options.temporal_radius as usize + 1,
    }
}

/// The algorithm's name with its settings.
fn algorithm_name(spec: &PassSpec) -> String {
    match spec {
        PassSpec::Nlmeans { algorithm, .. } => format!("Nlmeans({algorithm:?})"),
        PassSpec::Nl4d { options, .. } => format!("Nl4d({options:?})"),
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
