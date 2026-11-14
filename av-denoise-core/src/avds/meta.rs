use anyhow::Context;
use parquet::file::metadata::KeyValue;

use super::{AVDS_VERSION, AvdsMeta, PlanesMeasured};
use crate::{ChromaLayout, ChromaSiting};

pub(super) const VERSION_KEY: &str = "avds.version";
pub(super) const WRITER_KEY: &str = "avds.writer";
pub(super) const FRAME_RATE_KEY: &str = "frame_rate";
pub(super) const WIDTH_KEY: &str = "width";
pub(super) const HEIGHT_KEY: &str = "height";
pub(super) const BIT_DEPTH_KEY: &str = "bit_depth";
pub(super) const SUBSAMPLING_KEY: &str = "subsampling";
pub(super) const CHROMA_SITING_KEY: &str = "chroma_siting";
pub(super) const FRAMES_KEY: &str = "frames";
pub(super) const PLANES_MEASURED_KEY: &str = "planes_measured";

/// Every metadata pair except `frames`, which is only known once every row is written.
pub(super) fn metadata_pairs(meta: &AvdsMeta) -> Vec<KeyValue> {
    let (numerator, denominator) = meta.frame_rate;
    let frame_rate = format!("{numerator}/{denominator}");
    let width = meta.width.to_string();
    let height = meta.height.to_string();
    let bit_depth = meta.bit_depth.to_string();
    let subsampling = subsampling_name(meta.subsampling);
    let chroma_siting = siting_name(meta.chroma_siting);
    let planes_measured = meta.planes_measured.name();

    let pairs = [
        (VERSION_KEY, AVDS_VERSION),
        (WRITER_KEY, meta.writer.as_str()),
        (FRAME_RATE_KEY, frame_rate.as_str()),
        (WIDTH_KEY, width.as_str()),
        (HEIGHT_KEY, height.as_str()),
        (BIT_DEPTH_KEY, bit_depth.as_str()),
        (SUBSAMPLING_KEY, subsampling),
        (CHROMA_SITING_KEY, chroma_siting),
        (PLANES_MEASURED_KEY, planes_measured),
    ];

    let mut key_values = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        let key = key.to_owned();
        let value = value.to_owned();
        let key_value = KeyValue::new(key, value);
        key_values.push(key_value);
    }

    key_values
}

/// Reads the metadata from a file's pairs, refusing a major version other than this one's.
pub(super) fn meta_from_pairs(pairs: &[KeyValue]) -> Result<AvdsMeta, anyhow::Error> {
    let version = value_of(pairs, VERSION_KEY)?;
    let major = version.split('.').next().unwrap_or_default();
    if major != AVDS_VERSION {
        anyhow::bail!("the grain data file has version {version}, only version {AVDS_VERSION} can be read");
    }

    let writer = value_of(pairs, WRITER_KEY)?;
    let frame_rate = frame_rate_of(pairs)?;
    let width = number_of(pairs, WIDTH_KEY)?;
    let height = number_of(pairs, HEIGHT_KEY)?;
    let bit_depth = number_of(pairs, BIT_DEPTH_KEY)?;
    let frames = number_of(pairs, FRAMES_KEY)?;

    let subsampling_text = value_of(pairs, SUBSAMPLING_KEY)?;
    let subsampling = subsampling_from_name(subsampling_text).with_context(|| {
        format!("the grain data file has an unknown {SUBSAMPLING_KEY} {subsampling_text}")
    })?;

    let siting_text = value_of(pairs, CHROMA_SITING_KEY)?;
    let chroma_siting = siting_from_name(siting_text)
        .with_context(|| format!("the grain data file has an unknown {CHROMA_SITING_KEY} {siting_text}"))?;

    let planes_text = value_of(pairs, PLANES_MEASURED_KEY)?;
    let planes_measured = PlanesMeasured::from_name(planes_text)
        .with_context(|| format!("the grain data file has an unknown {PLANES_MEASURED_KEY} {planes_text}"))?;

    Ok(AvdsMeta {
        writer: writer.to_owned(),
        frame_rate,
        width,
        height,
        bit_depth,
        subsampling,
        chroma_siting,
        frames,
        planes_measured,
    })
}

fn value_of<'a>(pairs: &'a [KeyValue], key: &str) -> Result<&'a str, anyhow::Error> {
    let pair = pairs.iter().find(|pair| pair.key == key);
    let value = pair.and_then(|pair| pair.value.as_deref());

    value.with_context(|| format!("the grain data file has no {key} metadata"))
}

fn number_of<T: std::str::FromStr>(pairs: &[KeyValue], key: &str) -> Result<T, anyhow::Error> {
    let text = value_of(pairs, key)?;
    let parsed = text.parse().ok();

    parsed.with_context(|| format!("the grain data file's {key} metadata is not a number: {text}"))
}

fn frame_rate_of(pairs: &[KeyValue]) -> Result<(u64, u64), anyhow::Error> {
    let text = value_of(pairs, FRAME_RATE_KEY)?;
    let parts = text.split_once('/');
    let parsed = parts.and_then(|(numerator, denominator)| {
        let numerator = numerator.parse().ok()?;
        let denominator = denominator.parse().ok()?;
        Some((numerator, denominator))
    });

    parsed.with_context(|| format!("the grain data file's {FRAME_RATE_KEY} is not a ratio: {text}"))
}

fn subsampling_name(layout: ChromaLayout) -> &'static str {
    match layout {
        ChromaLayout::Gray => "400",
        ChromaLayout::Yuv420 => "420",
        ChromaLayout::Yuv422 => "422",
        ChromaLayout::Yuv444 => "444",
    }
}

fn subsampling_from_name(name: &str) -> Option<ChromaLayout> {
    match name {
        "400" => Some(ChromaLayout::Gray),
        "420" => Some(ChromaLayout::Yuv420),
        "422" => Some(ChromaLayout::Yuv422),
        "444" => Some(ChromaLayout::Yuv444),
        _ => None,
    }
}

fn siting_name(siting: ChromaSiting) -> &'static str {
    match siting {
        ChromaSiting::Left => "left",
        ChromaSiting::Center => "center",
        ChromaSiting::TopLeft => "top_left",
        ChromaSiting::Unknown => "unknown",
    }
}

fn siting_from_name(name: &str) -> Option<ChromaSiting> {
    match name {
        "left" => Some(ChromaSiting::Left),
        "center" => Some(ChromaSiting::Center),
        "top_left" => Some(ChromaSiting::TopLeft),
        "unknown" => Some(ChromaSiting::Unknown),
        _ => None,
    }
}
