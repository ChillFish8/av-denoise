use super::{AvdsFrame, AvdsMeta, scenes_from_rows};
use crate::{FrameGrain, build_table, frame_rows};

/// How [av1_table] fits the grain the rows describe.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct FitOptions {}

/// The AV1 film grain table the rows describe, as `filmgrn1` text.
///
/// Fails when the file's frame rate has a zero part, or when the rows hold frames but none of
/// them has the luma table measurements.
pub fn av1_table(
    meta: &AvdsMeta,
    rows: &[AvdsFrame],
    _options: &FitOptions,
) -> Result<String, anyhow::Error> {
    let (numerator, denominator) = meta.frame_rate;
    if numerator == 0 || denominator == 0 {
        anyhow::bail!("the grain data file's frame rate {numerator}/{denominator} is not valid");
    }

    let has_table = rows.iter().any(|row| row.table.is_some());
    if !rows.is_empty() && !has_table {
        anyhow::bail!("the grain data file holds no luma table measurements");
    }

    let scenes = scenes_from_rows(rows);
    let table = build_table(&scenes, meta.frame_rate);

    Ok(table)
}

/// The per-frame grain report the rows describe.
pub fn report_rows(rows: &[AvdsFrame]) -> Vec<FrameGrain> {
    let scenes = scenes_from_rows(rows);

    frame_rows(&scenes)
}
