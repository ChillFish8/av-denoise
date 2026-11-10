use super::{AvdsFrame, AvdsMeta, scenes_from_rows};
use crate::{FitOptions, FrameGrain, build_table, frame_rows};

/// The AV1 film grain table the rows describe, as `filmgrn1` text.
///
/// Fails when the file's frame rate has a zero part, when the fine noise share is not between 0
/// and 1, or when the rows hold frames but none of them has the luma table measurements.
pub fn av1_table(meta: &AvdsMeta, rows: &[AvdsFrame], options: &FitOptions) -> Result<String, anyhow::Error> {
    let (numerator, denominator) = meta.frame_rate;
    if numerator == 0 || denominator == 0 {
        anyhow::bail!("the grain data file's frame rate {numerator}/{denominator} is not valid");
    }

    let fine_noise = options.fine_noise;
    if !(0.0..=1.0).contains(&fine_noise) {
        anyhow::bail!("the fine noise share {fine_noise} is not between 0 and 1");
    }

    let has_table = rows.iter().any(|row| row.table.is_some());
    if !rows.is_empty() && !has_table {
        anyhow::bail!("the grain data file holds no luma table measurements");
    }

    let scenes = scenes_from_rows(rows);
    let table = build_table(&scenes, meta.frame_rate, options);

    Ok(table)
}

/// The per-frame grain report the rows describe.
pub fn report_rows(rows: &[AvdsFrame]) -> Vec<FrameGrain> {
    let scenes = scenes_from_rows(rows);

    frame_rows(&scenes)
}
