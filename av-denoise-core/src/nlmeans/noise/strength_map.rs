use super::curve::{CLIP_HIGH, CLIP_LOW, NoiseCurve, QUARTER_STATIC_GATE};
use super::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    RHO_SIGMA_GATE,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    temporal_stats_blocks,
    temporal_stats_record_len,
};
use crate::collab::geometry::strength_map_dims;

/// How much texture a quarter may carry and still count as flat, as a fraction of its own noise
/// variance.
///
/// Grain alone reads about 0.35 to 0.45 of it, so faint lines under heavy grain land above the cut.
const QUARTER_FLAT_FACTOR: f32 = 0.55;
/// A quarter noisier than this many times the curve at its luma is treated as motion, not grain.
const MOTION_FACTOR: f32 = 2.5;
/// The luma at and below which [StrengthMapParams::shadow_soften] applies in full.
const SHADOW_LOW: f32 = 128.0 / 255.0;
/// The luma at which the soften has faded back to 1.0.
const SHADOW_HIGH: f32 = 160.0 / 255.0;

/// The luma threshold multipliers a [QuarterClasses] turns into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StrengthMapParams {
    /// The multiplier for flat quarters.
    pub(crate) flat_boost: f32,
    /// The multiplier for other quarters at or below luma 128 of 255.
    pub(crate) shadow_soften: f32,
}

impl StrengthMapParams {
    /// Whether every multiplier these produce is exactly 1.0.
    pub(crate) fn is_identity(&self) -> bool {
        self.flat_boost == 1.0 && self.shadow_soften == 1.0
    }
}

/// One 8x8 quarter's class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QuarterClass {
    pub(crate) flat: bool,
    /// The quarter's mean luma, between 0 and 1.
    pub(crate) luma: f32,
}

/// Every 8x8 quarter of a frame, row-major.
///
/// A quarter lying wholly past the frame edge holds no pixels and has no class.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QuarterClasses {
    cols: usize,
    rows: usize,
    classes: Vec<Option<QuarterClass>>,
}

impl QuarterClasses {
    #[cfg(test)]
    pub(crate) fn from_classes(cols: usize, rows: usize, classes: Vec<Option<QuarterClass>>) -> Self {
        assert_eq!(classes.len(), cols * rows);
        Self { cols, rows, classes }
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// One luma threshold multiplier per quarter, row-major.
    ///
    /// Flat quarters get `flat_boost`. Other quarters get `shadow_soften` at or below luma 128 of
    /// 255, fading linearly back to 1.0 at 160. A quarter with no class gets 1.0.
    pub(crate) fn luma_multipliers(&self, params: StrengthMapParams) -> Vec<f32> {
        self.classes
            .iter()
            .map(|class| match class {
                Some(class) if class.flat => params.flat_boost,
                Some(class) => shadow_multiplier(class.luma, params.shadow_soften),
                None => 1.0,
            })
            .collect()
    }

    /// One chroma threshold multiplier per quarter, row-major. Flat quarters get `flat_boost`, and
    /// every other quarter gets 1.0.
    pub(crate) fn chroma_multipliers(&self, flat_boost: f32) -> Vec<f32> {
        self.classes
            .iter()
            .map(|class| match class {
                Some(class) if class.flat => flat_boost,
                _ => 1.0,
            })
            .collect()
    }
}

fn shadow_multiplier(luma: f32, shadow_soften: f32) -> f32 {
    let span = SHADOW_HIGH - SHADOW_LOW;
    let progress = ((luma - SHADOW_LOW) / span).clamp(0.0, 1.0);
    shadow_soften + (1.0 - shadow_soften) * progress
}

/// Classes every 8x8 quarter of a frame from its temporal-stats records.
///
/// A quarter is flat when it is static, carries grain, reads under [MOTION_FACTOR] times the noise
/// `curve` predicts at its luma, is unclipped, and its texture is under [QUARTER_FLAT_FACTOR] of its
/// own noise variance. Every quarter is classed, not only those of the blocks the curve accepted.
pub(in crate::nlmeans) fn classify_quarters(
    records: &[f32],
    stored_ch: u32,
    width: u32,
    height: u32,
    curve: &NoiseCurve,
) -> QuarterClasses {
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let quarters_base = (2 * stored_ch + TEMPORAL_QUARTER_BASE) as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let (map_cols, map_rows) = strength_map_dims(width, height);
    let cols = map_cols as usize;
    let rows = map_rows as usize;
    let mut classes = vec![None; cols * rows];

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_index = (block_y * blocks_x + block_x) as usize;
            let record = &records[block_index * record_len..(block_index + 1) * record_len];
            let block_w = (width - block_x * TEMPORAL_NOISE_BLOCK).min(TEMPORAL_NOISE_BLOCK);
            let block_h = (height - block_y * TEMPORAL_NOISE_BLOCK).min(TEMPORAL_NOISE_BLOCK);

            for quarter_index in 0..TEMPORAL_QUARTERS {
                let offset_x = (quarter_index % 2) * TEMPORAL_QUARTER_SIZE;
                let offset_y = (quarter_index / 2) * TEMPORAL_QUARTER_SIZE;
                let quarter_w = block_w.saturating_sub(offset_x).min(TEMPORAL_QUARTER_SIZE);
                let quarter_h = block_h.saturating_sub(offset_y).min(TEMPORAL_QUARTER_SIZE);
                let pixels = (quarter_w * quarter_h) as f32;
                if pixels == 0.0 {
                    continue;
                }

                let start = quarters_base + (quarter_index * TEMPORAL_QUARTER_FIELDS) as usize;
                let fields = &record[start..start + TEMPORAL_QUARTER_FIELDS as usize];
                let class = classify_quarter(fields, pixels, curve);

                let col = (2 * block_x + quarter_index % 2) as usize;
                let row = (2 * block_y + quarter_index / 2) as usize;
                classes[row * cols + col] = Some(class);
            }
        }
    }

    QuarterClasses { cols, rows, classes }
}

fn classify_quarter(fields: &[f32], pixels: f32, curve: &NoiseCurve) -> QuarterClass {
    let luma = fields[QUARTER_LUMA_SUM as usize] / pixels;
    let mean = fields[QUARTER_SUM_D as usize] / pixels;
    let mean_square = fields[QUARTER_SUM_D2 as usize] / pixels;
    let variance = (mean_square - mean * mean).max(0.0);
    let sigma = variance.sqrt() / std::f32::consts::SQRT_2;
    let expected = curve.sigma_at(luma);

    let static_quarter = mean.abs() < QUARTER_STATIC_GATE;
    let has_grain = sigma > RHO_SIGMA_GATE;
    let not_motion = sigma < MOTION_FACTOR * expected;
    let luma_min = fields[QUARTER_LUMA_MIN as usize];
    let luma_max = fields[QUARTER_LUMA_MAX as usize];
    let unclipped = luma_min >= CLIP_LOW && luma_max <= CLIP_HIGH;
    let smooth = fields[QUARTER_FLATNESS as usize] < QUARTER_FLAT_FACTOR * sigma * sigma;

    let flat = static_quarter && has_grain && not_motion && unclipped && smooth;
    QuarterClass { flat, luma }
}
