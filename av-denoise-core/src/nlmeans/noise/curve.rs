use super::{
    AcceptedBlock,
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    RHO_SIGMA_GATE,
    STATIC_GATE,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    median,
    sort_ascending,
    temporal_stats_record_len,
};

/// How many luma bins the noise curve spans.
pub const NOISE_CURVE_BINS: usize = 16;
/// The fewest quarters a bin needs before its median is trusted.
const MIN_QUARTERS_PER_BIN: usize = 32;
/// The fewest populated bins a frame needs before it gets a curve.
const MIN_POPULATED_BINS: usize = 3;
/// A quarter with a pixel at or below this luma may be clipped, so its noise reads low.
pub(super) const CLIP_LOW: f32 = 4.0 / 255.0;
/// A quarter with a pixel at or above this luma may be clipped, so its noise reads low.
pub(super) const CLIP_HIGH: f32 = 251.0 / 255.0;
/// How much texture a quarter may carry, as a fraction of the frame's noise variance.
const FLAT_FACTOR: f32 = 0.5;
/// The flat gate never tightens below one code of squared gradient.
const FLAT_FLOOR: f32 = (1.0 / 255.0) * (1.0 / 255.0);
/// How large a quarter's mean residual can be and still count as static.
///
/// A quarter averages 64 pixels rather than a block's 256, so the noise in its mean is twice as
/// large. The gate is widened by the same factor of 2 so static quarters pass as often as
/// static blocks.
pub(super) const QUARTER_STATIC_GATE: f32 = 2.0 * STATIC_GATE;

/// How much noisier each brightness level is than the frame's median quarter.
///
/// Bin `i` is centred at luma `(i + 0.5) / 16`, between 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseCurve {
    pub(crate) ratios: [f32; NOISE_CURVE_BINS],
    /// The median sigma of the quarters the curve was built from, which every ratio is relative to.
    pub(crate) sigma_quarter_median: f32,
}

impl NoiseCurve {
    /// The ratio at `luma`, interpolated between bin centres the same way the fused kernel reads it.
    pub(crate) fn ratio_at(&self, luma: f32) -> f32 {
        let bins = NOISE_CURVE_BINS as f32;
        let position = (luma * bins - 0.5).clamp(0.0, bins - 1.0);
        let lower = (position as usize).min(NOISE_CURVE_BINS - 2);
        let fraction = position - lower as f32;
        let lower_ratio = self.ratios[lower];
        let upper_ratio = self.ratios[lower + 1];
        lower_ratio + (upper_ratio - lower_ratio) * fraction
    }

    /// The noise sigma the curve predicts at `luma`.
    pub(crate) fn sigma_at(&self, luma: f32) -> f32 {
        self.ratio_at(luma) * self.sigma_quarter_median
    }
}

/// One 8x8 quarter that passed its own static gates.
struct CurveQuarter {
    sigma: f32,
    mean_luma: f32,
    flatness: f32,
    luma_min: f32,
    luma_max: f32,
}

/// Builds a frame's noise curve from the static, flat, unclipped quarters of its accepted blocks.
///
/// A quarter counts only when its own mean residual and sigma pass the static gates. The flat
/// gate scales with `sigma_block_median`, the frame's median block sigma. Each bin takes the
/// median sigma of its quarters, and empty bins are filled by linear interpolation between
/// their populated neighbours. Values past the first and last populated bins are held flat.
/// Every entry is then divided by the median sigma of all passing quarters.
pub(in crate::nlmeans) fn build_noise_curve(
    records: &[f32],
    stored_ch: u32,
    accepted: &[AcceptedBlock],
    sigma_block_median: f32,
) -> Option<NoiseCurve> {
    let quarters = static_quarters(records, stored_ch, accepted);
    if quarters.is_empty() {
        return None;
    }

    let mut quarter_sigmas: Vec<f32> = quarters.iter().map(|quarter| quarter.sigma).collect();
    sort_ascending(&mut quarter_sigmas);
    let sigma_quarter_median = median(&quarter_sigmas);
    if sigma_quarter_median <= 0.0 {
        return None;
    }

    let flat_variance = FLAT_FACTOR * sigma_block_median * sigma_block_median;
    let flat_limit = flat_variance.max(FLAT_FLOOR);

    let mut bins: [Vec<f32>; NOISE_CURVE_BINS] = Default::default();
    for quarter in &quarters {
        let textured = quarter.flatness > flat_limit;
        let clipped = quarter.luma_min < CLIP_LOW || quarter.luma_max > CLIP_HIGH;
        if textured || clipped {
            continue;
        }

        let scaled = quarter.mean_luma * NOISE_CURVE_BINS as f32;
        let bin = (scaled.max(0.0) as usize).min(NOISE_CURVE_BINS - 1);
        bins[bin].push(quarter.sigma);
    }

    let mut populated: Vec<(usize, f32)> = Vec::new();
    for (index, sigmas) in bins.iter_mut().enumerate() {
        if sigmas.len() < MIN_QUARTERS_PER_BIN {
            continue;
        }

        sort_ascending(sigmas);
        populated.push((index, median(sigmas)));
    }

    if populated.len() < MIN_POPULATED_BINS {
        return None;
    }

    let mut ratios = [0.0f32; NOISE_CURVE_BINS];
    for (index, ratio) in ratios.iter_mut().enumerate() {
        let sigma = interpolate(&populated, index);
        *ratio = sigma / sigma_quarter_median;
    }

    Some(NoiseCurve {
        ratios,
        sigma_quarter_median,
    })
}

/// The quarters of `accepted` whose own mean residual and sigma pass the static gates.
///
/// A quarter's mean residual is checked against [QUARTER_STATIC_GATE], and its sigma uses the
/// same formula as its block's. A quarter lying wholly outside the frame holds no pixels and is
/// skipped.
fn static_quarters(records: &[f32], stored_ch: u32, accepted: &[AcceptedBlock]) -> Vec<CurveQuarter> {
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let quarters_base = (2 * stored_ch + TEMPORAL_QUARTER_BASE) as usize;

    let mut quarters = Vec::new();
    for block in accepted {
        let record = &records[block.index * record_len..(block.index + 1) * record_len];

        for quarter_index in 0..TEMPORAL_QUARTERS {
            let offset_x = (quarter_index % 2) * TEMPORAL_QUARTER_SIZE;
            let offset_y = (quarter_index / 2) * TEMPORAL_QUARTER_SIZE;
            let quarter_w = block.width.saturating_sub(offset_x).min(TEMPORAL_QUARTER_SIZE);
            let quarter_h = block.height.saturating_sub(offset_y).min(TEMPORAL_QUARTER_SIZE);
            let pixels = (quarter_w * quarter_h) as f32;
            if pixels == 0.0 {
                continue;
            }

            let start = quarters_base + (quarter_index * TEMPORAL_QUARTER_FIELDS) as usize;
            let fields = &record[start..start + TEMPORAL_QUARTER_FIELDS as usize];

            let mean = fields[QUARTER_SUM_D as usize] / pixels;
            if mean.abs() >= QUARTER_STATIC_GATE {
                continue;
            }

            let mean_square = fields[QUARTER_SUM_D2 as usize] / pixels;
            let variance = (mean_square - mean * mean).max(0.0);
            let sigma = variance.sqrt() / std::f32::consts::SQRT_2;
            if sigma <= RHO_SIGMA_GATE {
                continue;
            }

            quarters.push(CurveQuarter {
                sigma,
                mean_luma: fields[QUARTER_LUMA_SUM as usize] / pixels,
                flatness: fields[QUARTER_FLATNESS as usize],
                luma_min: fields[QUARTER_LUMA_MIN as usize],
                luma_max: fields[QUARTER_LUMA_MAX as usize],
            });
        }
    }

    quarters
}

/// The curve's sigma at bin `index`, linear between populated bins and flat beyond them.
fn interpolate(populated: &[(usize, f32)], index: usize) -> f32 {
    let (first_index, first_sigma) = populated[0];
    let (last_index, last_sigma) = populated[populated.len() - 1];
    if index <= first_index {
        return first_sigma;
    }
    if index >= last_index {
        return last_sigma;
    }

    let upper = populated
        .iter()
        .position(|&(bin, _)| bin >= index)
        .expect("a populated bin lies at or above every interior index");
    let (high_index, high_sigma) = populated[upper];
    let (low_index, low_sigma) = populated[upper - 1];
    let span = (high_index - low_index) as f32;
    let fraction = (index - low_index) as f32 / span;
    low_sigma + (high_sigma - low_sigma) * fraction
}
