use super::{
    AcceptedBlock,
    TEMPORAL_LUMA_FLATNESS,
    TEMPORAL_LUMA_MAX,
    TEMPORAL_LUMA_MIN,
    TEMPORAL_LUMA_SUM,
    median,
    sort_ascending,
    temporal_stats_record_len,
};

/// How many luma bins the noise curve spans.
pub(crate) const NOISE_CURVE_BINS: usize = 16;
/// The fewest blocks a bin needs before its median is trusted.
const MIN_BLOCKS_PER_BIN: usize = 32;
/// The fewest populated bins a frame needs before it gets a curve.
const MIN_POPULATED_BINS: usize = 3;
/// A block with a pixel at or below this luma may be clipped, so its noise reads low.
const CLIP_LOW: f32 = 4.0 / 255.0;
/// A block with a pixel at or above this luma may be clipped, so its noise reads low.
const CLIP_HIGH: f32 = 251.0 / 255.0;
/// How much texture a block may carry, as a fraction of the frame's noise variance.
const FLAT_FACTOR: f32 = 0.5;
/// The flat gate never tightens below one code of squared gradient.
const FLAT_FLOOR: f32 = (1.0 / 255.0) * (1.0 / 255.0);

/// How much noisier each brightness level is than the frame's median block.
///
/// Bin `i` is centred at luma `(i + 0.5) / 16`, between 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseCurve {
    pub(crate) ratios: [f32; NOISE_CURVE_BINS],
}

/// Builds a frame's noise curve from its static, flat, unclipped blocks.
///
/// Each bin takes the median sigma of its blocks, and empty bins are filled by linear
/// interpolation between their populated neighbours. Values past the first and last populated
/// bins are held flat. Every entry is then divided by `sigma_median`.
pub(in crate::nlmeans) fn build_noise_curve(
    records: &[f32],
    stored_ch: u32,
    accepted: &[AcceptedBlock],
    sigma_median: f32,
) -> Option<NoiseCurve> {
    if sigma_median <= 0.0 {
        return None;
    }

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let luma_base = 2 * stored_ch as usize;
    let flat_limit = (FLAT_FACTOR * sigma_median * sigma_median).max(FLAT_FLOOR);

    let mut bins: [Vec<f32>; NOISE_CURVE_BINS] = Default::default();
    for block in accepted {
        let record = &records[block.index * record_len..(block.index + 1) * record_len];
        let flatness = record[luma_base + TEMPORAL_LUMA_FLATNESS as usize];
        let luma_min = record[luma_base + TEMPORAL_LUMA_MIN as usize];
        let luma_max = record[luma_base + TEMPORAL_LUMA_MAX as usize];
        let rejected = flatness > flat_limit || luma_min < CLIP_LOW || luma_max > CLIP_HIGH;
        if rejected {
            continue;
        }

        let mean_luma = record[luma_base + TEMPORAL_LUMA_SUM as usize] / block.pixels;
        let scaled = mean_luma * NOISE_CURVE_BINS as f32;
        let bin = (scaled.max(0.0) as usize).min(NOISE_CURVE_BINS - 1);
        bins[bin].push(block.sigmas[0]);
    }

    let mut populated: Vec<(usize, f32)> = Vec::new();
    for (index, sigmas) in bins.iter_mut().enumerate() {
        if sigmas.len() < MIN_BLOCKS_PER_BIN {
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
        *ratio = sigma / sigma_median;
    }

    Some(NoiseCurve { ratios })
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
