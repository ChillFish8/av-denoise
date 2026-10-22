use crate::noise::spatial::partials_len;
use crate::noise::temporal::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    STATIC_GATE,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    temporal_stats_blocks,
    temporal_stats_record_len,
};

/// The target statistics of one 8x8 quarter of a block record.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct QuarterSpec {
    pub mean: f32,
    pub sigma: f32,
    pub luma: f32,
    pub flatness: f32,
    pub luma_min: f32,
    pub luma_max: f32,
}

/// The target statistics of one block record.
///
/// `mean` is every channel's mean residual and `rho` channel 0's lag-1 correlation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BlockSpec {
    pub mean: f32,
    pub sigma: [f32; 3],
    pub rho: f32,
    pub quarters: [QuarterSpec; 4],
}

impl BlockSpec {
    /// A block whose channels all share `sigma`, with quiet quarters.
    pub(crate) fn uniform(mean: f32, sigma: f32, rho: f32) -> Self {
        Self {
            mean,
            sigma: [sigma; 3],
            rho,
            quarters: [QuarterSpec::default(); 4],
        }
    }
}

/// A value between 0 and 1 from a hash of `seed` and `index`.
pub(crate) fn hashed_unit(seed: u32, index: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;
    hash = hash.wrapping_mul(0x297A_2D39);
    hash ^= hash >> 15;

    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// The sums one channel of `pixels` residuals holds at `mean` and `sigma`.
fn channel_sums(mean: f32, sigma: f32, pixels: f32) -> (f32, f32) {
    let variance = 2.0 * sigma * sigma;
    let sum_d = mean * pixels;
    let sum_d2 = (variance + mean * mean) * pixels;

    (sum_d, sum_d2)
}

/// One slot's records, one per block of a `width` by `height` frame in row-major order.
pub(crate) fn frame_records(width: u32, height: u32, stored_ch: u32, blocks: &[BlockSpec]) -> Vec<f32> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let block_count = (blocks_x * blocks_y) as usize;
    let quarters_base = (2 * stored_ch + TEMPORAL_QUARTER_BASE) as usize;
    let stored = stored_ch as usize;

    let spec_count = blocks.len();
    assert_eq!(spec_count, block_count, "one spec per block");

    let mut records = vec![0.0f32; block_count * record_len];

    for (index, spec) in blocks.iter().enumerate() {
        let block_x = index as u32 % blocks_x;
        let block_y = index as u32 / blocks_x;
        let block_width = TEMPORAL_NOISE_BLOCK.min(width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = TEMPORAL_NOISE_BLOCK.min(height - block_y * TEMPORAL_NOISE_BLOCK);
        let pixels = (block_width * block_height) as f32;
        let pairs = (block_height * block_width.saturating_sub(1)) as f32;
        let record = &mut records[index * record_len..(index + 1) * record_len];

        for channel in 0..stored.min(3) {
            let (sum_d, sum_d2) = channel_sums(spec.mean, spec.sigma[channel], pixels);
            record[channel] = sum_d;
            record[stored + channel] = sum_d2;
        }

        let variance = 2.0 * spec.sigma[0] * spec.sigma[0];
        let mean_square = spec.mean * spec.mean;
        record[2 * stored] = (spec.rho * variance + mean_square) * pairs;

        for quarter_index in 0..TEMPORAL_QUARTERS {
            let offset_x = (quarter_index % 2) * TEMPORAL_QUARTER_SIZE;
            let offset_y = (quarter_index / 2) * TEMPORAL_QUARTER_SIZE;
            let quarter_width = block_width.saturating_sub(offset_x).min(TEMPORAL_QUARTER_SIZE);
            let quarter_height = block_height.saturating_sub(offset_y).min(TEMPORAL_QUARTER_SIZE);
            let quarter_pixels = (quarter_width * quarter_height) as f32;
            let quarter = spec.quarters[quarter_index as usize];
            let (sum_d, sum_d2) = channel_sums(quarter.mean, quarter.sigma, quarter_pixels);

            let start = quarters_base + (quarter_index * TEMPORAL_QUARTER_FIELDS) as usize;
            let fields = &mut record[start..start + TEMPORAL_QUARTER_FIELDS as usize];
            fields[QUARTER_SUM_D as usize] = sum_d;
            fields[QUARTER_SUM_D2 as usize] = sum_d2;
            fields[QUARTER_LUMA_SUM as usize] = quarter.luma * quarter_pixels;
            fields[QUARTER_FLATNESS as usize] = quarter.flatness;
            fields[QUARTER_LUMA_MIN as usize] = quarter.luma_min;
            fields[QUARTER_LUMA_MAX as usize] = quarter.luma_max;
        }
    }

    records
}

/// One slot's Immerkær partials with a hashed absolute response in every lane.
pub(crate) fn hashed_partials(width: u32, height: u32, seed: u32) -> Vec<f32> {
    let len = partials_len(width, height);
    let mut partials = Vec::with_capacity(len);

    for index in 0..len as u32 {
        let unit = hashed_unit(seed, index);
        let response = (1.0 + 4.0 * unit) / 255.0;
        partials.push(response * 256.0);
    }

    partials
}

/// A static block with hashed spread, a mean within half the static gate and sigmas between 2 and
/// 4 codes.
pub(crate) fn static_block(seed: u32, index: u32) -> BlockSpec {
    let mean_unit = hashed_unit(seed, 4 * index);
    let rho_unit = hashed_unit(seed, 4 * index + 1);
    let mean = (mean_unit - 0.5) * STATIC_GATE;
    let rho = 0.1 + 0.4 * rho_unit;

    let mut sigma = [0.0f32; 3];
    for (channel, channel_sigma) in sigma.iter_mut().enumerate() {
        let unit = hashed_unit(seed + channel as u32, 4 * index + 2);
        *channel_sigma = (2.0 + 2.0 * unit) / 255.0;
    }

    BlockSpec {
        mean,
        sigma,
        rho,
        ..BlockSpec::default()
    }
}

/// How many representable `f32` values lie between `left` and `right`.
///
/// A NaN on either side is as far apart as the gap can be.
pub(crate) fn ulp_gap(left: f32, right: f32) -> u32 {
    if left.is_nan() || right.is_nan() {
        return u32::MAX;
    }

    if left == right {
        return 0;
    }

    let left_bits = left.to_bits();
    let right_bits = right.to_bits();
    let left_wide = i64::from(left_bits);
    let right_wide = i64::from(right_bits);
    let gap = (left_wide - right_wide).unsigned_abs();
    let limit = u64::from(u32::MAX);

    gap.min(limit) as u32
}
