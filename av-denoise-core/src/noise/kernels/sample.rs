use std::f32::consts::SQRT_2;

use cubecl::prelude::*;

use super::select::{clear_batch, order_key, resolve_queries, select_result};
use crate::noise::curve::{MIN_POPULATED_BINS, MIN_QUARTERS_PER_BIN, QUARTER_STATIC_GATE};
use crate::noise::reading::{
    CURVE_QUERIES,
    CURVE_TAGS,
    IMMERKAER_QUERY_EXCLUDED,
    IMMERKAER_QUERY_REFERENCE,
    READING_ACCEPTED,
    READING_CURVE_VALID,
    READING_IMMERKAER_LOW,
    READING_POPULATED_BINS,
    READING_QUARTER_MEDIAN,
    READING_RATIOS,
    READING_RHO,
    READING_SAMPLE_VALID,
    READING_SIGMA,
    READING_SIGMA_LOW,
    READING_STATIC_FRACTION,
    READING_WORDS,
    SAMPLE_QUERIES,
    SAMPLE_QUERY_QUARTER,
    SAMPLE_QUERY_RHO,
    SAMPLE_TAGS,
    TAG_EXCLUDED,
    TAG_QUARTER,
    TAG_REFERENCE,
    TAG_RHO,
};
use crate::noise::select::{SELECT_CUBE_THREADS, SELECT_SLOT_WORDS, UNTAGGED};
use crate::noise::state::{
    BLOCK_ACCEPTED,
    BLOCK_CANDIDATE,
    BLOCK_FIELDS,
    BLOCK_VALUE_MEAN_LAG,
    BLOCK_VALUE_MEAN0,
    BLOCK_VALUE_PAIRS,
    BLOCK_VALUE_SPARE,
    BLOCK_VALUE_VAR0,
};
use crate::noise::temporal::{
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    RHO_SIGMA_GATE,
    SIGMA_OUTLIER_FACTOR,
    STATIC_GATE,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    temporal_stats_record_len,
};
use crate::noise::{IMMERKAER_BLOCK_X, IMMERKAER_BLOCK_Y};

const MIN_BIN_QUARTERS: u32 = MIN_QUARTERS_PER_BIN as u32;
const MIN_CURVE_BINS: u32 = MIN_POPULATED_BINS as u32;
const LAST_CURVE_BIN: u32 = CURVE_TAGS - 1;

/// The mean residual, its variance and the sigma of `pixels` residuals from their two sums.
///
/// The sigma halves the variance, since a residual of two frames carries the noise of both.
#[cube]
fn residual_sigma(sum_d: f32, sum_d2: f32, pixels: f32) -> (f32, f32, f32) {
    let mean = sum_d / pixels;
    let mean_square = sum_d2 / pixels;
    let variance = f32::max(mean_square - mean * mean, 0.0f32);
    let sigma = f32::sqrt(variance) / SQRT_2;

    (mean, variance, sigma)
}

/// Whether a quarter of `pixels` residuals passes the curve's static gates, with its sigma.
///
/// `start` is the first of the quarter's fields in `records`. A quarter with no pixels never passes.
#[cube]
pub(crate) fn static_quarter_sigma(records: &Array<f32>, start: u32, pixels: f32) -> (bool, f32) {
    let sum_d = records[(start + QUARTER_SUM_D) as usize];
    let sum_d2 = records[(start + QUARTER_SUM_D2) as usize];
    let has_pixels = pixels > 0.0f32;
    let divisor = select(has_pixels, pixels, 1.0f32);
    let (mean, _, sigma) = residual_sigma(sum_d, sum_d2, divisor);

    let quiet = f32::abs(mean) < QUARTER_STATIC_GATE;
    let measurable = sigma > RHO_SIGMA_GATE;

    (has_pixels && quiet && measurable, sigma)
}

/// The lower quartile from its floor and ceiling elements in a set of `count`.
#[cube]
fn quartile_from_pair(low: f32, high: f32, count: u32) -> f32 {
    let last = u32::max(count, 1u32) - 1;
    let fraction = f32::cast_from(last % 4) * 0.25f32;

    low + fraction * (high - low)
}

/// The median from its two middle elements.
#[cube]
pub(crate) fn median_from_pair(low: f32, high: f32) -> f32 {
    (low + high) * 0.5f32
}

/// Writes each Immerkær cube's sigmas and each block's gate, values and reference lane.
///
/// Thread `i` writes cube `i`'s three channel lanes at `c * cubes + i`, each tagged with its channel,
/// or `UNTAGGED` past the active channels or for a cube with no interior. With `temporal`, it also
/// derives block `i`'s per-channel sigma, mean residual, lag-1 mean, variance and pair count, flags
/// it a candidate when its mean residual passes the static gate, and writes its channel-0 sigma at
/// `3 * cubes + i`. A candidate above the measurable-noise gate joins the reference set and one at
/// or below it the excluded set. Without `temporal`, `records`, `block_values` and `block_flags`
/// are never touched.
///
/// The grid then clears the histogram and state of the immerkaer batch and, with `temporal` and
/// `curve`, of the sample and curve batches, so each reading's passes start from zero. Bindings of a
/// batch the comptime flags leave out are never touched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_block_derive(
    partials: &Array<f32>,
    records: &Array<f32>,
    block_values: &mut Array<f32>,
    block_flags: &mut Array<u32>,
    keys: &mut Array<u32>,
    tags: &mut Array<u32>,
    immerkaer_histogram: &mut Array<i32>,
    immerkaer_state: &mut Array<u32>,
    sample_histogram: &mut Array<i32>,
    sample_state: &mut Array<u32>,
    curve_histogram: &mut Array<i32>,
    curve_state: &mut Array<u32>,
    immerkaer_scale: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] temporal: bool,
    #[comptime] curve: bool,
    #[comptime] immerkaer_tag_count: u32,
    #[comptime] immerkaer_query_count: u32,
) {
    let cubes_x = comptime!(width.div_ceil(IMMERKAER_BLOCK_X));
    let cubes = comptime!(cubes_x * height.div_ceil(IMMERKAER_BLOCK_Y));
    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let blocks = comptime!(blocks_x * height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let record_len = comptime!(temporal_stats_record_len(stored_ch));

    let index = ABSOLUTE_POS_X;

    if index < cubes {
        let cube_x = index % cubes_x;
        let cube_y = index / cubes_x;
        let tile_x0 = cube_x * IMMERKAER_BLOCK_X;
        let tile_y0 = cube_y * IMMERKAER_BLOCK_Y;
        let tile_x1 = u32::min(tile_x0 + IMMERKAER_BLOCK_X, width);
        let tile_y1 = u32::min(tile_y0 + IMMERKAER_BLOCK_Y, height);
        let overlap_x0 = u32::max(tile_x0, 1u32);
        let overlap_y0 = u32::max(tile_y0, 1u32);
        let overlap_x1 = u32::min(tile_x1, comptime!(width - 1));
        let overlap_y1 = u32::min(tile_y1, comptime!(height - 1));

        let interior = overlap_x1 > overlap_x0 && overlap_y1 > overlap_y0;
        let span_x = select(interior, overlap_x1 - overlap_x0, 1u32);
        let span_y = select(interior, overlap_y1 - overlap_y0, 1u32);
        let area = f32::cast_from(span_x * span_y);
        let divisor = 6.0f32 * area;

        #[unroll]
        for channel in 0..3u32 {
            let sum = partials[(index * 4 + channel) as usize];
            let sigma = immerkaer_scale * sum / divisor;
            let active = interior && comptime!(channel < channels);
            let lane = channel * cubes + index;

            keys[lane as usize] = order_key(sigma);
            tags[lane as usize] = select(active, channel, UNTAGGED);
        }
    }

    if comptime!(temporal) && index < blocks {
        let block_x = index % blocks_x;
        let block_y = index / blocks_x;
        let block_width = u32::min(TEMPORAL_NOISE_BLOCK, width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = u32::min(TEMPORAL_NOISE_BLOCK, height - block_y * TEMPORAL_NOISE_BLOCK);
        let pixels = f32::cast_from(block_width * block_height);
        let pairs = f32::cast_from(block_height * (block_width - 1));

        let base = index * record_len;
        let value_base = index * BLOCK_FIELDS;

        let mut mean0 = 0.0f32;
        let mut var0 = 0.0f32;
        let mut sigma0 = 0.0f32;

        #[unroll]
        for channel in 0..3u32 {
            if comptime!(channel < channels) {
                let sum_d = records[(base + channel) as usize];
                let sum_d2 = records[(base + stored_ch + channel) as usize];
                let (mean, variance, sigma) = residual_sigma(sum_d, sum_d2, pixels);
                block_values[(value_base + channel) as usize] = sigma;

                if comptime!(channel == 0) {
                    mean0 = mean;
                    var0 = variance;
                    sigma0 = sigma;
                }
            } else {
                block_values[(value_base + channel) as usize] = 0.0f32;
            }
        }

        let has_pairs = pairs > 0.0f32;
        let pair_divisor = select(has_pairs, pairs, 1.0f32);
        let lag_sum = records[(base + 2 * stored_ch) as usize];
        let mean_lag = select(has_pairs, lag_sum / pair_divisor, 0.0f32);

        block_values[(value_base + BLOCK_VALUE_MEAN0) as usize] = mean0;
        block_values[(value_base + BLOCK_VALUE_MEAN_LAG) as usize] = mean_lag;
        block_values[(value_base + BLOCK_VALUE_VAR0) as usize] = var0;
        block_values[(value_base + BLOCK_VALUE_PAIRS) as usize] = pairs;
        block_values[(value_base + BLOCK_VALUE_SPARE) as usize] = 0.0f32;

        let candidate = f32::abs(mean0) < STATIC_GATE;
        let measurable = sigma0 > RHO_SIGMA_GATE;
        let set_tag = select(measurable, TAG_REFERENCE, TAG_EXCLUDED);
        let lane = 3 * cubes + index;

        block_flags[index as usize] = select(candidate, BLOCK_CANDIDATE, 0u32);
        keys[lane as usize] = order_key(sigma0);
        tags[lane as usize] = select(candidate, set_tag, UNTAGGED);
    }

    let stride = CUBE_COUNT_X * CUBE_DIM_X;

    clear_batch(
        immerkaer_histogram,
        immerkaer_state,
        index,
        stride,
        immerkaer_tag_count,
        immerkaer_query_count,
    );

    if comptime!(temporal) {
        clear_batch(
            sample_histogram,
            sample_state,
            index,
            stride,
            SAMPLE_TAGS,
            SAMPLE_QUERIES,
        );
    }

    if comptime!(curve) {
        clear_batch(
            curve_histogram,
            curve_state,
            index,
            stride,
            CURVE_TAGS,
            CURVE_QUERIES,
        );
    }
}

/// Accepts the candidate blocks under the outlier ceiling and writes every lane of the sample batch.
///
/// Each cube resolves the reference set's quartile pair, its first and last element, and the
/// excluded count from the immerkaer batch. The ceiling is the reference's lower quartile times
/// [SIGMA_OUTLIER_FACTOR](crate::noise::temporal::SIGMA_OUTLIER_FACTOR), or 0 for an empty
/// reference. When candidates were excluded and the reference has no spread, the ceiling is not
/// trusted and no block is accepted.
///
/// Thread `b` then writes block `b`'s lanes. An accepted block's sigma joins its channel's set, its
/// clamped correlation joins the correlation set when it has pixel pairs, and, with `luma_fields`,
/// each quarter that passes the static gates joins the quarter set. Every other lane is `UNTAGGED`.
/// Threads past the last block join the resolution and skip their writes.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_block_accept(
    records: &Array<f32>,
    block_values: &Array<f32>,
    block_flags: &mut Array<u32>,
    histogram: &Array<i32>,
    state: &Array<u32>,
    queries: &Array<u32>,
    tag_ranges: &Array<u32>,
    keys: &mut Array<u32>,
    tags: &mut Array<u32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] luma_fields: bool,
    #[comptime] tag_count: u32,
    #[comptime] query_count: u32,
) {
    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let blocks = comptime!(blocks_x * height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let record_len = comptime!(temporal_stats_record_len(stored_ch));
    let quarters_base = comptime!(2 * stored_ch + TEMPORAL_QUARTER_BASE);
    let resolved = comptime!(IMMERKAER_QUERY_EXCLUDED + 1 - IMMERKAER_QUERY_REFERENCE);

    let thread = UNIT_POS_X;
    let mut wanted = SharedMemory::<u32>::new(resolved as usize);
    let mut out = SharedMemory::<u32>::new(comptime!(query_count * SELECT_SLOT_WORDS) as usize);

    if thread < resolved {
        wanted[thread as usize] = IMMERKAER_QUERY_REFERENCE + thread;
    }

    sync_cube();

    resolve_queries(
        histogram,
        0u32,
        state,
        queries,
        tag_ranges,
        &wanted,
        resolved,
        2u32,
        &mut out,
        tag_count,
        query_count,
        SELECT_CUBE_THREADS,
    );

    let (quartile_low, reference_count) = select_result(&out, IMMERKAER_QUERY_REFERENCE);
    let (quartile_high, _) = select_result(&out, comptime!(IMMERKAER_QUERY_REFERENCE + 1));
    let (lowest, _) = select_result(&out, comptime!(IMMERKAER_QUERY_REFERENCE + 2));
    let (highest, _) = select_result(&out, comptime!(IMMERKAER_QUERY_REFERENCE + 3));
    let (_, excluded_count) = select_result(&out, IMMERKAER_QUERY_EXCLUDED);

    let quartile = quartile_from_pair(quartile_low, quartile_high, reference_count);
    let has_reference = reference_count > 0u32;
    let ceiling = select(has_reference, quartile * SIGMA_OUTLIER_FACTOR, 0.0f32);
    let has_spread = has_reference && highest > lowest;
    let trusted = excluded_count == 0u32 || has_spread;

    let index = ABSOLUTE_POS_X;

    if index < blocks {
        let block_x = index % blocks_x;
        let block_y = index / blocks_x;
        let block_width = u32::min(TEMPORAL_NOISE_BLOCK, width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = u32::min(TEMPORAL_NOISE_BLOCK, height - block_y * TEMPORAL_NOISE_BLOCK);
        let base = index * record_len;
        let value_base = index * BLOCK_FIELDS;

        let flags = block_flags[index as usize];
        let candidate = (flags & BLOCK_CANDIDATE) != 0u32;
        let sigma0 = block_values[value_base as usize];
        let under_ceiling = sigma0 <= ceiling;
        let measurable = sigma0 > RHO_SIGMA_GATE;
        let accepted = trusted && candidate && under_ceiling && measurable;

        block_flags[index as usize] = flags | select(accepted, BLOCK_ACCEPTED, 0u32);

        #[unroll]
        for channel in 0..3u32 {
            let sigma = block_values[(value_base + channel) as usize];
            let active = accepted && comptime!(channel < channels);
            let lane = channel * blocks + index;

            keys[lane as usize] = order_key(sigma);
            tags[lane as usize] = select(active, channel, UNTAGGED);
        }

        let mean0 = block_values[(value_base + BLOCK_VALUE_MEAN0) as usize];
        let mean_lag = block_values[(value_base + BLOCK_VALUE_MEAN_LAG) as usize];
        let var0 = block_values[(value_base + BLOCK_VALUE_VAR0) as usize];
        let pairs = block_values[(value_base + BLOCK_VALUE_PAIRS) as usize];
        let has_rho = accepted && pairs > 0.0f32;
        let variance = select(has_rho, var0, 1.0f32);
        let rho = (mean_lag - mean0 * mean0) / variance;
        let clamped_rho = f32::clamp(rho, 0.0f32, 1.0f32);
        let rho_lane = 3 * blocks + index;

        keys[rho_lane as usize] = order_key(clamped_rho);
        tags[rho_lane as usize] = select(has_rho, TAG_RHO, UNTAGGED);

        #[unroll]
        for quarter in 0..TEMPORAL_QUARTERS {
            let offset_x = comptime!((quarter % 2) * TEMPORAL_QUARTER_SIZE);
            let offset_y = comptime!((quarter / 2) * TEMPORAL_QUARTER_SIZE);
            let inside_x = block_width - u32::min(block_width, offset_x);
            let inside_y = block_height - u32::min(block_height, offset_y);
            let quarter_width = u32::min(inside_x, TEMPORAL_QUARTER_SIZE);
            let quarter_height = u32::min(inside_y, TEMPORAL_QUARTER_SIZE);
            let pixels = f32::cast_from(quarter_width * quarter_height);
            let start = base + quarters_base + comptime!(quarter * TEMPORAL_QUARTER_FIELDS);
            let lane = 4 * blocks + 4 * index + quarter;

            if comptime!(luma_fields) {
                let (passes, sigma) = static_quarter_sigma(records, start, pixels);
                let joins = accepted && passes;

                keys[lane as usize] = order_key(sigma);
                tags[lane as usize] = select(joins, TAG_QUARTER, UNTAGGED);
            } else {
                keys[lane as usize] = 0u32;
                tags[lane as usize] = UNTAGGED;
            }
        }
    }
}

/// Resolves every query of a slot's batches into the slot's reading.
///
/// The Immerkær words take each channel's lower quartile of its cube sigmas. With `temporal`, the
/// sample is valid when the accepted blocks make up at least
/// [STATIC_FRACTION_MIN](crate::noise::temporal::STATIC_FRACTION_MIN) of the frame and at least
/// one of them has a correlation. It then holds each channel's median and lower quartile, the
/// correlation median and the static fraction.
///
/// With `curve`, each bin holding at least
/// [MIN_QUARTERS_PER_BIN](crate::noise::curve::MIN_QUARTERS_PER_BIN) quarters takes their median.
/// The curve is valid with a sample, a positive static quarter median and at least
/// [MIN_POPULATED_BINS](crate::noise::curve::MIN_POPULATED_BINS) such bins. Each ratio is then the
/// bin's sigma, interpolated between populated bins and held flat past the outer ones, over the
/// quarter median. The populated bin count is always written. Every other word of a missing sample
/// or curve is written as 0.
///
/// `blocks` is the frame's block count and `min_accepted` the fewest accepted blocks that reach the
/// static fraction, both from the host. Bindings of a batch the comptime flags leave out are never
/// touched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_sample_finish(
    immerkaer_histogram: &Array<i32>,
    immerkaer_state: &Array<u32>,
    immerkaer_queries: &Array<u32>,
    immerkaer_tag_ranges: &Array<u32>,
    sample_histogram: &Array<i32>,
    sample_state: &Array<u32>,
    sample_queries: &Array<u32>,
    sample_tag_ranges: &Array<u32>,
    curve_histogram: &Array<i32>,
    curve_state: &Array<u32>,
    curve_queries: &Array<u32>,
    curve_tag_ranges: &Array<u32>,
    readings: &mut Array<f32>,
    slot: u32,
    blocks: f32,
    min_accepted: u32,
    #[comptime] immerkaer_tag_count: u32,
    #[comptime] immerkaer_query_count: u32,
    #[comptime] temporal: bool,
    #[comptime] curve: bool,
    #[comptime] cube_threads: u32,
) {
    let thread = UNIT_POS_X;
    let sample_wanted = comptime!(u32::max(immerkaer_query_count, SAMPLE_QUERIES));
    let wanted_len = comptime!(u32::max(sample_wanted, CURVE_QUERIES));
    let mut wanted = SharedMemory::<u32>::new(wanted_len as usize);
    let mut immerkaer_out =
        SharedMemory::<u32>::new(comptime!(immerkaer_query_count * SELECT_SLOT_WORDS) as usize);
    let mut sample_out = SharedMemory::<u32>::new(comptime!(SAMPLE_QUERIES * SELECT_SLOT_WORDS) as usize);
    let mut curve_out = SharedMemory::<u32>::new(comptime!(CURVE_QUERIES * SELECT_SLOT_WORDS) as usize);

    if thread < wanted_len {
        wanted[thread as usize] = thread;
    }

    sync_cube();

    resolve_queries(
        immerkaer_histogram,
        0u32,
        immerkaer_state,
        immerkaer_queries,
        immerkaer_tag_ranges,
        &wanted,
        immerkaer_query_count,
        2u32,
        &mut immerkaer_out,
        immerkaer_tag_count,
        immerkaer_query_count,
        cube_threads,
    );

    if comptime!(temporal) {
        resolve_queries(
            sample_histogram,
            0u32,
            sample_state,
            sample_queries,
            sample_tag_ranges,
            &wanted,
            SAMPLE_QUERIES,
            2u32,
            &mut sample_out,
            SAMPLE_TAGS,
            SAMPLE_QUERIES,
            cube_threads,
        );
    }

    if comptime!(curve) {
        resolve_queries(
            curve_histogram,
            0u32,
            curve_state,
            curve_queries,
            curve_tag_ranges,
            &wanted,
            CURVE_QUERIES,
            2u32,
            &mut curve_out,
            CURVE_TAGS,
            CURVE_QUERIES,
            cube_threads,
        );
    }

    let base = slot * READING_WORDS;
    let mut accepted = 0u32;
    let mut correlated = 0u32;

    if comptime!(temporal) {
        let (_, accepted_count) = select_result(&sample_out, 0u32);
        let (_, rho_count) = select_result(&sample_out, SAMPLE_QUERY_RHO);

        accepted = accepted_count;
        correlated = rho_count;
    }

    let valid = accepted >= min_accepted && correlated > 0u32;

    if thread == 0 {
        #[unroll]
        for channel in 0..3u32 {
            let (low, count) = select_result(&immerkaer_out, comptime!(2 * channel));
            let (high, _) = select_result(&immerkaer_out, comptime!(2 * channel + 1));
            let quartile = quartile_from_pair(low, high, count);

            readings[(base + READING_IMMERKAER_LOW + channel) as usize] = quartile;
        }

        let static_fraction = f32::cast_from(accepted) / blocks;
        let kept = select(valid, 1.0f32, 0.0f32);

        #[unroll]
        for channel in 0..3u32 {
            let first = comptime!(4 * channel);
            let (median_low, _) = select_result(&sample_out, first);
            let (median_high, _) = select_result(&sample_out, comptime!(first + 1));
            let (quartile_low, count) = select_result(&sample_out, comptime!(first + 2));
            let (quartile_high, _) = select_result(&sample_out, comptime!(first + 3));
            let sigma = median_from_pair(median_low, median_high);
            let sigma_low = quartile_from_pair(quartile_low, quartile_high, count);

            readings[(base + READING_SIGMA + channel) as usize] = select(valid, sigma, 0.0f32);
            readings[(base + READING_SIGMA_LOW + channel) as usize] = select(valid, sigma_low, 0.0f32);
        }

        let (rho_low, _) = select_result(&sample_out, SAMPLE_QUERY_RHO);
        let (rho_high, _) = select_result(&sample_out, comptime!(SAMPLE_QUERY_RHO + 1));
        let rho = median_from_pair(rho_low, rho_high);

        readings[(base + READING_SAMPLE_VALID) as usize] = kept;
        readings[(base + READING_RHO) as usize] = select(valid, rho, 0.0f32);
        readings[(base + READING_STATIC_FRACTION) as usize] = select(valid, static_fraction, 0.0f32);
        readings[(base + READING_ACCEPTED) as usize] = f32::cast_from(accepted);
    }

    if comptime!(curve) {
        let (quarter_low, quarter_count) = select_result(&sample_out, SAMPLE_QUERY_QUARTER);
        let (quarter_high, _) = select_result(&sample_out, comptime!(SAMPLE_QUERY_QUARTER + 1));
        let quarter_median = median_from_pair(quarter_low, quarter_high);

        let bin = u32::min(thread, LAST_CURVE_BIN);
        let (sigma, populated) = curve_bin_sigma(&curve_out, bin);

        let has_quarters = quarter_count > 0u32 && quarter_median > 0.0f32;
        let enough_bins = populated >= MIN_CURVE_BINS;
        let curve_valid = valid && has_quarters && enough_bins;
        let divisor = select(curve_valid, quarter_median, 1.0f32);
        let ratio = sigma / divisor;

        if thread < CURVE_TAGS {
            readings[(base + READING_RATIOS + thread) as usize] = select(curve_valid, ratio, 0.0f32);
        }

        if thread == 0 {
            readings[(base + READING_CURVE_VALID) as usize] = select(curve_valid, 1.0f32, 0.0f32);
            readings[(base + READING_QUARTER_MEDIAN) as usize] = select(curve_valid, quarter_median, 0.0f32);
            readings[(base + READING_POPULATED_BINS) as usize] = f32::cast_from(populated);
        }
    } else if thread == 0 {
        for word in READING_ACCEPTED + 1..READING_WORDS {
            readings[(base + word) as usize] = 0.0f32;
        }
    }
}

/// The curve's sigma at `bin` and how many bins are populated, from the resolved curve batch.
///
/// A bin is populated when it holds at least
/// [MIN_QUARTERS_PER_BIN](crate::noise::curve::MIN_QUARTERS_PER_BIN) quarters, and its sigma is
/// their median. Below the first populated bin the first one's sigma holds, and past the last the
/// last one's. Between them the sigma is linear between the nearest populated bin below `bin` and
/// the nearest at or above it, as the host's curve interpolates. The bins are walked in a runtime
/// loop, so every pick goes through `select`.
#[cube]
fn curve_bin_sigma(curve_out: &SharedMemory<u32>, bin: u32) -> (f32, u32) {
    let mut populated = 0u32;
    let mut first_index = 0u32;
    let mut first_sigma = 0.0f32;
    let mut last_index = 0u32;
    let mut last_sigma = 0.0f32;
    let mut low_index = 0u32;
    let mut low_sigma = 0.0f32;
    let mut high_index = 0u32;
    let mut high_sigma = 0.0f32;
    let mut has_high = false;

    for index in 0..CURVE_TAGS {
        let (median_low, count) = select_result(curve_out, 2 * index);
        let (median_high, _) = select_result(curve_out, 2 * index + 1);
        let median = median_from_pair(median_low, median_high);
        let filled = count >= MIN_BIN_QUARTERS;

        let is_first = filled && populated == 0u32;
        first_index = select(is_first, index, first_index);
        first_sigma = select(is_first, median, first_sigma);
        last_index = select(filled, index, last_index);
        last_sigma = select(filled, median, last_sigma);

        let below = filled && index < bin;
        low_index = select(below, index, low_index);
        low_sigma = select(below, median, low_sigma);

        let is_high = filled && index >= bin && !has_high;
        high_index = select(is_high, index, high_index);
        high_sigma = select(is_high, median, high_sigma);
        has_high = has_high || is_high;

        populated += select(filled, 1u32, 0u32);
    }

    let spread = high_index > low_index;
    let span_bins = select(spread, high_index - low_index, 1u32);
    let offset_bins = bin - u32::min(low_index, bin);
    let span = f32::cast_from(span_bins);
    let fraction = f32::cast_from(offset_bins) / span;
    let between = low_sigma + (high_sigma - low_sigma) * fraction;

    let past_last = select(bin >= last_index, last_sigma, between);
    let sigma = select(bin <= first_index, first_sigma, past_last);

    (sigma, populated)
}
