use std::f32::consts::SQRT_2;

use cubecl::prelude::*;

use super::select::{clear_batch, order_key, resolve_queries, select_result};
use crate::noise::curve::QUARTER_STATIC_GATE;
use crate::noise::reading::{
    IMMERKAER_QUERY_EXCLUDED,
    IMMERKAER_QUERY_REFERENCE,
    READING_ACCEPTED,
    READING_IMMERKAER_LOW,
    READING_RHO,
    READING_SAMPLE_VALID,
    READING_SIGMA,
    READING_SIGMA_LOW,
    READING_STATIC_FRACTION,
    READING_WORDS,
    SAMPLE_QUERIES,
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
fn median_from_pair(low: f32, high: f32) -> f32 {
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
    immerkaer_scale: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] temporal: bool,
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

/// Resolves every query of both batches into one slot's reading, then clears both batches.
///
/// The Immerkær words take each channel's lower quartile of its cube sigmas. With `temporal`, the
/// sample is valid when the accepted blocks make up at least
/// [STATIC_FRACTION_MIN](crate::noise::temporal::STATIC_FRACTION_MIN) of the frame and at least
/// one of them has a correlation. It then holds each channel's median and lower quartile, the
/// correlation median and the static fraction. Every word of a missing sample and every curve word
/// is written as 0.
///
/// `blocks` is the frame's block count and `min_accepted` the fewest accepted blocks that reach the
/// static fraction, both from the host. Without `temporal`, the sample batch's bindings are never
/// touched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_sample_finish(
    immerkaer_histogram: &mut Array<i32>,
    immerkaer_state: &mut Array<u32>,
    immerkaer_queries: &Array<u32>,
    immerkaer_tag_ranges: &Array<u32>,
    sample_histogram: &mut Array<i32>,
    sample_state: &mut Array<u32>,
    sample_queries: &Array<u32>,
    sample_tag_ranges: &Array<u32>,
    readings: &mut Array<f32>,
    slot: u32,
    blocks: f32,
    min_accepted: u32,
    #[comptime] immerkaer_tag_count: u32,
    #[comptime] immerkaer_query_count: u32,
    #[comptime] temporal: bool,
    #[comptime] cube_threads: u32,
) {
    let thread = UNIT_POS_X;
    let wanted_len = comptime!(u32::max(immerkaer_query_count, SAMPLE_QUERIES));
    let mut wanted = SharedMemory::<u32>::new(wanted_len as usize);
    let mut immerkaer_out =
        SharedMemory::<u32>::new(comptime!(immerkaer_query_count * SELECT_SLOT_WORDS) as usize);
    let mut sample_out = SharedMemory::<u32>::new(comptime!(SAMPLE_QUERIES * SELECT_SLOT_WORDS) as usize);

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

    if thread == 0 {
        let base = slot * READING_WORDS;

        #[unroll]
        for channel in 0..3u32 {
            let (low, count) = select_result(&immerkaer_out, comptime!(2 * channel));
            let (high, _) = select_result(&immerkaer_out, comptime!(2 * channel + 1));
            let quartile = quartile_from_pair(low, high, count);

            readings[(base + READING_IMMERKAER_LOW + channel) as usize] = quartile;
        }

        let mut accepted = 0u32;
        let mut correlated = 0u32;

        if comptime!(temporal) {
            let (_, accepted_count) = select_result(&sample_out, 0u32);
            let (_, rho_count) = select_result(&sample_out, SAMPLE_QUERY_RHO);

            accepted = accepted_count;
            correlated = rho_count;
        }

        let static_fraction = f32::cast_from(accepted) / blocks;
        let valid = accepted >= min_accepted && correlated > 0u32;
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

        for word in READING_ACCEPTED + 1..READING_WORDS {
            readings[(base + word) as usize] = 0.0f32;
        }
    }

    clear_batch(
        immerkaer_histogram,
        immerkaer_state,
        immerkaer_tag_count,
        immerkaer_query_count,
        cube_threads,
    );

    if comptime!(temporal) {
        clear_batch(
            sample_histogram,
            sample_state,
            SAMPLE_TAGS,
            SAMPLE_QUERIES,
            cube_threads,
        );
    }
}
