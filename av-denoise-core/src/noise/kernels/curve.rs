use cubecl::prelude::*;

use super::sample::{median_from_pair, static_quarter_sigma};
use super::select::{order_key, resolve_queries, select_result};
use crate::noise::curve::{CLIP_HIGH, CLIP_LOW, FLAT_FACTOR, FLAT_FLOOR, NOISE_CURVE_BINS};
use crate::noise::reading::{SAMPLE_QUERIES, SAMPLE_TAGS};
use crate::noise::select::{SELECT_CUBE_THREADS, SELECT_SLOT_WORDS, UNTAGGED};
use crate::noise::state::BLOCK_ACCEPTED;
use crate::noise::temporal::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    temporal_stats_record_len,
};

const CURVE_BINS: f32 = NOISE_CURVE_BINS as f32;
const LAST_BIN: f32 = (NOISE_CURVE_BINS - 1) as f32;
/// The sample batch's first queries, which hold channel 0's lower and upper median.
const SIGMA_MEDIAN_QUERIES: u32 = 2;

/// Writes every quarter's lane of the curve batch.
///
/// Each cube first resolves channel 0's median block sigma from the sample batch, which sets the
/// flat gate at `max(FLAT_FACTOR * median^2, FLAT_FLOOR)`. Thread `t` then writes quarter `t % 4`
/// of block `t / 4` at lane `t`. A quarter of an accepted block joins the bin of its mean luma when
/// it passes the static gates, its flatness is within the flat gate and no pixel sits past
/// [CLIP_LOW](crate::noise::curve::CLIP_LOW) or [CLIP_HIGH](crate::noise::curve::CLIP_HIGH). Every
/// other lane is `UNTAGGED`. Threads past the last quarter join the resolution and skip their
/// writes.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_curve_bins(
    records: &Array<f32>,
    block_flags: &Array<u32>,
    histogram: &Array<i32>,
    state: &Array<u32>,
    queries: &Array<u32>,
    tag_ranges: &Array<u32>,
    keys: &mut Array<u32>,
    tags: &mut Array<u32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
) {
    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let blocks = comptime!(blocks_x * height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let quarters = comptime!(blocks * TEMPORAL_QUARTERS);
    let record_len = comptime!(temporal_stats_record_len(stored_ch));
    let quarters_base = comptime!(2 * stored_ch + TEMPORAL_QUARTER_BASE);

    let thread = UNIT_POS_X;
    let mut wanted = SharedMemory::<u32>::new(SIGMA_MEDIAN_QUERIES as usize);
    let mut out = SharedMemory::<u32>::new(comptime!(SAMPLE_QUERIES * SELECT_SLOT_WORDS) as usize);

    if thread < SIGMA_MEDIAN_QUERIES {
        wanted[thread as usize] = thread;
    }

    sync_cube();

    resolve_queries(
        histogram,
        0u32,
        state,
        queries,
        tag_ranges,
        &wanted,
        SIGMA_MEDIAN_QUERIES,
        2u32,
        &mut out,
        SAMPLE_TAGS,
        SAMPLE_QUERIES,
        SELECT_CUBE_THREADS,
    );

    let (median_low, _) = select_result(&out, 0u32);
    let (median_high, _) = select_result(&out, 1u32);
    let sigma_median = median_from_pair(median_low, median_high);
    let flat_variance = FLAT_FACTOR * sigma_median * sigma_median;
    let flat_limit = f32::max(flat_variance, FLAT_FLOOR);

    let index = ABSOLUTE_POS_X;

    if index < quarters {
        let block = index / TEMPORAL_QUARTERS;
        let quarter = index % TEMPORAL_QUARTERS;
        let block_x = block % blocks_x;
        let block_y = block / blocks_x;
        let block_width = u32::min(TEMPORAL_NOISE_BLOCK, width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = u32::min(TEMPORAL_NOISE_BLOCK, height - block_y * TEMPORAL_NOISE_BLOCK);

        let offset_x = (quarter % 2) * TEMPORAL_QUARTER_SIZE;
        let offset_y = (quarter / 2) * TEMPORAL_QUARTER_SIZE;
        let inside_x = block_width - u32::min(block_width, offset_x);
        let inside_y = block_height - u32::min(block_height, offset_y);
        let quarter_width = u32::min(inside_x, TEMPORAL_QUARTER_SIZE);
        let quarter_height = u32::min(inside_y, TEMPORAL_QUARTER_SIZE);
        let pixels = f32::cast_from(quarter_width * quarter_height);
        let start = block * record_len + quarters_base + quarter * TEMPORAL_QUARTER_FIELDS;

        let (passes, sigma) = static_quarter_sigma(records, start, pixels);
        let accepted = (block_flags[block as usize] & BLOCK_ACCEPTED) != 0u32;

        let luma_sum = records[(start + QUARTER_LUMA_SUM) as usize];
        let flatness = records[(start + QUARTER_FLATNESS) as usize];
        let luma_min = records[(start + QUARTER_LUMA_MIN) as usize];
        let luma_max = records[(start + QUARTER_LUMA_MAX) as usize];

        let divisor = select(pixels > 0.0f32, pixels, 1.0f32);
        let mean_luma = luma_sum / divisor;
        let scaled = f32::max(mean_luma * CURVE_BINS, 0.0f32);
        let clamped = f32::min(scaled, LAST_BIN);
        let bin = u32::cast_from(clamped);

        let textured = flatness > flat_limit;
        let clipped = luma_min < CLIP_LOW || luma_max > CLIP_HIGH;
        let joins = accepted && passes && !textured && !clipped;

        keys[index as usize] = order_key(sigma);
        tags[index as usize] = select(joins, bin, UNTAGGED);
    }
}
