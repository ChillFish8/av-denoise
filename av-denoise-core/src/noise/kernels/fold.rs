use cubecl::prelude::*;

use crate::noise::correlation::CORRELATION_FACTOR_TABLE;
use crate::noise::curve::NOISE_CURVE_BINS;
use crate::noise::derived::{
    DERIVED_CONFIDENCE_SAD_FLOOR,
    DERIVED_CURVE_VALID,
    DERIVED_H2_INV_NORM,
    DERIVED_INPUT_NOISE_OFFSET,
    DERIVED_MAP_MODE,
    DERIVED_NOISE_OFFSET,
    DERIVED_PILOT_H2_INV_NORM,
    DERIVED_RHO,
    DERIVED_SAD_FLOOR,
    DERIVED_SIGMA,
    DERIVED_WEIGHT_SCALE,
    DERIVED_WORDS,
};
use crate::noise::estimator::{EMA_ALPHA, SIGMA_FLOOR};
use crate::noise::fold::{
    FOLD_BORROW_AHEAD,
    FOLD_CONFIDENCE_SAD_PER_SIGMA,
    FOLD_CURVE_APPLIES,
    FOLD_H2_AUTO,
    FOLD_H2_FIXED,
    FOLD_H2_PER_INVERSE_VARIANCE,
    FOLD_IMMERKAER_FACTOR,
    FOLD_MAIN_OFFSET,
    FOLD_MAP_KIND,
    FOLD_OFFSET_PER_VARIANCE,
    FOLD_PILOT_H2_FACTOR,
    FOLD_SAD_PER_SIGMA,
    FOLD_SEGMENT_RECIPROCALS,
    FOLD_SIGMA_SCALE,
    FOLD_WEIGHT_FLOOR,
    FOLD_WEIGHT_G_MAX,
    STATE_CLASSES_PRESENT,
    STATE_CURVE_VALID,
    STATE_LOW,
    STATE_LOW_UNBOOSTED,
    STATE_MEDIAN,
    STATE_QUARTER_MEDIAN,
    STATE_RATIOS,
    STATE_RHO,
    STATE_RHO_VALID,
    STATE_TEMPORAL_ONLY,
};
use crate::noise::reading::{
    READING_CURVE_VALID,
    READING_IMMERKAER_LOW,
    READING_QUARTER_MEDIAN,
    READING_RATIOS,
    READING_RHO,
    READING_SAMPLE_VALID,
    READING_SIGMA,
    READING_SIGMA_LOW,
    READING_WORDS,
};
use crate::noise::spatial::RESULTS_PER_SLOT;
use crate::noise::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};

const _: () = assert!(
    CORRELATION_FACTOR_TABLE.len() == 4,
    "the fold interpolates exactly three correlation segments"
);

const RHO_0: f32 = CORRELATION_FACTOR_TABLE[0].0;
const RHO_1: f32 = CORRELATION_FACTOR_TABLE[1].0;
const RHO_2: f32 = CORRELATION_FACTOR_TABLE[2].0;
const RHO_3: f32 = CORRELATION_FACTOR_TABLE[3].0;
const FACTOR_0: f32 = CORRELATION_FACTOR_TABLE[0].1;
const FACTOR_1: f32 = CORRELATION_FACTOR_TABLE[1].1;
const FACTOR_2: f32 = CORRELATION_FACTOR_TABLE[2].1;
const FACTOR_3: f32 = CORRELATION_FACTOR_TABLE[3].1;
const FACTOR_SPAN_0: f32 = FACTOR_1 - FACTOR_0;
const FACTOR_SPAN_1: f32 = FACTOR_2 - FACTOR_1;
const FACTOR_SPAN_2: f32 = FACTOR_3 - FACTOR_2;
/// The weight the previous smoothed value keeps.
const EMA_KEEP: f32 = 1.0 - EMA_ALPHA;
const RESULT_WORDS: u32 = RESULTS_PER_SLOT as u32;
const CURVE_BINS: u32 = NOISE_CURVE_BINS as u32;
/// The derived words past the rho, written as 0.
const DERIVED_SPARE: u32 = DERIVED_RHO + 1;
/// The shared words thread 0 hands the offset tables, the noise offset, the input noise offset
/// and the rho.
const TABLE_INPUTS: u32 = 3;

/// The factor that scales a temporal sigma up for grain correlation `rho`.
///
/// It interpolates the correlation table linearly, with `rho` clamped to the table's range. Each
/// segment's fraction multiplies by the host's reciprocal of the segment width.
#[cube]
fn correlation_factor(rho: f32, fold_settings: &Array<f32>) -> f32 {
    let x = f32::clamp(rho, RHO_0, RHO_3);
    let first_reciprocal = fold_settings[FOLD_SEGMENT_RECIPROCALS as usize];
    let second_reciprocal = fold_settings[(FOLD_SEGMENT_RECIPROCALS + 1) as usize];
    let third_reciprocal = fold_settings[(FOLD_SEGMENT_RECIPROCALS + 2) as usize];

    let first_fraction = (x - RHO_0) * first_reciprocal;
    let second_fraction = (x - RHO_1) * second_reciprocal;
    let third_fraction = (x - RHO_2) * third_reciprocal;
    let first = FACTOR_0 + first_fraction * FACTOR_SPAN_0;
    let second = FACTOR_1 + second_fraction * FACTOR_SPAN_1;
    let third = FACTOR_2 + third_fraction * FACTOR_SPAN_2;

    let later = select(x <= RHO_2, second, third);
    select(x <= RHO_1, first, later)
}

/// The share of the noise floor that applies at the candidate `dx`, `dy` under grain correlation
/// `rho`.
///
/// The centre gives 0. Every other candidate gives 1 with `rho` at or below 0, and otherwise
/// `1 - rho^distance`.
#[cube]
fn spatial_offset_factor(dx: i32, dy: i32, rho: f32) -> f32 {
    let squared = dx * dx + dy * dy;
    let distance = f32::sqrt(f32::cast_from(squared));
    let log_rho = f32::ln(rho);
    let shared = f32::exp(distance * log_rho);
    let correlated = 1.0f32 - shared;
    let off_centre = select(rho <= 0.0f32, 1.0f32, correlated);

    select(squared == 0, 0.0f32, off_centre)
}

/// Writes both offset tables of a `radius` search window across cube 0's threads.
///
/// Thread 0 of cube 0 puts the noise offset, the input noise offset and the rho in
/// `table_inputs` first. Only cube 0 holds them, and every thread of a cube takes the same branch,
/// so the barrier stays uniform within each cube. Entry `index` holds the candidate at
/// `dx = index % side - radius` and `dy = index / side - radius`, with `side = 2 * radius + 1`.
/// The spatial table scales the noise offset by the factor at the rho, and the pilot table scales
/// the input noise offset by the factor at a rho of 0.
#[cube]
fn write_offset_tables(
    table_inputs: &SharedMemory<f32>,
    spatial_table: &mut Array<f32>,
    pilot_table: &mut Array<f32>,
    #[comptime] radius: u32,
) {
    if CUBE_POS_X == 0 {
        sync_cube();

        let noise_offset = table_inputs[0];
        let input_offset = table_inputs[1];
        let rho = table_inputs[2];
        let side = comptime!(2 * radius + 1);
        let entries = comptime!(side * side);
        let centre = comptime!(radius as i32);
        let mut index = UNIT_POS_X;

        loop {
            if index >= entries {
                break;
            }

            let column = i32::cast_from(index % side);
            let row = i32::cast_from(index / side);
            let dx = column - centre;
            let dy = row - centre;
            let spatial_factor = spatial_offset_factor(dx, dy, rho);
            let pilot_factor = spatial_offset_factor(dx, dy, 0.0f32);

            spatial_table[index as usize] = noise_offset * spatial_factor;
            pilot_table[index as usize] = input_offset * pilot_factor;
            index += CUBE_DIM_X;
        }
    }
}

/// One smoothing step of a chain from `raw`, floored at the sigma floor.
///
/// A chain's first update takes `raw` outright.
#[cube]
fn smooth(previous: f32, raw: f32, valid: bool) -> f32 {
    let blended = EMA_ALPHA * raw + EMA_KEEP * previous;
    let smoothed = select(valid, blended, raw);

    f32::max(smoothed, SIGMA_FLOOR)
}

/// The places one fold reads its readings from.
#[derive(CubeType, Clone, Copy)]
struct FoldInputs {
    /// The centre's first reading word.
    centre_base: u32,
    /// The centre's first results word.
    results_base: u32,
    /// The lender's first reading word.
    lender_base: u32,
    /// Whether the fold takes the lender's temporal sample.
    has_sample: bool,
    /// The correlation factor of the lender's sample.
    factor: f32,
    sigma_scale: f32,
    immerkaer_factor: f32,
}

/// Folds `channel` into every chain and writes its derived sigma.
///
/// A channel past the active count writes 0 to every chain and to its derived sigma.
#[cube]
fn fold_channel(
    readings: &Array<f32>,
    results: &Array<f32>,
    state: &mut Array<f32>,
    derived: &mut Array<f32>,
    inputs: &FoldInputs,
    #[comptime] channel: u32,
    #[comptime] channels: u32,
) {
    let median_slot = (STATE_MEDIAN + 1 + channel) as usize;
    let low_slot = (STATE_LOW + 1 + channel) as usize;
    let unboosted_slot = (STATE_LOW_UNBOOSTED + 1 + channel) as usize;
    let temporal_slot = (STATE_TEMPORAL_ONLY + 1 + channel) as usize;

    if comptime!(channel < channels) {
        let abs_sum = results[(inputs.results_base + channel) as usize];
        let immerkaer = inputs.immerkaer_factor * abs_sum;
        let immerkaer_low = readings[(inputs.centre_base + READING_IMMERKAER_LOW + channel) as usize];
        let sigma = readings[(inputs.lender_base + READING_SIGMA + channel) as usize];
        let sigma_low = readings[(inputs.lender_base + READING_SIGMA_LOW + channel) as usize];

        let boosted = f32::max(immerkaer, sigma * inputs.factor);
        let boosted_low = f32::max(immerkaer_low, sigma_low * inputs.factor);
        let unboosted_low = f32::max(immerkaer_low, sigma_low);
        let raw = select(inputs.has_sample, boosted, immerkaer) * inputs.sigma_scale;
        let raw_low = select(inputs.has_sample, boosted_low, immerkaer_low) * inputs.sigma_scale;
        let raw_unboosted = select(inputs.has_sample, unboosted_low, immerkaer_low) * inputs.sigma_scale;
        let raw_temporal = sigma * inputs.sigma_scale;

        let median_valid = state[STATE_MEDIAN as usize] != 0.0f32;
        let low_valid = state[STATE_LOW as usize] != 0.0f32;
        let unboosted_valid = state[STATE_LOW_UNBOOSTED as usize] != 0.0f32;
        let temporal_valid = state[STATE_TEMPORAL_ONLY as usize] != 0.0f32;

        let median = smooth(state[median_slot], raw, median_valid);
        let low = smooth(state[low_slot], raw_low, low_valid);
        let unboosted = smooth(state[unboosted_slot], raw_unboosted, unboosted_valid);
        let temporal_previous = state[temporal_slot];
        let temporal_update = smooth(temporal_previous, raw_temporal, temporal_valid);
        let temporal = select(inputs.has_sample, temporal_update, temporal_previous);
        let temporal_after = temporal_valid || inputs.has_sample;

        state[median_slot] = median;
        state[low_slot] = low;
        state[unboosted_slot] = unboosted;
        state[temporal_slot] = temporal;
        derived[(DERIVED_SIGMA + channel) as usize] = select(temporal_after, temporal, unboosted);
    } else {
        state[median_slot] = 0.0f32;
        state[low_slot] = 0.0f32;
        state[unboosted_slot] = 0.0f32;
        state[temporal_slot] = 0.0f32;
        derived[(DERIVED_SIGMA + channel) as usize] = 0.0f32;
    }
}

/// The logical position whose sample a fold of `centre` takes.
///
/// Unless `seeding`, a centre with no sample borrows, with borrowing on, the first later position
/// whose reading has one. It reads only the readings and the settings.
#[cube]
fn lender_position(
    readings: &Array<f32>,
    fold_settings: &Array<f32>,
    ring_start: u32,
    centre: u32,
    seeding: bool,
    #[comptime] frames: u32,
) -> u32 {
    let centre_slot = (ring_start + centre) % frames;
    let centre_base = centre_slot * READING_WORDS;
    let centre_has_sample = readings[(centre_base + READING_SAMPLE_VALID) as usize] != 0.0f32;
    let borrow_ahead = fold_settings[FOLD_BORROW_AHEAD as usize] != 0.0f32;
    let borrows = borrow_ahead && !seeding && !centre_has_sample;

    let mut lender = centre;
    let mut found = false;
    for logical in centre + 1..frames {
        let slot = (ring_start + logical) % frames;
        let has_sample = readings[(slot * READING_WORDS + READING_SAMPLE_VALID) as usize] != 0.0f32;
        let takes = borrows && has_sample && !found;

        lender = select(takes, logical, lender);
        found = found || takes;
    }

    lender
}

/// Folds one reading into the stream state and writes the derived values, the current curve, the
/// current map and the offset tables.
///
/// The centre slot is `(ring_start + centre) % frames`. Unless `seed` is set, a centre with no
/// sample borrows, with borrowing on, the first reading after it in the ring that has one. Each
/// chain takes the larger of the centre's Immerkær reading and the lender's sample, boosted for
/// correlation on the boosted chains, scales it by `sigma_scale` and smooths it into its running
/// value. A lender with a sample replaces the curve and the classes with its own, and its map is
/// copied when it has classes. A seed folds the centre's Immerkær reading alone and keeps the
/// curve, the classes and the map.
///
/// Every thread picks the lender from the readings and the settings, which no cube writes, so the
/// cubes need no ordering between them. Thread 0 of cube 0 does the fold. With a map, every
/// thread of the grid copies its grid-stride share of the lender's slot map. `map_len` 0 leaves
/// both map bindings untouched.
///
/// With a `table_radius` above 0, thread 0 shares the new noise offsets and rho with cube 0, whose
/// threads then write both offset tables between them. A `table_radius` of 0 leaves both table
/// bindings untouched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a fold scalar or a comptime shape"
)]
pub fn noise_fold(
    readings: &Array<f32>,
    results: &Array<f32>,
    fold_settings: &Array<f32>,
    state: &mut Array<f32>,
    derived: &mut Array<f32>,
    curve: &mut Array<f32>,
    slot_maps: &Array<f32>,
    current_map: &mut Array<f32>,
    spatial_table: &mut Array<f32>,
    pilot_table: &mut Array<f32>,
    ring_start: u32,
    centre: u32,
    seed: u32,
    #[comptime] frames: u32,
    #[comptime] channels: u32,
    #[comptime] map_len: u32,
    #[comptime] table_radius: u32,
) {
    let mut table_inputs = SharedMemory::<f32>::new(TABLE_INPUTS as usize);
    let seeding = seed != 0u32;
    let lender = lender_position(readings, fold_settings, ring_start, centre, seeding, frames);
    let lender_slot = (ring_start + lender) % frames;
    let lender_base = lender_slot * READING_WORDS;
    let lender_has_sample = readings[(lender_base + READING_SAMPLE_VALID) as usize] != 0.0f32;
    let has_sample = lender_has_sample && !seeding;
    let lender_curve = readings[(lender_base + READING_CURVE_VALID) as usize];

    if CUBE_POS_X == 0 && UNIT_POS_X == 0 {
        let centre_slot = (ring_start + centre) % frames;
        let centre_base = centre_slot * READING_WORDS;
        let sample_rho = readings[(lender_base + READING_RHO) as usize];
        let factor = correlation_factor(sample_rho, fold_settings);

        let inputs = FoldInputs {
            centre_base,
            results_base: centre_slot * RESULT_WORDS,
            lender_base,
            has_sample,
            factor,
            sigma_scale: fold_settings[FOLD_SIGMA_SCALE as usize],
            immerkaer_factor: fold_settings[FOLD_IMMERKAER_FACTOR as usize],
        };

        fold_channel(readings, results, state, derived, &inputs, 0u32, channels);
        fold_channel(readings, results, state, derived, &inputs, 1u32, channels);
        fold_channel(readings, results, state, derived, &inputs, 2u32, channels);

        let temporal_valid = state[STATE_TEMPORAL_ONLY as usize] != 0.0f32;
        let temporal_after = temporal_valid || has_sample;
        state[STATE_MEDIAN as usize] = 1.0f32;
        state[STATE_LOW as usize] = 1.0f32;
        state[STATE_LOW_UNBOOSTED as usize] = 1.0f32;
        state[STATE_TEMPORAL_ONLY as usize] = select(temporal_after, 1.0f32, 0.0f32);

        let rho_valid = state[STATE_RHO_VALID as usize] != 0.0f32;
        let rho_previous = state[STATE_RHO as usize];
        let rho_blended = EMA_ALPHA * sample_rho + EMA_KEEP * rho_previous;
        let rho_update = select(rho_valid, rho_blended, sample_rho);
        let rho = select(has_sample, rho_update, rho_previous);
        let rho_after = rho_valid || has_sample;
        state[STATE_RHO as usize] = rho;
        state[STATE_RHO_VALID as usize] = select(rho_after, 1.0f32, 0.0f32);

        let lender_quarter_median = readings[(lender_base + READING_QUARTER_MEDIAN) as usize];
        let curve_flag = select(has_sample, lender_curve, state[STATE_CURVE_VALID as usize]);
        let quarter_median = select(
            has_sample,
            lender_quarter_median,
            state[STATE_QUARTER_MEDIAN as usize],
        );
        let classes_flag = select(has_sample, lender_curve, state[STATE_CLASSES_PRESENT as usize]);
        state[STATE_CURVE_VALID as usize] = curve_flag;
        state[STATE_QUARTER_MEDIAN as usize] = quarter_median;
        state[STATE_CLASSES_PRESENT as usize] = classes_flag;

        let curve_applies = fold_settings[FOLD_CURVE_APPLIES as usize] != 0.0f32;
        let curve_valid = curve_flag != 0.0f32 && curve_applies;

        for bin in 0..CURVE_BINS {
            let lender_ratio = readings[(lender_base + READING_RATIOS + bin) as usize];
            let ratio = select(has_sample, lender_ratio, state[(STATE_RATIOS + bin) as usize]);

            state[(STATE_RATIOS + bin) as usize] = ratio;
            curve[bin as usize] = select(curve_valid, ratio, 0.0f32);
        }

        let sigma0 = derived[DERIVED_SIGMA as usize];
        let g_max = fold_settings[FOLD_WEIGHT_G_MAX as usize];
        let weight_floor = fold_settings[FOLD_WEIGHT_FLOOR as usize];
        let norm = sigma0 * sigma0 * g_max * g_max;
        let usable = !norm.is_nan() && !norm.is_inf() && norm > weight_floor;

        let map_kind = u32::cast_from(fold_settings[FOLD_MAP_KIND as usize]);
        let classes_present = classes_flag != 0.0f32;
        let luma_map = map_kind == STRENGTH_MAP_LUMA && curve_valid;
        let chroma_map = map_kind == STRENGTH_MAP_ALL && classes_present;
        let other_mode = select(chroma_map, STRENGTH_MAP_ALL, STRENGTH_MAP_OFF);
        let map_mode = select(luma_map, STRENGTH_MAP_LUMA, other_mode);

        let mut median_sum = 0.0f32;
        let mut low_sum = 0.0f32;
        #[unroll]
        for channel in 0..channels {
            let median = state[(STATE_MEDIAN + 1 + channel) as usize];
            let low = state[(STATE_LOW + 1 + channel) as usize];

            median_sum += median * median;
            low_sum += low * low;
        }

        let median0 = state[(STATE_MEDIAN + 1) as usize];
        let h2_auto = fold_settings[FOLD_H2_AUTO as usize] != 0.0f32;
        let per_inverse_variance = fold_settings[FOLD_H2_PER_INVERSE_VARIANCE as usize];
        let channel_count = comptime!(channels as f32);
        let h2_measured = per_inverse_variance * channel_count / median_sum;
        let h2 = select(h2_auto, h2_measured, fold_settings[FOLD_H2_FIXED as usize]);
        let input_offset = fold_settings[FOLD_OFFSET_PER_VARIANCE as usize] * low_sum;
        let main_offset = fold_settings[FOLD_MAIN_OFFSET as usize] != 0.0f32;
        let noise_offset = select(main_offset, input_offset, 0.0f32);
        let derived_rho = select(rho_after, rho, 0.0f32);

        derived[DERIVED_WEIGHT_SCALE as usize] = select(usable, norm, weight_floor);
        derived[DERIVED_CURVE_VALID as usize] = select(curve_valid, 1.0f32, 0.0f32);
        derived[DERIVED_MAP_MODE as usize] = f32::cast_from(map_mode);
        derived[DERIVED_SAD_FLOOR as usize] = fold_settings[FOLD_SAD_PER_SIGMA as usize] * median0;
        derived[DERIVED_CONFIDENCE_SAD_FLOOR as usize] =
            fold_settings[FOLD_CONFIDENCE_SAD_PER_SIGMA as usize] * median0;
        derived[DERIVED_H2_INV_NORM as usize] = h2;
        derived[DERIVED_PILOT_H2_INV_NORM as usize] = h2 * fold_settings[FOLD_PILOT_H2_FACTOR as usize];
        derived[DERIVED_NOISE_OFFSET as usize] = noise_offset;
        derived[DERIVED_INPUT_NOISE_OFFSET as usize] = input_offset;
        derived[DERIVED_RHO as usize] = derived_rho;

        for word in DERIVED_SPARE..DERIVED_WORDS {
            derived[word as usize] = 0.0f32;
        }

        if comptime!(table_radius > 0) {
            table_inputs[0] = noise_offset;
            table_inputs[1] = input_offset;
            table_inputs[2] = derived_rho;
        }
    }

    if comptime!(table_radius > 0) {
        write_offset_tables(&table_inputs, spatial_table, pilot_table, table_radius);
    }

    if comptime!(map_len > 0) {
        let copies = has_sample && lender_curve != 0.0f32;
        let source = lender_slot * map_len;
        let stride = CUBE_COUNT_X * CUBE_DIM_X;
        let mut index = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

        if copies {
            loop {
                if index >= map_len {
                    break;
                }

                current_map[index as usize] = slot_maps[(source + index) as usize];
                index += stride;
            }
        }
    }
}

/// Clears every validity flag of a stream state, so the next fold starts a new stream.
///
/// The values behind the flags are left in place.
#[cube(launch_unchecked)]
pub fn noise_reset(state: &mut Array<f32>) {
    if UNIT_POS_X == 0 {
        state[STATE_MEDIAN as usize] = 0.0f32;
        state[STATE_LOW as usize] = 0.0f32;
        state[STATE_LOW_UNBOOSTED as usize] = 0.0f32;
        state[STATE_TEMPORAL_ONLY as usize] = 0.0f32;
        state[STATE_RHO_VALID as usize] = 0.0f32;
        state[STATE_CURVE_VALID as usize] = 0.0f32;
        state[STATE_CLASSES_PRESENT as usize] = 0.0f32;
    }
}
