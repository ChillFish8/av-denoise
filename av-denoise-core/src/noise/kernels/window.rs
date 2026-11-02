use cubecl::prelude::*;

use super::derive::{FoldedNoise, TABLE_INPUTS, correlation_factor, write_derived, write_offset_tables};
use crate::noise::curve::NOISE_CURVE_BINS;
use crate::noise::derived::DERIVED_SIGMA;
use crate::noise::estimator::SIGMA_FLOOR;
use crate::noise::fold::{
    FOLD_CURVE_APPLIES,
    FOLD_IMMERKAER_FACTOR,
    FOLD_SIGMA_SCALE,
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
use crate::noise::history::NOISE_WINDOW_LEN;
use crate::noise::reading::{
    READING_CURVE_VALID,
    READING_IMMERKAER_LOW,
    READING_QUARTER_MEDIAN,
    READING_RATIOS,
    READING_RHO,
    READING_SAMPLE_VALID_BASE,
    READING_SIGMA,
    READING_SIGMA_LOW,
    READING_WORDS,
};
use crate::noise::spatial::RESULTS_PER_SLOT;
use crate::noise::{NOISE_BLOCK_1D, NOISE_CHANNELS};

const CURVE_BINS: u32 = NOISE_CURVE_BINS as u32;
/// The chain fed by the boosted median reading.
const CHAIN_MEDIAN: u32 = 0;
/// The chain fed by the boosted lower-quartile reading.
const CHAIN_LOW: u32 = 1;
/// The chain fed by the unboosted lower-quartile reading.
const CHAIN_LOW_UNBOOSTED: u32 = 2;
/// The chain fed by the temporal median alone, from frames with a sample.
const CHAIN_TEMPORAL_ONLY: u32 = 3;
const CHAINS: u32 = 4;
/// The shared median of chain `chain` on channel `channel` sits at
/// `chain * NOISE_CHANNELS + channel`.
const MEDIAN_CHAIN_WORDS: u32 = CHAINS * NOISE_CHANNELS;
const MEDIAN_RHO: u32 = MEDIAN_CHAIN_WORDS;
const MEDIAN_QUARTER: u32 = MEDIAN_RHO + 1;
const MEDIAN_RATIOS: u32 = MEDIAN_QUARTER + 1;
/// The shared words cube 0 keeps its medians in.
const MEDIAN_WORDS: u32 = MEDIAN_RATIOS + CURVE_BINS;

const _: () = assert!(
    NOISE_BLOCK_1D >= MEDIAN_WORDS,
    "cube 0 takes one median per thread"
);

/// The frame qualifies with any reading.
const GATE_ANY: u32 = 0;
/// The frame qualifies with a sample of the channel.
const GATE_SAMPLE: u32 = 1;
/// The frame qualifies with a sample of the channel and a curve.
const GATE_CURVE: u32 = 2;

/// Where a window fold reads its frames from.
#[derive(CubeType, Clone, Copy)]
struct Window {
    first: u32,
    last: u32,
}

/// Whether the frame in history slot `slot` passes `gate` on channel `channel`.
///
/// The sample gate takes the channel's own sample. The curve gate takes the channel's sample and
/// the frame's curve.
#[cube]
fn passes(records: &[f32], slot: u32, gate: u32, channel: u32) -> bool {
    let base = slot * READING_WORDS;
    let has_sample = records[(base + READING_SAMPLE_VALID_BASE + channel) as usize] != 0.0f32;
    let has_curve = records[(base + READING_CURVE_VALID) as usize] != 0.0f32;
    let curve_passes = has_sample && has_curve;
    let sample_passes = select(gate == GATE_SAMPLE, has_sample, curve_passes);

    select(gate == GATE_ANY, true, sample_passes)
}

/// The frames of `window` that pass `gate` on channel `channel`.
#[cube]
fn passing_frames(
    records: &[f32],
    window: &Window,
    gate: u32,
    channel: u32,
    #[comptime] capacity: u32,
) -> u32 {
    let mut count = 0u32;
    let mut frame = window.first;

    while frame <= window.last {
        let slot = frame % capacity;
        let counted = passes(records, slot, gate, channel);

        count += select(counted, 1u32, 0u32);
        frame += 1u32;
    }

    count
}

/// The median of the first `count` entries of `values`, sorting them in place.
///
/// An even count gives the mean of the two middle entries, and a count of 0 gives 0.
#[cube]
fn sorted_median(values: &mut Array<f32>, count: u32) -> f32 {
    let mut index = 1u32;
    while index < count {
        let key = values[index as usize];
        let mut position = index;
        let mut moving = true;

        while moving {
            let below = values[(position - 1u32) as usize];
            let shifts = below > key;

            if shifts {
                values[position as usize] = below;
                position -= 1u32;
            }

            moving = shifts && position > 0u32;
        }

        values[position as usize] = key;
        index += 1u32;
    }

    let mut median = 0.0f32;
    if count > 0u32 {
        let lower = values[((count - 1u32) / 2u32) as usize];
        let upper = values[(count / 2u32) as usize];
        median = (lower + upper) * 0.5f32;
    }

    median
}

/// The median of word `offset` of each frame's `stride`-word row of `source`, over the frames of
/// `window` that pass `gate` on channel 0.
#[cube]
fn gated_median(
    source: &[f32],
    records: &[f32],
    window: &Window,
    stride: u32,
    offset: u32,
    gate: u32,
    #[comptime] capacity: u32,
) -> f32 {
    let mut values = Array::<f32>::new(NOISE_WINDOW_LEN as usize);
    let mut count = 0u32;
    let mut frame = window.first;

    while frame <= window.last {
        let slot = frame % capacity;

        if passes(records, slot, gate, 0u32) {
            values[count as usize] = source[(slot * stride + offset) as usize];
            count += 1u32;
        }

        frame += 1u32;
    }

    sorted_median(&mut values, count)
}

/// One frame's reading for `chain` on `channel`, as the fold combines and scales it.
///
/// A frame without a sample of `channel` gives the channel's Immerkær reading alone.
#[cube]
fn chain_reading(
    records: &[f32],
    results: &[f32],
    fold_settings: &[f32],
    slot: u32,
    chain: u32,
    channel: u32,
) -> f32 {
    let base = slot * READING_WORDS;
    let has_sample = records[(base + READING_SAMPLE_VALID_BASE + channel) as usize] != 0.0f32;
    let rho = records[(base + READING_RHO) as usize];
    let factor = correlation_factor(rho, fold_settings);
    let sigma_scale = fold_settings[FOLD_SIGMA_SCALE as usize];
    let immerkaer_factor = fold_settings[FOLD_IMMERKAER_FACTOR as usize];

    let results_per_slot = comptime!(RESULTS_PER_SLOT as u32);
    let abs_sum = results[(slot * results_per_slot + channel) as usize];
    let immerkaer = immerkaer_factor * abs_sum;
    let immerkaer_low = records[(base + READING_IMMERKAER_LOW + channel) as usize];
    let sigma = records[(base + READING_SIGMA + channel) as usize];
    let sigma_low = records[(base + READING_SIGMA_LOW + channel) as usize];

    let boosted = f32::max(immerkaer, sigma * factor);
    let boosted_low = f32::max(immerkaer_low, sigma_low * factor);
    let unboosted_low = f32::max(immerkaer_low, sigma_low);
    let raw = select(has_sample, boosted, immerkaer) * sigma_scale;
    let raw_low = select(has_sample, boosted_low, immerkaer_low) * sigma_scale;
    let raw_unboosted = select(has_sample, unboosted_low, immerkaer_low) * sigma_scale;
    let raw_temporal = sigma * sigma_scale;

    let low_reading = select(chain == CHAIN_LOW, raw_low, raw_unboosted);
    let later_reading = select(chain == CHAIN_TEMPORAL_ONLY, raw_temporal, low_reading);

    select(chain == CHAIN_MEDIAN, raw, later_reading)
}

/// The median of `chain` on `channel` over `window`, floored at the sigma floor.
///
/// The temporal-only chain takes only the frames with a sample of `channel`, and gives 0 without
/// one.
#[cube]
fn chain_median(
    records: &[f32],
    results: &[f32],
    fold_settings: &[f32],
    window: &Window,
    chain: u32,
    channel: u32,
    #[comptime] capacity: u32,
) -> f32 {
    let mut values = Array::<f32>::new(NOISE_WINDOW_LEN as usize);
    let gate = select(chain == CHAIN_TEMPORAL_ONLY, GATE_SAMPLE, GATE_ANY);
    let mut count = 0u32;
    let mut frame = window.first;

    while frame <= window.last {
        let slot = frame % capacity;

        if passes(records, slot, gate, channel) {
            values[count as usize] = chain_reading(records, results, fold_settings, slot, chain, channel);
            count += 1u32;
        }

        frame += 1u32;
    }

    let median = sorted_median(&mut values, count);
    let floored = f32::max(median, SIGMA_FLOOR);

    select(count > 0u32, floored, 0.0f32)
}

/// Writes cube 0's median of median word `task` into `medians`.
///
/// A chain past the active channels writes 0.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, the window or a comptime shape"
)]
fn window_median_task(
    records: &[f32],
    results: &[f32],
    fold_settings: &[f32],
    medians: &mut Shared<[f32]>,
    window: &Window,
    task: u32,
    #[comptime] capacity: u32,
    #[comptime] channels: u32,
) {
    if task < MEDIAN_CHAIN_WORDS {
        let chain = task / NOISE_CHANNELS;
        let channel = task % NOISE_CHANNELS;
        let mut median = 0.0f32;

        if channel < channels {
            median = chain_median(records, results, fold_settings, window, chain, channel, capacity);
        }

        medians[task as usize] = median;
    } else if task < MEDIAN_WORDS {
        let rho_word = READING_RHO;
        let quarter_word = READING_QUARTER_MEDIAN;
        let ratio_word = READING_RATIOS + task - MEDIAN_RATIOS;
        let curve_word = select(task == MEDIAN_QUARTER, quarter_word, ratio_word);
        let word = select(task == MEDIAN_RHO, rho_word, curve_word);
        let gate = select(task == MEDIAN_RHO, GATE_SAMPLE, GATE_CURVE);

        medians[task as usize] = gated_median(records, records, window, READING_WORDS, word, gate, capacity);
    }
}

/// One bit per active channel, set when a frame of `window` has a sample of that channel.
///
/// `sampled` is channel 0's count of such frames.
#[cube]
fn sampled_channels(
    records: &[f32],
    window: &Window,
    sampled: u32,
    #[comptime] capacity: u32,
    #[comptime] channels: u32,
) -> u32 {
    let mut mask = select(sampled > 0u32, 1u32, 0u32);

    #[unroll]
    for channel in 1..channels {
        let count = passing_frames(records, window, GATE_SAMPLE, channel, capacity);
        let bit = comptime!(1u32 << channel);

        mask |= select(count > 0u32, bit, 0u32);
    }

    mask
}

/// Writes the windowed chains, rho and curve into `state`, and every derived value.
///
/// A channel without a sampled frame takes its unboosted low chain as its sigma. Its temporal-only
/// value is the same while another channel holds the chain, and 0 while no channel does. The rho
/// follows channel 0's samples.
///
/// Run by thread 0 of cube 0 once every median is in `medians`.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a window count or a comptime shape"
)]
fn write_window_state(
    medians: &Shared<[f32]>,
    fold_settings: &[f32],
    state: &mut [f32],
    derived: &mut [f32],
    curve: &mut [f32],
    table_inputs: &mut Shared<[f32]>,
    sampled_mask: u32,
    curve_present: bool,
    #[comptime] channels: u32,
    #[comptime] table_radius: u32,
) {
    let rho_valid = (sampled_mask & 1u32) != 0u32;
    let any_sampled = sampled_mask != 0u32;
    let mut median_sum = 0.0f32;
    let mut low_sum = 0.0f32;

    #[unroll]
    for channel in 0..NOISE_CHANNELS {
        let median = medians[(CHAIN_MEDIAN * NOISE_CHANNELS + channel) as usize];
        let low = medians[(CHAIN_LOW * NOISE_CHANNELS + channel) as usize];
        let unboosted = medians[(CHAIN_LOW_UNBOOSTED * NOISE_CHANNELS + channel) as usize];
        let temporal = medians[(CHAIN_TEMPORAL_ONLY * NOISE_CHANNELS + channel) as usize];
        let bit = comptime!(1u32 << channel);
        let channel_sampled = (sampled_mask & bit) != 0u32;
        let fallback = select(any_sampled, unboosted, 0.0f32);

        state[(STATE_MEDIAN + 1 + channel) as usize] = median;
        state[(STATE_LOW + 1 + channel) as usize] = low;
        state[(STATE_LOW_UNBOOSTED + 1 + channel) as usize] = unboosted;
        state[(STATE_TEMPORAL_ONLY + 1 + channel) as usize] = select(channel_sampled, temporal, fallback);
        derived[(DERIVED_SIGMA + channel) as usize] = select(channel_sampled, temporal, unboosted);
    }

    #[unroll]
    for channel in 0..channels {
        let median = medians[(CHAIN_MEDIAN * NOISE_CHANNELS + channel) as usize];
        let low = medians[(CHAIN_LOW * NOISE_CHANNELS + channel) as usize];

        median_sum += median * median;
        low_sum += low * low;
    }

    let temporal_flag = select(any_sampled, 1.0f32, 0.0f32);
    let rho_flag = select(rho_valid, 1.0f32, 0.0f32);
    let curve_flag = select(curve_present, 1.0f32, 0.0f32);
    let rho = select(rho_valid, medians[MEDIAN_RHO as usize], 0.0f32);
    let quarter_median = select(curve_present, medians[MEDIAN_QUARTER as usize], 0.0f32);

    state[STATE_MEDIAN as usize] = 1.0f32;
    state[STATE_LOW as usize] = 1.0f32;
    state[STATE_LOW_UNBOOSTED as usize] = 1.0f32;
    state[STATE_TEMPORAL_ONLY as usize] = temporal_flag;
    state[STATE_RHO_VALID as usize] = rho_flag;
    state[STATE_RHO as usize] = rho;
    state[STATE_CURVE_VALID as usize] = curve_flag;
    state[STATE_QUARTER_MEDIAN as usize] = quarter_median;

    let curve_applies = fold_settings[FOLD_CURVE_APPLIES as usize] != 0.0f32;
    let curve_valid = curve_present && curve_applies;

    for bin in 0..CURVE_BINS {
        let median_ratio = medians[(MEDIAN_RATIOS + bin) as usize];
        let ratio = select(curve_present, median_ratio, 0.0f32);

        state[(STATE_RATIOS + bin) as usize] = ratio;
        curve[bin as usize] = select(curve_valid, ratio, 0.0f32);
    }

    let folded = FoldedNoise {
        sigma0: derived[DERIVED_SIGMA as usize],
        median0: medians[(CHAIN_MEDIAN * NOISE_CHANNELS) as usize],
        median_sum,
        low_sum,
        curve_valid,
        rho,
    };

    write_derived(
        &folded,
        fold_settings,
        derived,
        table_inputs,
        channels,
        table_radius,
    );
}

/// Folds the history frames `first..=last` into the stream state, the derived values, the current
/// curve and the offset tables, in one launch.
///
/// Stream frame `frame` sits in history slot `frame % capacity`. Each chain is the median over the
/// window of each frame's reading, combined and scaled as one fold of that frame alone would, and
/// floored at the sigma floor. Every frame feeds the median and low chains. Each channel's
/// temporal-only chain takes only the frames with a sample of that channel. The rho takes only the
/// frames with a sample of channel 0. The window holds a curve when at least half of those frames
/// have one, and the ratios and the quarter median are then the elementwise median over them.
///
/// Cube 0's threads each take one median between them, then thread 0 writes the state and the
/// derived values. With a `table_radius` above 0, cube 0 then writes both offset tables. A
/// `table_radius` of 0 leaves both table bindings untouched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a window bound or a comptime shape"
)]
pub fn noise_window_fold(
    records: &[f32],
    results: &[f32],
    fold_settings: &[f32],
    state: &mut [f32],
    derived: &mut [f32],
    curve: &mut [f32],
    spatial_table: &mut [f32],
    pilot_table: &mut [f32],
    first: u32,
    last: u32,
    #[comptime] capacity: u32,
    #[comptime] channels: u32,
    #[comptime] table_radius: u32,
) {
    comptime! {
        assert!(channels <= NOISE_CHANNELS, "noise_window_fold needs a slot per channel");
    }

    let mut table_inputs = Shared::<[f32]>::new_slice(TABLE_INPUTS as usize);
    let mut medians = Shared::<[f32]>::new_slice(MEDIAN_WORDS as usize);
    let window = Window { first, last };
    let sampled = passing_frames(records, &window, GATE_SAMPLE, 0u32, capacity);
    let curved = passing_frames(records, &window, GATE_CURVE, 0u32, capacity);
    let curve_present = sampled > 0u32 && 2u32 * curved >= sampled;

    if CUBE_POS_X == 0 {
        window_median_task(
            records,
            results,
            fold_settings,
            &mut medians,
            &window,
            UNIT_POS_X,
            capacity,
            channels,
        );

        sync_cube();

        if UNIT_POS_X == 0 {
            let sampled_mask = sampled_channels(records, &window, sampled, capacity, channels);

            write_window_state(
                &medians,
                fold_settings,
                state,
                derived,
                curve,
                &mut table_inputs,
                sampled_mask,
                curve_present,
                channels,
                table_radius,
            );
        }
    }

    if comptime!(table_radius > 0) {
        write_offset_tables(&table_inputs, spatial_table, pilot_table, table_radius);
    }
}
