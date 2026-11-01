#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicU32;

use cubecl::prelude::*;
use cubecl::server::Handle;

#[cfg(test)]
pub(crate) use self::tests::SlotStats;
use super::NOISE_CHANNELS;
use super::curve::{MIN_QUARTERS_PER_BIN, NOISE_CURVE_BINS};
use super::derived::DerivedValues;
use super::fold::{FoldSettings, NoiseSettings, OffsetTables, STATE_WORDS, fold_words};
use super::history::{ReadingHistory, history_capacity};
use super::reading::{
    CURVE_TAGS,
    IMMERKAER_SPATIAL_TAGS,
    READING_WORDS,
    curve_queries,
    immerkaer_queries,
    immerkaer_tags,
    sample_queries,
    sample_tags,
};
use super::select::{SelectBatch, histogram_len};
use super::spatial::{IMMERKAER_LANES, RESULTS_PER_SLOT, noise_partials_slot_stride_bytes, partials_len};
use super::temporal::{
    TEMPORAL_QUARTERS,
    min_static_blocks,
    temporal_stats_blocks,
    temporal_stats_buf_bytes,
};
use crate::engine::BufferSize;
use crate::engine::align::StorageAlign;

/// The plane geometry one noise state measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NoiseShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// Look-ahead slots, one stats region and one reading each.
    pub frames: u32,
    /// The temporal radius of the frame ring the readings feed, which sets how far past a window
    /// the history reaches.
    pub ring_radius: u32,
    /// Whether temporal statistics are measured, which needs a temporal radius of at least 1.
    pub temporal: bool,
    pub align: StorageAlign,
}

/// The block value of channel 0's mean residual, after one sigma per channel.
pub(crate) const BLOCK_VALUE_MEAN0: u32 = NOISE_CHANNELS;
/// The block value of channel 0's mean lag-1 product.
pub(crate) const BLOCK_VALUE_MEAN_LAG: u32 = BLOCK_VALUE_MEAN0 + 1;
/// The block value of channel 0's residual variance.
pub(crate) const BLOCK_VALUE_VAR0: u32 = BLOCK_VALUE_MEAN_LAG + 1;
/// The block value of the pixel pair count.
pub(crate) const BLOCK_VALUE_PAIRS: u32 = BLOCK_VALUE_VAR0 + 1;
/// The `f32` values block derive keeps per block.
pub(crate) const BLOCK_FIELDS: u32 = BLOCK_VALUE_PAIRS + 1;
/// The sample batch's run of correlation lanes, after one run of sigma lanes per channel.
///
/// Each run holds one lane per block, and run `r` starts at lane `r * blocks`.
pub(crate) const SAMPLE_RUN_RHO: u32 = NOISE_CHANNELS;
/// The sample batch's first run of quarter sigma lanes.
///
/// The quarters fill [TEMPORAL_QUARTERS](crate::noise::temporal::TEMPORAL_QUARTERS) runs, with each
/// block's quarters side by side.
pub(crate) const SAMPLE_RUN_QUARTERS: u32 = SAMPLE_RUN_RHO + 1;
/// The sample batch's run of channel 0's accepted flag lanes.
///
/// Channel `c`'s run is `c` past it.
pub(crate) const SAMPLE_RUN_ACCEPTED: u32 = SAMPLE_RUN_QUARTERS + TEMPORAL_QUARTERS;
/// The curve batch's lanes per block, one per static quarter.
pub(crate) const CURVE_LANES_PER_BLOCK: u32 = TEMPORAL_QUARTERS;
/// The block flag of a block whose channel 0 mean residual passes the static gate.
pub(crate) const BLOCK_CANDIDATE: u32 = 1;
/// The block flag of a block channel 0 accepts as static noise.
pub(crate) const BLOCK_ACCEPTED: u32 = 2;
/// The bits each channel's block flags take.
const BLOCK_CHANNEL_FLAG_BITS: u32 = 2;

/// Channel `channel`'s copy of a channel 0 block flag.
pub(crate) const fn channel_flag(flag: u32, channel: u32) -> u32 {
    flag << (BLOCK_CHANNEL_FLAG_BITS * channel)
}

/// The sample batch's lanes per block.
pub(crate) const fn sample_lanes_per_block(channels: u32) -> u32 {
    SAMPLE_RUN_ACCEPTED + channels
}

/// What one slot's reading measures.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ReadingSettings {
    /// Whether the stats carry the luma quarter fields the curve reads.
    pub luma_fields: bool,
}

/// The noise values every consumer reads and, when the noise is measured, the buffers one ring of
/// frames is measured in.
pub(crate) struct NoiseState {
    pub(super) client: Client,
    pub(super) shape: NoiseShape,
    pub(super) settings: NoiseSettings,
    /// The [DERIVED_WORDS](crate::noise::derived::DERIVED_WORDS) values every consumer reads.
    pub(super) derived: Handle,
    /// The current curve's ratios when the consumer applies it, else zeros.
    pub(super) curve_ratios: Handle,
    /// The NLM offset tables, when the fold settings ask for them.
    pub(super) offset_tables: Option<OffsetTables>,
    /// The fixed derived values last uploaded, which host-built offset tables follow.
    pub(super) fixed_values: DerivedValues,
    /// The measurement buffers, `None` for a state that holds fixed values.
    pub(super) measurement: Option<Measurement>,
}

/// The buffers one ring of frames is measured and folded in.
pub(super) struct Measurement {
    /// The first stage of each slot's Immerkær estimate, one padded region per slot.
    pub(super) partials: Handle,
    /// The per-channel Immerkær totals, one row per slot.
    pub(super) results: Handle,
    /// The temporal residual statistics, one padded region per slot, when they are measured.
    pub(super) stats: Option<Handle>,
    /// One reading of [READING_WORDS](crate::noise::reading::READING_WORDS) values per slot.
    pub(super) readings: Handle,
    /// Each Immerkær cube's sigmas and, with temporal stats, each candidate block's per-channel
    /// sigmas and texture.
    pub(super) immerkaer_batch: SelectBatch,
    /// The per-block scratch and the sample batch, when temporal stats are measured.
    pub(super) sample: Option<SampleScratch>,
    /// Each flat, unclipped static quarter's sigma tagged with its luma bin, when temporal stats
    /// and the luma quarter fields are measured.
    pub(super) curve: Option<SelectBatch>,
    /// The fewest accepted blocks a sample needs to reach the static fraction.
    pub(super) min_accepted: u32,
    /// The [FOLD_WORDS](crate::noise::fold::FOLD_WORDS) settings the fold reads.
    pub(super) fold_settings: Handle,
    /// The [STATE_WORDS](crate::noise::fold::STATE_WORDS) chains, rho and curve the last fold
    /// wrote.
    pub(super) fold_state: Handle,
    /// The readings of the latest stream frames, which every fold reads its window from.
    pub(super) history: ReadingHistory,
    /// The stream frames recorded since the last reset.
    pub(super) recorded: AtomicU32,
}

/// The buffers one slot's temporal sample is built in.
pub(super) struct SampleScratch {
    /// [BLOCK_FIELDS](crate::noise::state::BLOCK_FIELDS) values per block.
    pub(super) block_values: Handle,
    /// One flag word per block.
    pub(super) block_flags: Handle,
    /// Each block's kept sigma and accepted flag per channel, plus its correlation and static
    /// quarter sigmas.
    pub(super) batch: SelectBatch,
}

impl NoiseState {
    /// A state for `shape` whose derived buffer holds `initial` until the first fold.
    pub(crate) fn new(
        client: &Client,
        shape: NoiseShape,
        settings: NoiseSettings,
        initial: DerivedValues,
    ) -> Self {
        let offset_tables = settings.fold.offset_tables;
        let mut state = Self::fixed(client, shape, initial, offset_tables);
        state.settings = settings;
        state.start_measuring();

        state
    }

    /// A state that measures nothing and holds `derived` and a curve of zeros.
    ///
    /// With `offset_tables` it also holds the offset tables of that search radius, built from
    /// `derived` at a rho of 0. Every queue is a no-op, and a snapshot holds `derived` with no
    /// chains.
    pub(crate) fn fixed(
        client: &Client,
        shape: NoiseShape,
        derived: DerivedValues,
        offset_tables: Option<u32>,
    ) -> Self {
        let fixed_values = derived;
        let tables = offset_tables.map(|radius| OffsetTables::from_host(client, radius, fixed_values));
        let fold = FoldSettings {
            offset_tables,
            ..FoldSettings::default()
        };
        let settings = NoiseSettings {
            fold,
            ..NoiseSettings::default()
        };

        let derived_words = derived.to_words();
        let derived_bytes = f32::as_bytes(&derived_words);
        let derived = client.create_from_slice(derived_bytes);

        let curve_zeros = [0.0f32; NOISE_CURVE_BINS];
        let curve_bytes = f32::as_bytes(&curve_zeros);
        let curve_ratios = client.create_from_slice(curve_bytes);

        Self {
            client: client.clone(),
            shape,
            settings,
            derived,
            curve_ratios,
            offset_tables: tables,
            fixed_values,
            measurement: None,
        }
    }

    /// Whether the state measures the noise rather than holding fixed values.
    pub(crate) fn is_measured(&self) -> bool {
        self.measurement.is_some()
    }

    /// Allocates the measurement buffers for the current settings, so the state measures the noise
    /// from its next queued slot.
    ///
    /// The derived buffer keeps its values until the first fold.
    pub(crate) fn start_measuring(&mut self) {
        let measurement = measurement_buffers(&self.client, self.shape, self.settings);
        self.measurement = Some(measurement);
    }

    /// The measurement buffers.
    ///
    /// # Panics
    ///
    /// Panics on a state that holds fixed values.
    pub(super) fn measurement(&self) -> &Measurement {
        self.measurement
            .as_ref()
            .expect("a fixed noise state has no measurement buffers")
    }

    /// Replaces the derived values of a state that holds fixed values, and rebuilds its offset
    /// tables from them.
    ///
    /// # Panics
    ///
    /// Panics on a state that measures the noise.
    pub(crate) fn set_fixed(&mut self, derived: DerivedValues) {
        let measured = self.is_measured();
        assert!(!measured, "a measured noise state writes its own derived values");

        let derived_words = derived.to_words();
        let derived_bytes = f32::as_bytes(&derived_words);
        self.derived = self.client.create_from_slice(derived_bytes);
        self.fixed_values = derived;

        let radius = self.settings.fold.offset_tables;
        self.offset_tables = radius.map(|radius| OffsetTables::from_host(&self.client, radius, derived));
    }

    pub(crate) fn settings(&self) -> NoiseSettings {
        self.settings
    }

    /// Replaces the settings before the first queued slot.
    ///
    /// A new offset table radius rebuilds the tables from the last fixed derived values. A measured
    /// state allocates or drops the curve batch to match, and rewrites the fold settings.
    pub(crate) fn set_settings(&mut self, settings: NoiseSettings) {
        let radius = settings.fold.offset_tables;
        let table_radius = self.offset_tables.as_ref().map(|tables| tables.radius);

        if radius != table_radius {
            let fixed_values = self.fixed_values;
            self.offset_tables =
                radius.map(|radius| OffsetTables::from_host(&self.client, radius, fixed_values));
        }

        self.settings = settings;

        let client = self.client.clone();
        let shape = self.shape;
        let Some(measurement) = self.measurement.as_mut() else {
            return;
        };

        let reading = settings.reading;
        let has_curve = measurement.curve.is_some();
        let wants_curve = shape.temporal && reading.luma_fields;

        if wants_curve && !has_curve {
            measurement.curve = curve_batch(&client, shape, reading);
        }

        if !wants_curve {
            measurement.curve = None;
        }

        measurement.fold_settings = fold_settings_buffer(&client, shape, settings);
    }

    /// The derived values every consumer reads.
    pub(crate) fn derived(&self) -> &Handle {
        &self.derived
    }

    /// The current curve's ratios when the consumer applies it, else zeros.
    pub(crate) fn curve(&self) -> &Handle {
        &self.curve_ratios
    }

    /// The main pass's noise-floor table, one entry per search candidate, `None` without tables.
    pub(crate) fn spatial_offsets(&self) -> Option<&Handle> {
        self.offset_tables.as_ref().map(|tables| &tables.spatial)
    }

    /// The NLM pilot's noise-floor table, one entry per search candidate, `None` without tables.
    pub(crate) fn pilot_offsets(&self) -> Option<&Handle> {
        self.offset_tables.as_ref().map(|tables| &tables.pilot)
    }

    /// The partials ring, offset to the start of `slot`.
    pub(super) fn partials_slot(&self, slot: u32) -> Handle {
        let stride = noise_partials_slot_stride_bytes(self.shape.width, self.shape.height, self.shape.align);
        let start = u64::from(slot) * stride;

        self.measurement().partials.clone().offset_start(start)
    }
}

/// The measurement buffers of a state for `shape` built with `settings`.
fn measurement_buffers(client: &Client, shape: NoiseShape, settings: NoiseSettings) -> Measurement {
    let frames = u64::from(shape.frames);
    let partials_stride = noise_partials_slot_stride_bytes(shape.width, shape.height, shape.align);
    let partials_bytes = partials_stride * frames;
    let partials = client.empty(partials_bytes as usize);

    let results_len = shape.frames as usize * RESULTS_PER_SLOT;
    let results = client.empty(results_len * size_of::<f32>());

    let stats = if shape.temporal {
        let stats_bytes = temporal_stats_buf_bytes(
            shape.width,
            shape.height,
            shape.channels,
            shape.frames,
            shape.align,
        );
        let stats = client.empty(stats_bytes);
        Some(stats)
    } else {
        None
    };

    let readings_len = (shape.frames * READING_WORDS) as usize;
    let reading_zeros = vec![0.0f32; readings_len];
    let reading_bytes = f32::as_bytes(&reading_zeros);
    let readings = client.create_from_slice(reading_bytes);

    let reading = settings.reading;
    let immerkaer_batch = immerkaer_batch(client, shape);
    let sample = shape.temporal.then(|| sample_scratch(client, shape));
    let curve = curve_batch(client, shape, reading);
    let blocks = block_count(shape);
    let min_accepted = min_static_blocks(blocks);

    let fold_settings = fold_settings_buffer(client, shape, settings);

    let state_zeros = [0.0f32; STATE_WORDS as usize];
    let state_bytes = f32::as_bytes(&state_zeros);
    let fold_state = client.create_from_slice(state_bytes);

    let temporal_radius = shape.ring_radius;
    let history = ReadingHistory::new(client, temporal_radius);

    Measurement {
        partials,
        results,
        stats,
        readings,
        immerkaer_batch,
        sample,
        curve,
        min_accepted,
        fold_settings,
        fold_state,
        history,
        recorded: AtomicU32::new(0),
    }
}

fn fold_settings_buffer(client: &Client, shape: NoiseShape, settings: NoiseSettings) -> Handle {
    let words = fold_words(shape, settings);
    let bytes = f32::as_bytes(&words);
    client.create_from_slice(bytes)
}

/// Immerkær cubes in one frame of `shape`.
pub(super) fn cube_count(shape: NoiseShape) -> u32 {
    let lanes = partials_len(shape.width, shape.height) as u32;
    lanes / IMMERKAER_LANES
}

/// Temporal-stats blocks in one frame of `shape`.
pub(super) fn block_count(shape: NoiseShape) -> u32 {
    let (blocks_x, blocks_y) = temporal_stats_blocks(shape.width, shape.height);
    blocks_x * blocks_y
}

/// The immerkaer batch's lanes, one per cube for each channel plus, with temporal stats, two per
/// block for each active channel, its sigma and its texture.
fn immerkaer_lanes(shape: NoiseShape) -> u32 {
    let cubes = cube_count(shape);
    let block_lanes = if shape.temporal {
        2 * shape.channels * block_count(shape)
    } else {
        0
    };

    NOISE_CHANNELS * cubes + block_lanes
}

fn immerkaer_tag_count(shape: NoiseShape) -> u32 {
    if shape.temporal {
        immerkaer_tags(shape.channels)
    } else {
        IMMERKAER_SPATIAL_TAGS
    }
}

fn immerkaer_batch(client: &Client, shape: NoiseShape) -> SelectBatch {
    let len = immerkaer_lanes(shape);
    let tag_count = immerkaer_tag_count(shape);
    let min_counts = vec![0u32; tag_count as usize];
    let queries = immerkaer_queries(shape.temporal, shape.channels);

    SelectBatch::new(client, len, &min_counts, &queries)
}

/// The names one select batch's buffers are sized under.
struct BatchNames {
    keys: &'static str,
    tags: &'static str,
    histogram: &'static str,
}

const IMMERKAER_BATCH: BatchNames = BatchNames {
    keys: "noise immerkaer keys",
    tags: "noise immerkaer tags",
    histogram: "noise immerkaer histogram",
};

const SAMPLE_BATCH: BatchNames = BatchNames {
    keys: "noise sample keys",
    tags: "noise sample tags",
    histogram: "noise sample histogram",
};

const CURVE_BATCH: BatchNames = BatchNames {
    keys: "noise curve keys",
    tags: "noise curve tags",
    histogram: "noise curve histogram",
};

const READINGS: &str = "noise readings";
const BLOCK_VALUES: &str = "noise block values";
const BLOCK_FLAGS: &str = "noise block flags";
const STREAM_STATE: &str = "noise stream state";
const HISTORY_RECORDS: &str = "noise history records";
const HISTORY_RESULTS: &str = "noise history results";

/// The element count of every reading and stream buffer a noise state for `shape` holds.
///
/// The curve batch is counted whenever temporal stats are measured, which is the most any settings
/// allocate.
pub(crate) fn reading_buffer_sizes(shape: NoiseShape) -> Vec<BufferSize> {
    let frames = u64::from(shape.frames);
    let block_total = block_count(shape);
    let blocks = u64::from(block_total);
    let mut buffers = Vec::new();

    let readings = frames.checked_mul(u64::from(READING_WORDS));
    buffers.push((READINGS, readings));

    let immerkaer_lane_count = immerkaer_lanes(shape);
    let immerkaer_len = u64::from(immerkaer_lane_count);
    let immerkaer_tags = immerkaer_tag_count(shape);
    let immerkaer_queries = immerkaer_queries(shape.temporal, shape.channels).len() as u32;
    push_batch_sizes(
        &mut buffers,
        &IMMERKAER_BATCH,
        Some(immerkaer_len),
        immerkaer_tags,
        immerkaer_queries,
    );

    if shape.temporal {
        let values = blocks.checked_mul(u64::from(BLOCK_FIELDS));
        buffers.push((BLOCK_VALUES, values));
        buffers.push((BLOCK_FLAGS, Some(blocks)));

        let sample_lanes = sample_lanes_per_block(shape.channels);
        let sample_len = blocks.checked_mul(u64::from(sample_lanes));
        let sample_tags = sample_tags(shape.channels);
        let sample_queries = sample_queries(shape.channels).len() as u32;
        push_batch_sizes(
            &mut buffers,
            &SAMPLE_BATCH,
            sample_len,
            sample_tags,
            sample_queries,
        );

        let curve_len = blocks.checked_mul(u64::from(CURVE_LANES_PER_BLOCK));
        let curve_queries = curve_queries().len() as u32;
        push_batch_sizes(&mut buffers, &CURVE_BATCH, curve_len, CURVE_TAGS, curve_queries);
    }

    let state_words = u64::from(STATE_WORDS);
    buffers.push((STREAM_STATE, Some(state_words)));

    let radius = shape.ring_radius;
    let history_slots = u64::from(history_capacity(radius));
    let history_records = history_slots.checked_mul(u64::from(READING_WORDS));
    let history_results = history_slots.checked_mul(RESULTS_PER_SLOT as u64);
    buffers.push((HISTORY_RECORDS, history_records));
    buffers.push((HISTORY_RESULTS, history_results));

    buffers
}

fn push_batch_sizes(
    buffers: &mut Vec<BufferSize>,
    names: &BatchNames,
    len: Option<u64>,
    tag_count: u32,
    query_count: u32,
) {
    let histogram = histogram_len(tag_count, query_count);
    let histogram_words = u64::from(histogram);

    buffers.push((names.keys, len));
    buffers.push((names.tags, len));
    buffers.push((names.histogram, Some(histogram_words)));
}

/// The block scratch and the sample batch of [sample_lanes_per_block] lanes per block.
fn sample_scratch(client: &Client, shape: NoiseShape) -> SampleScratch {
    let blocks = block_count(shape);
    let values_bytes = (blocks * BLOCK_FIELDS) as usize * size_of::<f32>();
    let flags_bytes = blocks as usize * size_of::<u32>();
    let tag_count = sample_tags(shape.channels);
    let min_counts = vec![0u32; tag_count as usize];
    let queries = sample_queries(shape.channels);
    let lanes = sample_lanes_per_block(shape.channels);
    let len = lanes * blocks;
    let batch = SelectBatch::new(client, len, &min_counts, &queries);

    SampleScratch {
        block_values: client.empty(values_bytes),
        block_flags: client.empty(flags_bytes),
        batch,
    }
}

/// The curve batch of [CURVE_LANES_PER_BLOCK] quarter lanes per block, with temporal stats and the
/// luma quarter fields.
fn curve_batch(client: &Client, shape: NoiseShape, settings: ReadingSettings) -> Option<SelectBatch> {
    if !shape.temporal || !settings.luma_fields {
        return None;
    }

    let blocks = block_count(shape);
    let min_count = MIN_QUARTERS_PER_BIN as u32;
    let min_counts = vec![min_count; CURVE_TAGS as usize];
    let queries = curve_queries();
    let len = CURVE_LANES_PER_BLOCK * blocks;
    let batch = SelectBatch::new(client, len, &min_counts, &queries);

    Some(batch)
}
