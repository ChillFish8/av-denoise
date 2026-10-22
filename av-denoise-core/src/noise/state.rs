use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NOISE_BLOCK_1D;
use super::curve::MIN_QUARTERS_PER_BIN;
#[cfg(test)]
use super::reading::ReadingRecord;
use super::reading::{
    CURVE_TAGS,
    IMMERKAER_SPATIAL_TAGS,
    IMMERKAER_TAGS,
    READING_WORDS,
    SAMPLE_TAGS,
    curve_queries,
    immerkaer_queries,
    sample_queries,
};
use super::select::SelectBatch;
#[cfg(test)]
use super::select::{SelectQuery, order_key_host};
use super::spatial::{
    RESULTS_PER_SLOT,
    noise_partials_slot_stride_bytes,
    noise_partials_slot_view,
    noise_results_slot_view,
    partials_len,
};
use super::strength_map::{QuarterSettings, StrengthMapParams};
#[cfg(test)]
use super::temporal::temporal_stats_slot_len;
use super::temporal::{
    min_static_blocks,
    temporal_stats_blocks,
    temporal_stats_buf_bytes,
    temporal_stats_slot_view,
};
use crate::collab::geometry::strength_map_dims;
use crate::engine::align::StorageAlign;

/// The plane geometry one noise state measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NoiseShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    /// Ring slots, one stats region and one reading each.
    pub frames: u32,
    /// Whether temporal statistics are measured, which needs a temporal radius of at least 1.
    pub temporal: bool,
    pub align: StorageAlign,
}

/// Which strength map a reading builds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no production caller builds a chroma strength map")
)]
pub(crate) enum StrengthMapKind {
    #[default]
    Off,
    Luma(StrengthMapParams),
    Chroma {
        flat_boost: f32,
    },
}

/// The `f32` values block derive keeps per block, starting with the three channel sigmas.
pub(crate) const BLOCK_FIELDS: u32 = 8;
/// The block value of channel 0's mean residual.
pub(crate) const BLOCK_VALUE_MEAN0: u32 = 3;
/// The block value of channel 0's mean lag-1 product.
pub(crate) const BLOCK_VALUE_MEAN_LAG: u32 = 4;
/// The block value of channel 0's residual variance.
pub(crate) const BLOCK_VALUE_VAR0: u32 = 5;
/// The block value of the pixel pair count.
pub(crate) const BLOCK_VALUE_PAIRS: u32 = 6;
/// The unused last block value.
pub(crate) const BLOCK_VALUE_SPARE: u32 = 7;
/// The block flag of a block whose mean residual passes the static gate.
pub(crate) const BLOCK_CANDIDATE: u32 = 1;
/// The block flag of a block accepted as static noise.
pub(crate) const BLOCK_ACCEPTED: u32 = 2;
/// The class flag of a map quarter that holds pixels.
pub(crate) const CLASS_PRESENT: u32 = 1;
/// The class flag of a flat map quarter, before the texture veto.
pub(crate) const CLASS_FLAT: u32 = 2;
/// The class flag of a full map quarter that holds a strong line.
pub(crate) const CLASS_LINE: u32 = 4;
/// The `f32` words of one map quarter's structure tensor, `xx`, `yy` then `xy`.
pub(crate) const QUARTER_TENSOR_WORDS: u32 = 3;

/// What one slot's reading measures.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ReadingSettings {
    /// Whether the stats carry the luma quarter fields the curve and the classes read.
    pub luma_fields: bool,
    pub quarter: QuarterSettings,
    pub map: StrengthMapKind,
}

/// The noise measurement buffers of one ring of frames.
pub(crate) struct NoiseState<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) shape: NoiseShape,
    pub(super) settings: ReadingSettings,
    /// The first stage of each slot's Immerkær estimate, one padded region per slot.
    pub(super) partials: Handle,
    /// The per-channel Immerkær totals, one row per slot.
    pub(super) results: Handle,
    /// The temporal residual statistics, one padded region per slot, when they are measured.
    pub(super) stats: Option<Handle>,
    /// One reading of [READING_WORDS](crate::noise::reading::READING_WORDS) values per slot.
    pub(super) readings: Handle,
    /// Each Immerkær cube's sigmas and, with temporal stats, each candidate block's channel-0 sigma.
    pub(super) immerkaer_batch: SelectBatch,
    /// The per-block scratch and the sample batch, when temporal stats are measured.
    pub(super) sample: Option<SampleScratch>,
    /// Each flat, unclipped static quarter's sigma tagged with its luma bin, when temporal stats
    /// and the luma quarter fields are measured.
    pub(super) curve: Option<SelectBatch>,
    /// The quarter classes and the per-slot strength maps, when a strength map is built.
    pub(super) classes: Option<ClassScratch>,
    /// The fewest accepted blocks a sample needs to reach the static fraction.
    pub(super) min_accepted: u32,
}

/// The buffers one slot's temporal sample is built in.
pub(super) struct SampleScratch {
    /// [BLOCK_FIELDS](crate::noise::state::BLOCK_FIELDS) values per block.
    pub(super) block_values: Handle,
    /// One flag word per block.
    pub(super) block_flags: Handle,
    /// Each accepted block's sigmas, correlation and static quarter sigmas.
    pub(super) batch: SelectBatch,
}

/// The buffers one slot's strength map is built in, one entry per map quarter.
pub(super) struct ClassScratch {
    /// One word of `CLASS_*` flags per quarter.
    pub(super) classes: Handle,
    /// Each quarter's mean luma.
    pub(super) lumas: Handle,
    /// [QUARTER_TENSOR_WORDS](crate::noise::state::QUARTER_TENSOR_WORDS) values per quarter.
    pub(super) tensors: Handle,
    /// One threshold multiplier per quarter for every slot, created as ones.
    pub(super) slot_maps: Handle,
}

impl<R: Runtime> NoiseState<R> {
    pub(crate) fn new(client: &ComputeClient<R>, shape: NoiseShape, settings: ReadingSettings) -> Self {
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
                shape.stored_ch,
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

        let immerkaer_batch = immerkaer_batch(client, shape);
        let sample = shape.temporal.then(|| sample_scratch(client, shape));
        let curve = curve_batch(client, shape, settings);
        let classes = class_scratch(client, shape, settings);
        let blocks = block_count(shape);
        let min_accepted = min_static_blocks(blocks);

        Self {
            client: client.clone(),
            shape,
            settings,
            partials,
            results,
            stats,
            readings,
            immerkaer_batch,
            sample,
            curve,
            classes,
            min_accepted,
        }
    }

    pub(crate) fn shape(&self) -> NoiseShape {
        self.shape
    }

    pub(crate) fn settings(&self) -> ReadingSettings {
        self.settings
    }

    /// Replaces the settings before the first queued slot, allocating or dropping the curve batch
    /// and the class scratch to match.
    pub(crate) fn set_settings(&mut self, settings: ReadingSettings) {
        let has_curve = self.curve.is_some();
        let wants_curve = self.shape.temporal && settings.luma_fields;

        if wants_curve && !has_curve {
            self.curve = curve_batch(&self.client, self.shape, settings);
        }

        if !wants_curve {
            self.curve = None;
        }

        let has_classes = self.classes.is_some();
        let wants_classes = settings.map != StrengthMapKind::Off;

        if wants_classes && !has_classes {
            self.classes = class_scratch(&self.client, self.shape, settings);
        }

        if !wants_classes {
            self.classes = None;
        }

        self.settings = settings;
    }

    /// The results row, the partials and, with temporal stats, the records of `slot`.
    pub(crate) fn reading_handles(&self, slot: u32) -> Vec<Handle> {
        let shape = self.shape;
        let row = noise_results_slot_view(&self.results, shape.frames, slot);
        let partials = noise_partials_slot_view(
            &self.partials,
            shape.width,
            shape.height,
            shape.frames,
            slot,
            shape.align,
        );
        let mut handles = vec![row, partials];

        if let Some(stats_buf) = self.stats.as_ref() {
            let stats = temporal_stats_slot_view(
                stats_buf,
                shape.width,
                shape.height,
                shape.stored_ch,
                shape.frames,
                slot,
                shape.align,
            );
            handles.push(stats);
        }

        handles
    }

    /// The partials ring, offset to the start of `slot`.
    pub(super) fn partials_slot(&self, slot: u32) -> Handle {
        let stride = noise_partials_slot_stride_bytes(self.shape.width, self.shape.height, self.shape.align);
        let start = u64::from(slot) * stride;

        self.partials.clone().offset_start(start)
    }

    #[cfg(test)]
    pub(crate) fn partials(&self) -> &Handle {
        &self.partials
    }

    #[cfg(test)]
    pub(crate) fn results(&self) -> &Handle {
        &self.results
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> Option<&Handle> {
        self.stats.as_ref()
    }
}

/// Immerkær cubes in one frame of `shape`.
pub(super) fn cube_count(shape: NoiseShape) -> u32 {
    let lanes = partials_len(shape.width, shape.height) as u32;
    lanes / 4
}

/// Temporal-stats blocks in one frame of `shape`.
pub(super) fn block_count(shape: NoiseShape) -> u32 {
    let (blocks_x, blocks_y) = temporal_stats_blocks(shape.width, shape.height);
    blocks_x * blocks_y
}

/// Map quarters in one frame of `shape`.
pub(super) fn quarter_count(shape: NoiseShape) -> u32 {
    let (cols, rows) = strength_map_dims(shape.width, shape.height);
    cols * rows
}

/// The cubes of a launch with one thread per map quarter.
pub(super) fn quarter_grid(shape: NoiseShape) -> u32 {
    let quarters = quarter_count(shape);
    quarters.div_ceil(NOISE_BLOCK_1D)
}

/// The immerkaer batch, `3 * cubes` lanes plus, with temporal stats, one lane per block.
fn immerkaer_batch<R: Runtime>(client: &ComputeClient<R>, shape: NoiseShape) -> SelectBatch {
    let cubes = cube_count(shape);
    let block_lanes = if shape.temporal { block_count(shape) } else { 0 };
    let len = 3 * cubes + block_lanes;
    let tag_count = if shape.temporal {
        IMMERKAER_TAGS
    } else {
        IMMERKAER_SPATIAL_TAGS
    };
    let min_counts = vec![0u32; tag_count as usize];
    let queries = immerkaer_queries(shape.temporal);

    SelectBatch::new(client, len, &min_counts, &queries)
}

/// The block scratch and the sample batch of eight lanes per block.
fn sample_scratch<R: Runtime>(client: &ComputeClient<R>, shape: NoiseShape) -> SampleScratch {
    let blocks = block_count(shape);
    let values_bytes = (blocks * BLOCK_FIELDS) as usize * size_of::<f32>();
    let flags_bytes = blocks as usize * size_of::<u32>();
    let min_counts = vec![0u32; SAMPLE_TAGS as usize];
    let queries = sample_queries();
    let batch = SelectBatch::new(client, 8 * blocks, &min_counts, &queries);

    SampleScratch {
        block_values: client.empty(values_bytes),
        block_flags: client.empty(flags_bytes),
        batch,
    }
}

/// The curve batch of four quarter lanes per block, with temporal stats and the luma quarter fields.
fn curve_batch<R: Runtime>(
    client: &ComputeClient<R>,
    shape: NoiseShape,
    settings: ReadingSettings,
) -> Option<SelectBatch> {
    if !shape.temporal || !settings.luma_fields {
        return None;
    }

    let blocks = block_count(shape);
    let min_count = MIN_QUARTERS_PER_BIN as u32;
    let min_counts = vec![min_count; CURVE_TAGS as usize];
    let queries = curve_queries();
    let batch = SelectBatch::new(client, 4 * blocks, &min_counts, &queries);

    Some(batch)
}

/// The class scratch and the slot maps of ones, when `settings` builds a strength map.
fn class_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    shape: NoiseShape,
    settings: ReadingSettings,
) -> Option<ClassScratch> {
    if settings.map == StrengthMapKind::Off {
        return None;
    }

    let quarters = quarter_count(shape) as usize;
    let word = size_of::<f32>();
    let tensor_words = quarters * QUARTER_TENSOR_WORDS as usize;
    let map_ones = vec![1.0f32; quarters * shape.frames as usize];
    let map_bytes = f32::as_bytes(&map_ones);

    let scratch = ClassScratch {
        classes: client.empty(quarters * word),
        lumas: client.empty(quarters * word),
        tensors: client.empty(tensor_words * word),
        slot_maps: client.create_from_slice(map_bytes),
    };

    Some(scratch)
}

/// One slot's stats a test state starts with.
#[cfg(test)]
pub(crate) struct SlotStats<'a> {
    pub partials: &'a [f32],
    pub results_row: [f32; 4],
    pub records: &'a [f32],
}

#[cfg(test)]
impl<R: Runtime> NoiseState<R> {
    /// A state whose slot 0 holds `partials`, `results_row` and, with temporal stats, `records`.
    ///
    /// Every other slot and the padding between slots read 0.
    pub(crate) fn from_stats_for_test(
        client: &ComputeClient<R>,
        shape: NoiseShape,
        settings: ReadingSettings,
        partials: &[f32],
        results_row: [f32; 4],
        records: &[f32],
    ) -> Self {
        let slot = SlotStats {
            partials,
            results_row,
            records,
        };

        Self::from_slots_for_test(client, shape, settings, &[slot])
    }

    /// A state whose first slots hold `slots` in order, with every other slot and the padding at 0.
    pub(crate) fn from_slots_for_test(
        client: &ComputeClient<R>,
        shape: NoiseShape,
        settings: ReadingSettings,
        slots: &[SlotStats<'_>],
    ) -> Self {
        let mut state = Self::new(client, shape, settings);
        let frames = shape.frames as usize;
        let word = size_of::<f32>();
        let partials_words = partials_len(shape.width, shape.height);
        let records_words = temporal_stats_slot_len(shape.width, shape.height, shape.stored_ch);

        let partials_stride = noise_partials_slot_stride_bytes(shape.width, shape.height, shape.align);
        let partials_stride_words = partials_stride as usize / word;
        let mut partials_ring = vec![0.0f32; partials_stride_words * frames];
        let mut results_ring = vec![0.0f32; RESULTS_PER_SLOT * frames];

        let stats_bytes = temporal_stats_buf_bytes(
            shape.width,
            shape.height,
            shape.stored_ch,
            shape.frames,
            shape.align,
        );
        let stats_stride_words = stats_bytes / word / frames;
        let mut stats_ring = vec![0.0f32; stats_bytes / word];

        for (index, slot) in slots.iter().enumerate() {
            let slot_partials = slot.partials.len();
            assert_eq!(slot_partials, partials_words, "one slot's partials");

            let partials_start = index * partials_stride_words;
            let partials_end = partials_start + partials_words;
            partials_ring[partials_start..partials_end].copy_from_slice(slot.partials);

            let results_start = index * RESULTS_PER_SLOT;
            let results_end = results_start + RESULTS_PER_SLOT;
            results_ring[results_start..results_end].copy_from_slice(&slot.results_row);

            if shape.temporal {
                let slot_records = slot.records.len();
                assert_eq!(slot_records, records_words, "one slot's records");

                let records_start = index * stats_stride_words;
                let records_end = records_start + records_words;
                stats_ring[records_start..records_end].copy_from_slice(slot.records);
            }
        }

        state.partials = upload_ring(client, &partials_ring);
        state.results = upload_ring(client, &results_ring);

        if shape.temporal {
            let stats = upload_ring(client, &stats_ring);
            state.stats = Some(stats);
        }

        state
    }

    /// Fills every batch's keys and tags and the block scratch with values no reading may keep.
    ///
    /// Every tag joins channel 1 or 2 with a key far above any sigma, and every block reads as an
    /// accepted candidate with garbage values.
    pub(crate) fn poison_for_test(&mut self) {
        let client = self.client.clone();
        let immerkaer_queries = immerkaer_queries(self.shape.temporal);
        self.immerkaer_batch = poisoned_batch(&client, &self.immerkaer_batch, 0, &immerkaer_queries);

        if let Some(sample) = self.sample.as_mut() {
            let blocks = block_count(self.shape) as usize;
            let values = vec![7.5f32; blocks * BLOCK_FIELDS as usize];
            let flags = vec![BLOCK_CANDIDATE | BLOCK_ACCEPTED; blocks];
            let value_bytes = f32::as_bytes(&values);
            let flag_bytes = u32::as_bytes(&flags);

            sample.block_values = client.create_from_slice(value_bytes);
            sample.block_flags = client.create_from_slice(flag_bytes);
            let sample_queries = sample_queries();
            sample.batch = poisoned_batch(&client, &sample.batch, 0, &sample_queries);
        }

        if let Some(curve) = self.curve.as_ref() {
            let curve_queries = curve_queries();
            let min_count = MIN_QUARTERS_PER_BIN as u32;
            let poisoned = poisoned_batch(&client, curve, min_count, &curve_queries);
            self.curve = Some(poisoned);
        }

        if let Some(scratch) = self.classes.as_mut() {
            let quarters = quarter_count(self.shape) as usize;
            let tensor_words = quarters * QUARTER_TENSOR_WORDS as usize;
            let map_words = quarters * self.shape.frames as usize;
            let flags = vec![CLASS_PRESENT | CLASS_FLAT | CLASS_LINE; quarters];
            let lumas = vec![7.5f32; quarters];
            let tensors = vec![7.5f32; tensor_words];
            let maps = vec![7.5f32; map_words];

            let flag_bytes = u32::as_bytes(&flags);
            scratch.classes = client.create_from_slice(flag_bytes);
            scratch.lumas = upload_ring(&client, &lumas);
            scratch.tensors = upload_ring(&client, &tensors);
            scratch.slot_maps = upload_ring(&client, &maps);
        }
    }

    pub(crate) fn has_curve_batch(&self) -> bool {
        self.curve.is_some()
    }

    /// Reads every lane's tag of the curve batch back, empty without one.
    pub(crate) fn read_curve_tags(&self) -> Vec<u32> {
        let Some(curve) = self.curve.as_ref() else {
            return Vec::new();
        };

        let handle = curve.tags().clone();
        let bytes = self.client.read_one(handle).expect("the curve tags read back");
        let tags = u32::from_bytes(&bytes);

        tags.to_vec()
    }

    /// Reads the strength map of `slot` back, empty without a strength map.
    pub(crate) fn read_slot_map(&self, slot: u32) -> Vec<f32> {
        let Some(scratch) = self.classes.as_ref() else {
            return Vec::new();
        };

        let row_bytes = u64::from(quarter_count(self.shape)) * size_of::<f32>() as u64;
        let total_bytes = u64::from(self.shape.frames) * row_bytes;
        let start = u64::from(slot) * row_bytes;
        let end_trim = total_bytes - start - row_bytes;
        let view = scratch.slot_maps.clone().offset_start(start).offset_end(end_trim);
        let bytes = self.client.read_one(view).expect("the slot map reads back");
        let map = f32::from_bytes(&bytes);

        map.to_vec()
    }

    /// Reads every map quarter's class flags back, empty without a strength map.
    pub(crate) fn read_quarter_flags(&self) -> Vec<u32> {
        let Some(scratch) = self.classes.as_ref() else {
            return Vec::new();
        };

        let handle = scratch.classes.clone();
        let bytes = self.client.read_one(handle).expect("the quarter flags read back");
        let flags = u32::from_bytes(&bytes);

        flags.to_vec()
    }

    /// Reads every map quarter's mean luma back, empty without a strength map.
    pub(crate) fn read_quarter_lumas(&self) -> Vec<f32> {
        let Some(scratch) = self.classes.as_ref() else {
            return Vec::new();
        };

        let handle = scratch.lumas.clone();
        let bytes = self.client.read_one(handle).expect("the quarter lumas read back");
        let lumas = f32::from_bytes(&bytes);

        lumas.to_vec()
    }

    /// Reads the reading of `slot` back.
    pub(crate) fn read_reading(&self, slot: u32) -> ReadingRecord {
        let row_bytes = u64::from(READING_WORDS) * size_of::<f32>() as u64;
        let total_bytes = u64::from(self.shape.frames) * row_bytes;
        let start = u64::from(slot) * row_bytes;
        let end_trim = total_bytes - start - row_bytes;
        let view = self.readings.clone().offset_start(start).offset_end(end_trim);
        let bytes = self.client.read_one(view).expect("the reading reads back");
        let words = f32::from_bytes(&bytes);

        ReadingRecord::from_words(words)
    }

    /// Reads the partials, the results row and the records of `slot` back.
    ///
    /// The records are empty without temporal stats.
    pub(crate) fn read_slot_stats(&self, slot: u32) -> (Vec<f32>, [f32; 4], Vec<f32>) {
        let handles = self.reading_handles(slot);
        let buffers = self.client.read(handles);
        let row = f32::from_bytes(&buffers[0]);
        let partials = f32::from_bytes(&buffers[1]).to_vec();
        let records = buffers
            .get(2)
            .map(|bytes| f32::from_bytes(bytes).to_vec())
            .unwrap_or_default();

        let mut results_row = [0.0f32; RESULTS_PER_SLOT];
        results_row.copy_from_slice(row);

        (partials, results_row, records)
    }

    /// Reads every block's flags back, empty without temporal stats.
    pub(crate) fn read_block_flags(&self) -> Vec<u32> {
        let Some(sample) = self.sample.as_ref() else {
            return Vec::new();
        };

        let handle = sample.block_flags.clone();
        let bytes = self.client.read_one(handle).expect("the block flags read back");
        let flags = u32::from_bytes(&bytes);

        flags.to_vec()
    }
}

#[cfg(test)]
fn upload_ring<R: Runtime>(client: &ComputeClient<R>, ring: &[f32]) -> Handle {
    let bytes = f32::as_bytes(ring);
    client.create_from_slice(bytes)
}

/// A batch shaped like `batch` whose every element is tagged 1 or 2 with the key of 1.0.
///
/// Every tag keeps `min_count`.
#[cfg(test)]
fn poisoned_batch<R: Runtime>(
    client: &ComputeClient<R>,
    batch: &SelectBatch,
    min_count: u32,
    queries: &[SelectQuery],
) -> SelectBatch {
    let len = batch.len() as usize;
    let key = order_key_host(1.0);
    let keys = vec![key; len];
    let tags: Vec<u32> = (0..len).map(|index| 1 + (index % 2) as u32).collect();
    let min_counts = vec![min_count; batch.tag_count() as usize];

    SelectBatch::from_elements(client, &keys, &tags, &min_counts, queries)
}
