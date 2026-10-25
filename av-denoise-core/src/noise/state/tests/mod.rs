use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_ACCEPTED,
    BLOCK_CANDIDATE,
    BLOCK_FIELDS,
    CLASS_FLAT,
    CLASS_LINE,
    CLASS_PRESENT,
    Measurement,
    NoiseShape,
    NoiseState,
    QUARTER_TENSOR_WORDS,
    ReadingSettings,
    block_count,
    quarter_count,
};
#[cfg(any(feature = "vulkan", feature = "metal"))]
use super::{
    BLOCK_FLAGS,
    BLOCK_VALUES,
    BatchNames,
    CURRENT_MAP,
    CURVE_BATCH,
    IMMERKAER_BATCH,
    QUARTER_CLASSES,
    QUARTER_LUMAS,
    QUARTER_TENSORS,
    READINGS,
    SAMPLE_BATCH,
    SLOT_MAPS,
    STREAM_STATE,
};
#[cfg(any(feature = "vulkan", feature = "metal"))]
use crate::noise::correlation::build_spatial_offset_lut;
use crate::noise::curve::MIN_QUARTERS_PER_BIN;
use crate::noise::derived::DerivedValues;
use crate::noise::fold::NoiseSettings;
use crate::noise::reading::{READING_WORDS, ReadingRecord, curve_queries, immerkaer_queries, sample_queries};
use crate::noise::select::{SelectBatch, SelectQuery, order_key_host};
use crate::noise::spatial::{
    RESULTS_PER_SLOT,
    noise_partials_slot_stride_bytes,
    noise_partials_slot_view,
    noise_results_slot_view,
    partials_len,
};
use crate::noise::temporal::{temporal_stats_buf_bytes, temporal_stats_slot_len, temporal_stats_slot_view};

/// One slot's stats a test state starts with.
pub(crate) struct SlotStats<'a> {
    pub partials: &'a [f32],
    pub results_row: [f32; 4],
    pub records: &'a [f32],
}

impl<R: Runtime> NoiseState<R> {
    /// The measurement buffers, mutably.
    ///
    /// # Panics
    ///
    /// Panics on a state that holds fixed values.
    fn measurement_mut(&mut self) -> &mut Measurement {
        self.measurement
            .as_mut()
            .expect("a fixed noise state has no measurement buffers")
    }

    pub(crate) fn shape(&self) -> NoiseShape {
        self.shape
    }

    /// The results row, the partials and, with temporal stats, the records of `slot`.
    pub(crate) fn reading_handles(&self, slot: u32) -> Vec<Handle> {
        let shape = self.shape;
        let measurement = self.measurement();
        let row = noise_results_slot_view(&measurement.results, shape.frames, slot);
        let partials = noise_partials_slot_view(
            &measurement.partials,
            shape.width,
            shape.height,
            shape.frames,
            slot,
            shape.align,
        );
        let mut handles = vec![row, partials];

        if let Some(stats_buf) = measurement.stats.as_ref() {
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

    pub(crate) fn partials(&self) -> &Handle {
        &self.measurement().partials
    }

    pub(crate) fn results(&self) -> &Handle {
        &self.measurement().results
    }

    pub(crate) fn stats(&self) -> Option<&Handle> {
        self.measurement().stats.as_ref()
    }

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
        let noise_settings = NoiseSettings {
            reading: settings,
            ..NoiseSettings::default()
        };
        let initial = DerivedValues::default();
        let mut state = Self::new(client, shape, noise_settings, initial);
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

        let measurement = state.measurement_mut();
        measurement.partials = upload_ring(client, &partials_ring);
        measurement.results = upload_ring(client, &results_ring);

        if shape.temporal {
            let stats = upload_ring(client, &stats_ring);
            measurement.stats = Some(stats);
        }

        state
    }

    /// A state whose slots hold `readings`, the Immerkær abs sums `immerkaer_sums` and, with a
    /// strength map, the slot maps `slot_maps`, one entry per slot.
    pub(crate) fn with_readings_for_test(
        client: &ComputeClient<R>,
        shape: NoiseShape,
        settings: NoiseSettings,
        readings: &[ReadingRecord],
        immerkaer_sums: &[[f32; 4]],
        slot_maps: &[Vec<f32>],
    ) -> Self {
        let mut state = Self::new(client, shape, settings, DerivedValues::default());
        state.set_readings_for_test(readings, immerkaer_sums, slot_maps);

        state
    }

    /// Replaces every slot's reading, Immerkær abs sums and, with a strength map, slot map.
    ///
    /// The stream state, the derived values, the curve and the current map are kept.
    pub(crate) fn set_readings_for_test(
        &mut self,
        readings: &[ReadingRecord],
        immerkaer_sums: &[[f32; 4]],
        slot_maps: &[Vec<f32>],
    ) {
        let frames = self.shape.frames as usize;
        let reading_count = readings.len();
        let sums_count = immerkaer_sums.len();
        assert_eq!(reading_count, frames, "one reading per slot");
        assert_eq!(sums_count, frames, "one row of sums per slot");

        let words: Vec<f32> = readings.iter().flat_map(ReadingRecord::to_words).collect();
        let sums: Vec<f32> = immerkaer_sums.iter().flatten().copied().collect();
        let quarters = quarter_count(self.shape) as usize;
        let client = self.client.clone();
        let measurement = self.measurement_mut();
        measurement.readings = upload_ring(&client, &words);
        measurement.results = upload_ring(&client, &sums);

        if let Some(scratch) = measurement.classes.as_mut() {
            let map_count = slot_maps.len();
            assert_eq!(map_count, frames, "one map per slot");

            let maps: Vec<f32> = slot_maps.iter().flatten().copied().collect();
            let map_words = maps.len();
            assert_eq!(map_words, quarters * frames, "one multiplier per quarter");

            scratch.slot_maps = upload_ring(&client, &maps);
        }
    }

    /// Rebuilds a fixed state's spatial table at `rho`, as a fold would after a temporal sample.
    ///
    /// # Panics
    ///
    /// Panics on a state that measures the noise or keeps no offset tables.
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    pub(crate) fn set_fixed_rho_for_test(&mut self, rho: f32) {
        let measured = self.is_measured();
        assert!(!measured, "a measured noise state's folds write its own tables");

        let noise_offset = self.fixed_values.noise_offset;
        let tables = self
            .offset_tables
            .as_mut()
            .expect("the state keeps offset tables");
        let entries = build_spatial_offset_lut(tables.radius, rho, noise_offset);
        let bytes = f32::as_bytes(&entries);

        tables.spatial = self.client.create_from_slice(bytes);
    }

    /// Reads the current strength map back.
    pub(crate) fn read_current_map(&self) -> Vec<f32> {
        let handle = self.strength_map().clone();
        let bytes = self.client.read_one(handle).expect("the current map reads back");
        let map = f32::from_bytes(&bytes);

        map.to_vec()
    }

    /// Reads the current curve buffer back.
    pub(crate) fn read_curve(&self) -> Vec<f32> {
        let handle = self.curve().clone();
        let bytes = self.client.read_one(handle).expect("the curve reads back");
        let ratios = f32::from_bytes(&bytes);

        ratios.to_vec()
    }

    /// Fills every batch's keys and tags and the block scratch with values no reading may keep.
    ///
    /// Every tag joins channel 1 or 2 with a key far above any sigma, and every block reads as an
    /// accepted candidate with garbage values.
    pub(crate) fn poison_for_test(&mut self) {
        let client = self.client.clone();
        let shape = self.shape;
        let immerkaer_queries = immerkaer_queries(shape.temporal);
        let measurement = self.measurement_mut();
        let immerkaer_batch = &measurement.immerkaer_batch;
        measurement.immerkaer_batch = poisoned_batch(&client, immerkaer_batch, 0, &immerkaer_queries);

        if let Some(sample) = measurement.sample.as_mut() {
            let blocks = block_count(shape) as usize;
            let values = vec![7.5f32; blocks * BLOCK_FIELDS as usize];
            let flags = vec![BLOCK_CANDIDATE | BLOCK_ACCEPTED; blocks];
            let value_bytes = f32::as_bytes(&values);
            let flag_bytes = u32::as_bytes(&flags);

            sample.block_values = client.create_from_slice(value_bytes);
            sample.block_flags = client.create_from_slice(flag_bytes);
            let sample_queries = sample_queries();
            sample.batch = poisoned_batch(&client, &sample.batch, 0, &sample_queries);
        }

        if let Some(curve) = measurement.curve.as_ref() {
            let curve_queries = curve_queries();
            let min_count = MIN_QUARTERS_PER_BIN as u32;
            let poisoned = poisoned_batch(&client, curve, min_count, &curve_queries);
            measurement.curve = Some(poisoned);
        }

        if let Some(scratch) = measurement.classes.as_mut() {
            let quarters = quarter_count(shape) as usize;
            let tensor_words = quarters * QUARTER_TENSOR_WORDS as usize;
            let map_words = quarters * shape.frames as usize;
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

    /// Every buffer [reading_buffer_sizes](crate::noise::state::reading_buffer_sizes) counts, by the same
    /// names, `None` where the settings allocate none.
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    pub(crate) fn buffers_for_test(&self) -> Vec<(&'static str, Option<&Handle>)> {
        let measurement = self.measurement();
        let sample = measurement.sample.as_ref();
        let curve = measurement.curve.as_ref();
        let classes = measurement.classes.as_ref();
        let mut buffers = vec![(READINGS, Some(&measurement.readings))];

        push_batch_handles(&mut buffers, &IMMERKAER_BATCH, Some(&measurement.immerkaer_batch));

        if self.shape.temporal {
            let values = sample.map(|scratch| &scratch.block_values);
            let flags = sample.map(|scratch| &scratch.block_flags);
            let sample_batch = sample.map(|scratch| &scratch.batch);
            buffers.push((BLOCK_VALUES, values));
            buffers.push((BLOCK_FLAGS, flags));
            push_batch_handles(&mut buffers, &SAMPLE_BATCH, sample_batch);
            push_batch_handles(&mut buffers, &CURVE_BATCH, curve);
        }

        let class_flags = classes.map(|scratch| &scratch.classes);
        let lumas = classes.map(|scratch| &scratch.lumas);
        let tensors = classes.map(|scratch| &scratch.tensors);
        let slot_maps = classes.map(|scratch| &scratch.slot_maps);
        buffers.push((QUARTER_CLASSES, class_flags));
        buffers.push((QUARTER_LUMAS, lumas));
        buffers.push((QUARTER_TENSORS, tensors));
        buffers.push((SLOT_MAPS, slot_maps));
        buffers.push((CURRENT_MAP, Some(&self.current_map)));
        buffers.push((STREAM_STATE, Some(&measurement.fold_state)));

        buffers
    }

    pub(crate) fn has_curve_batch(&self) -> bool {
        self.measurement().curve.is_some()
    }

    /// Reads every lane's tag of the curve batch back, empty without one.
    pub(crate) fn read_curve_tags(&self) -> Vec<u32> {
        let Some(curve) = self.measurement().curve.as_ref() else {
            return Vec::new();
        };

        let handle = curve.tags().clone();
        let bytes = self.client.read_one(handle).expect("the curve tags read back");
        let tags = u32::from_bytes(&bytes);

        tags.to_vec()
    }

    /// Reads the strength map of `slot` back, empty without a strength map.
    pub(crate) fn read_slot_map(&self, slot: u32) -> Vec<f32> {
        let Some(scratch) = self.measurement().classes.as_ref() else {
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
        let Some(scratch) = self.measurement().classes.as_ref() else {
            return Vec::new();
        };

        let handle = scratch.classes.clone();
        let bytes = self.client.read_one(handle).expect("the quarter flags read back");
        let flags = u32::from_bytes(&bytes);

        flags.to_vec()
    }

    /// Reads every map quarter's mean luma back, empty without a strength map.
    pub(crate) fn read_quarter_lumas(&self) -> Vec<f32> {
        let Some(scratch) = self.measurement().classes.as_ref() else {
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
        let readings = &self.measurement().readings;
        let view = readings.clone().offset_start(start).offset_end(end_trim);
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
        let Some(sample) = self.measurement().sample.as_ref() else {
            return Vec::new();
        };

        let handle = sample.block_flags.clone();
        let bytes = self.client.read_one(handle).expect("the block flags read back");
        let flags = u32::from_bytes(&bytes);

        flags.to_vec()
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
fn push_batch_handles<'a>(
    buffers: &mut Vec<(&'static str, Option<&'a Handle>)>,
    names: &BatchNames,
    batch: Option<&'a SelectBatch>,
) {
    let keys = batch.map(SelectBatch::keys);
    let tags = batch.map(SelectBatch::tags);
    let histogram = batch.map(SelectBatch::histogram);

    buffers.push((names.keys, keys));
    buffers.push((names.tags, tags));
    buffers.push((names.histogram, histogram));
}

fn upload_ring<R: Runtime>(client: &ComputeClient<R>, ring: &[f32]) -> Handle {
    let bytes = f32::as_bytes(ring);
    client.create_from_slice(bytes)
}

/// A batch shaped like `batch` whose every element is tagged 1 or 2 with the key of 1.0.
///
/// Every tag keeps `min_count`.
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
