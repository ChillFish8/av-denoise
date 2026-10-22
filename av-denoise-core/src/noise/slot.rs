use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NOISE_BLOCK_1D;
use super::kernels::{noise_block_accept, noise_block_derive, noise_sample_finish};
use super::reading::READING_WORDS;
use super::select::{
    SELECT_CUBE_THREADS,
    SELECT_QUERY_WORDS,
    SELECT_TAG_WORDS,
    SelectBatch,
    resolve_cube_threads,
};
use super::spatial::{
    NoiseCtx,
    immerkaer_sigma_scale,
    noise_partials_slot_view,
    partials_len,
    run_noise_estimate,
};
use super::state::{BLOCK_FIELDS, NoiseState, block_count, cube_count};
use super::temporal::{
    TemporalStatsCtx,
    run_temporal_noise_stats,
    temporal_stats_slot_len,
    temporal_stats_slot_view,
    zero_temporal_stats_slot,
};
use crate::tune::gpu::bind;

impl<R: Runtime> NoiseState<R> {
    /// Queues the Immerkær estimate and the temporal statistics for the frame in `slot` of `input`.
    ///
    /// `previous` is the slot the residual is taken against. `None`, for a stream's first frame
    /// or a duplicated slot, zeroes the record.
    pub(crate) fn queue_stats(
        &self,
        input: &Handle,
        slot: u32,
        previous: Option<u32>,
    ) -> Result<(), anyhow::Error> {
        self.queue_immerkaer(input, slot)?;

        let Some(stats_buf) = self.stats.as_ref() else {
            return Ok(());
        };

        let shape = self.shape;
        let Some(slot_prev) = previous else {
            zero_temporal_stats_slot(
                &self.client,
                stats_buf,
                shape.width,
                shape.height,
                shape.stored_ch,
                slot,
                shape.align,
            );
            return Ok(());
        };

        let stats_ctx = TemporalStatsCtx {
            width: shape.width,
            height: shape.height,
            stored_ch: shape.stored_ch,
            frame_count: shape.frames,
            slot_new: slot,
            slot_prev,
            input_buf: input,
            stats_buf,
            align: shape.align,
        };
        let luma_fields = self.settings.luma_fields;

        run_temporal_noise_stats(&self.client, &stats_ctx, luma_fields)
            .context("temporal noise stats dispatch failed")
    }

    fn queue_immerkaer(&self, input: &Handle, slot: u32) -> Result<(), anyhow::Error> {
        let shape = self.shape;
        let partials_slot = self.partials_slot(slot);

        let noise_ctx = NoiseCtx {
            width: shape.width,
            height: shape.height,
            channels: shape.channels,
            stored_ch: shape.stored_ch,
            frame_count: shape.frames,
            frame: slot,
            slot,
            input_buf: input,
            partials_buf: &partials_slot,
            results_buf: &self.results,
        };

        run_noise_estimate(&self.client, &noise_ctx).context("noise estimate dispatch failed")
    }
}

impl<R: Runtime> NoiseState<R> {
    /// Queues the reading of the stats in `slot` into the slot's reading words.
    ///
    /// With temporal stats that is block derive, the immerkaer batch, block accept, the sample batch
    /// and the finish, 9 launches. Without them it is block derive, the immerkaer batch and the
    /// finish, 5 launches.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no production code queues a slot reading")
    )]
    pub(crate) fn queue_reading(&self, slot: u32) -> Result<(), anyhow::Error> {
        self.queue_block_derive(slot);
        self.queue_immerkaer_batch();
        self.queue_block_accept(slot);
        self.queue_sample_batch();
        self.queue_sample_finish(slot);

        Ok(())
    }

    /// Queues the Immerkær cube lanes and, with temporal stats, every block's derived values.
    pub(crate) fn queue_block_derive(&self, slot: u32) {
        let shape = self.shape;
        let cubes = cube_count(shape);
        let blocks = if shape.temporal { block_count(shape) } else { 0 };
        let threads = cubes.max(blocks);
        let grid = threads.div_ceil(NOISE_BLOCK_1D);
        let batch = &self.immerkaer_batch;
        let batch_len = batch.len() as usize;
        let (partials, partials_words) = self.partials_view(slot);
        let block_args = self.block_args(slot, &partials, partials_words);
        let scale = immerkaer_sigma_scale();

        unsafe {
            noise_block_derive::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(NOISE_BLOCK_1D),
                ArrayArg::from_raw_parts(partials, partials_words),
                block_args.records,
                block_args.values,
                block_args.flags,
                bind(batch.keys(), batch_len),
                bind(batch.tags(), batch_len),
                scale,
                shape.width,
                shape.height,
                shape.channels,
                shape.stored_ch,
                shape.temporal,
            );
        }
    }

    /// Queues the immerkaer batch's three histogram passes.
    pub(crate) fn queue_immerkaer_batch(&self) {
        self.immerkaer_batch.queue(&self.client);
    }

    /// Queues the outlier ceiling and the sample batch's lanes, with temporal stats.
    pub(crate) fn queue_block_accept(&self, slot: u32) {
        let Some(sample) = self.sample.as_ref() else {
            return;
        };

        let shape = self.shape;
        let blocks = block_count(shape);
        let grid = blocks.div_ceil(SELECT_CUBE_THREADS);
        let reference = &self.immerkaer_batch;
        let reference_args = batch_args::<R>(reference);
        let (records, records_len) = self.records_view(slot).expect("a sample implies temporal stats");
        let values_len = (blocks * BLOCK_FIELDS) as usize;
        let batch_len = sample.batch.len() as usize;
        let luma_fields = self.settings.luma_fields;

        unsafe {
            noise_block_accept::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(SELECT_CUBE_THREADS),
                ArrayArg::from_raw_parts(records, records_len),
                bind(&sample.block_values, values_len),
                bind(&sample.block_flags, blocks as usize),
                reference_args.histogram,
                reference_args.state,
                reference_args.queries,
                reference_args.tag_ranges,
                bind(sample.batch.keys(), batch_len),
                bind(sample.batch.tags(), batch_len),
                shape.width,
                shape.height,
                shape.channels,
                shape.stored_ch,
                luma_fields,
                reference.tag_count(),
                reference.query_count(),
            );
        }
    }

    /// Queues the sample batch's three histogram passes, with temporal stats.
    pub(crate) fn queue_sample_batch(&self) {
        if let Some(sample) = self.sample.as_ref() {
            sample.batch.queue(&self.client);
        }
    }

    /// Queues the finish, which writes the reading of `slot` and clears both batches.
    pub(crate) fn queue_sample_finish(&self, slot: u32) {
        let shape = self.shape;
        let threads = resolve_cube_threads(&self.client);
        let immerkaer = &self.immerkaer_batch;
        let sample = self.sample_batch().unwrap_or(immerkaer);
        let immerkaer_args = batch_args::<R>(immerkaer);
        let sample_args = batch_args::<R>(sample);
        let blocks = block_count(shape) as f32;
        let readings_len = (shape.frames * READING_WORDS) as usize;

        // Without temporal stats the finish never touches the sample bindings, so the immerkaer
        // batch stands in for them.
        unsafe {
            noise_sample_finish::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(threads),
                immerkaer_args.histogram,
                immerkaer_args.state,
                immerkaer_args.queries,
                immerkaer_args.tag_ranges,
                sample_args.histogram,
                sample_args.state,
                sample_args.queries,
                sample_args.tag_ranges,
                bind(&self.readings, readings_len),
                slot,
                blocks,
                self.min_accepted,
                immerkaer.tag_count(),
                immerkaer.query_count(),
                shape.temporal,
                threads,
            );
        }
    }

    /// The histogram and the state of every batch a reading uses, with their lengths in words.
    pub(crate) fn reading_batch_buffers(&self) -> Vec<(Handle, u32)> {
        let mut buffers = Vec::new();
        let batches = [Some(&self.immerkaer_batch), self.sample_batch()];

        for batch in batches.into_iter().flatten() {
            buffers.push((batch.histogram().clone(), batch.histogram_len()));
            buffers.push((batch.state().clone(), batch.state_len()));
        }

        buffers
    }

    fn sample_batch(&self) -> Option<&SelectBatch> {
        self.sample.as_ref().map(|sample| &sample.batch)
    }

    /// The records of `slot` with their length in words.
    fn records_view(&self, slot: u32) -> Option<(Handle, usize)> {
        let shape = self.shape;
        let stats_buf = self.stats.as_ref()?;
        let records = temporal_stats_slot_view(
            stats_buf,
            shape.width,
            shape.height,
            shape.stored_ch,
            shape.frames,
            slot,
            shape.align,
        );
        let records_len = temporal_stats_slot_len(shape.width, shape.height, shape.stored_ch);

        Some((records, records_len))
    }

    /// The partials of `slot` with their length in words.
    fn partials_view(&self, slot: u32) -> (Handle, usize) {
        let shape = self.shape;
        let partials = noise_partials_slot_view(
            &self.partials,
            shape.width,
            shape.height,
            shape.frames,
            slot,
            shape.align,
        );
        let partials_words = partials_len(shape.width, shape.height);

        (partials, partials_words)
    }

    /// Block derive's record, value and flag bindings.
    ///
    /// Without temporal stats the kernel never touches them, so the partials of `slot` stand in.
    fn block_args(&self, slot: u32, partials: &Handle, partials_words: usize) -> BlockArgs<R> {
        let records = self.records_view(slot);
        let (Some((records, records_len)), Some(sample)) = (records, self.sample.as_ref()) else {
            return BlockArgs {
                records: bind(partials, partials_words),
                values: bind(partials, partials_words),
                flags: bind(partials, partials_words),
            };
        };

        let blocks = block_count(self.shape);
        let values_len = (blocks * BLOCK_FIELDS) as usize;

        BlockArgs {
            records: bind(&records, records_len),
            values: bind(&sample.block_values, values_len),
            flags: bind(&sample.block_flags, blocks as usize),
        }
    }
}

/// The per-block bindings of block derive.
struct BlockArgs<R: Runtime> {
    records: ArrayArg<R>,
    values: ArrayArg<R>,
    flags: ArrayArg<R>,
}

/// The bindings a consumer resolves a select batch through.
struct BatchArgs<R: Runtime> {
    histogram: ArrayArg<R>,
    state: ArrayArg<R>,
    queries: ArrayArg<R>,
    tag_ranges: ArrayArg<R>,
}

fn batch_args<R: Runtime>(batch: &SelectBatch) -> BatchArgs<R> {
    let histogram_len = batch.histogram_len() as usize;
    let state_len = batch.state_len() as usize;
    let query_words = (batch.query_count() * SELECT_QUERY_WORDS) as usize;
    let tag_words = (batch.tag_count() * SELECT_TAG_WORDS) as usize;

    BatchArgs {
        histogram: bind(batch.histogram(), histogram_len),
        state: bind(batch.state(), state_len),
        queries: bind(batch.query_table(), query_words),
        tag_ranges: bind(batch.tag_table(), tag_words),
    }
}
