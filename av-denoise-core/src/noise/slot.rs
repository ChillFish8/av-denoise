use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::spatial::{NoiseCtx, run_noise_estimate};
use super::state::NoiseState;
use super::temporal::{TemporalStatsCtx, run_temporal_noise_stats, zero_temporal_stats_slot};

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
