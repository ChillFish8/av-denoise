use std::sync::Arc;

use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use super::readings::{ReadingShape, SlotNoiseReading};
use crate::engine::EngineTiming;
use crate::engine::timing::{DeviceTimed, DeviceTimer};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::params::sigma_eff;
use crate::nlmeans::prefilter::PrefilterMode;
use crate::noise::{
    EMA_ALPHA,
    NoiseCtx,
    NoiseCurve,
    QuarterClasses,
    RESULTS_PER_SLOT,
    TEMPORAL_QUARTER_SIZE,
    TemporalNoiseSample,
    TemporalStatsCtx,
    build_spatial_offset_lut,
    correlation_factor,
    noise_partials_slot_stride_bytes,
    noise_partials_slot_view,
    noise_results_slot_view,
    run_noise_estimate,
    run_temporal_noise_stats,
    sigma_from_abs_sum,
    temporal_stats_slot_view,
    zero_temporal_stats_slot,
};

impl<R: Runtime> DeviceTimed for NlmDenoiser<R> {
    fn device_timer(&mut self) -> &mut DeviceTimer {
        &mut self.device_timer
    }
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Host and device time this denoiser has spent.
    pub(crate) fn timing(&mut self) -> EngineTiming {
        let (device_time, device_samples) = self.device_timer.settle();
        let stalls = self.noise_readings.stalls();

        EngineTiming {
            readback_wait: stalls.wait(),
            readback_waits: stalls.waits(),
            device_time,
            device_samples,
        }
    }

    pub(crate) fn sample_device_time(&mut self, every: u32) {
        let timing_method = self.client.properties().timing_method;
        self.device_timer.set_timing_method(timing_method);
        self.device_timer.set_every(every);
    }

    /// Queues the Immerkær noise estimate for `slot` when automatic estimation is active.
    pub(super) fn run_noise_estimate_for_slot(&self, slot: u32) -> Result<(), anyhow::Error> {
        let (Some(partials_buf), Some(results_buf)) =
            (self.noise_partials.as_ref(), self.noise_results.as_ref())
        else {
            return Ok(());
        };

        let stride = noise_partials_slot_stride_bytes(self.width, self.height, self.align);
        let partials_slot = partials_buf.clone().offset_start((slot as u64) * stride);

        let noise_ctx = NoiseCtx {
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            frame_count: self.params.total_frames(),
            frame: slot,
            slot,
            input_buf: &self.input_buf,
            partials_buf: &partials_slot,
            results_buf,
        };

        run_noise_estimate::<R>(&self.client, &noise_ctx).context("noise estimate dispatch failed")
    }

    /// Queues the temporal residual statistics between `slot` and the slot before it.
    ///
    /// A stream's first frame has no predecessor, so its record is zeroed instead.
    pub(super) fn run_temporal_stats_for_slot(&self, slot: u32) -> Result<(), anyhow::Error> {
        let Some(stats_buf) = self.temporal_stats_buf.as_ref() else {
            return Ok(());
        };

        if self.ring_head == 0 {
            self.zero_temporal_stats_for_slot(slot);
            return Ok(());
        }

        let total_frames = self.params.total_frames();
        let slot_prev = (slot + total_frames - 1) % total_frames;

        let stats_ctx = TemporalStatsCtx {
            width: self.width,
            height: self.height,
            stored_ch: self.params.channels.storage_count(),
            frame_count: total_frames,
            slot_new: slot,
            slot_prev,
            input_buf: &self.input_buf,
            stats_buf,
            align: self.align,
        };

        run_temporal_noise_stats::<R>(&self.client, &stats_ctx, self.luma_noise_fields)
            .context("temporal noise stats dispatch failed")
    }

    pub(crate) fn set_luma_noise_fields(&mut self, on: bool) {
        self.luma_noise_fields = on;
    }

    pub(crate) fn set_flat_texture_cut(&mut self, cut: Option<f32>) {
        self.quarter_settings.texture_cut = cut;
    }

    /// Sets the line ring's radius in pixels.
    ///
    /// Within the ring around strong lines, the luma flat map skips shadow soften and the flat texture
    /// cut. The radius rounds up to whole 8x8 quarters. `None` or `0` turns the ring off.
    pub(crate) fn set_line_ring(&mut self, radius_px: Option<u32>) {
        let active_radius = radius_px.filter(|&radius| radius > 0);
        let quarters = active_radius.map(|radius| radius.div_ceil(TEMPORAL_QUARTER_SIZE) as usize);
        self.quarter_settings.line_ring = quarters;
    }

    pub(crate) fn current_noise_curve(&self) -> Option<NoiseCurve> {
        self.noise_curve
    }

    pub(crate) fn current_quarter_classes(&self) -> Option<&QuarterClasses> {
        self.quarter_classes.as_ref()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn flat_texture_cut(&self) -> Option<f32> {
        self.quarter_settings.texture_cut
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn line_ring(&self) -> Option<usize> {
        self.quarter_settings.line_ring
    }

    /// Zeroes a duplicated slot's temporal stats record.
    ///
    /// A duplicate matches the slot before it, so zeroes are the measured answer at less cost.
    pub(super) fn zero_temporal_stats_for_slot(&self, slot: u32) {
        let Some(stats_buf) = self.temporal_stats_buf.as_ref() else {
            return;
        };

        zero_temporal_stats_slot::<R>(
            &self.client,
            stats_buf,
            self.width,
            self.height,
            self.params.channels.storage_count(),
            slot,
            self.align,
        );
    }

    /// The handles one slot's noise reading reads, in [SlotNoiseReading::from_bytes] order.
    ///
    /// `None` when automatic noise estimation is off.
    pub(in crate::nlmeans) fn noise_reading_handles(&self, slot: u32) -> Option<Vec<Handle>> {
        let (Some(partials_buf), Some(results_buf)) =
            (self.noise_partials.as_ref(), self.noise_results.as_ref())
        else {
            return None;
        };

        let frame_count = self.params.total_frames();
        let row = noise_results_slot_view(results_buf, frame_count, slot);
        let partials = noise_partials_slot_view(
            partials_buf,
            self.width,
            self.height,
            frame_count,
            slot,
            self.align,
        );
        let mut handles = vec![row, partials];

        if let Some(stats_buf) = self.temporal_stats_buf.as_ref() {
            let stored_ch = self.params.channels.storage_count();
            let stats = temporal_stats_slot_view(
                stats_buf,
                self.width,
                self.height,
                stored_ch,
                frame_count,
                slot,
                self.align,
            );
            handles.push(stats);
        }

        Some(handles)
    }

    pub(in crate::nlmeans) fn reading_shape(&self) -> ReadingShape {
        ReadingShape {
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            with_curve: self.luma_noise_fields,
            settings: self.quarter_settings,
        }
    }

    /// Starts reading back the noise that push `push_index` measured into `slot`.
    ///
    /// It runs right after the slot's noise kernels are queued, so the copy lands while later work
    /// runs.
    pub(super) fn start_noise_reading(&mut self, slot: u32, push_index: usize) {
        let Some(handles) = self.noise_reading_handles(slot) else {
            return;
        };

        let shape = self.reading_shape();
        self.noise_readings
            .start(&self.client, slot, push_index, handles, shape);
    }

    /// The push that filled logical ring position `logical`, counting duplicates.
    fn push_index_at(&self, logical: u32) -> usize {
        let total_frames = self.params.total_frames() as usize;
        debug_assert!(
            self.ring_head >= total_frames,
            "the ring is full whenever a reading is folded"
        );

        self.ring_head - total_frames + logical as usize
    }

    /// The reading push `push_index` left in `slot`, waiting for it if it has not landed.
    fn slot_reading(&mut self, slot: u32, push_index: usize) -> Result<Arc<SlotNoiseReading>, anyhow::Error> {
        let stored = self.noise_readings.get(slot, push_index)?;
        if let Some(reading) = stored {
            return Ok(reading);
        }

        // Every write to a slot starts its reading, so this only covers a slot nothing has read yet.
        self.noise_readings.count_fallback();
        self.start_noise_reading(slot, push_index);

        let started = self.noise_readings.get(slot, push_index)?;
        let reading = started.ok_or_else(|| anyhow::anyhow!("no noise reading for slot {slot}"))?;

        Ok(reading)
    }

    /// The noise reading of logical ring position `logical`.
    pub(in crate::nlmeans) fn noise_reading_at(
        &mut self,
        logical: u32,
    ) -> Result<Arc<SlotNoiseReading>, anyhow::Error> {
        let slot = self.ring_slot(logical);
        let push_index = self.push_index_at(logical);

        self.slot_reading(slot, push_index)
    }

    /// Marks the start of an engine call, so the readings it starts are not polled during it.
    pub(crate) fn begin_call(&mut self) {
        self.call_start_push = self.ring_head;
    }

    /// Turns landed readings that earlier engine calls started into values.
    ///
    /// Readings the current call started are left alone, since on CUDA and ROCm a poll waits for
    /// the copy.
    pub(crate) fn process_landed_readings(&mut self) -> Result<(), anyhow::Error> {
        self.noise_readings.process_landed(self.call_start_push)
    }

    /// Folds the first frame's noise estimate straight away so push-time work has a real sigma.
    ///
    /// Push-time stages such as the NLM pilot run before the first submit folds an estimate. This
    /// waits for the push's own reading, which was queued just before.
    ///
    /// It must run after this push starts its reading and before `advance_ring` moves
    /// `frames_loaded`, which the first-frame check reads. The first submit folds the same frame
    /// again, which is harmless because it reproduces these values to within rounding.
    pub(super) fn seed_noise_estimate_if_first_frame(&mut self, slot: u32) -> Result<(), anyhow::Error> {
        if self.frames_loaded != 0 || self.noise_results.is_none() {
            return Ok(());
        }

        let reading = self
            .slot_reading(slot, self.ring_head)
            .context("noise-estimate seed readback failed")?;

        // The first frame has no temporal record, so Immerkær alone seeds it.
        self.fold_noise_estimate(reading.immerkaer_sums, None, reading.immerkaer_low);

        Ok(())
    }

    /// Folds one slot's noise readings into every estimator chain and refreshes the derived values.
    ///
    /// Each chain takes the larger of an Immerkær reading and a temporal reading, so the temporal
    /// side can raise an estimate but an unreliable shot cannot lower it. The median chain drives
    /// the strength. The low chain reads lower quartiles and drives the noise offsets, where
    /// reading too high scrubs fine texture.
    ///
    /// The unboosted chain skips the correlation boost for consumers that square sigma into a
    /// threshold. The temporal-only chain also skips the spatial maximum, because a spatial mask
    /// reads repeating texture as noise.
    fn fold_noise_estimate(
        &mut self,
        immerkaer_sums: [f32; RESULTS_PER_SLOT],
        temporal: Option<TemporalNoiseSample>,
        immerkaer_low: [f32; 3],
    ) {
        let channels = self.params.channels.count() as usize;

        let mut raw = [0.0f32; 3];
        for (c, sigma) in raw.iter_mut().enumerate().take(channels) {
            *sigma = sigma_from_abs_sum(immerkaer_sums[c], self.width, self.height);
        }

        let mut raw_low = immerkaer_low;
        let mut raw_low_unboosted = immerkaer_low;
        let mut raw_temporal_only: Option<[f32; 3]> = None;

        if let Some(sample) = temporal {
            let factor = correlation_factor(sample.rho);
            for c in 0..channels {
                raw[c] = raw[c].max(sample.sigma[c] * factor);
                raw_low[c] = raw_low[c].max(sample.sigma_low[c] * factor);
                raw_low_unboosted[c] = raw_low_unboosted[c].max(sample.sigma_low[c]);
            }

            raw_temporal_only = Some(sample.sigma);

            let rho = match self.rho_smoothed {
                None => sample.rho,
                Some(previous) => EMA_ALPHA * sample.rho + (1.0 - EMA_ALPHA) * previous,
            };
            self.rho_smoothed = Some(rho);
        }

        // The user's scale applies after combining and before smoothing, so everything derived
        // follows it.
        let sigma_scale = self.params.hq.map_or(1.0, |hq| hq.sigma_scale);
        for c in 0..channels {
            raw[c] *= sigma_scale;
            raw_low[c] *= sigma_scale;
            raw_low_unboosted[c] *= sigma_scale;
        }

        if let Some(raw_temporal) = raw_temporal_only.as_mut() {
            for sigma in raw_temporal.iter_mut().take(channels) {
                *sigma *= sigma_scale;
            }
        }

        let updated = self.noise_estimator.update(&raw[..channels]);
        let mut smoothed = [0.0f32; 3];
        smoothed[..channels].copy_from_slice(updated);

        let updated_low = self.noise_estimator_low.update(&raw_low[..channels]);
        let mut smoothed_low = [0.0f32; 3];
        smoothed_low[..channels].copy_from_slice(updated_low);

        self.noise_estimator_low_unboosted
            .update(&raw_low_unboosted[..channels]);

        if let Some(raw_temporal) = raw_temporal_only {
            self.noise_estimator_temporal_only
                .update(&raw_temporal[..channels]);
        }

        let effective_sigma = sigma_eff(&smoothed[..channels], self.params.channels);
        self.h2_inv_norm = self.params.h2_inv_norm_with(Some(effective_sigma));
        self.input_noise_offset = self.params.noise_offset_with(Some(&smoothed_low[..channels]));
        self.noise_offset = match self.params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => self.input_noise_offset,
        };

        // Motion estimation treats channel 0 as luma.
        self.sigma_y = smoothed[0];
    }

    /// Refreshes the derived filter values from the window centre's noise reading.
    ///
    /// The reading was queued when the slot was written, so it has almost always landed.
    pub(super) fn update_noise_estimate(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        debug_assert!(
            center_t < self.params.total_frames(),
            "center_t must be a logical ring position"
        );

        let centre = self.noise_reading_at(center_t)?;
        let lender = self.borrow_reading_ahead(center_t, &centre)?;

        // The curve only changes alongside a trustworthy sample.
        if lender.temporal.sample.is_some() {
            self.noise_curve = lender.temporal.curve;
            self.quarter_classes = lender.temporal.classes.clone();
        }

        self.fold_noise_estimate(
            centre.immerkaer_sums,
            lender.temporal.sample,
            centre.immerkaer_low,
        );

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn noise_reading_fallbacks(&self) -> u64 {
        self.noise_readings.fallbacks()
    }

    /// Points `spatial_offset_lut` at the table `noise_offset` and `rho_smoothed` imply.
    ///
    /// The table uploads only when it changed.
    pub(super) fn rebuild_spatial_offset_lut(&mut self) {
        let rho = self.rho_smoothed.unwrap_or(0.0);
        let offsets = build_spatial_offset_lut(self.params.search_radius, rho, self.noise_offset);
        let offset_bytes = f32::as_bytes(&offsets);

        self.spatial_offset_lut.update(&self.client, offset_bytes);
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn spatial_offset_lut_for_test(&self) -> (&MirroredBuffer, Vec<f32>) {
        let rho = self.rho_smoothed.unwrap_or(0.0);
        let expected = build_spatial_offset_lut(self.params.search_radius, rho, self.noise_offset);

        (&self.spatial_offset_lut, expected)
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn pilot_offset_lut_for_test(&self) -> &MirroredBuffer {
        &self.pilot_offset_lut
    }

    /// The median chain's smoothed per-channel sigma.
    ///
    /// A fixed `sigma_override` is returned for every channel. Before the first estimate, and
    /// on the fast path, it is zero.
    pub fn current_sigmas(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        let mut sigmas = [0.0f32; 3];
        if let Some(smoothed) = self.noise_estimator.current() {
            sigmas[..channels].copy_from_slice(&smoothed[..channels]);
        }

        sigmas
    }

    /// The low chain's smoothed per-channel sigma without the correlation boost.
    ///
    /// A fixed `sigma_override` is returned for every channel. Before the first estimate, and
    /// on the fast path, it is zero.
    pub fn current_sigmas_low_unboosted(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        let mut sigmas = [0.0f32; 3];
        if let Some(smoothed) = self.noise_estimator_low_unboosted.current() {
            sigmas[..channels].copy_from_slice(&smoothed[..channels]);
        }

        sigmas
    }

    /// The temporal median's smoothed per-channel sigma.
    ///
    /// A fixed `sigma_override` is returned for every channel. Until a trustworthy temporal
    /// reading lands, this falls back to [Self::current_sigmas_low_unboosted], since zero would
    /// under-filter.
    pub fn current_sigmas_temporal_only(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        if let Some(smoothed) = self.noise_estimator_temporal_only.current() {
            let mut sigmas = [0.0f32; 3];
            sigmas[..channels].copy_from_slice(&smoothed[..channels]);
            return sigmas;
        }

        self.current_sigmas_low_unboosted()
    }
}
