use std::sync::Arc;

use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use super::readings::{ReadingShape, SlotNoiseReading};
use crate::engine::EngineTiming;
use crate::engine::align::StorageAlign;
use crate::engine::timing::{DeviceTimed, DeviceTimer};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::params::{NlmParams, sigma_eff};
use crate::nlmeans::prefilter::PrefilterMode;
use crate::noise::{
    NoiseCurve,
    NoiseShape,
    QuarterClasses,
    RESULTS_PER_SLOT,
    ReadingSettings,
    TEMPORAL_QUARTER_SIZE,
    TemporalNoiseSample,
    build_spatial_offset_lut,
    sigma_from_abs_sum,
};

/// The geometry the front's noise state measures.
///
/// `None` unless the noise is estimated automatically.
pub(super) fn measured_noise_shape(
    params: &NlmParams,
    width: u32,
    height: u32,
    align: StorageAlign,
) -> Option<NoiseShape> {
    let auto_noise = params.hq.is_some_and(|hq| hq.sigma_override.is_none());
    if !auto_noise {
        return None;
    }

    let shape = NoiseShape {
        width,
        height,
        channels: params.channels.count(),
        stored_ch: params.channels.storage_count(),
        frames: params.total_frames(),
        // The temporal estimator needs a neighbour frame to take a difference against.
        temporal: params.temporal_radius >= 1,
        align,
    };
    Some(shape)
}

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

    /// Queues the noise statistics for the frame just written to `slot`.
    ///
    /// A stream's first frame has no predecessor, so its temporal record is zeroed instead.
    pub(super) fn queue_noise_stats(&self, slot: u32) -> Result<(), anyhow::Error> {
        let Some(noise) = self.noise.as_ref() else {
            return Ok(());
        };

        let total_frames = self.params.total_frames();
        let slot_prev = (slot + total_frames - 1) % total_frames;
        let previous = (self.ring_head != 0).then_some(slot_prev);

        noise.queue_stats(&self.input_buf, slot, previous)
    }

    /// Queues the noise statistics for a duplicated `slot`, whose temporal record is zeroed.
    pub(super) fn queue_duplicate_noise_stats(&self, slot: u32) -> Result<(), anyhow::Error> {
        let Some(noise) = self.noise.as_ref() else {
            return Ok(());
        };

        noise.queue_stats(&self.input_buf, slot, None)
    }

    /// Applies `change` to the reading settings when the noise is estimated automatically.
    fn change_reading_settings(&mut self, change: impl FnOnce(&mut ReadingSettings)) {
        let Some(noise) = self.noise.as_mut() else {
            return;
        };

        let mut settings = noise.settings();
        change(&mut settings);
        noise.set_settings(settings);
    }

    pub(crate) fn set_luma_noise_fields(&mut self, on: bool) {
        self.change_reading_settings(|settings| settings.luma_fields = on);
    }

    pub(crate) fn set_flat_texture_cut(&mut self, cut: Option<f32>) {
        self.change_reading_settings(|settings| settings.quarter.texture_cut = cut);
    }

    /// Sets the line ring's radius in pixels.
    ///
    /// Within the ring around strong lines, the luma flat map skips shadow soften and the flat texture
    /// cut. The radius rounds up to whole 8x8 quarters. `None` or `0` turns the ring off.
    pub(crate) fn set_line_ring(&mut self, radius_px: Option<u32>) {
        let active_radius = radius_px.filter(|&radius| radius > 0);
        let quarters = active_radius.map(|radius| radius.div_ceil(TEMPORAL_QUARTER_SIZE) as usize);
        self.change_reading_settings(|settings| settings.quarter.line_ring = quarters);
    }

    pub(crate) fn current_noise_curve(&self) -> Option<NoiseCurve> {
        self.host_noise.curve
    }

    pub(crate) fn current_quarter_classes(&self) -> Option<&QuarterClasses> {
        self.host_noise.classes.as_ref()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn flat_texture_cut(&self) -> Option<f32> {
        let settings = self.noise.as_ref().map(|noise| noise.settings());
        settings.and_then(|settings| settings.quarter.texture_cut)
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn line_ring(&self) -> Option<usize> {
        let settings = self.noise.as_ref().map(|noise| noise.settings());
        settings.and_then(|settings| settings.quarter.line_ring)
    }

    /// The handles one slot's noise reading reads, in [SlotNoiseReading::from_bytes] order.
    ///
    /// `None` when automatic noise estimation is off.
    pub(in crate::nlmeans) fn noise_reading_handles(&self, slot: u32) -> Option<Vec<Handle>> {
        let noise = self.noise.as_ref()?;
        let handles = noise.reading_handles(slot);

        Some(handles)
    }

    /// The shape one slot's noise reading is read with.
    ///
    /// `None` when automatic noise estimation is off.
    pub(in crate::nlmeans) fn reading_shape(&self) -> Option<ReadingShape> {
        let noise = self.noise.as_ref()?;
        let shape = noise.shape();
        let settings = noise.settings();

        let reading_shape = ReadingShape {
            width: shape.width,
            height: shape.height,
            channels: shape.channels,
            stored_ch: shape.stored_ch,
            with_curve: settings.luma_fields,
            settings: settings.quarter,
        };
        Some(reading_shape)
    }

    /// Starts reading back the noise that push `push_index` measured into `slot`.
    ///
    /// It runs right after the slot's noise kernels are queued, so the copy lands while later work
    /// runs.
    pub(super) fn start_noise_reading(&mut self, slot: u32, push_index: usize) {
        let Some(handles) = self.noise_reading_handles(slot) else {
            return;
        };

        let Some(shape) = self.reading_shape() else {
            return;
        };

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
        if self.frames_loaded != 0 || self.noise.is_none() {
            return Ok(());
        }

        let reading = self
            .slot_reading(slot, self.ring_head)
            .context("noise-estimate seed readback failed")?;

        // The first frame has no temporal record, so Immerkær alone seeds it.
        self.fold_noise_estimate(reading.immerkaer_sums, None, reading.immerkaer_low);

        Ok(())
    }

    /// Folds one slot's noise readings into the host estimate and refreshes the derived values.
    ///
    /// The median chain drives the strength. The low chain reads lower quartiles and drives the
    /// noise offsets, where reading too high scrubs fine texture.
    fn fold_noise_estimate(
        &mut self,
        immerkaer_sums: [f32; RESULTS_PER_SLOT],
        temporal: Option<TemporalNoiseSample>,
        immerkaer_low: [f32; 3],
    ) {
        let channels = self.params.channels.count() as usize;

        let mut immerkaer = [0.0f32; 3];
        for (c, sigma) in immerkaer.iter_mut().enumerate().take(channels) {
            *sigma = sigma_from_abs_sum(immerkaer_sums[c], self.width, self.height);
        }

        let sigma_scale = self.params.hq.map_or(1.0, |hq| hq.sigma_scale);
        let fold = self
            .host_noise
            .fold(immerkaer, temporal, immerkaer_low, channels, sigma_scale);

        let effective_sigma = sigma_eff(&fold.median[..channels], self.params.channels);
        self.h2_inv_norm = self.params.h2_inv_norm_with(Some(effective_sigma));
        self.input_noise_offset = self.params.noise_offset_with(Some(&fold.low[..channels]));
        self.noise_offset = match self.params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => self.input_noise_offset,
        };

        // Motion estimation treats channel 0 as luma.
        self.sigma_y = fold.median[0];
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
        self.host_noise.adopt(&lender.temporal);

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

    /// Points `spatial_offset_lut` at the table `noise_offset` and the smoothed rho imply.
    ///
    /// The table uploads only when it changed.
    pub(super) fn rebuild_spatial_offset_lut(&mut self) {
        let rho = self.host_noise.rho.unwrap_or(0.0);
        let offsets = build_spatial_offset_lut(self.params.search_radius, rho, self.noise_offset);
        let offset_bytes = f32::as_bytes(&offsets);

        self.spatial_offset_lut.update(&self.client, offset_bytes);
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn spatial_offset_lut_for_test(&self) -> (&MirroredBuffer, Vec<f32>) {
        let rho = self.host_noise.rho.unwrap_or(0.0);
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
        self.host_noise.sigmas(channels)
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
        self.host_noise.sigmas_low_unboosted(channels)
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
        self.host_noise.sigmas_temporal_only(channels)
    }
}
