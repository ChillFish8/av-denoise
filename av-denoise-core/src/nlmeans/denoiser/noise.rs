use anyhow::Context;
use cubecl::prelude::*;

use super::NlmDenoiser;
use crate::nlmeans::noise::{
    EMA_ALPHA,
    NoiseCtx,
    NoiseCurve,
    QuarterClasses,
    TEMPORAL_QUARTER_SIZE,
    TemporalNoiseReading,
    TemporalNoiseSample,
    TemporalStatsCtx,
    build_spatial_offset_lut,
    correlation_factor,
    noise_partials_slot_stride_bytes,
    partials_len,
    read_temporal_stats_slot,
    run_noise_estimate,
    run_temporal_noise_stats,
    sigma_block_p25_from_partials,
    sigma_from_abs_sum,
    temporal_noise_reading,
    zero_temporal_stats_slot,
};
use crate::nlmeans::params::sigma_eff;
use crate::nlmeans::prefilter::PrefilterMode;

impl<R: Runtime> NlmDenoiser<R> {
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

    #[cfg(test)]
    pub(crate) fn flat_texture_cut(&self) -> Option<f32> {
        self.quarter_settings.texture_cut
    }

    #[cfg(test)]
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

    /// Folds the first frame's noise estimate straight away so push-time work has a real sigma.
    ///
    /// Push-time stages such as the NLM pilot run before the first submit folds an estimate.
    /// One blocking read here keeps them off the construction-time fallback strength.
    ///
    /// It must run after this push queues its estimate and before `advance_ring` moves
    /// `frames_loaded`, which the first-frame check reads. The first submit folds the same frame
    /// again, which is harmless because it reproduces these values to within rounding.
    pub(super) fn seed_noise_estimate_if_first_frame(&mut self, slot: u32) -> Result<(), anyhow::Error> {
        if self.frames_loaded != 0 {
            return Ok(());
        }

        let Some(results_buf) = self.noise_results.as_ref() else {
            return Ok(());
        };

        let bytes = self
            .client
            .read_one(results_buf.clone())
            .context("noise-estimate seed readback failed")?;
        let data = f32::from_bytes(&bytes);

        // The first frame has no temporal record, so Immerkær alone seeds it.
        let immerkaer_low = self
            .read_noise_partials_low(slot)
            .context("noise-partials seed readback failed")?;
        self.fold_noise_estimate(data, slot as usize, None, immerkaer_low);

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
        data: &[f32],
        slot: usize,
        temporal: Option<TemporalNoiseSample>,
        immerkaer_low: [f32; 3],
    ) {
        let channels = self.params.channels.count() as usize;
        let base = slot * 4;

        let mut raw = [0.0f32; 3];
        for (c, sigma) in raw.iter_mut().enumerate().take(channels) {
            *sigma = sigma_from_abs_sum(data[base + c], self.width, self.height);
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

    /// Refreshes the derived filter values from the window centre's noise estimate.
    ///
    /// The estimate was queued when the slot was first written, so the blocking read lands on
    /// finished work.
    pub(super) fn update_noise_estimate(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        debug_assert!(
            center_t < self.params.total_frames(),
            "center_t must be a logical ring position"
        );

        let results_buf = self
            .noise_results
            .as_ref()
            .expect("noise_results allocated when auto noise is active")
            .clone();

        let bytes = self
            .client
            .read_one(results_buf)
            .map_err(|error| anyhow::anyhow!("noise-estimate results readback failed: {error}"))?;
        let data = f32::from_bytes(&bytes);

        let center_slot = self.phys_frame(center_t as i32) as usize;

        let reading = self.borrow_reading_ahead(center_t)?;
        let immerkaer_low = self.read_noise_partials_low(center_slot as u32)?;

        // The curve only changes alongside a trustworthy sample.
        if reading.sample.is_some() {
            self.noise_curve = reading.curve;
            self.quarter_classes = reading.classes;
        }

        self.fold_noise_estimate(data, center_slot, reading.sample, immerkaer_low);

        Ok(())
    }

    /// Reads one slot's noise partials back and reduces them to the low chain's estimate.
    ///
    /// The ring handle is sliced to the slot, so the transfer skips the rest of the ring.
    fn read_noise_partials_low(&self, slot: u32) -> Result<[f32; 3], anyhow::Error> {
        let partials_buf = self
            .noise_partials
            .as_ref()
            .expect("noise_partials allocated when auto noise is active");

        let slot_len_bytes = partials_len(self.width, self.height) as u64 * size_of::<f32>() as u64;
        let stride = noise_partials_slot_stride_bytes(self.width, self.height, self.align);
        let total_bytes = self.params.total_frames() as u64 * stride;
        let start = (slot as u64) * stride;
        let end_trim = total_bytes - start - slot_len_bytes;

        let sliced = partials_buf.clone().offset_start(start).offset_end(end_trim);
        let bytes = self
            .client
            .read_one(sliced)
            .map_err(|error| anyhow::anyhow!("noise partials readback failed: {error}"))?;
        let data = f32::from_bytes(&bytes);

        let sigmas =
            sigma_block_p25_from_partials(data, self.params.channels.count(), self.width, self.height);
        Ok(sigmas)
    }

    /// Reads a slot's temporal residual statistics back and combines them into one reading.
    ///
    /// Every part is `None` when the temporal estimator is inactive or too little of the frame
    /// held still. The curve also needs `luma_noise_fields` on.
    pub(in crate::nlmeans) fn read_temporal_noise(
        &self,
        slot: u32,
    ) -> Result<TemporalNoiseReading, anyhow::Error> {
        let Some(stats_buf) = self.temporal_stats_buf.as_ref() else {
            return Ok(TemporalNoiseReading {
                sample: None,
                curve: None,
                classes: None,
            });
        };

        let stored_ch = self.params.channels.storage_count();
        let channels = self.params.channels.count();
        let frame_count = self.params.total_frames();

        let records = read_temporal_stats_slot::<R>(
            &self.client,
            stats_buf,
            self.width,
            self.height,
            stored_ch,
            frame_count,
            slot,
            self.align,
        )?;

        let reading = temporal_noise_reading(
            &records,
            channels,
            stored_ch,
            self.width,
            self.height,
            self.luma_noise_fields,
            self.quarter_settings,
        );
        Ok(reading)
    }

    /// Rebuilds `spatial_offset_lut` from `noise_offset` and `rho_smoothed`.
    pub(super) fn rebuild_spatial_offset_lut(&mut self) {
        let offsets = build_spatial_offset_lut(
            self.params.search_radius,
            self.rho_smoothed.unwrap_or(0.0),
            self.noise_offset,
        );
        let offset_bytes = f32::as_bytes(&offsets);
        self.spatial_offset_lut = self.client.create_from_slice(offset_bytes);
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
