use cubecl::server::Handle;

use super::NlmDenoiser;
use crate::collab::kernels::aggregate::weight_scale_with_g_max;
use crate::engine::EngineTiming;
use crate::engine::align::StorageAlign;
use crate::engine::timing::{DeviceTimed, DeviceTimer};
use crate::motion::sad_noise_floor;
use crate::nlmeans::dispatch::{mc_sad_noise_floor_sigma, pilot_h2_inv_norm};
use crate::nlmeans::params::NlmParams;
use crate::nlmeans::prefilter::PrefilterMode;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::noise::NoiseCurve;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::noise::NoiseSnapshot;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::noise::NoiseState;
use crate::noise::{
    DerivedCoefficients,
    DerivedValues,
    FoldSettings,
    NoiseConsumer,
    NoiseSettings,
    NoiseShape,
    ReadingSettings,
    STRENGTH_MAP_OFF,
    StatsMotion,
    StrengthMapKind,
    TEMPORAL_QUARTER_SIZE,
    WindowBounds,
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
    if !measures_noise(params) {
        return None;
    }

    let shape = noise_shape(params, width, height, align);
    Some(shape)
}

/// Whether the front measures the noise, which it does unless HQ is off or the sigma is pinned.
pub(super) fn measures_noise(params: &NlmParams) -> bool {
    params.hq.is_some_and(|hq| hq.sigma_override.is_none())
}

/// The geometry of the front's noise state, whether or not it measures the noise.
pub(super) fn noise_shape(params: &NlmParams, width: u32, height: u32, align: StorageAlign) -> NoiseShape {
    NoiseShape {
        width,
        height,
        channels: params.channels.count(),
        stored_ch: params.channels.storage_count(),
        frames: params.total_frames(),
        // The temporal estimator needs a neighbour frame to take a difference against.
        temporal: params.temporal_radius >= 1,
        align,
    }
}

impl DeviceTimed for NlmDenoiser {
    fn device_timer(&mut self) -> &mut DeviceTimer {
        &mut self.device_timer
    }
}

impl NlmDenoiser {
    /// Device time this denoiser has spent.
    pub(crate) fn timing(&mut self) -> EngineTiming {
        let (device_time, device_samples) = self.device_timer.settle();

        EngineTiming {
            device_time,
            device_samples,
        }
    }

    pub(crate) fn sample_device_time(&mut self, every: u32) {
        let timing_method = self.client.properties().timing_method;
        self.device_timer.set_timing_method(timing_method);
        self.device_timer.set_every(every);
    }

    /// Queues the noise statistics and the slot reading for stream frame `frame`, just written to
    /// `slot`, and records the reading.
    ///
    /// A stream's first frame has no predecessor, so its temporal record is zeroed instead. With
    /// motion compensation the record keeps only the Immerkær values, since only a pass's
    /// compensated re-read may give the frame a temporal sample.
    pub(super) fn queue_noise_stats(&self, slot: u32, frame: u32) -> Result<(), anyhow::Error> {
        let total_frames = self.params.total_frames();
        let slot_prev = (slot + total_frames - 1) % total_frames;
        let previous = (self.ring_head != 0).then_some(slot_prev);

        self.noise.queue_stats(&self.input_buf, slot, previous)?;
        self.noise.queue_reading(slot)?;

        if self.mc_ctx.is_some() {
            self.noise.record_spatial_reading(slot, frame);
        } else {
            self.noise.record_reading(slot, frame);
        }

        Ok(())
    }

    /// Queues the noise statistics and the slot reading for a duplicated `slot`, whose temporal
    /// record is zeroed.
    pub(super) fn queue_duplicate_noise_stats(&self, slot: u32) -> Result<(), anyhow::Error> {
        self.noise.queue_stats(&self.input_buf, slot, None)?;
        self.noise.queue_reading(slot)?;

        Ok(())
    }

    /// Re-measures the centre of a pass at `center_t` against an adjacent frame along the pass's
    /// motion, then re-reads and re-records its slot.
    ///
    /// It prefers the previous frame and falls back to the next one, which a scene's first frame
    /// needs. It needs a measured noise state, motion compensation and an adjacent frame in the
    /// window. A centre pushed as a duplicate keeps its zeroed record, and so does a centre whose
    /// next frame is a copy of itself.
    pub(in crate::nlmeans) fn queue_compensated_noise_reading(
        &self,
        center_t: u32,
    ) -> Result<(), anyhow::Error> {
        if !self.noise.is_measured() {
            return Ok(());
        }

        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        let Some((offset, neighbour_idx)) = self.adjacent_neighbour(center_t) else {
            return Ok(());
        };

        let centre_slot = self.phys_frame(center_t as i32);
        if self.duplicate_slots[centre_slot as usize] {
            return Ok(());
        }

        // A duplicated next frame is a copy of the centre, so it holds no residual to measure.
        let neighbour_slot = self.phys_frame(center_t as i32 + offset);
        let copies_centre = offset > 0 && self.duplicate_slots[neighbour_slot as usize];
        if copies_centre {
            return Ok(());
        }

        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");
        let neighbours = self.params.total_frames() - 1;
        let mv_stride = motion_ctx.mv_stride();
        let motion = StatsMotion {
            mv_field,
            mv_len: (neighbours * mv_stride) as usize,
            mv_offset: neighbour_idx * mv_stride,
            step: motion_ctx.step,
            blocks_x: motion_ctx.blocks_x,
            blocks_y: motion_ctx.blocks_y,
        };

        self.noise
            .queue_compensated_stats(&self.input_buf, centre_slot, neighbour_slot, motion)?;
        self.noise.queue_reading(centre_slot)?;

        let frame = self.slot_frames[centre_slot as usize];
        self.noise.record_reading(centre_slot, frame);

        Ok(())
    }

    /// Applies `change` to the noise settings and refreshes the fixed derived values to match.
    fn change_noise_settings(&mut self, change: impl FnOnce(&mut NoiseSettings)) {
        let mut settings = self.noise.settings();
        change(&mut settings);
        self.noise.set_settings(settings);

        self.refresh_fixed_derived();
    }

    fn change_reading_settings(&mut self, change: impl FnOnce(&mut ReadingSettings)) {
        self.change_noise_settings(|settings| change(&mut settings.reading));
    }

    /// Sets the strength map each slot reading builds, before the first push.
    pub(crate) fn set_strength_map(&mut self, kind: StrengthMapKind) {
        self.change_reading_settings(|settings| settings.map = kind);
    }

    /// Sets what the consuming denoiser makes of the noise, before the first push.
    pub(crate) fn set_noise_consumer(&mut self, consumer: NoiseConsumer) {
        self.change_noise_settings(|settings| settings.fold.consumer = consumer);
    }

    /// Sets whether the noise state keeps the NLM offset tables of the search window, before the
    /// first push.
    pub(crate) fn set_offset_tables(&mut self, on: bool) {
        let radius = on.then_some(self.params.search_radius);
        self.change_noise_settings(|settings| settings.fold.offset_tables = radius);
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

    /// The current noise curve, read back from the GPU state.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn current_noise_curve(&self) -> Option<NoiseCurve> {
        let snapshot = self.gpu_snapshot_for_test();
        snapshot.curve.map(|(ratios, sigma_quarter_median)| NoiseCurve {
            ratios,
            sigma_quarter_median,
        })
    }

    /// Whether the current estimate holds quarter classes.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn current_classes_present(&self) -> bool {
        let snapshot = self.gpu_snapshot_for_test();
        snapshot.classes_present
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    fn gpu_snapshot_for_test(&self) -> NoiseSnapshot {
        self.noise.snapshot().expect("the GPU noise state reads back")
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn noise_settings_for_test(&self) -> NoiseSettings {
        self.noise.settings()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn flat_texture_cut(&self) -> Option<f32> {
        let settings = self.measured_noise().map(|noise| noise.settings().reading);
        settings.and_then(|settings| settings.quarter.texture_cut)
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn line_ring(&self) -> Option<usize> {
        let settings = self.measured_noise().map(|noise| noise.settings().reading);
        settings.and_then(|settings| settings.quarter.line_ring)
    }

    /// The noise state, `None` unless it measures the noise.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn measured_noise(&self) -> Option<&NoiseState> {
        let measured = self.noise.is_measured();
        measured.then_some(&self.noise)
    }

    /// Folds the stream's first frame alone straight away so push-time work has a real sigma.
    ///
    /// Push-time stages such as the NLM pilot run before the first submit folds an estimate.
    ///
    /// It must run after this push records its reading and before `advance_ring` moves
    /// `frames_loaded`, which the first-frame check reads.
    pub(super) fn seed_noise_estimate_if_first_frame(&mut self) -> Result<(), anyhow::Error> {
        if self.frames_loaded != 0 || !self.noise.is_measured() {
            return Ok(());
        }

        let first_frame = WindowBounds::around(0, 0);
        self.noise.queue_window_fold(first_frame)
    }

    /// Folds the noise window around the stream frame at logical ring position `center_t`.
    ///
    /// With motion compensation, frames ahead of the centre hold readings their passes have not
    /// re-measured yet, so the window stops at the centre. Otherwise it reaches every recorded
    /// frame ahead, up to the window radius.
    pub(in crate::nlmeans) fn update_noise_estimate(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        debug_assert!(
            center_t < self.params.total_frames(),
            "center_t must be a logical ring position"
        );

        let centre_slot = self.phys_frame(center_t as i32);
        let centre = self.slot_frames[centre_slot as usize];
        let recorded = self.noise.stream_frame_count();
        let newest_recorded = recorded.saturating_sub(1);
        let newest = if self.mc_ctx.is_some() {
            centre
        } else {
            newest_recorded
        };
        let bounds = WindowBounds::around(centre, newest);

        self.noise.queue_window_fold(bounds)
    }

    /// The noise settings a new noise state starts from, with the default reading and consumer and
    /// the offset tables of the search window.
    pub(super) fn initial_noise_settings(&self) -> NoiseSettings {
        let sigma_scale = self.params.hq.map_or(1.0, |hq| hq.sigma_scale);
        let coefficients = self.derived_coefficients();
        let fold = FoldSettings {
            sigma_scale,
            consumer: NoiseConsumer::default(),
            coefficients,
            offset_tables: Some(self.params.search_radius),
        };

        NoiseSettings {
            reading: ReadingSettings::default(),
            fold,
        }
    }

    /// The factors the GPU fold turns folded sigmas into derived values with.
    ///
    /// Each factor is a host formula evaluated at a sigma of 1, which the fold then scales by the
    /// folded sigmas. A factor whose pass does not run is 0.
    fn derived_coefficients(&self) -> DerivedCoefficients {
        let params = &self.params;
        let auto_strength = params.hq.is_some_and(|hq| hq.auto_strength);
        let h2_per_inverse_variance = auto_strength.then(|| params.h2_inv_norm_with(Some(1.0)));
        let h2_fixed = params.h2_inv_norm_with(None);

        // One channel at a sigma of 1, which the fold scales by the sum of squared sigmas.
        let unit_sigma = [1.0f32];
        let offset_per_variance = params.noise_offset_with(Some(&unit_sigma));

        let (pilot_h2_factor, main_offset) = match params.prefilter {
            PrefilterMode::NlmSpatial { strength_scale } => {
                let factor = pilot_h2_inv_norm(1.0, strength_scale);
                (factor, false)
            },
            _ => (1.0, true),
        };

        let sad_per_sigma = self.mc_ctx.as_ref().map_or(0.0, |motion_ctx| {
            let residual_fraction = mc_sad_noise_floor_sigma(params.prefilter, 1.0);
            sad_noise_floor(motion_ctx.blksize, residual_fraction)
        });
        let confidence_sad_per_sigma = self
            .confidence_ctx
            .as_ref()
            .map_or(0.0, |geometry| sad_noise_floor(geometry.blksize, 1.0));

        DerivedCoefficients {
            h2_per_inverse_variance,
            h2_fixed,
            pilot_h2_factor,
            offset_per_variance,
            main_offset,
            sad_per_sigma,
            confidence_sad_per_sigma,
        }
    }

    /// Fills the noise state's settings and derived values from the front's parameters.
    ///
    /// With `measured` set the state starts measuring the noise, and its folds write the derived
    /// values from then on.
    pub(super) fn init_noise(&mut self, measured: bool) {
        let settings = self.initial_noise_settings();
        self.noise.set_settings(settings);

        let initial = self.fixed_derived_values();
        self.noise.set_fixed(initial);

        if measured {
            self.noise.start_measuring();
        }
    }

    /// Writes the fixed derived values into a fixed noise state.
    ///
    /// A measured state's folds write its own derived values.
    fn refresh_fixed_derived(&mut self) {
        if self.noise.is_measured() {
            return;
        }

        let values = self.fixed_derived_values();
        self.noise.set_fixed(values);
    }

    /// The derived values buffer the kernels bind, the noise state's own.
    pub(crate) fn derived_handle(&self) -> Handle {
        self.noise.derived().clone()
    }

    /// The main pass's noise-floor table, one entry per search candidate.
    ///
    /// # Panics
    ///
    /// Panics on a front whose noise state keeps no offset tables.
    pub(in crate::nlmeans) fn spatial_offsets_handle(&self) -> Handle {
        let offsets = self
            .noise
            .spatial_offsets()
            .expect("an NLM front's noise state keeps the offset tables");

        offsets.clone()
    }

    /// The NLM pilot's noise-floor table, one entry per search candidate.
    ///
    /// # Panics
    ///
    /// Panics on a front whose noise state keeps no offset tables.
    pub(in crate::nlmeans) fn pilot_offsets_handle(&self) -> Handle {
        let offsets = self
            .noise
            .pilot_offsets()
            .expect("a front with the NLM pilot keeps the offset tables");

        offsets.clone()
    }

    /// The noise curve buffer the kernels bind, zeros while the consumer applies no curve.
    pub(crate) fn noise_curve_handle(&self) -> Handle {
        self.noise.curve().clone()
    }

    /// The strength map buffer the kernels bind, one multiplier per map quarter.
    pub(crate) fn strength_map_handle(&self) -> Handle {
        self.noise.strength_map().clone()
    }

    /// The derived values of a front with no estimate, at the pinned sigma or at zero.
    ///
    /// A pinned or fast front keeps these values. There is no curve and no map, and the grain reads
    /// as white noise.
    pub(super) fn fixed_derived_values(&self) -> DerivedValues {
        let params = &self.params;
        let consumer = self.noise.settings().fold.consumer;
        let pinned = params.hq.and_then(|hq| hq.sigma_override);
        let sigma_y = pinned.unwrap_or(0.0);
        let weight = weight_scale_with_g_max(sigma_y, consumer.weight_g_max);

        let h2_inv_norm = params.h2_inv_norm();
        let input_noise_offset = params.noise_offset();
        // Under `NlmSpatial` the main pass compares pilot outputs, which carry no noise floor.
        let (pilot_h2, noise_offset) = match params.prefilter {
            PrefilterMode::NlmSpatial { strength_scale } => {
                let pilot = pilot_h2_inv_norm(h2_inv_norm, strength_scale);
                (pilot, 0.0)
            },
            _ => (h2_inv_norm, input_noise_offset),
        };

        let sad_floor = self.mc_ctx.as_ref().map_or(0.0, |motion_ctx| {
            let sigma = mc_sad_noise_floor_sigma(params.prefilter, sigma_y);
            sad_noise_floor(motion_ctx.blksize, sigma)
        });
        let confidence_sad_floor = self
            .confidence_ctx
            .as_ref()
            .map_or(0.0, |geometry| sad_noise_floor(geometry.blksize, sigma_y));

        DerivedValues {
            sigma: [sigma_y; 3],
            weight_scale: weight,
            curve_valid: false,
            map_mode: STRENGTH_MAP_OFF,
            sad_noise_floor: sad_floor,
            confidence_sad_noise_floor: confidence_sad_floor,
            h2_inv_norm,
            pilot_h2_inv_norm: pilot_h2,
            noise_offset,
            input_noise_offset,
            rho: 0.0,
        }
    }

    /// The GPU noise state and derived values.
    ///
    /// A front with fixed values has no chains. It waits for every queued fold.
    ///
    /// # Panics
    ///
    /// Panics if the readback fails.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn noise_snapshot(&self) -> Option<NoiseSnapshot> {
        let snapshot = self.noise.snapshot().expect("the GPU noise state reads back");

        Some(snapshot)
    }

    /// The median chain's folded per-channel sigma.
    ///
    /// A fixed `sigma_override` is returned for every channel. Before the first estimate, and
    /// on the fast path, it is zero.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn current_sigmas(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        let snapshot = self.gpu_snapshot_for_test();
        let median = snapshot.median.unwrap_or_default();
        leading_channels(median, channels)
    }

    /// The low chain's folded per-channel sigma without the correlation boost.
    ///
    /// A fixed `sigma_override` is returned for every channel. Before the first estimate, and
    /// on the fast path, it is zero.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn current_sigmas_low_unboosted(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        let snapshot = self.gpu_snapshot_for_test();
        let low_unboosted = snapshot.low_unboosted.unwrap_or_default();
        leading_channels(low_unboosted, channels)
    }

    /// The temporal median's folded per-channel sigma.
    ///
    /// A fixed `sigma_override` is returned for every channel. Until a trustworthy temporal
    /// reading lands, this falls back to the unboosted low chain, since zero would under-filter.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn current_sigmas_temporal_only(&self) -> [f32; 3] {
        if let Some(sigma) = self.params.hq.and_then(|hq| hq.sigma_override) {
            return [sigma; 3];
        }

        let channels = self.params.channels.count() as usize;
        let snapshot = self.gpu_snapshot_for_test();
        let low_unboosted = snapshot.low_unboosted.unwrap_or_default();
        let temporal_only = snapshot.temporal_only.unwrap_or(low_unboosted);
        leading_channels(temporal_only, channels)
    }
}

/// The first `channels` values of `sigmas`, with the rest at zero.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
fn leading_channels(sigmas: [f32; 3], channels: usize) -> [f32; 3] {
    let mut leading = [0.0f32; 3];
    leading[..channels].copy_from_slice(&sigmas[..channels]);

    leading
}
