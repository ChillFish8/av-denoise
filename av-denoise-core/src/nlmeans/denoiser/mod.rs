mod machinery;
mod noise;
mod ring;
mod stages;

use cubecl::prelude::*;
use cubecl::server::Handle;

pub(crate) use self::machinery::RingView;
use super::align::StorageAlign;
use super::motion::{self, MotionCtx, MotionEstimation};
use super::noise::{
    NoiseCurve,
    NoiseEstimator,
    QuarterClasses,
    build_spatial_offset_lut,
    noise_partials_slot_stride_bytes,
    temporal_stats_buf_bytes,
};
use super::params::{NlmParams, SEPARABLE_THRESHOLD, validate_dimensions};
use super::prefilter::PrefilterMode;
use crate::engine::{DevicePlane, IngestTarget, SampleFormat, ingest};

/// A denoised frame still resident on the GPU.
///
/// `handle` points at one of the denoiser's two output slots and stays valid until that slot is
/// reused, so holding more than two at once lets a later submit overwrite an earlier frame.
pub struct GpuOutput {
    pub handle: Handle,
}

/// The stateful NLMeans denoiser that owns the GPU buffers.
///
/// Each push ingests one frame into a ring of frames, and each submit cleans the centre frame
/// using the neighbours around it.
pub struct NlmDenoiser<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) params: NlmParams,
    pub(super) width: u32,
    pub(super) height: u32,
    /// The byte alignment every per-slot buffer view starts on.
    pub(super) align: StorageAlign,

    /// Frames pushed including duplicates, which modulo the window size picks the next slot.
    pub(super) ring_head: usize,
    /// Frames loaded so far, capped at the window size.
    pub(super) frames_loaded: usize,
    /// Real pushes in the current stream, not counting the duplicates added at either end.
    pub(super) real_pushes: usize,

    /// The frame ring, one slot per frame in the window.
    pub(super) input_buf: Handle,
    /// The prefiltered ring the `_ref` distance kernels read, shaped like `input_buf`.
    pub(super) reference_buf: Option<Handle>,
    /// A 4-byte handle bound for planes a kernel never reads.
    pub(super) placeholder: Handle,
    /// The weighted pixel sum, one entry per stored channel per pixel.
    pub(super) accum: Handle,
    pub(super) weight_sum: Handle,
    pub(super) max_weight: Handle,
    /// Weight scratch for the path that compares a frame against itself.
    pub(super) weight_buf: Handle,
    /// The raw forward distance on the separable path.
    pub(super) raw_fwd: Handle,
    /// The raw backward distance on the separable path.
    pub(super) raw_bwd: Handle,
    /// The forward row sums on the separable path.
    pub(super) tmp_hsum: Handle,
    /// The backward row sums on the separable path.
    pub(super) tmp_hsum_bwd: Handle,
    /// Two output buffers used in turn, so one frame's kernels overlap the previous readback.
    pub(super) outputs: [Handle; 2],
    pub(super) next_output_slot: usize,

    /// Scales patch distances into weights, one over the squared strength times the patch area.
    pub(super) h2_inv_norm: f32,
    /// The distance floor the main pass subtracts before weighting.
    ///
    /// It is zero under `NlmSpatial`, because two pilot outputs carry no noise floor and
    /// subtracting one would overweight patches that do not really match.
    pub(super) noise_offset: f32,
    /// The distance floor for comparisons against noisy input pixels, which the pilot always uses.
    pub(super) input_noise_offset: f32,
    pub use_separable: bool,
    pub(super) use_reference: bool,

    /// The smoothed grain correlation between neighbouring pixels.
    ///
    /// The first temporal sample sets it directly so the opening frames are not under-corrected.
    /// `None` reads as white noise.
    pub(super) rho_smoothed: Option<f32>,
    /// One noise-floor offset per search candidate, row-major, for the self-comparison kernels.
    pub(super) spatial_offset_lut: Handle,

    /// Per-ring-slot scratch for the first stage of the noise estimate.
    ///
    /// Each slot keeps a frame's partials intact until that frame reaches the centre and is
    /// folded.
    pub(super) noise_partials: Option<Handle>,
    /// The per-channel Immerkær totals for each ring slot.
    pub(super) noise_results: Option<Handle>,
    /// Per-ring-slot temporal residual statistics, one record per spatial block.
    pub(super) temporal_stats_buf: Option<Handle>,
    /// Smooths the median chain's estimate, which feeds `h2_inv_norm` and `sigma_y`.
    pub(super) noise_estimator: NoiseEstimator,
    /// Smooths the low chain's estimate, which feeds the noise offsets.
    ///
    /// The low chain reads lower-quartile statistics, because reading the noise too high there
    /// destroys detail.
    pub(super) noise_estimator_low: NoiseEstimator,
    /// Smooths the low chain's estimate without the correlation boost.
    ///
    /// A consumer that squares sigma into a shrinkage threshold pays for an over-read twice, so
    /// it reads this instead.
    pub(super) noise_estimator_low_unboosted: NoiseEstimator,
    /// Smooths the temporal median alone, with no spatial maximum and no correlation boost.
    ///
    /// It only updates on a fold with a trustworthy temporal sample.
    pub(super) noise_estimator_temporal_only: NoiseEstimator,
    /// The latest centre frame's luma noise curve, if one could be built.
    pub(super) noise_curve: Option<NoiseCurve>,
    /// The latest centre frame's quarter classes, present exactly when `noise_curve` is.
    pub(super) quarter_classes: Option<QuarterClasses>,

    pub(super) mc_ctx: Option<MotionCtx>,
    /// The motion-shifted input ring the temporal kernels read neighbours from.
    pub(super) compensated_input_buf: Option<Handle>,
    /// The motion-shifted reference ring.
    pub(super) compensated_reference_buf: Option<Handle>,
    /// Motion vectors, one slice per neighbour with two `i32` components per block.
    ///
    /// Neighbours behind the centre fill the first half of the slices.
    pub(super) mv_field_buf: Option<Handle>,
    /// Adjacent-frame motion fields for chained estimation, by slot, then direction, then block.
    ///
    /// Direction 0 runs from the older frame to the newer one.
    pub(super) pair_ring_buf: Option<Handle>,
    /// The luma pyramid, indexed by level, then frame, then pixel.
    pub(super) pyramid_input: Option<Handle>,
    /// The luma pyramid built from the reference ring.
    pub(super) pyramid_reference: Option<Handle>,

    /// Block geometry for the confidence pass that runs without motion compensation.
    pub(super) confidence_ctx: Option<MotionCtx>,
    /// Per-block match confidence, laid out like `mv_field_buf` with one `f32` per block.
    pub(super) confidence_buf: Option<Handle>,
    /// The single-level luma pyramid ring for the confidence-only pass.
    pub(super) confidence_pyramid: Option<Handle>,
    /// Discarded motion vectors from the confidence-only pass.
    pub(super) confidence_mv_scratch: Option<Handle>,
    /// The fine block-match kernel's confidence argument when confidence weighting is off.
    ///
    /// The kernel drops the confidence write at compile time in that case, so this buffer is
    /// never indexed.
    pub(super) confidence_dummy: Handle,
    /// The smoothed luma sigma, which feeds the confidence noise floor.
    pub(super) sigma_y: f32,

    /// Whether the temporal stats kernel runs its four luma-only lanes.
    ///
    /// Leaving them off roughly halves the kernel's cost at 1080p.
    pub(super) luma_noise_fields: bool,

    /// The cut the luma flat map vetoes textured quarters at, or `None` for no veto.
    pub(super) flat_texture_cut: Option<f32>,

    /// Whether the stream's edges run off-centre passes instead of copied padding.
    ///
    /// With it on, a stream gets no leading copies and a centre with no temporal reading
    /// borrows the nearest one ahead.
    pub(super) shifted_edges: bool,
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Builds a new denoiser.
    ///
    /// # Panics
    ///
    /// Panics if the parameters or the frame dimensions are invalid. [Nlmeans](crate::Nlmeans)
    /// checks both and returns a `Result` instead.
    pub fn new(client: &ComputeClient<R>, params: NlmParams, width: u32, height: u32) -> Self {
        params
            .validate()
            .expect("invalid NlmParams, call params.validate() first to get this as a Result");
        validate_dimensions(width, height)
            .expect("unsupported frame dimensions, call validate_dimensions first to get this as a Result");

        let align = StorageAlign::from_client(client);
        let stored_ch = params.channels.storage_count();
        let total_frames = params.total_frames();
        let pixels = (width * height) as usize;
        let frame_bytes = pixels * stored_ch as usize * size_of::<f32>();
        let scalar_bytes = pixels * size_of::<f32>();

        let input_buf = client.empty(frame_bytes * total_frames as usize);
        let reference_buf = if params.prefilter.needs_reference_buf() {
            let reference = client.empty(frame_bytes * total_frames as usize);
            Some(reference)
        } else {
            None
        };

        let placeholder = client.empty(4);
        let accum = client.empty(frame_bytes);
        let weight_sum = client.empty(scalar_bytes);
        let max_weight = client.empty(scalar_bytes);
        let weight_buf = client.empty(scalar_bytes);
        let raw_fwd = client.empty(scalar_bytes);
        let raw_bwd = client.empty(scalar_bytes);
        let tmp_hsum = client.empty(scalar_bytes);
        let tmp_hsum_bwd = client.empty(scalar_bytes);
        let outputs = [client.empty(frame_bytes), client.empty(frame_bytes)];

        let h2_inv_norm = params.h2_inv_norm();
        let input_noise_offset = params.noise_offset();
        // Under `NlmSpatial` the main pass compares pilot outputs, which carry no noise floor.
        let noise_offset = match params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => input_noise_offset,
        };
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let use_reference = params.prefilter.needs_reference_buf();

        let rho_smoothed: Option<f32> = None;
        let initial_offsets = build_spatial_offset_lut(params.search_radius, 0.0, noise_offset);
        let initial_offset_bytes = f32::as_bytes(&initial_offsets);
        let spatial_offset_lut = client.create_from_slice(initial_offset_bytes);

        let auto_noise = params.hq.is_some_and(|hq| hq.sigma_override.is_none());
        let (noise_partials, noise_results) = if auto_noise {
            let partials_ring_bytes =
                noise_partials_slot_stride_bytes(width, height, align) * total_frames as u64;
            let n_results = (total_frames * 4) as usize;
            let partials = client.empty(partials_ring_bytes as usize);
            let results = client.empty(n_results * size_of::<f32>());
            (Some(partials), Some(results))
        } else {
            (None, None)
        };

        // The temporal estimator needs a neighbour frame to take a difference against.
        let temporal_stats_buf = if auto_noise && params.temporal_radius >= 1 {
            let stats_bytes = temporal_stats_buf_bytes(width, height, stored_ch, total_frames, align);
            let stats = client.empty(stats_bytes);
            Some(stats)
        } else {
            None
        };

        let mc_ctx = if params.motion_compensation.is_active() && params.temporal_radius > 0 {
            MotionCtx::new(params.motion_compensation, width, height, align)
        } else {
            None
        };

        let (
            compensated_input_buf,
            compensated_reference_buf,
            mv_field_buf,
            pyramid_input,
            pyramid_reference,
        ) = if let Some(motion_ctx) = mc_ctx.as_ref() {
            let compensated_input = client.empty(frame_bytes * total_frames as usize);
            let compensated_reference = if use_reference {
                let reference = client.empty(frame_bytes * total_frames as usize);
                Some(reference)
            } else {
                None
            };

            let neighbours = (2 * params.temporal_radius) as u64;
            let mv_field_bytes = neighbours * motion_ctx.mv_field_bytes_per_neighbour();
            let mv_field = client.empty(mv_field_bytes as usize);

            let pyramid_pixels =
                motion::pyramid_pixels_per_frame(width, height, motion_ctx.pyramid_levels, motion_ctx.align);
            let pyramid_bytes = pyramid_pixels * total_frames as usize * size_of::<f32>();
            let input_pyramid = client.empty(pyramid_bytes);
            let reference_pyramid = if use_reference {
                let pyramid = client.empty(pyramid_bytes);
                Some(pyramid)
            } else {
                None
            };

            (
                Some(compensated_input),
                compensated_reference,
                Some(mv_field),
                Some(input_pyramid),
                reference_pyramid,
            )
        } else {
            (None, None, None, None, None)
        };

        let estimation = params
            .motion_compensation
            .resolved_estimation(params.temporal_radius);
        let is_chained = matches!(estimation, Some(MotionEstimation::Chained { .. }));
        let pair_ring_buf = if is_chained {
            mc_ctx.as_ref().map(|motion_ctx| {
                let pair_ring_slots = motion::pair_ring_slot_count(params.temporal_radius) as u64;
                let pair_ring_bytes = pair_ring_slots * motion_ctx.pair_slot_bytes();
                client.empty(pair_ring_bytes as usize)
            })
        } else {
            None
        };

        // This also gates the motion-compensated path, so no submit pays for an unread
        // confidence write.
        let confidence_active =
            params.hq.is_some_and(|hq| hq.temporal_confidence) && params.temporal_radius > 0;

        let confidence_only_active = confidence_active && mc_ctx.is_none();
        let confidence_ctx = confidence_only_active.then(|| MotionCtx::confidence_only(width, height, align));

        let confidence_geometry = if confidence_active {
            mc_ctx.as_ref().or(confidence_ctx.as_ref())
        } else {
            None
        };
        let confidence_buf = confidence_geometry.map(|geometry| {
            let neighbours = (2 * params.temporal_radius) as u64;
            let confidence_bytes = neighbours * geometry.confidence_bytes_per_neighbour();
            client.empty(confidence_bytes as usize)
        });

        let confidence_dummy = client.empty(size_of::<f32>());

        let (confidence_pyramid, confidence_mv_scratch) = if let Some(geometry) = confidence_ctx.as_ref() {
            let pyramid_pixels =
                motion::pyramid_pixels_per_frame(width, height, geometry.pyramid_levels, geometry.align);
            let pyramid_bytes = pyramid_pixels * total_frames as usize * size_of::<f32>();
            let mv_scratch_len = geometry.mv_slots_per_neighbour() * 2 * size_of::<i32>();
            let pyramid = client.empty(pyramid_bytes);
            let mv_scratch = client.empty(mv_scratch_len);
            (Some(pyramid), Some(mv_scratch))
        } else {
            (None, None)
        };

        let sigma_y = params.hq.and_then(|hq| hq.sigma_override).unwrap_or(0.0);

        Self {
            client: client.clone(),
            params,
            width,
            height,
            align,
            ring_head: 0,
            frames_loaded: 0,
            real_pushes: 0,
            input_buf,
            reference_buf,
            placeholder,
            accum,
            weight_sum,
            max_weight,
            weight_buf,
            raw_fwd,
            raw_bwd,
            tmp_hsum,
            tmp_hsum_bwd,
            outputs,
            next_output_slot: 0,
            h2_inv_norm,
            noise_offset,
            input_noise_offset,
            use_separable,
            use_reference,
            rho_smoothed,
            spatial_offset_lut,
            noise_partials,
            noise_results,
            temporal_stats_buf,
            noise_estimator: NoiseEstimator::default(),
            noise_estimator_low: NoiseEstimator::default(),
            noise_estimator_low_unboosted: NoiseEstimator::default(),
            noise_estimator_temporal_only: NoiseEstimator::default(),
            noise_curve: None,
            quarter_classes: None,
            mc_ctx,
            compensated_input_buf,
            compensated_reference_buf,
            mv_field_buf,
            pair_ring_buf,
            pyramid_input,
            pyramid_reference,
            confidence_ctx,
            confidence_buf,
            confidence_pyramid,
            confidence_mv_scratch,
            confidence_dummy,
            sigma_y,
            luma_noise_fields: false,
            flat_texture_cut: None,
            shifted_edges: false,
        }
    }

    pub(crate) fn placeholder(&self) -> &Handle {
        &self.placeholder
    }

    /// Ingests `planes` into the next ring slot and runs every per-frame stage on it.
    pub(crate) fn push_planes(
        &mut self,
        planes: &[DevicePlane<'_>],
        format: SampleFormat,
    ) -> Result<(), anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        let slot = self.ring_head % total_frames;
        let pixels = self.width * self.height;
        let stored_ch = self.params.channels.storage_count();
        let frame_len = pixels * stored_ch;
        let target = IngestTarget {
            ring: &self.input_buf,
            ring_len: total_frames * frame_len as usize,
            offset: slot as u32 * frame_len,
            pixels,
            channels: self.params.channels.count(),
            stored_ch,
        };

        ingest(&self.client, planes, format, &self.placeholder, target);
        self.run_post_upload_stages(slot)
    }

    /// Queues the denoise kernels for the current window without reading the result back.
    ///
    /// Returns `Ok(None)` while the window is still filling.
    pub fn denoise_submit_gpu(&mut self) -> Result<Option<GpuOutput>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        if self.noise_results.is_some() {
            self.update_noise_estimate(self.params.temporal_radius)?;
        }

        self.rebuild_spatial_offset_lut();

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        self.run_denoise_kernels(slot)?;

        let output = GpuOutput {
            handle: self.outputs[slot].clone(),
        };
        Ok(Some(output))
    }

    /// How many tail frames the end-of-stream drain must emit.
    ///
    /// It is zero in spatial mode and for a stream with no pushes.
    pub(crate) fn flush_target(&self) -> usize {
        let temporal_radius = self.params.temporal_radius as usize;
        if temporal_radius == 0 || self.real_pushes == 0 {
            0
        } else {
            self.real_pushes.min(temporal_radius)
        }
    }

    pub(crate) fn real_pushes(&self) -> usize {
        self.real_pushes
    }

    /// Runs one end-of-stream drain step, duplicating the last frame forward and submitting.
    ///
    /// Returns `Ok(None)` while the duplicates still fill a window that never filled during
    /// pushing. Later steps always return `Some`, so callers stop after [Self::flush_target]
    /// outputs. It needs at least one pushed frame and a temporal radius above 0.
    pub(crate) fn flush_step_gpu(&mut self) -> Result<Option<GpuOutput>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;

        self.duplicate_last_frame()?;
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }

        self.denoise_submit_gpu()
    }

    /// Resets the stream state so the next push starts a fresh temporal stream.
    ///
    /// The GPU buffers are left alone, because a fresh stream writes every slot before reading it.
    pub fn reset_stream_state(&mut self) {
        self.ring_head = 0;
        self.frames_loaded = 0;
        self.next_output_slot = 0;
        self.real_pushes = 0;
        self.noise_estimator.reset();
        self.noise_estimator_low.reset();
        self.noise_estimator_low_unboosted.reset();
        self.noise_estimator_temporal_only.reset();
        self.noise_curve = None;
        self.quarter_classes = None;
        self.rho_smoothed = None;
    }
}
