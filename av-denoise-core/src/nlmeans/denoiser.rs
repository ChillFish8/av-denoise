use cubecl::prelude::*;
use cubecl::server::Handle;

use super::align::StorageAlign;
use super::kernels::{gpu_copy, gpu_unpack_wire};
use super::motion::{self, MotionCtx, MotionEstimation, build_pyramid_for_slot, run_pyramid_build};
use super::noise::{
    EMA_ALPHA,
    NoiseCtx,
    NoiseEstimator,
    TemporalNoiseSample,
    TemporalStatsCtx,
    aggregate_temporal_noise_stats,
    build_spatial_offset_lut,
    correlation_factor,
    noise_partials_slot_stride_bytes,
    partials_len,
    read_temporal_stats_slot,
    run_noise_estimate,
    run_temporal_noise_stats,
    sigma_block_p25_from_partials,
    sigma_from_abs_sum,
    temporal_stats_buf_bytes,
    zero_temporal_stats_slot,
};
use super::params::{NlmParams, SEPARABLE_THRESHOLD, sigma_eff, validate_dimensions};
use super::pending::{Pending, empty_output, start_readback};
use super::prefilter::{PrefilterCtx, PrefilterMode, run_prefilter};
use super::{BLOCK_1D, Depth, MAX_GRID_1D};
use crate::denoiser::{DenoiserError, FrameOutput, OutputFormat};

/// A denoised frame that has finished its kernels but is still resident
/// on the GPU.
///
/// [`NlmDenoiser::denoise_submit_gpu`] returns this instead of starting a
/// readback, for a caller that queues more GPU work against the frame
/// rather than pulling it back to the host straight away.
///
/// `handle` points at one of the denoiser's two output slots, and it
/// stays valid only until that slot is reused. A denoiser only has two
/// output slots, so at most two outstanding `GpuOutput`s (or
/// [`Pending`]s, which are built from the same slots) may exist for one
/// denoiser at a time. Submitting a third before an earlier one is
/// consumed reuses its slot and silently corrupts it.
pub struct GpuOutput {
    /// The GPU buffer holding the denoised frame.
    pub handle: Handle,
    /// Which of the denoiser's two output slots `handle` came from.
    pub slot: usize,
}

/// Handles and geometry a collaborative stage needs to read the ring.
///
/// [`NlmDenoiser::submit_machinery`] and
/// [`NlmDenoiser::flush_step_machinery`] build this instead of running
/// any NLM denoising kernel, so a caller that wants the frame ring, the
/// motion fields, and the confidence scores without the NLM weighting
/// itself can read them straight from here.
///
/// The handles are views into the denoiser's own buffers, valid until
/// the next push or the next machinery step reuses the slots they point
/// into.
pub(crate) struct RingView {
    /// The whole input ring, indexable by physical frame slot.
    pub input: Handle,
    /// Chained motion fields, one per neighbour index.
    pub mv_field: Handle,
    /// Per-block confidence, one plane per neighbour index.
    pub confidence: Handle,
    /// Physical ring slot of the centre frame.
    pub centre_slot: u32,
    /// Physical ring slot per logical offset k, indexed by
    /// `neighbour_idx_for_k(radius, k)`.
    ///
    /// This stays a host `Vec` rather than a GPU buffer, because the
    /// grouping kernel that consumes it indexes it per candidate on the
    /// device, and it is the caller's job to upload it, not this one's.
    pub neighbour_slots: Vec<u32>,
    /// `i32` element stride between neighbours in `mv_field`.
    pub mv_stride: u32,
    /// `f32` element stride between neighbours in `confidence`.
    pub conf_stride: u32,
}

/// The stateful NLMeans denoiser that owns the GPU buffers.
///
/// It keeps a ring of frames in `input_buf`. Each `push_frame` uploads
/// one frame into the ring, and each `denoise` cleans the current centre
/// frame using the neighbours around it.
pub struct NlmDenoiser<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) params: NlmParams,
    pub(super) width: u32,
    pub(super) height: u32,
    /// The byte alignment every per-slot buffer view has to start on,
    /// read from `client`'s runtime at construction. See
    /// [`StorageAlign`].
    pub(super) align: StorageAlign,

    /// A running count of frames pushed. Taken modulo the window size it
    /// gives the next physical slot in `input_buf` to overwrite.
    pub(super) ring_head: usize,
    /// How many frames are loaded so far, capped at the window size.
    pub(super) frames_loaded: usize,
    /// How many real pushes the current stream has seen, not counting
    /// the duplicates the denoiser adds at either end.
    ///
    /// [`Self::reset_stream_state`] clears this.
    pub(super) real_pushes: usize,

    /// The frame ring itself, one slot per frame in the window.
    pub(super) input_buf: Handle,
    /// The reference ring, shaped exactly like `input_buf`.
    ///
    /// It only exists when a prefilter is set, and it supplies the
    /// distances the `_ref` kernels read.
    pub(super) reference_buf: Option<Handle>,
    /// CPU scratch for repacking 3-channel YUV into 4 lanes. Empty when
    /// no padding is needed.
    pub(super) padding_scratch: Vec<f32>,
    /// CPU scratch the wire push concatenates its planes into, so one
    /// push costs one transfer whatever the channel mode. Reused, so it
    /// allocates nothing after the first frame.
    pub(super) upload_scratch: Vec<u8>,
    /// The weighted-pixel accumulator, one entry per stored channel per
    /// pixel.
    pub(super) accum: Handle,
    /// The total weight at each pixel.
    pub(super) weight_sum: Handle,
    /// The largest neighbour weight at each pixel.
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
    /// Two denoised output buffers, used in turn.
    ///
    /// A new submit writes into the next slot while the previous one may
    /// still be reading back, which lets one frame's kernels overlap
    /// with the frame before it.
    pub(super) outputs: [Handle; 2],
    /// Which output slot the next submit writes into.
    pub(super) next_output_slot: usize,
    /// The format every readback this denoiser starts comes back in.
    pub(super) output_format: OutputFormat,
    /// Packed-word destinations, one per entry of `outputs`, allocated
    /// only in wire mode. They rotate on the same slot counter, so each
    /// is free again exactly when the `f32` slot it is packed from is.
    pub(super) wire_outputs: Option<[Handle; 2]>,
    /// CPU scratch the blocking `denoise()` and `flush()` paths reuse
    /// through `Pending::wait_into`, so they do not allocate per frame.
    ///
    /// It holds `output_format`'s variant, so `wait_into` keeps its
    /// allocation rather than replacing it.
    pub(super) output_scratch: FrameOutput,

    pub(super) h2_inv_norm: f32,
    /// The distance floor the main pass subtracts before weighting.
    ///
    /// It is zero for `NlmSpatial`, because comparing one pilot output
    /// against another no longer carries a noise floor, so subtracting
    /// one would overweight patches that do not really match.
    ///
    /// Every other prefilter mode leaves it equal to
    /// `input_noise_offset`.
    pub(super) noise_offset: f32,
    /// The distance floor for comparisons against noisy input pixels.
    ///
    /// The pilot pass always uses this value, because its own inputs
    /// still carry the full noise floor even when `noise_offset` has
    /// been zeroed for the main pass.
    pub(super) input_noise_offset: f32,
    pub use_separable: bool,
    pub(super) use_reference: bool,

    /// How correlated the grain is between neighbouring pixels,
    /// smoothed over time on the same schedule as the sigma estimate.
    ///
    /// It is `None` until the stream's first temporal sample, following
    /// the same seeding rule as `NoiseEstimator`. That first sample sets
    /// the value directly rather than blending up from an assumed zero,
    /// so a stream does not spend its opening frames under-corrected.
    ///
    /// It only updates when a temporal sample exists, so the fast path
    /// and a fixed `sigma_override` leave it `None` for the whole
    /// stream. That reads as zero, meaning white noise and no
    /// correction.
    ///
    /// With `hq.windowed_noise_estimation` set, a fold with no temporal
    /// sample clears this to `None` instead of leaving it where it was,
    /// the same reasoning `noise_estimator_temporal_only` documents:
    /// coasting on an older window's reading is itself a form of
    /// cross-window history.
    pub(super) rho_smoothed: Option<f32>,
    /// One noise-floor offset per search candidate, for the weighting
    /// kernels that compare a frame against itself.
    ///
    /// The table covers the whole search window, laid out row-major.
    ///
    /// It is rebuilt from `noise_offset` and `rho_smoothed` on every
    /// submit. See `Self::rebuild_spatial_offset_lut`.
    ///
    /// While `rho_smoothed` is unset every entry equals the flat
    /// `noise_offset` scalar this table replaced.
    pub(super) spatial_offset_lut: Handle,

    /// Scratch for the first stage of the noise estimate, one slot per
    /// ring position.
    ///
    /// It only exists when the noise level is measured automatically,
    /// meaning HQ is on and no fixed sigma was given.
    ///
    /// Giving each ring position its own slot keeps a frame's partials
    /// intact between the push that queues them and the later fold that
    /// reads them back once that frame reaches the centre. One shared
    /// region would be overwritten long before then.
    pub(super) noise_partials: Option<Handle>,
    /// The per-channel Immerkær totals for each ring slot, gated the
    /// same way as `noise_partials`.
    pub(super) noise_results: Option<Handle>,
    /// The temporal residual statistics for each ring slot, one record
    /// per spatial block.
    ///
    /// Each record holds the sum and sum of squares per channel, plus
    /// the lag-1 total that reveals correlated grain.
    ///
    /// It only exists when the noise level is measured automatically and
    /// the temporal radius is at least 1, because with no neighbour
    /// there is nothing to take a difference against.
    pub(super) temporal_stats_buf: Option<Handle>,
    /// Smooths the median chain's raw per-frame estimate into a steady
    /// per-channel sigma, which feeds `h2_inv_norm` and `sigma_y`.
    ///
    /// It never updates when the noise level is not measured
    /// automatically.
    pub(super) noise_estimator: NoiseEstimator,
    /// Smooths the low chain's raw per-frame estimate into a steady
    /// per-channel sigma, which feeds only `input_noise_offset` and
    /// `noise_offset`.
    ///
    /// The low chain reads more cautiously than the median chain,
    /// combining a lower-quartile spatial statistic with a
    /// lower-quartile temporal one. Its consumers are the ones where
    /// reading the noise too high destroys detail.
    ///
    /// It is inert under the same condition as `noise_estimator`.
    pub(super) noise_estimator_low: NoiseEstimator,
    /// Smooths the low chain's raw per-frame estimate a second time,
    /// built the same way `noise_estimator_low` is except the temporal
    /// reading never passes through `correlation_factor`.
    ///
    /// A consumer that squares this sigma into a shrinkage threshold
    /// pays for any over-read twice over, so it can read this estimator
    /// instead of `noise_estimator_low` to get the temporal reading on
    /// its own, without a second correction stacked on top of it.
    ///
    /// It is inert under the same condition as `noise_estimator`.
    pub(super) noise_estimator_low_unboosted: NoiseEstimator,
    /// Smooths the temporal reading on its own, with no maximum taken
    /// against an Immerkær spatial reading and no correlation boost.
    ///
    /// It only updates on a fold that has a temporal sample trustworthy
    /// enough for `aggregate_temporal_noise_stats` to produce one. A fold
    /// with no such sample, whether because the temporal radius is zero
    /// or because too little of the frame held still, leaves this
    /// estimator exactly where it was. `current_sigmas_temporal_only`
    /// treats "never updated" as "no trustworthy reading has arrived
    /// yet" and falls back to `noise_estimator_low_unboosted` for that
    /// case.
    ///
    /// With `hq.windowed_noise_estimation` set, a fold with no
    /// trustworthy sample clears this estimator instead of leaving it
    /// where it was: "keeps going between folds" is itself a form of
    /// history carried past the current window, the same thing
    /// window-local estimation exists to remove from the other chains.
    ///
    /// It is inert under the same condition as `noise_estimator`.
    pub(super) noise_estimator_temporal_only: NoiseEstimator,

    /// The motion-compensation geometry, present while motion
    /// compensation is active.
    pub(super) mc_ctx: Option<MotionCtx>,
    /// The shifted input ring, shaped exactly like `input_buf`.
    ///
    /// The temporal kernels read their neighbours from here, and the
    /// centre slot is a straight copy of `input_buf`.
    pub(super) compensated_input_buf: Option<Handle>,
    /// The shifted reference ring, shaped like
    /// `compensated_input_buf`, present when a prefilter is active.
    pub(super) compensated_reference_buf: Option<Handle>,
    /// The motion field, with one slice per neighbour and two `i32`
    /// components per block.
    ///
    /// The first half of the slices hold the neighbours behind the
    /// centre frame, and the second half those ahead of it.
    pub(super) mv_field_buf: Option<Handle>,
    /// The ring of adjacent-frame motion fields, indexed by slot, then
    /// direction, then block.
    ///
    /// Direction 0 runs from the older frame to the newer one, and
    /// direction 1 the other way.
    ///
    /// It only exists when `MotionEstimation::Chained` is active, and
    /// the direct path never touches it.
    ///
    /// `motion::pair_ring_slot_count` explains why the slot count is
    /// exactly enough, and `Self::pair_slot` shows how a frame's place
    /// in the push sequence picks a slot.
    pub(super) pair_ring_buf: Option<Handle>,
    /// The luma pyramid, indexed by level, then frame, then pixel.
    pub(super) pyramid_input: Option<Handle>,
    /// The same pyramid built from the reference ring, present when a
    /// prefilter is active.
    pub(super) pyramid_reference: Option<Handle>,

    /// The block geometry for the confidence pass that runs without
    /// motion compensation.
    ///
    /// It only exists when confidence weighting is on and motion
    /// compensation is off. When motion compensation is on, `mc_ctx`
    /// supplies the geometry instead.
    pub(super) confidence_ctx: Option<MotionCtx>,
    /// The per-block match confidence, with one slice per neighbour
    /// laid out like `mv_field_buf` but holding a single `f32` per
    /// block.
    ///
    /// It exists whenever confidence weighting is on, whichever context
    /// supplied the geometry.
    pub(super) confidence_buf: Option<Handle>,
    /// A single-level luma pyramid ring feeding the confidence pass that
    /// runs without motion compensation. It exists alongside
    /// `confidence_ctx`.
    pub(super) confidence_pyramid: Option<Handle>,
    /// Somewhere to throw away the motion vector that pass produces,
    /// since nothing shifts by it without motion compensation. It exists
    /// alongside `confidence_ctx`.
    pub(super) confidence_mv_scratch: Option<Handle>,
    /// A small placeholder passed as the fine block-match kernel's
    /// confidence argument when confidence weighting is off but motion
    /// compensation still runs.
    ///
    /// The kernel drops the confidence write at compile time in that
    /// case, so this buffer is never indexed and its size does not
    /// matter. It is tiny, so unlike the buffers above it is always
    /// allocated.
    pub(super) confidence_dummy: Handle,
    /// The smoothed sigma for channel 0, the plane motion estimation
    /// treats as luma, which feeds the confidence noise floor.
    ///
    /// It is zero unless HQ is on. A fixed `sigma_override` sets it once
    /// at construction, while automatic estimation refreshes it every
    /// submit.
    pub(super) sigma_y: f32,
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Builds a new denoiser.
    ///
    /// # Panics
    ///
    /// This panics if the parameters or the frame dimensions are
    /// invalid. The high-level [`crate::Denoiser`] checks both first and
    /// reports them as a `Result`, so most callers should use that
    /// instead.
    pub fn new(client: &ComputeClient<R>, params: NlmParams, width: u32, height: u32) -> Self {
        Self::with_output_format(client, params, width, height, OutputFormat::F32)
    }

    /// Builds a new denoiser whose readbacks come back in
    /// `output_format`.
    ///
    /// [`OutputFormat::Wire`] gives the denoiser a packed-word buffer
    /// per output slot, so a readback quantises on the GPU and only the
    /// wire bytes cross the bus.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::new`].
    pub fn with_output_format(
        client: &ComputeClient<R>,
        params: NlmParams,
        width: u32,
        height: u32,
        output_format: OutputFormat,
    ) -> Self {
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
            Some(client.empty(frame_bytes * total_frames as usize))
        } else {
            None
        };

        let padding_scratch = if params.channels.count() != stored_ch {
            vec![0.0f32; pixels * stored_ch as usize]
        } else {
            Vec::new()
        };

        let accum = client.empty(frame_bytes);
        let weight_sum = client.empty(scalar_bytes);
        let max_weight = client.empty(scalar_bytes);
        let weight_buf = client.empty(scalar_bytes);
        let raw_fwd = client.empty(scalar_bytes);
        let raw_bwd = client.empty(scalar_bytes);
        let tmp_hsum = client.empty(scalar_bytes);
        let tmp_hsum_bwd = client.empty(scalar_bytes);
        let outputs = [client.empty(frame_bytes), client.empty(frame_bytes)];
        let wire_outputs = match output_format {
            OutputFormat::F32 => None,
            OutputFormat::Wire { depth } => {
                let samples = pixels as u32 * params.channels.count();
                let words = samples.div_ceil(depth.wire_pack().samples_per_word()) as usize;
                Some([
                    client.empty(words * size_of::<u32>()),
                    client.empty(words * size_of::<u32>()),
                ])
            },
        };

        let h2_inv_norm = params.h2_inv_norm();
        let input_noise_offset = params.noise_offset();
        // The pilot pass compares noisy input pixels, so it always
        // keeps the full noise floor. Under `NlmSpatial` the main pass
        // compares one pilot output against another, which no longer
        // carries that floor, so subtracting it would overweight
        // patches that do not really match.
        let noise_offset = match params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => input_noise_offset,
        };
        let output_scratch = empty_output(pixels, params.channels.count(), output_format);
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let use_reference = params.prefilter.needs_reference_buf();

        // This stays unset until the first temporal sample lands, so
        // the initial table matches the flat `noise_offset` scalar it
        // replaces exactly.
        let rho_smoothed: Option<f32> = None;
        let spatial_offset_lut = client.create_from_slice(f32::as_bytes(&build_spatial_offset_lut(
            params.search_radius,
            0.0,
            noise_offset,
        )));

        // Automatic noise estimation only runs when HQ is on and the
        // caller has not pinned a fixed sigma. The fast path and the
        // fixed-sigma path allocate neither buffer and never launch the
        // estimate kernels.
        let auto_noise = params.hq.is_some_and(|hq| hq.sigma_override.is_none());
        let (noise_partials, noise_results) = if auto_noise {
            let partials_ring_bytes =
                noise_partials_slot_stride_bytes(width, height, align) * total_frames as u64;
            let n_results = (total_frames * 4) as usize;
            (
                Some(client.empty(partials_ring_bytes as usize)),
                Some(client.empty(n_results * size_of::<f32>())),
            )
        } else {
            (None, None)
        };

        // The temporal residual estimator also needs a real neighbour
        // to take a difference against, so it stays inert at a temporal
        // radius of 0 even when automatic estimation is on.
        let temporal_stats_buf = if auto_noise && params.temporal_radius >= 1 {
            Some(client.empty(temporal_stats_buf_bytes(
                width,
                height,
                stored_ch,
                total_frames,
                align,
            )))
        } else {
            None
        };

        // The motion-compensation buffers, allocated only when motion
        // compensation is active and the temporal window reaches past
        // the centre frame. A spatial-only pass never touches them.
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
        ) = if let Some(ctx) = mc_ctx.as_ref() {
            let comp_in = client.empty(frame_bytes * total_frames as usize);
            let comp_ref = if use_reference {
                Some(client.empty(frame_bytes * total_frames as usize))
            } else {
                None
            };
            let neighbours = (2 * params.temporal_radius) as u64;
            let mv_field = client.empty((neighbours * ctx.mv_field_bytes_per_neighbour()) as usize);
            let pyramid_pixels =
                motion::pyramid_pixels_per_frame(width, height, ctx.pyramid_levels, ctx.align);
            let pyr_in_bytes = pyramid_pixels * total_frames as usize * size_of::<f32>();
            let pyr_in = client.empty(pyr_in_bytes);
            let pyr_ref = if use_reference {
                Some(client.empty(pyr_in_bytes))
            } else {
                None
            };
            (Some(comp_in), comp_ref, Some(mv_field), Some(pyr_in), pyr_ref)
        } else {
            (None, None, None, None, None)
        };

        // The pair ring is allocated only when `Chained` estimation is
        // active, either because it was asked for or because `Auto`
        // resolved to it at this temporal radius, and only on top of
        // motion compensation already being on. The direct path never
        // reads or writes it.
        let is_chained = matches!(
            params
                .motion_compensation
                .resolved_estimation(params.temporal_radius),
            Some(MotionEstimation::Chained { .. })
        );
        let pair_ring_buf = if is_chained {
            mc_ctx.as_ref().map(|ctx| {
                let pair_ring_slots = motion::pair_ring_slot_count(params.temporal_radius) as u64;
                let bytes = pair_ring_slots * ctx.pair_slot_bytes();
                client.empty(bytes as usize)
            })
        } else {
            None
        };

        // Confidence weighting, in either of its two forms, only runs
        // when HQ has it enabled and the temporal window reaches past
        // the centre frame.
        //
        // This check applies with motion compensation on as well.
        // Without it, every submit would pay for the fine kernel's
        // confidence write whether or not anything read the result.
        let confidence_active =
            params.hq.is_some_and(|hq| hq.temporal_confidence) && params.temporal_radius > 0;

        // Geometry for the confidence-only pass, needed only when
        // motion compensation is not already supplying block geometry
        // through its own analyse pass.
        //
        // This costs real extra work, because it needs its own luma
        // pyramid ring and a block-match kernel per neighbour.
        let confidence_only_active = confidence_active && mc_ctx.is_none();
        let confidence_ctx = confidence_only_active.then(|| MotionCtx::confidence_only(width, height, align));

        // The confidence buffer uses whichever block geometry is
        // available, but only when confidence weighting is on.
        let confidence_geometry = if confidence_active {
            mc_ctx.as_ref().or(confidence_ctx.as_ref())
        } else {
            None
        };
        let confidence_buf = confidence_geometry.map(|ctx| {
            let neighbours = (2 * params.temporal_radius) as u64;
            client.empty((neighbours * ctx.confidence_bytes_per_neighbour()) as usize)
        });
        // Always allocated, tiny, and reused whenever the fine
        // block-match kernel runs without writing confidence.
        let confidence_dummy = client.empty(size_of::<f32>());

        let (confidence_pyramid, confidence_mv_scratch) = if let Some(ctx) = confidence_ctx.as_ref() {
            let pyramid_pixels =
                motion::pyramid_pixels_per_frame(width, height, ctx.pyramid_levels, ctx.align);
            let pyr_bytes = pyramid_pixels * total_frames as usize * size_of::<f32>();
            let mv_scratch_len = ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>();
            (Some(client.empty(pyr_bytes)), Some(client.empty(mv_scratch_len)))
        } else {
            (None, None)
        };

        // A fixed `sigma_override` is the only source before the first
        // estimate lands, and automatic estimation refreshes this every
        // submit. See `update_noise_estimate`.
        //
        // The fast path leaves it at zero, which
        // `motion::sad_noise_floor` turns into a zero floor, exactly
        // what a caller with no estimate should get.
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
            padding_scratch,
            upload_scratch: Vec::new(),
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
            output_format,
            wire_outputs,
            output_scratch,
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
        }
    }

    /// Pushes a new frame into the ring buffer.
    ///
    /// `frame` holds `width * height * channels` `f32` values in
    /// `[0, 1]`. A 3-channel frame is repacked into 4 lanes through a
    /// reused CPU scratch buffer.
    ///
    /// For `PrefilterMode::External` use
    /// [`Self::push_frame_with_reference`] instead.
    pub fn push_frame(&mut self, frame: &[f32]) {
        assert!(
            !matches!(self.params.prefilter, PrefilterMode::External),
            "push_frame_with_reference is required when prefilter == External"
        );

        let slot = self.upload_into(&self.input_buf.clone(), frame);
        self.run_post_upload_stages(slot);
    }

    /// Pushes a new frame held as wire bytes, one slice per channel.
    ///
    /// Each plane holds `width * height` samples at `depth`, and the GPU
    /// normalises them and interleaves the planes. `planes` runs Y, U, V
    /// for a fused frame and U, V for a chroma pair.
    ///
    /// For `PrefilterMode::External` use
    /// [`Self::push_frame_with_reference`] instead.
    pub fn push_frame_wire(&mut self, planes: &[&[u8]], depth: Depth) {
        assert!(
            !matches!(self.params.prefilter, PrefilterMode::External),
            "push_frame_with_reference is required when prefilter == External"
        );

        let slot = self.upload_wire_into(&self.input_buf.clone(), planes, depth);
        self.run_post_upload_stages(slot);
    }

    /// The work every push runs once its frame is in `slot`, from the
    /// noise estimate through to the ring advance.
    fn run_post_upload_stages(&mut self, slot: usize) {
        self.run_noise_estimate_for_slot(slot as u32);
        self.run_temporal_stats_for_slot(slot as u32);
        self.seed_noise_estimate_if_first_frame(slot as u32);

        if let PrefilterMode::NlmSpatial { strength_scale } = self.params.prefilter {
            self.run_nlm_spatial_pilot(slot as u32, strength_scale)
                .expect("nlm spatial pilot dispatch failed");
        } else if self.params.prefilter.is_gpu_internal() {
            self.run_prefilter_for_slot(slot);
        }

        self.build_pyramids_for_slot(slot as u32);
        self.build_confidence_pyramid_for_slot(slot as u32);
        self.run_pair_analyse_for_slot(slot as u32);

        self.advance_ring();
        self.prime_leading_edge_if_first();
    }

    /// Pushes a new frame together with a reference image the caller
    /// prefiltered itself.
    ///
    /// This is what `PrefilterMode::External` needs. Both slices hold
    /// `width * height * channels` `f32` values in `[0, 1]`.
    pub fn push_frame_with_reference(&mut self, frame: &[f32], reference: &[f32]) {
        assert!(
            matches!(self.params.prefilter, PrefilterMode::External),
            "push_frame_with_reference requires prefilter == External"
        );

        let slot = self.upload_into(&self.input_buf.clone(), frame);
        let reference_buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist for External prefilter")
            .clone();
        self.upload_into_slot(&reference_buf, reference, slot);

        // The same order as `push_frame`. The noise estimate and its
        // first-frame seed run before anything that could read the
        // sigma. Building the pyramids only needs the reference upload
        // just above, not the noise estimate, so the ordering does not
        // change what either step sees.
        self.run_noise_estimate_for_slot(slot as u32);
        self.run_temporal_stats_for_slot(slot as u32);
        self.seed_noise_estimate_if_first_frame(slot as u32);

        self.build_pyramids_for_slot(slot as u32);
        self.build_confidence_pyramid_for_slot(slot as u32);
        self.run_pair_analyse_for_slot(slot as u32);

        self.advance_ring();
        self.prime_leading_edge_if_first();
    }

    /// Uploads `frame` into the next ring slot of `dst` and returns the
    /// physical slot it wrote.
    fn upload_into(&mut self, dst: &Handle, frame: &[f32]) -> usize {
        let total_frames = self.params.total_frames() as usize;
        let slot = self.ring_head % total_frames;
        self.upload_into_slot(dst, frame, slot);
        slot
    }

    fn upload_into_slot(&mut self, dst: &Handle, frame: &[f32], slot: usize) {
        let channels = self.params.channels.count() as usize;
        let stored_ch = self.params.channels.storage_count() as usize;
        let pixels = self.width as usize * self.height as usize;
        let expected = pixels * channels;

        assert_eq!(
            frame.len(),
            expected,
            "frame size mismatch: expected {expected}, got {}",
            frame.len()
        );

        let staging = if channels == stored_ch {
            self.client.create_from_slice(f32::as_bytes(frame))
        } else {
            for i in 0..pixels {
                let dst_off = i * stored_ch;
                let src_off = i * channels;
                self.padding_scratch[dst_off..dst_off + channels]
                    .copy_from_slice(&frame[src_off..src_off + channels]);
            }
            self.client
                .create_from_slice(f32::as_bytes(&self.padding_scratch))
        };

        self.copy_frame_into_slot(dst, slot, &staging, 0, 1);
    }

    /// Uploads one wire-byte frame into the next ring slot of `dst` and
    /// returns the physical slot it wrote.
    ///
    /// The planes are concatenated into `upload_scratch` and uploaded as
    /// one buffer, so a push costs one transfer whatever the channel
    /// mode. The scratch is reused, so the concatenation allocates
    /// nothing after the first frame.
    fn upload_wire_into(&mut self, dst: &Handle, planes: &[&[u8]], depth: Depth) -> usize {
        let total_frames = self.params.total_frames() as usize;
        let slot = self.ring_head % total_frames;
        self.upload_wire_into_slot(dst, planes, depth, slot);
        slot
    }

    fn upload_wire_into_slot(&mut self, dst: &Handle, planes: &[&[u8]], depth: Depth, slot: usize) {
        let channels = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let pixels = self.width * self.height;
        let plane_bytes = pixels as usize * depth.bytes_per_sample();

        assert_eq!(
            planes.len(),
            channels as usize,
            "plane count mismatch: expected {channels}, got {}",
            planes.len()
        );

        // Ten and Twelve share a byte width, so the plane-length check
        // below cannot tell them apart. A wrong depth here divides by the
        // wrong maximum and darkens the whole frame without failing
        // anything else, so it is pinned against the depth this denoiser
        // returns frames in.
        if let OutputFormat::Wire { depth: out_depth } = self.output_format {
            assert_eq!(
                depth, out_depth,
                "wire push depth {depth:?} does not match the denoiser's output depth {out_depth:?}"
            );
        }

        self.upload_scratch.clear();
        for plane in planes {
            assert_eq!(
                plane.len(),
                plane_bytes,
                "plane size mismatch: expected {plane_bytes}, got {}",
                plane.len()
            );
            debug_assert!(
                wire_samples_in_range(plane, depth),
                "a sample is larger than {depth:?} can express"
            );
            self.upload_scratch.extend_from_slice(plane);
        }

        // The kernel reads whole words, so a plane that ends mid-word
        // needs its last word backed by real storage.
        let words = self.upload_scratch.len().div_ceil(size_of::<u32>());
        self.upload_scratch.resize(words * size_of::<u32>(), 0);

        let src = self.client.create_from_slice(&self.upload_scratch);

        let elements = pixels * stored_ch;
        let total_frames = self.params.total_frames() as usize;
        let grid = elements.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        // One `wire_pack` for both, since a hand-paired maximum and lane
        // width decode the wrong bits.
        let pack = depth.wire_pack();

        unsafe {
            gpu_unpack_wire::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(src, words),
                ArrayArg::from_raw_parts(dst.clone(), total_frames * elements as usize),
                pack.max(),
                slot as u32 * elements,
                pixels,
                channels,
                stored_ch,
                pack.samples_per_word(),
                elements,
                total_threads,
            )
        };
    }

    fn run_prefilter_for_slot(&self, slot: usize) {
        let reference_buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist for GPU prefilter");

        let ctx = PrefilterCtx {
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            frame_count: self.params.total_frames(),
            frame: slot as u32,
            input_buf: &self.input_buf,
            reference_buf,
        };

        run_prefilter::<R>(self.params.prefilter, &self.client, &ctx).expect("prefilter dispatch failed");
    }

    /// Builds the motion-estimation pyramid for `slot` on the input
    /// ring, and on the reference ring when there is one.
    ///
    /// This does nothing when motion compensation is off.
    fn build_pyramids_for_slot(&self, slot: u32) {
        let Some(ctx) = self.mc_ctx.as_ref() else {
            return;
        };

        let stored_ch = self.params.channels.storage_count();
        let frame_count = self.params.total_frames();

        if let Some(pyr) = self.pyramid_input.as_ref() {
            build_pyramid_for_slot::<R>(
                &self.client,
                ctx,
                self.width,
                self.height,
                frame_count,
                slot,
                &self.input_buf,
                pyr,
                stored_ch,
            )
            .expect("input pyramid build dispatch failed");
        }

        if let (Some(pyr_ref), Some(ref_buf)) = (self.pyramid_reference.as_ref(), self.reference_buf.as_ref())
        {
            build_pyramid_for_slot::<R>(
                &self.client,
                ctx,
                self.width,
                self.height,
                frame_count,
                slot,
                ref_buf,
                pyr_ref,
                stored_ch,
            )
            .expect("reference pyramid build dispatch failed");
        }
    }

    /// Extracts the luma plane for `slot` into the pyramid the
    /// confidence pass uses when motion compensation is off.
    ///
    /// This does nothing unless that pass is active.
    ///
    /// It always reads `input_buf`, even with a prefilter set. Comparing
    /// the raw input keeps this path simple, rather than duplicating the
    /// reference ring's pyramid.
    ///
    /// It calls `run_pyramid_build` directly rather than going through
    /// [`Self::build_pyramids_for_slot`]. That helper only touches the
    /// motion-compensation pyramids, and its context is never present at
    /// the same time as this one, so it would return without building
    /// anything.
    fn build_confidence_pyramid_for_slot(&self, slot: u32) {
        let (Some(ctx), Some(pyr)) = (self.confidence_ctx.as_ref(), self.confidence_pyramid.as_ref()) else {
            return;
        };

        run_pyramid_build::<R>(
            &self.client,
            ctx,
            self.width,
            self.height,
            self.params.total_frames(),
            slot,
            &self.input_buf,
            pyr,
            self.params.channels.storage_count(),
        )
        .expect("confidence pyramid build dispatch failed");
    }

    /// Whether `Chained` motion estimation is in use, either because it
    /// was asked for or because `Auto` resolved to it at this temporal
    /// radius.
    ///
    /// `resolved_estimation` is the one place that decision is made.
    ///
    /// This is separate from whether motion compensation itself is
    /// active, which callers still have to check, because that also
    /// needs a temporal radius above 0.
    pub(super) fn is_chained(&self) -> bool {
        matches!(
            self.params
                .motion_compensation
                .resolved_estimation(self.params.temporal_radius),
            Some(MotionEstimation::Chained { .. })
        )
    }

    /// Measures motion between the slot a push just wrote and the one
    /// before it, storing both directions into the pair ring.
    ///
    /// This does nothing unless `Chained` estimation is active, and it
    /// also does nothing for a stream's very first frame, which has no
    /// older partner to pair against.
    ///
    /// Composition covers that first gap by reading the priming
    /// duplicate's zero-filled pair instead. See
    /// [`Self::zero_pair_slot_for_duplicate`].
    fn run_pair_analyse_for_slot(&self, newer_slot: u32) {
        if self.ring_head == 0 {
            return;
        }
        let Some(mc) = self.mc_ctx.as_ref() else {
            return;
        };
        if !self.is_chained() {
            return;
        }
        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");

        // Match against the cleaner of the two buffers, exactly as
        // `run_motion_compensation` does on the direct path.
        let pyramid = self.pyramid_reference.as_ref().unwrap_or_else(|| {
            self.pyramid_input
                .as_ref()
                .expect("pyramid_input allocated when mc_ctx is Some")
        });

        let total_frames = self.params.total_frames();
        let older_slot = (newer_slot + total_frames - 1) % total_frames;
        let pair_slot = self.pair_slot(0);

        motion::run_pair_analyse::<R>(
            &self.client,
            mc,
            self.width,
            self.height,
            total_frames,
            older_slot,
            newer_slot,
            pair_slot,
            pyramid,
            pair_ring,
            &self.confidence_dummy,
        )
        .expect("pair analyse dispatch failed");
    }

    /// Fills the pair-ring slot for a duplicated frame with zeroes,
    /// which happens while priming a stream and during the
    /// end-of-stream flush.
    ///
    /// This does nothing unless `Chained` estimation is active.
    fn zero_pair_slot_for_duplicate(&self) {
        let Some(mc) = self.mc_ctx.as_ref() else {
            return;
        };
        if !self.is_chained() {
            return;
        }
        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");
        let pair_slot = self.pair_slot(0);
        motion::zero_pair_slot::<R>(&self.client, mc, pair_ring, pair_slot);
    }

    /// Queues the Immerkær noise estimate for `slot` on the input ring.
    ///
    /// This does nothing unless automatic noise estimation is active.
    ///
    /// The results are normally read back later, in
    /// [`Self::denoise_submit`], once `slot` reaches the centre of the
    /// temporal window. A stream's very first frame is read immediately
    /// as well. See [`Self::seed_noise_estimate_if_first_frame`].
    fn run_noise_estimate_for_slot(&self, slot: u32) {
        let (Some(partials_buf), Some(results_buf)) =
            (self.noise_partials.as_ref(), self.noise_results.as_ref())
        else {
            return;
        };

        let stride = noise_partials_slot_stride_bytes(self.width, self.height, self.align);
        let partials_slot = partials_buf.clone().offset_start((slot as u64) * stride);

        let ctx = NoiseCtx {
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

        run_noise_estimate::<R>(&self.client, &ctx).expect("noise estimate dispatch failed");
    }

    /// Queues the temporal residual statistics for `slot`, comparing it
    /// against the slot immediately before it in the ring.
    ///
    /// This does nothing unless the temporal estimator is active, and it
    /// also does nothing for a stream's very first frame, which has no
    /// predecessor to compare against. That matches the check in
    /// [`Self::run_pair_analyse_for_slot`].
    ///
    /// The centre slot's statistics are read back and combined later, in
    /// [`Self::update_noise_estimate`].
    fn run_temporal_stats_for_slot(&self, slot: u32) {
        let Some(stats_buf) = self.temporal_stats_buf.as_ref() else {
            return;
        };
        if self.ring_head == 0 {
            return;
        }

        let total_frames = self.params.total_frames();
        let slot_prev = (slot + total_frames - 1) % total_frames;

        let ctx = TemporalStatsCtx {
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

        run_temporal_noise_stats::<R>(&self.client, &ctx).expect("temporal noise stats dispatch failed");
    }

    /// Fills a duplicated slot's temporal-stats region with zeroes.
    ///
    /// A duplicate holds exactly the same pixels as the slot before it,
    /// so measuring the difference would only ever produce an all-zero
    /// record. Writing the zeroes is the cheaper way to the same answer.
    ///
    /// This does nothing unless the temporal estimator is active.
    fn zero_temporal_stats_for_slot(&self, slot: u32) {
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

    /// Reads the noise estimate once, for a stream's very first frame,
    /// so push-time work has a real sigma to use.
    ///
    /// Automatic estimation normally refreshes the derived filter
    /// parameters at submit time, in [`Self::update_noise_estimate`].
    /// But push-time GPU work that reads them, namely the NLM pilot,
    /// runs before the first submit ever happens.
    ///
    /// Without this, that work would run on the absolute-strength
    /// fallback chosen at construction for every frame up to the first
    /// submit. One blocking read of the estimate this push just queued
    /// fixes it from frame one onward.
    ///
    /// The first frame is spotted through `frames_loaded`, the same
    /// counter [`Self::prime_leading_edge_if_first`] checks, but read
    /// here before [`Self::advance_ring`] moves it on. It applies at
    /// every temporal radius, not only when priming happens.
    ///
    /// The first submit folds the same frame's estimate in a second
    /// time, which reproduces these values to within floating-point
    /// rounding rather than exactly.
    fn seed_noise_estimate_if_first_frame(&mut self, slot: u32) {
        if self.frames_loaded != 0 {
            return;
        }
        let Some(results_buf) = self.noise_results.as_ref() else {
            return;
        };

        let bytes = self
            .client
            .read_one(results_buf.clone())
            .expect("noise-estimate seed readback failed");
        let data = f32::from_bytes(&bytes);

        // The stream's first frame has no predecessor, so
        // `run_temporal_stats_for_slot` never ran for it and this
        // slot's stats region is unwritten. Seed from Immerkær alone.
        let imm_low = self
            .read_noise_partials_low(slot)
            .expect("noise-partials seed readback failed");
        self.fold_noise_estimate(data, slot as usize, None, imm_low);
    }

    /// Folds one ring slot's noise totals into both estimator chains and
    /// recomputes everything derived from them.
    ///
    /// Those derived values are `h2_inv_norm`, `noise_offset`,
    /// `input_noise_offset`, and `sigma_y`.
    ///
    /// [`Self::seed_noise_estimate_if_first_frame`] and
    /// [`Self::update_noise_estimate`] both call this. They differ only
    /// in how they obtain the readings and which slot they pass.
    ///
    /// # The estimator chains
    ///
    /// Each chain starts from an Immerkær reading and takes the larger
    /// of that and a temporal-residual reading, when one is available.
    ///
    /// The temporal estimator sees correlated grain the Immerkær mask
    /// reads too low, but a shot with little static content, because of
    /// motion or a scene change, makes its reading unreliable. Taking
    /// the larger value lets it raise an estimate but never lower one.
    ///
    /// The chains differ only in which statistic they read. The median
    /// chain takes the frame-mean Immerkær total and the per-block
    /// median of the temporal reading. The low chain takes Immerkær's
    /// own lower-quartile block statistic and the temporal lower
    /// quartile.
    ///
    /// `noise_offset` weighs patch distances by the square of the sigma,
    /// so reading too high there scrubs fine texture. The low chain's
    /// cautious statistics keep it from over-reading on shots where
    /// texture leaks into the temporal residuals.
    ///
    /// The strength and the confidence floor stay on the median chain,
    /// because that is what the dark-footage calibration validated.
    ///
    /// A third estimator, `noise_estimator_low_unboosted`, folds the
    /// same lower-quartile temporal reading as the low chain but skips
    /// the correlation boost described below. It exists for a consumer
    /// that squares its sigma into a threshold, where a boost meant to
    /// offset a spatial estimator's blind spot on correlated grain would
    /// otherwise be applied a second time to a temporal reading that
    /// already tracks that grain directly.
    ///
    /// A fourth estimator, `noise_estimator_temporal_only`, folds the
    /// temporal reading by itself, with neither the maximum against the
    /// Immerkær spatial reading nor the correlation boost. It exists for
    /// the same squaring consumer, for a stronger reason than the boost
    /// alone: a spatial mask reads regularly repeating texture the same
    /// way it reads noise, and taking the maximum against it lets that
    /// misreading through no matter how accurate the temporal side is.
    /// This estimator only folds a new value on a fold whose temporal
    /// sample was trustworthy enough for `aggregate_temporal_noise_stats`
    /// to produce, and otherwise keeps whatever it last held.
    ///
    /// # Grain correlation
    ///
    /// The temporal sample's correlation figure folds into
    /// `rho_smoothed` once, whichever chain reads it. That value feeds
    /// the spatial-offset table.
    ///
    /// A stream's first fold sets it directly rather than blending from
    /// an assumed zero, the same convention `NoiseEstimator` uses for
    /// its own first sample.
    ///
    /// It stays unset on the fast path and with a fixed sigma, because
    /// no temporal sample ever arrives there.
    ///
    /// # Window-local estimation
    ///
    /// With `hq.windowed_noise_estimation` set, every chain and
    /// `rho_smoothed` take this fold's own sample outright instead of
    /// blending it into their running state, the same way each does for
    /// its very first sample. `noise_estimator_temporal_only` and
    /// `rho_smoothed` also stop keeping their last reading on a fold
    /// that has no temporal sample of its own, clearing instead, since
    /// that "keep going" behaviour is itself a form of cross-window
    /// history. See
    /// [`crate::nlmeans::HqParams::windowed_noise_estimation`].
    fn fold_noise_estimate(
        &mut self,
        data: &[f32],
        slot: usize,
        temporal: Option<TemporalNoiseSample>,
        imm_low: [f32; 3],
    ) {
        let channels = self.params.channels.count() as usize;
        let base = slot * 4;

        let mut raw = [0.0f32; 3];
        for (c, s) in raw.iter_mut().enumerate().take(channels) {
            *s = sigma_from_abs_sum(data[base + c], self.width, self.height);
        }
        let mut raw_low = imm_low;
        let mut raw_low_unboosted = imm_low;
        let mut raw_temporal_only: Option<[f32; 3]> = None;

        // Resolved once per fold rather than re-read per estimator
        // below, and `false` on every call the default configuration
        // makes, since `hq.windowed_noise_estimation` is `false` unless
        // a caller set it. See `HqParams::windowed_noise_estimation`.
        let windowed = self.params.hq.is_some_and(|hq| hq.windowed_noise_estimation);

        if let Some(sample) = temporal {
            let factor = correlation_factor(sample.rho);
            for c in 0..channels {
                raw[c] = raw[c].max(sample.sigma[c] * factor);
                raw_low[c] = raw_low[c].max(sample.sigma_low[c] * factor);
                raw_low_unboosted[c] = raw_low_unboosted[c].max(sample.sigma_low[c]);
            }
            raw_temporal_only = Some(sample.sigma_low);
            self.rho_smoothed = Some(
                if windowed {
                    sample.rho
                } else {
                    match self.rho_smoothed {
                        None => sample.rho,
                        Some(prev) => EMA_ALPHA * sample.rho + (1.0 - EMA_ALPHA) * prev,
                    }
                },
            );
        } else if windowed {
            // Window-local estimation must not let an earlier push's
            // correlation reading leak into a fold that has no temporal
            // sample of its own, the same reasoning that gates
            // `noise_estimator_temporal_only` above. Without this,
            // `rho_smoothed` keeps whatever an earlier window last
            // measured, so `spatial_offset_lut` would depend on how
            // many pushes preceded this fold rather than only the
            // current window's own content.
            self.rho_smoothed = None;
        }

        // The user's nudge on the measured noise level. It applies
        // after the two readings are combined and before the smoothing
        // step, so it scales the smoothed estimate and everything
        // derived from it.
        //
        // This is only reached when no fixed sigma was given, so the HQ
        // parameters are always present here.
        let sigma_scale = self.params.hq.map_or(1.0, |hq| hq.sigma_scale);
        for c in 0..channels {
            raw[c] *= sigma_scale;
            raw_low[c] *= sigma_scale;
            raw_low_unboosted[c] *= sigma_scale;
        }
        if let Some(raw_t) = raw_temporal_only.as_mut() {
            for s in raw_t.iter_mut().take(channels) {
                *s *= sigma_scale;
            }
        }

        let updated = self.noise_estimator.update(&raw[..channels], windowed);
        let mut smoothed = [0.0f32; 3];
        smoothed[..channels].copy_from_slice(updated);

        let updated_low = self.noise_estimator_low.update(&raw_low[..channels], windowed);
        let mut smoothed_low = [0.0f32; 3];
        smoothed_low[..channels].copy_from_slice(updated_low);

        self.noise_estimator_low_unboosted
            .update(&raw_low_unboosted[..channels], windowed);

        match raw_temporal_only {
            Some(raw_t) => {
                self.noise_estimator_temporal_only
                    .update(&raw_t[..channels], windowed);
            },
            // Window-local estimation must not let an earlier push's
            // trustworthy reading leak into a fold that has none of its
            // own, or the target frame's result would depend on how
            // many pushes preceded it in this call, the same history
            // dependence window-local estimation exists to remove from
            // the other chains. See `noise_estimator_temporal_only`.
            None if windowed => self.noise_estimator_temporal_only.reset(),
            None => {},
        }

        let eff = sigma_eff(&smoothed[..channels], self.params.channels);
        self.h2_inv_norm = self.params.h2_inv_norm_with(Some(eff));
        self.input_noise_offset = self.params.noise_offset_with(Some(&smoothed_low[..channels]));
        self.noise_offset = match self.params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => self.input_noise_offset,
        };
        // Channel 0 is whatever motion estimation already treats as
        // luma, as `nlm_mc_extract_luma` shows, so the confidence floor
        // uses the median chain's estimate for that same plane.
        self.sigma_y = smoothed[0];
    }

    fn advance_ring(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        self.ring_head += 1;
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }
        self.real_pushes += 1;
    }

    /// Copies one frame from a slot of `src` into a slot of `dst`,
    /// entirely on the GPU.
    ///
    /// `dst` has to use the same ring layout as `input_buf`.
    /// `src_slots` is how many frames `src` holds, which is 1 for a
    /// frame-sized staging buffer.
    ///
    /// Both handles are bound whole, and the kernel picks the slots
    /// through its own offset arguments. Binding a slot directly would
    /// need its byte offset to be a multiple of the GPU's
    /// `min_storage_buffer_offset_alignment`, and a
    /// `width * height * stored_ch` frame stride rarely lands on one.
    fn copy_frame_into_slot(
        &self,
        dst: &Handle,
        slot: usize,
        src: &Handle,
        src_slot: usize,
        src_slots: usize,
    ) {
        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let dst_slots = self.params.total_frames() as usize;

        let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        unsafe {
            gpu_copy::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(src.clone(), src_slots * frame_size as usize),
                ArrayArg::from_raw_parts(dst.clone(), dst_slots * frame_size as usize),
                src_slot as u32 * frame_size,
                slot as u32 * frame_size,
                frame_size,
                total_threads,
            )
        };
    }

    /// Copies the very first pushed frame into the leading ring slots,
    /// so the temporal window starts out balanced rather than dropping
    /// the opening frames.
    ///
    /// [`Self::flush`] does the same thing at the other end of the
    /// stream.
    fn prime_leading_edge_if_first(&mut self) {
        let r = self.params.temporal_radius as usize;

        if r == 0 || self.frames_loaded != 1 {
            return;
        }

        for _ in 0..r {
            self.duplicate_last_frame();
            self.frames_loaded += 1;
        }
    }

    /// Copies the most recently pushed frame into the next ring slot.
    ///
    /// This runs at the end of a stream, keeping the window full as the
    /// real frames ahead of the centre run out.
    ///
    /// Slots never overlap, so copying inside the same buffer is safe.
    ///
    /// The reference ring is copied in step when it exists, so the
    /// weights are never computed from a stale slot.
    fn duplicate_last_frame(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        let last_slot = (self.ring_head - 1) % total_frames;
        let next_slot = self.ring_head % total_frames;

        let input_buf = self.input_buf.clone();
        self.copy_frame_into_slot(&input_buf, next_slot, &input_buf, last_slot, total_frames);

        // Skipped for `NlmSpatial`, because the pilot dispatch below
        // rebuilds this slot's reference from scratch and would
        // overwrite the copy straight away.
        if !matches!(self.params.prefilter, PrefilterMode::NlmSpatial { .. })
            && let Some(reference_buf) = self.reference_buf.clone()
        {
            self.copy_frame_into_slot(&reference_buf, next_slot, &reference_buf, last_slot, total_frames);
        }

        // Keep the pyramid and the noise estimate for the duplicated
        // slot in step too, so a later denoise sees valid state at
        // every ring slot it visits rather than whatever an older frame
        // left behind at this position.
        //
        // The NLM pilot needs the same treatment, or the duplicated
        // slot's reference would keep whatever an older frame last
        // wrote there.
        if let PrefilterMode::NlmSpatial { strength_scale } = self.params.prefilter {
            self.run_nlm_spatial_pilot(next_slot as u32, strength_scale)
                .expect("nlm spatial pilot dispatch failed");
        }
        self.build_pyramids_for_slot(next_slot as u32);
        self.build_confidence_pyramid_for_slot(next_slot as u32);
        self.run_noise_estimate_for_slot(next_slot as u32);
        self.zero_temporal_stats_for_slot(next_slot as u32);
        // Runs before `ring_head` advances, so `pair_slot(0)` reads the
        // same pre-advance `ring_head` as `run_pair_analyse_for_slot`
        // (see `Self::pair_slot`).
        self.zero_pair_slot_for_duplicate();

        self.ring_head += 1;
    }

    /// Queues the denoise kernels for the current window, without
    /// reading the result back.
    ///
    /// Use this instead of [`Self::denoise_submit`] when the output has
    /// more GPU work ahead of it, so the round trip to the host can be
    /// skipped until the value the caller actually wants is ready. The
    /// returned [`GpuOutput`] documents the lifetime the caller has to
    /// respect.
    ///
    /// Returns `Ok(None)` while the temporal window is still filling.
    pub fn denoise_submit_gpu(&mut self) -> Result<Option<GpuOutput>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        if self.noise_results.is_some() {
            self.update_noise_estimate()?;
        }
        self.rebuild_spatial_offset_lut();

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        self.run_denoise_kernels(slot)?;

        Ok(Some(GpuOutput {
            handle: self.outputs[slot].clone(),
            slot,
        }))
    }

    /// Runs the per-submit machinery a collaborative stage builds on,
    /// without launching any NLM denoising kernel.
    ///
    /// This refreshes the noise estimate the same way
    /// [`Self::denoise_submit_gpu`] does, then runs the same per-neighbour
    /// motion estimate [`Self::run_motion_compensation`] runs, minus the
    /// `run_compensate` shift into the compensated buffers. The returned
    /// [`RingView`] reads the unmodified input ring directly, so a
    /// caller predicts where a patch moved from the motion field and
    /// searches around that prediction itself, rather than reading
    /// content a warp has already resampled.
    ///
    /// Returns `Ok(None)` while the temporal window is still filling, the
    /// same condition [`Self::denoise_submit_gpu`] checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the denoiser was not built with motion
    /// compensation and temporal confidence both active, since a
    /// [`RingView`] has nothing meaningful to hand back otherwise.
    pub(crate) fn submit_machinery(&mut self) -> Result<Option<RingView>, DenoiserError> {
        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        if self.noise_results.is_some() {
            self.update_noise_estimate()?;
        }

        let center_t = self.params.temporal_radius;
        let neighbour_slots = self.run_motion_machinery(center_t)?;

        let mc = self.mc_ctx.as_ref().ok_or_else(|| {
            DenoiserError::Other(anyhow::anyhow!(
                "submit_machinery requires motion compensation to be active"
            ))
        })?;
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some")
            .clone();
        let confidence = self
            .confidence_buf
            .as_ref()
            .ok_or_else(|| {
                DenoiserError::Other(anyhow::anyhow!(
                    "submit_machinery requires HQ temporal confidence to be active"
                ))
            })?
            .clone();

        Ok(Some(RingView {
            input: self.input_buf.clone(),
            mv_field,
            confidence,
            centre_slot: self.phys_frame(center_t as i32),
            neighbour_slots,
            mv_stride: (mc.mv_field_bytes_per_neighbour() / size_of::<i32>() as u64) as u32,
            conf_stride: (mc.confidence_bytes_per_neighbour() / size_of::<f32>() as u64) as u32,
        }))
    }

    /// Flush-mode counterpart of [`Self::submit_machinery`], duplicating
    /// the trailing frame the same way [`Self::flush_step_gpu`] does,
    /// minus the NLM launches.
    ///
    /// Returns `Ok(None)` while the very first duplicates are still
    /// filling out a window that never reached its full size during
    /// pushing, the same condition [`Self::flush_step_gpu`] documents.
    pub(crate) fn flush_step_machinery(&mut self) -> Result<Option<RingView>, DenoiserError> {
        let total_frames = self.params.total_frames() as usize;

        self.duplicate_last_frame();
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }

        self.submit_machinery()
    }

    /// The motion-compensation geometry the last [`Self::submit_machinery`]
    /// or [`Self::flush_step_machinery`] call used.
    ///
    /// # Panics
    ///
    /// Panics if the denoiser was not built with motion compensation
    /// active. Only call this on a denoiser [`Self::submit_machinery`]
    /// has already returned `Some` for.
    pub(crate) fn motion_ctx(&self) -> &MotionCtx {
        self.mc_ctx
            .as_ref()
            .expect("motion_ctx called without motion compensation active")
    }

    /// `thsad(blksize, thsad_scale)` in normalised SAD units, the same
    /// threshold [`Self::submit_machinery`] scores confidence against.
    ///
    /// # Panics
    ///
    /// Panics under the same condition as [`Self::motion_ctx`].
    pub(crate) fn thsad_value(&self) -> f32 {
        let blksize = self.motion_ctx().blksize;
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        motion::thsad(blksize, thsad_scale)
    }

    /// The compute client this denoiser dispatches kernels through, for
    /// a collaborative stage that reads a [`RingView`]'s handles back or
    /// launches its own kernels against them.
    pub(crate) fn compute_client(&self) -> &cubecl::client::ComputeClient<R> {
        &self.client
    }

    /// Queues the denoise kernels for the current window and starts the
    /// readback.
    ///
    /// Returns a [`Pending`] whose `wait()` produces the denoised frame.
    ///
    /// There are two output handles, so a caller can keep two `Pending`s
    /// in flight and let one frame's kernels overlap the previous
    /// frame's readback.
    ///
    /// A third concurrent submit would reuse the oldest pending frame's
    /// output handle and quietly corrupt the results. The high-level
    /// [`crate::Denoiser`] holds callers to that limit through its
    /// `MAX_PENDING` constant.
    ///
    /// The frame comes back in the [`OutputFormat`] this denoiser was
    /// built with. [`OutputFormat::Wire`] quantises and packs the frame
    /// on the GPU before the readback, so only the wire bytes cross the
    /// bus.
    ///
    /// Returns `Ok(None)` while the temporal window is still filling.
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        let Some(output) = self.denoise_submit_gpu()? else {
            return Ok(None);
        };

        // Start the readback right away, so the GPU-side copy is queued
        // before the caller dispatches the next frame's kernels.
        let pixels = (self.width * self.height) as usize;
        Ok(Some(start_readback(
            &self.client,
            output.handle,
            self.wire_outputs.as_ref().map(|w| &w[output.slot]),
            self.params.channels.count(),
            self.params.channels.storage_count(),
            pixels,
            self.output_format,
        )))
    }

    /// The packed-word destinations, which are `Some` only in wire mode.
    #[cfg(test)]
    pub(crate) fn wire_outputs_for_test(&self) -> Option<&[Handle; 2]> {
        self.wire_outputs.as_ref()
    }

    /// The smoothed per-channel sigma estimate NLMeans is currently
    /// filtering with.
    ///
    /// This is `sigma_override` broadcast to every channel when HQ
    /// pinned a fixed sigma. Otherwise it is the median chain's smoothed
    /// estimate once one has landed, and zeros before that first
    /// estimate and on the fast path where no estimate ever runs.
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

    /// The smoothed per-channel sigma estimate from the low chain, with
    /// the correlation boost left out of its temporal reading.
    ///
    /// This is `sigma_override` broadcast to every channel when HQ
    /// pinned a fixed sigma, the same as [`Self::current_sigmas`].
    /// Otherwise it is `noise_estimator_low_unboosted`'s smoothed
    /// estimate once one has landed, and zeros before that first
    /// estimate and on the fast path where no estimate ever runs.
    ///
    /// See the "The estimator chains" section of [`Self::fold_noise_estimate`]
    /// for why a consumer would want this instead of
    /// [`Self::current_sigmas`].
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

    /// The smoothed per-channel sigma estimate from the temporal reading
    /// alone, with no maximum taken against an Immerkær spatial reading
    /// and no correlation boost.
    ///
    /// This is `sigma_override` broadcast to every channel when HQ
    /// pinned a fixed sigma, the same as the other chains. Otherwise it
    /// is `noise_estimator_temporal_only`'s smoothed estimate, once a
    /// fold has arrived with a temporal sample trustworthy enough for
    /// `aggregate_temporal_noise_stats` to produce one.
    ///
    /// Before that first trustworthy reading, this falls back to
    /// [`Self::current_sigmas_low_unboosted`] instead of reading zero,
    /// which would under-filter. Two situations reach that fallback: a
    /// temporal radius of zero, where no temporal sample ever exists
    /// because there is no neighbouring frame to difference against, and
    /// any push where too little of the frame held still for
    /// `aggregate_temporal_noise_stats` to trust its own reading. Once a
    /// trustworthy reading has landed the smoothed estimate keeps going
    /// on later folds that individually lack one, the same way the other
    /// chains keep going between folds.
    ///
    /// See the "The estimator chains" section of [`Self::fold_noise_estimate`]
    /// for why a consumer would want this instead of
    /// [`Self::current_sigmas_low_unboosted`].
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

    /// Refreshes the derived filter parameters from the centre slot's
    /// noise estimate.
    ///
    /// That estimate was queued several pushes ago, when this slot was
    /// first written. See [`Self::run_noise_estimate_for_slot`].
    ///
    /// The blocking read therefore lands on work the GPU has already
    /// finished, rather than stalling the pipeline behind a fresh
    /// dispatch.
    fn update_noise_estimate(&mut self) -> Result<(), anyhow::Error> {
        let results_buf = self
            .noise_results
            .as_ref()
            .expect("noise_results allocated when auto noise is active")
            .clone();

        let bytes = self
            .client
            .read_one(results_buf)
            .map_err(|e| anyhow::anyhow!("noise-estimate results readback failed: {e}"))?;
        let data = f32::from_bytes(&bytes);

        let center_t = self.params.temporal_radius;
        let center_slot = self.phys_frame(center_t as i32) as usize;

        let temporal = self.read_temporal_noise_sample(center_slot as u32)?;
        let imm_low = self.read_noise_partials_low(center_slot as u32)?;

        self.fold_noise_estimate(data, center_slot, temporal, imm_low);

        Ok(())
    }

    /// Reads one ring slot's noise partials back and reduces them to the
    /// low chain's per-channel estimate.
    ///
    /// The shared ring handle is sliced by byte offset, so the transfer
    /// only covers one slot rather than the whole ring. This matches
    /// [`read_temporal_stats_slot`].
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
            .map_err(|e| anyhow::anyhow!("noise partials readback failed: {e}"))?;
        let data = f32::from_bytes(&bytes);

        Ok(sigma_block_p25_from_partials(
            data,
            self.params.channels.count(),
            self.width,
            self.height,
        ))
    }

    /// Reads the centre slot's temporal residual statistics back and
    /// combines them into one sample.
    ///
    /// Returns `None` when the temporal estimator is inactive, which
    /// happens at a temporal radius of 0 or with a fixed sigma, and also
    /// when the combining step itself declines to produce a sample. See
    /// [`aggregate_temporal_noise_stats`].
    fn read_temporal_noise_sample(&self, slot: u32) -> Result<Option<TemporalNoiseSample>, anyhow::Error> {
        let Some(stats_buf) = self.temporal_stats_buf.as_ref() else {
            return Ok(None);
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

        Ok(aggregate_temporal_noise_stats(
            &records,
            channels,
            stored_ch,
            self.width,
            self.height,
        ))
    }

    /// Rebuilds `spatial_offset_lut` from the current `noise_offset` and
    /// `rho_smoothed`.
    ///
    /// This runs once per submit, after `noise_offset` has been
    /// refreshed, so the table and the scalar it comes from never
    /// disagree.
    ///
    /// The rebuild is cheap, covering at most 289 floats.
    fn rebuild_spatial_offset_lut(&mut self) {
        let lut = build_spatial_offset_lut(
            self.params.search_radius,
            self.rho_smoothed.unwrap_or(0.0),
            self.noise_offset,
        );
        self.spatial_offset_lut = self.client.create_from_slice(f32::as_bytes(&lut));
    }

    /// Submits the denoise and waits for it, all in one call.
    ///
    /// Prefer [`Self::denoise_submit`] when the caller can hold a frame
    /// in flight, which lets one frame's kernels overlap the previous
    /// frame's readback.
    ///
    /// Returns `Ok(None)` while not enough frames have been pushed.
    ///
    /// The frame comes back in the [`OutputFormat`] this denoiser was
    /// built with.
    ///
    /// On success the return borrows a reusable internal buffer. Copy it
    /// out if the data has to survive another call into the denoiser.
    pub fn denoise(&mut self) -> Result<Option<&FrameOutput>, anyhow::Error> {
        let Some(pending) = self.denoise_submit()? else {
            return Ok(None);
        };
        // The scratch already holds this denoiser's format, so
        // `wait_into` refills it and it keeps its allocation.
        pending.wait_into(&mut self.output_scratch)?;
        Ok(Some(&self.output_scratch))
    }

    /// How many tail frames a call to [`Self::flush`] must emit for the
    /// stream pushed so far.
    ///
    /// While frames were being pushed the backend produced one output
    /// per push beyond the temporal radius, or none at all if the
    /// stream was shorter than that. This is the remaining difference,
    /// so a caller driving [`Self::flush_step_gpu`] directly knows how
    /// many `Some` results to collect before the stream is fully
    /// drained.
    ///
    /// It reads as zero for spatial mode, where there is no trailing
    /// context to drain, and for a stream that has not pushed anything
    /// yet.
    pub(crate) fn flush_target(&self) -> usize {
        let temporal_radius = self.params.temporal_radius as usize;
        if temporal_radius == 0 || self.real_pushes == 0 {
            0
        } else {
            self.real_pushes.min(temporal_radius)
        }
    }

    /// How many genuine `push_frame`/`push_frame_with_reference` calls
    /// the current stream has seen, not counting the duplicates
    /// [`Self::prime_leading_edge_if_first`] and [`Self::flush`] add at
    /// either end.
    ///
    /// A collaborative stage built on top of [`Self::submit_machinery`]
    /// reads this to size its own end-of-stream drain, the way
    /// [`Self::flush_target`] sizes this front end's.
    pub(crate) fn real_pushes(&self) -> usize {
        self.real_pushes
    }

    /// Runs one step of the end-of-stream drain. It duplicates the most
    /// recently pushed frame forward and submits the window that
    /// results.
    ///
    /// Returns `Ok(None)` while the very first duplicates are still
    /// filling out a window that never reached its full size during
    /// pushing. Every step after the window is full returns `Ok(Some)`,
    /// so a caller has to stop on its own once it has collected
    /// [`Self::flush_target`] outputs, not on seeing `None` again.
    ///
    /// [`Self::flush`] is a loop over this method. A caller that wants
    /// the tail frames to stay on the GPU for further work, rather than
    /// making a round trip through the host, can drive this directly
    /// instead and check its own count against [`Self::flush_target`].
    ///
    /// This assumes there is a frame to duplicate, which means
    /// `flush_target() > 0`. Calling it on spatial mode, or before any
    /// frame has been pushed, duplicates a frame that was never written.
    pub(crate) fn flush_step_gpu(&mut self) -> Result<Option<GpuOutput>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;

        self.duplicate_last_frame();
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }

        self.denoise_submit_gpu()
    }

    /// Produces the frames still held at the end of a stream.
    ///
    /// For the last few frames the temporal window is kept full by
    /// repeating the final frame.
    ///
    /// `sink` is called once per frame produced, and the frame it
    /// receives is only valid for that call. It arrives in the
    /// [`OutputFormat`] this denoiser was built with, quantised by the
    /// same pack kernel as every streaming frame.
    pub fn flush(&mut self, mut sink: impl FnMut(&FrameOutput)) -> Result<(), anyhow::Error> {
        let target = self.flush_target();
        let mut emitted = 0usize;
        let pixels = (self.width * self.height) as usize;

        // Every output slot is free here. A caller reaches a flush only
        // once its streaming readbacks have landed, and the readback
        // below blocks, so no other readback is ever reading the slot
        // this step is handed.
        while emitted < target {
            if let Some(output) = self.flush_step_gpu()? {
                let pending = start_readback(
                    &self.client,
                    output.handle,
                    self.wire_outputs.as_ref().map(|w| &w[output.slot]),
                    self.params.channels.count(),
                    self.params.channels.storage_count(),
                    pixels,
                    self.output_format,
                );
                pending.wait_into(&mut self.output_scratch)?;
                sink(&self.output_scratch);
                emitted += 1;
            }
        }

        // Leave the denoiser ready for a fresh stream of the same
        // shape. The GPU buffers stay allocated and are overwritten one
        // slot at a time as new frames arrive, and
        // `prime_leading_edge_if_first` refills the leading edge as
        // soon as the new stream's first frame lands.
        self.reset_stream_state();

        Ok(())
    }

    /// Resets the stream-tracking indices, so the next push starts a
    /// fresh temporal stream.
    ///
    /// The GPU buffers are deliberately left alone. Like the pyramid and
    /// noise-estimate buffers, the pair ring is always written before it
    /// is read.
    ///
    /// A fresh stream's opening pushes overwrite every slot they touch
    /// before anything reads it, so content from the previous stream is
    /// never seen.
    pub fn reset_stream_state(&mut self) {
        self.ring_head = 0;
        self.frames_loaded = 0;
        self.next_output_slot = 0;
        self.real_pushes = 0;
        self.noise_estimator.reset();
        self.noise_estimator_low.reset();
        self.noise_estimator_low_unboosted.reset();
        self.noise_estimator_temporal_only.reset();
        self.rho_smoothed = None;
    }

    /// The physical slot holding the oldest frame in the window.
    ///
    /// This is only meaningful once a full window has been pushed.
    pub(super) fn ring_start(&self) -> u32 {
        let total_frames = self.params.total_frames() as usize;
        (self.ring_head % total_frames) as u32
    }

    /// Turns a logical frame index within the window into its physical
    /// slot inside `input_buf`.
    pub(super) fn phys_frame(&self, logical: i32) -> u32 {
        let total_frames = self.params.total_frames() as i32;
        let wrapped = logical.rem_euclid(total_frames);
        ((self.ring_start() as i32 + wrapped).rem_euclid(total_frames)) as u32
    }

    /// The pair-ring slot holding the gap between two neighbouring
    /// frames in the window.
    ///
    /// It reduces `ring_head`, the running count of frames pushed
    /// including duplicates, by the pair-ring size rather than by the
    /// window size `Self::phys_frame` uses.
    ///
    /// Two callers reach the same slot for the same physical pair.
    ///
    /// At push time, with a gap index of 0 and `ring_head` still at the
    /// value the frame just written was given, it returns the slot that
    /// frame's pair with its predecessor belongs in.
    ///
    /// At compose time, with `ring_head` already advanced and the gap
    /// index measured out from the window's centre, it returns the slot
    /// an earlier push wrote.
    ///
    /// The two differ only in how far `ring_head` has moved since the
    /// pair was created, and the gap index cancels exactly that much, so
    /// the sum lands on the same slot either way.
    pub(super) fn pair_slot(&self, gap_index: i32) -> u32 {
        let radius = self.params.temporal_radius as i32;
        debug_assert!(
            radius > 0,
            "pair ring is only meaningful when temporal_radius > 0"
        );
        let n = 2 * radius;
        ((self.ring_head as i32 + gap_index).rem_euclid(n)) as u32
    }
}

/// True when every sample in `plane` fits the range `depth` expresses.
///
/// `gpu_unpack_wire` is branch-free and divides every sample by the
/// depth's maximum, so a larger sample normalises above 1.0 and reaches
/// the filter as a value no clean frame can hold. Only 10 and 12-bit can
/// carry one, in the unused high bits of a 16-bit lane, so 8-bit is
/// always in range.
fn wire_samples_in_range(plane: &[u8], depth: Depth) -> bool {
    if depth.bytes_per_sample() == 1 {
        return true;
    }

    let max = depth.max_value() as u32;
    plane
        .as_chunks::<2>()
        .0
        .iter()
        .all(|&s| u32::from(u16::from_le_bytes(s)) <= max)
}
