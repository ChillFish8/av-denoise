use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::gpu_copy;
use super::motion::{self, MotionCtx, MotionEstimation, build_pyramid_for_slot, run_pyramid_build};
use super::noise::{NoiseCtx, NoiseEstimator, partials_len, run_noise_estimate, sigma_from_abs_sum};
use super::params::{NlmParams, SEPARABLE_THRESHOLD, sigma_eff};
use super::pending::Pending;
use super::prefilter::{PrefilterCtx, PrefilterMode, run_prefilter};
use super::{BLOCK_1D, MAX_GRID_1D};

/// Stateful NLMeans denoiser. Maintains a ring of frames in
/// `input_buf`; each `push_frame` uploads one frame, each `denoise`
/// processes the current center frame using its temporal neighbourhood.
pub struct NlmDenoiser<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) params: NlmParams,
    pub(super) width: u32,
    pub(super) height: u32,

    /// Monotonic count of frames pushed; `% total_frames` is the next
    /// physical slot in `input_buf` to overwrite.
    pub(super) ring_head: usize,
    /// Frames loaded so far, capped at `total_frames`.
    pub(super) frames_loaded: usize,
    /// Number of real `push_frame` / `push_frame_with_reference` calls
    /// in the current stream (does not count internal leading/trailing
    /// duplicates). Reset by [`Self::reset_stream_state`].
    pub(super) real_pushes: usize,

    /// `[total_frames * height * width * stored_ch]` ring buffer.
    pub(super) input_buf: Handle,
    /// Reference ring buffer with the same shape as `input_buf`. Used
    /// only when `params.prefilter != None`; supplies the distance
    /// signal for the `_ref` kernel variants.
    pub(super) reference_buf: Option<Handle>,
    /// CPU scratch for YUV-→4-lane repacking. Empty when no padding needed.
    pub(super) padding_scratch: Vec<f32>,
    /// `[pixels * stored_ch]` weighted-pixel accumulator.
    pub(super) accum: Handle,
    /// `[pixels]` total weight per pixel.
    pub(super) weight_sum: Handle,
    /// `[pixels]` max neighbour weight per pixel.
    pub(super) max_weight: Handle,
    /// `[pixels]` weight scratch used by the symmetric (k=0) path.
    pub(super) weight_buf: Handle,
    /// `[pixels]` raw fwd distance (separable path).
    pub(super) raw_fwd: Handle,
    /// `[pixels]` raw bwd distance (separable path).
    pub(super) raw_bwd: Handle,
    /// `[pixels]` hsum intermediate, fwd direction (separable path).
    pub(super) tmp_hsum: Handle,
    /// `[pixels]` hsum intermediate, bwd direction (separable path).
    pub(super) tmp_hsum_bwd: Handle,
    /// Double-buffered `[pixels * stored_ch]` denoised output. A new
    /// `denoise_submit()` writes into `outputs[next_output_slot]` while
    /// the previous slot may still be draining via `read_async`, letting
    /// frame N+1's kernels overlap with frame N's readback.
    pub(super) outputs: [Handle; 2],
    /// Index of the next output slot to write into.
    pub(super) next_output_slot: usize,
    /// CPU scratch reused by the sync `denoise()` path via
    /// `Pending::wait_into`. Avoids a per-frame allocation.
    pub(super) output_scratch: Vec<f32>,

    pub(super) h2_inv_norm: f32,
    /// Distance floor fed to the main-pass weighting kernels. Zero for
    /// `NlmSpatial`, because pilot-vs-pilot distances no longer carry
    /// the 2σ² floor, so subtracting it there would overweight
    /// mismatched patches. Equal to `input_noise_offset` for every
    /// other prefilter mode.
    pub(super) noise_offset: f32,
    /// Distance floor for comparisons against noisy input pixels. The
    /// pilot pass always uses this value, since its own inputs still
    /// carry the full noise floor even when `noise_offset` has been
    /// zeroed for the main pass.
    pub(super) input_noise_offset: f32,
    pub use_separable: bool,
    pub(super) use_reference: bool,

    /// Stage-1 noise-estimate scratch, `[partials_len(width, height)]`
    /// f32s. `Some` only when the noise level is measured
    /// automatically (`hq` is set and `sigma_override` is `None`).
    /// Reused across pushes. Each `run_noise_estimate` call fully
    /// overwrites it before the reduce kernel reads it back.
    pub(super) noise_partials: Option<Handle>,
    /// Per-ring-slot Immerkær totals, `[total_frames * 4]` f32s. Same
    /// gating as `noise_partials`.
    pub(super) noise_results: Option<Handle>,
    /// Smooths the raw per-frame noise estimate into a stable
    /// per-channel sigma. Inert (never updated) when noise is not
    /// measured automatically.
    pub(super) noise_estimator: NoiseEstimator,

    /// Cached motion-compensation context. `Some` when MC is active.
    pub(super) mc_ctx: Option<MotionCtx>,
    /// `[total_frames * height * width * stored_ch]` warped input ring,
    /// matching `input_buf`. Temporal (k≠0) kernels read neighbours
    /// from here; the centre slot is a straight copy of `input_buf`.
    pub(super) compensated_input_buf: Option<Handle>,
    /// Same shape as `compensated_input_buf`, mirroring the reference
    /// ring when a prefilter is active.
    pub(super) compensated_reference_buf: Option<Handle>,
    /// Per-neighbour MV field. Layout:
    /// `[2·temporal_radius][blocks_y * blocks_x * 2]` `i32`. Neighbour
    /// indices `0..R` are the backward k = -R..-1; `R..2R` are forward
    /// k = +1..+R.
    pub(super) mv_field_buf: Option<Handle>,
    /// Adjacent-frame pair-motion ring, laid out
    /// `[2·temporal_radius][2][blocks_y][blocks_x][2]` `i32` (outer
    /// index is the pair slot, next is direction, 0 = older→newer, 1 =
    /// newer→older). `Some` only when `MotionEstimation::Chained` is
    /// active. The direct path never touches this buffer. See
    /// `motion::pair_ring_slot_count` for why `2·temporal_radius` slots
    /// is exactly enough, and `Self::pair_slot` for how a slot is
    /// resolved from a frame's position in the push sequence.
    pub(super) pair_ring_buf: Option<Handle>,
    /// Luma-only pyramid storage:
    /// `[pyramid_levels][total_frames][level_w * level_h]` `f32`.
    pub(super) pyramid_input: Option<Handle>,
    /// Same shape as `pyramid_input`, built from the reference ring
    /// when a prefilter is active.
    pub(super) pyramid_reference: Option<Handle>,

    /// Block geometry for the no-MC confidence pass. `Some` only when
    /// confidence weighting is active (HQ, `temporal_confidence: true`,
    /// `temporal_radius > 0`) and motion compensation is not. The
    /// MC-active case reuses `mc_ctx`'s geometry instead.
    pub(super) confidence_ctx: Option<MotionCtx>,
    /// Per-neighbour block-match confidence. Layout mirrors
    /// `mv_field_buf`, `[2·temporal_radius][blocks_y * blocks_x]`
    /// `f32`. `Some` only when confidence weighting is active (see
    /// `confidence_ctx`), whether the block geometry comes from
    /// `mc_ctx` or from the no-MC confidence pass.
    pub(super) confidence_buf: Option<Handle>,
    /// Luma-only single-level pyramid ring feeding the no-MC
    /// confidence pass. `Some` only alongside `confidence_ctx`.
    pub(super) confidence_pyramid: Option<Handle>,
    /// Discard sink for the no-MC confidence pass's mandatory MV
    /// write. Nothing warps by it without motion compensation. `Some`
    /// only alongside `confidence_ctx`.
    pub(super) confidence_mv_scratch: Option<Handle>,
    /// Small placeholder buffer passed as the fine block-match
    /// kernel's `confidence` argument whenever confidence weighting is
    /// inactive but motion compensation still runs. The kernel's
    /// `write_confidence` comptime flag skips indexing into it
    /// entirely in that case, so its size never matters. Always
    /// allocated (trivially small), unlike the confidence-specific
    /// buffers above.
    pub(super) confidence_dummy: Handle,
    /// Smoothed sigma for channel 0 (the plane motion estimation
    /// treats as luma), feeding the confidence noise floor. Zero
    /// unless HQ is active. A fixed `sigma_override` seeds it once at
    /// construction, auto estimation refreshes it every submit.
    pub(super) sigma_y: f32,
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Build a new denoiser.
    ///
    /// **Panics** if `params.validate()` fails, the high-level [`crate::Denoiser`]
    /// runs validation first and surfaces errors as `Result`, so most callers should prefer that.
    pub fn new(client: &ComputeClient<R>, params: NlmParams, width: u32, height: u32) -> Self {
        params
            .validate()
            .expect("invalid NlmParams; call params.validate() first to surface this as Result");

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

        let h2_inv_norm = params.h2_inv_norm();
        let input_noise_offset = params.noise_offset();
        // The pilot pass compares noisy input pixels, so it always
        // keeps the full noise floor. Main-pass distances under
        // `NlmSpatial` are pilot-vs-pilot, which no longer carries
        // that floor, so subtracting it there would overweight
        // mismatched patches.
        let noise_offset = match params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => input_noise_offset,
        };
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let use_reference = params.prefilter.needs_reference_buf();
        let output_scratch_cap = pixels * params.channels.count() as usize;

        // Auto noise estimation only runs when HQ is on and the caller
        // hasn't pinned a fixed sigma. The fast path and the
        // sigma-override path allocate neither buffer nor ever launch
        // the estimate kernels.
        let auto_noise = params.hq.is_some_and(|hq| hq.sigma_override.is_none());
        let (noise_partials, noise_results) = if auto_noise {
            let n_partials = partials_len(width, height);
            let n_results = (total_frames * 4) as usize;
            (
                Some(client.empty(n_partials * size_of::<f32>())),
                Some(client.empty(n_results * size_of::<f32>())),
            )
        } else {
            (None, None)
        };

        // Motion-compensation buffers. Only allocated when MC is
        // active *and* the temporal window is non-trivial (k=0 path
        // would never touch them).
        let mc_ctx = if params.motion_compensation.is_active() && params.temporal_radius > 0 {
            MotionCtx::new(params.motion_compensation, width, height)
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
            let neighbours = (2 * params.temporal_radius) as usize;
            let mv_field = client.empty(neighbours * ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>());
            let pyramid_pixels = motion::pyramid_pixels_per_frame(width, height, ctx.pyramid_levels);
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
        // active (explicitly, or via `Auto` resolving to it at this
        // temporal radius), on top of `mc_ctx` already being `Some`.
        // The direct path never reads or writes it.
        let is_chained = matches!(
            params
                .motion_compensation
                .resolved_estimation(params.temporal_radius),
            Some(MotionEstimation::Chained { .. })
        );
        let pair_ring_buf = if is_chained {
            mc_ctx.as_ref().map(|ctx| {
                let pair_ring_slots = motion::pair_ring_slot_count(params.temporal_radius) as usize;
                let bytes = pair_ring_slots * ctx.pair_slot_len() as usize * size_of::<i32>();
                client.empty(bytes)
            })
        } else {
            None
        };

        // Confidence weighting (in either its MC-active or its no-MC
        // form) is active only when HQ has `temporal_confidence: true`
        // and the temporal window is non-trivial. This gate applies
        // even when MC is active. Without it, every MC-active submit
        // would pay for the fine kernel's confidence write whether or
        // not anything consumes it.
        let confidence_active =
            params.hq.is_some_and(|hq| hq.temporal_confidence) && params.temporal_radius > 0;

        // Confidence-only geometry, only needed when MC isn't already
        // supplying block geometry (and thus MVs and confidence via
        // its own analyse pass). This incurs real extra work beyond
        // the MC-active case. It needs its own luma pyramid ring and a
        // block-match kernel per neighbour.
        let confidence_only_active = confidence_active && mc_ctx.is_none();
        let confidence_ctx = confidence_only_active.then(|| MotionCtx::confidence_only(width, height));

        // The confidence buffer piggybacks on whichever block geometry
        // is available, but only when confidence weighting is active.
        let confidence_geometry = if confidence_active {
            mc_ctx.as_ref().or(confidence_ctx.as_ref())
        } else {
            None
        };
        let confidence_buf = confidence_geometry.map(|ctx| {
            let neighbours = (2 * params.temporal_radius) as u64;
            client.empty((neighbours * ctx.confidence_bytes_per_neighbour()) as usize)
        });
        // Always allocated, trivially small, and reused whenever the
        // fine block-match kernel runs with `write_confidence: false`.
        let confidence_dummy = client.empty(size_of::<f32>());

        let (confidence_pyramid, confidence_mv_scratch) = if let Some(ctx) = confidence_ctx.as_ref() {
            let pyramid_pixels = motion::pyramid_pixels_per_frame(width, height, ctx.pyramid_levels);
            let pyr_bytes = pyramid_pixels * total_frames as usize * size_of::<f32>();
            let mv_scratch_len = ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>();
            (Some(client.empty(pyr_bytes)), Some(client.empty(mv_scratch_len)))
        } else {
            (None, None)
        };

        // `sigma_override` is the only source before the first noise
        // estimate lands. Auto estimation refreshes this every submit
        // (see `update_noise_estimate`). The fast path (`hq: None`)
        // leaves it at zero, which `motion::sad_noise_floor` turns into
        // a zero floor exactly as callers with no estimate should get.
        let sigma_y = params.hq.and_then(|hq| hq.sigma_override).unwrap_or(0.0);

        Self {
            client: client.clone(),
            params,
            width,
            height,
            ring_head: 0,
            frames_loaded: 0,
            real_pushes: 0,
            input_buf,
            reference_buf,
            padding_scratch,
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
            output_scratch: Vec::with_capacity(output_scratch_cap),
            h2_inv_norm,
            noise_offset,
            input_noise_offset,
            use_separable,
            use_reference,
            noise_partials,
            noise_results,
            noise_estimator: NoiseEstimator::default(),
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

    /// Push a new frame into the ring buffer. `frame` must hold
    /// `width * height * channels` f32 values normalised to [0, 1].
    /// YUV padding (3→4 lanes) is repacked through a reused CPU scratch.
    ///
    /// For `PrefilterMode::External` use
    /// [`Self::push_frame_with_reference`] instead.
    pub fn push_frame(&mut self, frame: &[f32]) {
        assert!(
            !matches!(self.params.prefilter, PrefilterMode::External),
            "push_frame_with_reference is required when prefilter == External"
        );

        let slot = self.upload_into(&self.input_buf.clone(), frame);

        self.run_noise_estimate_for_slot(slot as u32);
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

    /// Push a new frame together with an externally-prefiltered
    /// reference. Required when `prefilter == External`; both slices
    /// must hold `width * height * channels` f32 values in [0, 1].
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
        self.build_pyramids_for_slot(slot as u32);
        self.build_confidence_pyramid_for_slot(slot as u32);
        self.run_pair_analyse_for_slot(slot as u32);
        self.run_noise_estimate_for_slot(slot as u32);
        self.advance_ring();
        self.prime_leading_edge_if_first();
    }

    /// Upload `frame` into the next ring slot of `dst`. Returns the
    /// physical slot index written.
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

        self.copy_frame_into_slot(dst, &staging, slot);
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

    /// Build the per-frame motion-estimation pyramid for `slot` on
    /// both the input and (when present) the reference rings. No-op
    /// when MC is disabled.
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

    /// Extract the level-0 luma plane for `slot` into the no-MC
    /// confidence pyramid. No-op unless the no-MC confidence pass is
    /// active. Always reads `input_buf`, even under a prefilter. The
    /// no-MC path keeps confidence simple by comparing raw input
    /// rather than duplicating the reference ring's pyramid.
    ///
    /// Calls `run_pyramid_build` directly rather than going through
    /// [`Self::build_pyramids_for_slot`]'s `build_pyramid_for_slot`
    /// helper. That helper only ever touches `mc_ctx`'s own pyramid
    /// buffers (`pyramid_input`, `pyramid_reference`), and
    /// `confidence_ctx` is only `Some` when `mc_ctx` is `None`, so it
    /// would return immediately without ever building this method's
    /// own `confidence_pyramid`.
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

    /// Whether `Chained` motion estimation is configured, explicitly or
    /// via `Auto` resolving to it at this denoiser's temporal radius
    /// (see `resolved_estimation`, the single source every
    /// estimation-dependent decision goes through). Orthogonal to
    /// `mc_ctx.is_some()`, which callers still need to check
    /// separately, since `mc_ctx` also requires `temporal_radius > 0`.
    pub(super) fn is_chained(&self) -> bool {
        matches!(
            self.params
                .motion_compensation
                .resolved_estimation(self.params.temporal_radius),
            Some(MotionEstimation::Chained { .. })
        )
    }

    /// Run the adjacent-frame pair analyse for the physical input-ring
    /// slot just written by `push_frame`/`push_frame_with_reference`,
    /// storing both directions' motion fields into the pair ring at
    /// `Self::pair_slot(0)`. No-op unless `Chained` estimation is
    /// active, and for the very first frame of a stream (`ring_head ==
    /// 0`), which has no older partner to pair against. Composition
    /// for that gap instead reads the priming duplicate's zero-filled
    /// pair (see [`Self::zero_pair_slot_for_duplicate`]).
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

        // Use the cleaner of the two buffers for motion estimation,
        // exactly as `run_motion_compensation` does for the direct path.
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

    /// Zero-fill the pair-ring slot for a duplicated ring slot (stream
    /// priming or end-of-stream flush). No-op unless `Chained`
    /// estimation is active.
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

    /// Queue the Immerkær noise estimate for `slot` on the input ring.
    /// No-op unless auto noise estimation is active. The read of these
    /// results normally happens later in [`Self::denoise_submit`], once
    /// `slot` reaches the centre of the temporal window. The stream's
    /// very first frame also gets an immediate read, see
    /// [`Self::seed_noise_estimate_if_first_frame`].
    fn run_noise_estimate_for_slot(&self, slot: u32) {
        let (Some(partials_buf), Some(results_buf)) =
            (self.noise_partials.as_ref(), self.noise_results.as_ref())
        else {
            return;
        };

        let ctx = NoiseCtx {
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            frame_count: self.params.total_frames(),
            frame: slot,
            slot,
            input_buf: &self.input_buf,
            partials_buf,
            results_buf,
        };

        run_noise_estimate::<R>(&self.client, &ctx).expect("noise estimate dispatch failed");
    }

    /// One-time σ bootstrap for the very first frame of a stream. Auto
    /// noise estimation normally updates `h2_inv_norm` / `noise_offset`
    /// / `input_noise_offset` from [`Self::update_noise_estimate`] at
    /// submit time, but any push-time GPU work that reads them (the
    /// nlm-spatial pilot) runs before the first submit ever happens.
    /// Without this, that work would run on the absolute-strength
    /// fallback set at construction for every frame up to the first
    /// submit. One blocking read of the estimate this push just queued
    /// for `slot` fixes that from frame one onward. Detects "first
    /// frame of the stream" from `frames_loaded`, the same counter
    /// [`Self::prime_leading_edge_if_first`] checks, but reads it here
    /// before [`Self::advance_ring`] increments it, and applies for
    /// every `temporal_radius` rather than only when priming happens.
    /// The first submit's [`Self::update_noise_estimate`] folds the
    /// same frame's estimate into the EMA a second time, which only
    /// reproduces this seed's values up to floating-point rounding,
    /// not bit-exactly.
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

        let channels = self.params.channels.count() as usize;
        let base = slot as usize * 4;

        let mut raw = [0.0f32; 3];
        for (c, s) in raw.iter_mut().enumerate().take(channels) {
            *s = sigma_from_abs_sum(data[base + c], self.width, self.height);
        }

        let updated = self.noise_estimator.update(&raw[..channels]);
        let mut smoothed = [0.0f32; 3];
        smoothed[..channels].copy_from_slice(updated);

        let eff = sigma_eff(&smoothed[..channels], self.params.channels);
        self.h2_inv_norm = self.params.h2_inv_norm_with(Some(eff));
        self.input_noise_offset = self.params.noise_offset_with(Some(&smoothed[..channels]));
        self.noise_offset = match self.params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => self.input_noise_offset,
        };
        // Channel 0 is whatever motion estimation already treats as
        // luma (see `nlm_mc_extract_luma`), so the confidence floor
        // uses the same plane's noise estimate.
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

    /// GPU→GPU copy of one frame from `src` into `dst` at the given
    /// physical slot. `dst` must have ring-buffer layout matching
    /// `input_buf` (`total_frames * height * width * stored_ch`).
    fn copy_frame_into_slot(&self, dst: &Handle, src: &Handle, slot: usize) {
        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let byte_offset = (slot as u64) * (frame_size as u64) * (size_of::<f32>() as u64);
        let dst_handle = dst.clone().offset_start(byte_offset);

        let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        unsafe {
            gpu_copy::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(src.clone(), frame_size as usize),
                ArrayArg::from_raw_parts(dst_handle.clone(), frame_size as usize),
                frame_size,
                total_threads,
            )
        };
    }

    /// Mirror the very first pushed frame into the `R` leading ring
    /// slots so the temporal window starts symmetric instead of dropping
    /// the first `R` logical frames. Mirrors the trailing-edge logic in
    /// [`Self::flush`].
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

    /// Duplicate the most recently pushed frame into the next ring slot.
    /// Used at end-of-stream to keep the window full while future
    /// context shrinks. Slots never overlap, so the in-buffer copy is
    /// well-defined. The reference ring is duplicated in lockstep when
    /// active, so weight calculation never falls back to a stale slot.
    fn duplicate_last_frame(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        let last_slot = (self.ring_head - 1) % total_frames;
        let next_slot = self.ring_head % total_frames;

        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let bytes_per_slot = (frame_size as u64) * (size_of::<f32>() as u64);

        let input_src = self
            .input_buf
            .clone()
            .offset_start((last_slot as u64) * bytes_per_slot);
        self.copy_frame_into_slot(&self.input_buf.clone(), &input_src, next_slot);

        // Skipped for `NlmSpatial`: the pilot dispatch below recomputes
        // this slot's reference from scratch, so the byte copy would
        // just be overwritten immediately.
        if !matches!(self.params.prefilter, PrefilterMode::NlmSpatial { .. })
            && let Some(reference_buf) = self.reference_buf.clone()
        {
            let ref_src = reference_buf
                .clone()
                .offset_start((last_slot as u64) * bytes_per_slot);
            self.copy_frame_into_slot(&reference_buf, &ref_src, next_slot);
        }

        // Keep the motion-estimation pyramid and noise estimate for the
        // duplicated slot in lockstep so a subsequent denoise sees valid
        // state for every ring slot it visits, not whatever an older
        // frame left behind at this physical position. The nlm-spatial
        // pilot needs the same treatment, otherwise the duplicated
        // slot's reference would keep whatever an older frame at this
        // physical position last wrote there.
        if let PrefilterMode::NlmSpatial { strength_scale } = self.params.prefilter {
            self.run_nlm_spatial_pilot(next_slot as u32, strength_scale)
                .expect("nlm spatial pilot dispatch failed");
        }
        self.build_pyramids_for_slot(next_slot as u32);
        self.build_confidence_pyramid_for_slot(next_slot as u32);
        self.run_noise_estimate_for_slot(next_slot as u32);
        // Runs before `ring_head` advances, so `pair_slot(0)` reads the
        // same pre-advance `ring_head` as `run_pair_analyse_for_slot`
        // (see `Self::pair_slot`).
        self.zero_pair_slot_for_duplicate();

        self.ring_head += 1;
    }

    /// Queue denoise kernels for the current window and kick off an
    /// async readback. Returns a [`Pending`] whose `wait()` produces the
    /// denoised frame.
    ///
    /// Output handles are double-buffered (`outputs: [Handle; 2]`), so
    /// the caller may keep up to `self.outputs.len()` (= 2) `Pending`s
    /// in flight at once, so frame N+1's kernels overlap frame N's
    /// readback. A third concurrent submit would alias the oldest
    /// pending's output handle and silently corrupt results, so the
    /// high-level [`crate::Denoiser`] enforces that cap via its
    /// `MAX_PENDING` constant.
    ///
    /// Returns `Ok(None)` if the temporal window is not yet filled.
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        if self.noise_results.is_some() {
            self.update_noise_estimate()?;
        }

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        self.run_denoise_kernels(slot)?;

        // Call `read_async` eagerly so the GPU-side copy is queued before
        // the caller dispatches the next frame's kernels. The future is
        // wrapped in an `async move` that owns a cloned `ComputeClient`
        // (cheap: it's `Arc`-shared internally). That owned client lives
        // inside the future's state machine, so the resulting future is
        // genuinely `'static` and the `Pending` may outlive the
        // `NlmDenoiser` without any lifetime gymnastics.
        let handle = self.outputs[slot].clone();
        let client = self.client.clone();
        let fut = Box::pin(async move { client.read_async(vec![handle]).await });

        let pixels = (self.width * self.height) as usize;
        Ok(Some(Pending {
            fut,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            pixels,
            _marker: PhantomData,
        }))
    }

    /// Refresh `h2_inv_norm` / `noise_offset` from the centre slot's
    /// noise estimate. The centre slot's estimate was queued
    /// `temporal_radius` pushes earlier (see
    /// [`Self::run_noise_estimate_for_slot`]), so this blocking read
    /// lands on work the GPU already finished instead of stalling the
    /// pipeline behind a fresh dispatch.
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
        let channels = self.params.channels.count() as usize;
        let base = center_slot * 4;

        let mut raw = [0.0f32; 3];
        for (c, slot) in raw.iter_mut().enumerate().take(channels) {
            *slot = sigma_from_abs_sum(data[base + c], self.width, self.height);
        }

        let updated = self.noise_estimator.update(&raw[..channels]);
        let mut smoothed = [0.0f32; 3];
        smoothed[..channels].copy_from_slice(updated);

        let eff = sigma_eff(&smoothed[..channels], self.params.channels);
        self.h2_inv_norm = self.params.h2_inv_norm_with(Some(eff));
        self.input_noise_offset = self.params.noise_offset_with(Some(&smoothed[..channels]));
        self.noise_offset = match self.params.prefilter {
            PrefilterMode::NlmSpatial { .. } => 0.0,
            _ => self.input_noise_offset,
        };
        self.sigma_y = smoothed[0];

        Ok(())
    }

    /// Synchronous convenience wrapper: submits + immediately waits.
    /// Prefer [`Self::denoise_submit`] when the caller can hold one frame
    /// in flight, letting frame N+1's kernels overlap with frame N's
    /// readback.
    ///
    /// Returns `Ok(None)` if not enough frames have been pushed yet.
    /// On success returns `Ok(Some(&[f32]))` borrowing a reusable
    /// internal buffer; copy it out (e.g. `to_vec()`) if you need to
    /// hold the data across another `denoise`/`flush`/`push_frame` call.
    pub fn denoise(&mut self) -> Result<Option<&[f32]>, anyhow::Error> {
        let Some(pending) = self.denoise_submit()? else {
            return Ok(None);
        };
        pending.wait_into(&mut self.output_scratch)?;
        Ok(Some(self.output_scratch.as_slice()))
    }

    /// Flush remaining frames at end-of-stream. For the last `d` frames
    /// the temporal window is clamped by duplicating the last frame.
    /// `sink` is invoked once per produced frame; the borrowed slice is
    /// only valid for that call.
    pub fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        let temporal_radius = self.params.temporal_radius as usize;
        let total_frames = self.params.total_frames() as usize;

        // Spatial mode has no trailing context to drain.
        if temporal_radius == 0 || self.real_pushes == 0 {
            self.reset_stream_state();
            return Ok(());
        }

        // During pushes the backend submits `real_pushes - R` denoises
        // (zero when `real_pushes <= R`). flush must produce the
        // remainder so the caller gets exactly `real_pushes` outputs.
        let target = self.real_pushes.min(temporal_radius);
        let mut emitted = 0usize;

        // Partial window: pad with trailing duplicates of the last
        // pushed frame so the temporal neighbourhood is complete, then
        // emit centred denoises. Each padded step shifts the centre
        // forward by one, so we may emit several outputs before
        // crossing into the regular trailing-tail loop.
        while self.frames_loaded < total_frames && emitted < target {
            self.duplicate_last_frame();
            self.frames_loaded += 1;
            if self.frames_loaded == total_frames
                && let Some(pending) = self.denoise_submit()?
            {
                pending.wait_into(&mut self.output_scratch)?;
                sink(self.output_scratch.as_slice());
                emitted += 1;
            }
        }

        // Trailing window: full ring, shrinking future context. Each
        // iteration duplicates the most recent frame and emits one more
        // centred denoise.
        while emitted < target {
            self.duplicate_last_frame();
            if let Some(pending) = self.denoise_submit()? {
                pending.wait_into(&mut self.output_scratch)?;
                sink(self.output_scratch.as_slice());
                emitted += 1;
            }
        }

        // Leave the denoiser ready for a fresh stream of the same shape.
        // GPU buffers stay allocated, they get overwritten slot-by-slot
        // as new frames arrive, and `prime_leading_edge_if_first` re-fills
        // the leading edge once `frames_loaded == 1` on the new stream.
        self.reset_stream_state();

        Ok(())
    }

    /// Reset stream-tracking indices so the next `push_frame` begins a
    /// fresh temporal stream. GPU buffers are intentionally not cleared.
    /// `pair_ring_buf` relies on the same write-before-read convention as
    /// the pyramid and noise-estimate buffers. A fresh stream's first
    /// pushes fully overwrite every slot they touch before anything
    /// reads it back, so leftover content from the previous stream is
    /// never observed.
    pub(crate) fn reset_stream_state(&mut self) {
        self.ring_head = 0;
        self.frames_loaded = 0;
        self.next_output_slot = 0;
        self.real_pushes = 0;
        self.noise_estimator.reset();
    }

    /// Physical slot of logical frame 0 (oldest frame in the window).
    /// Defined only once a full window has been pushed.
    pub(super) fn ring_start(&self) -> u32 {
        let total_frames = self.params.total_frames() as usize;
        (self.ring_head % total_frames) as u32
    }

    /// Resolve a logical frame index in `[0, total_frames)` to its
    /// physical slot inside `input_buf`.
    pub(super) fn phys_frame(&self, logical: i32) -> u32 {
        let total_frames = self.params.total_frames() as i32;
        let wrapped = logical.rem_euclid(total_frames);
        ((self.ring_start() as i32 + wrapped).rem_euclid(total_frames)) as u32
    }

    /// Pair-ring slot for the gap between window-relative logical
    /// frames `gap_index` and `gap_index + 1`. Reduces the current
    /// `ring_head`, the monotonic count of frames pushed so far
    /// (including duplicates), by `2 * temporal_radius` instead of the
    /// `total_frames` modulus `Self::phys_frame` uses for the frame
    /// ring.
    ///
    /// Called two ways that resolve to the same slot for the same
    /// physical pair. At push time, with `gap_index = 0` and the
    /// pre-advance `ring_head` (the generation of the frame just
    /// written), it gives the slot that frame's pair with its
    /// immediate predecessor belongs in. At compose time, with the
    /// post-push `ring_head` and `gap_index` measured from the
    /// window's centre, it gives the slot a past push already wrote
    /// to. The two calls only differ in how far `ring_head` has
    /// advanced since the pair was created, and `gap_index` exactly
    /// offsets that advance, so `ring_head + gap_index` lands on the
    /// same value mod `2 * temporal_radius` either way.
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
