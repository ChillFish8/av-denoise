//! Stateful NLMeans denoiser: ring-buffer state, per-frame upload,
//! pyramid build, and the public push / recv / flush API. The actual
//! kernel dispatch lives in [`super::dispatch`] in a sibling
//! `impl<R: Runtime> NlmDenoiser<R>` block — fields are `pub(super)`
//! so that file can read them directly.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use super::kernels::gpu_copy;
use super::motion::{self, MotionCtx, build_pyramid_for_slot};
use super::params::{NlmParams, SEPARABLE_THRESHOLD};
use super::pending::{Pending, ReadFuture};
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
    /// `Pending::wait_into` — avoids a per-frame allocation.
    pub(super) output_scratch: Vec<f32>,

    pub(super) h2_inv_norm: f32,
    pub use_separable: bool,
    pub(super) use_reference: bool,

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
    /// Luma-only pyramid storage:
    /// `[pyramid_levels][total_frames][level_w * level_h]` `f32`.
    pub(super) pyramid_input: Option<Handle>,
    /// Same shape as `pyramid_input`, built from the reference ring
    /// when a prefilter is active.
    pub(super) pyramid_reference: Option<Handle>,
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Build a new denoiser. **Panics** if `params.validate()` fails;
    /// the high-level [`crate::Denoiser`] runs validation first and
    /// surfaces errors as `Result`, so most callers should prefer that.
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
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let use_reference = params.prefilter.needs_reference_buf();
        let output_scratch_cap = pixels * params.channels.count() as usize;

        // Motion-compensation buffers. Only allocated when MC is
        // active *and* the temporal window is non-trivial (k=0 path
        // would never touch them).
        let mc_ctx = if params.motion_compensation.is_active() && params.temporal_radius > 0 {
            MotionCtx::new(params.motion_compensation, width, height)
        } else {
            None
        };
        let (compensated_input_buf, compensated_reference_buf, mv_field_buf, pyramid_input, pyramid_reference) =
            if let Some(ctx) = mc_ctx.as_ref() {
                let comp_in = client.empty(frame_bytes * total_frames as usize);
                let comp_ref = if use_reference {
                    Some(client.empty(frame_bytes * total_frames as usize))
                } else {
                    None
                };
                let neighbours = (2 * params.temporal_radius) as usize;
                let mv_field = client.empty(
                    neighbours * ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>(),
                );
                let pyramid_pixels =
                    motion::pyramid_pixels_per_frame(width, height, ctx.pyramid_levels);
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

        Self {
            client: client.clone(),
            params,
            width,
            height,
            ring_head: 0,
            frames_loaded: 0,
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
            use_separable,
            use_reference,
            mc_ctx,
            compensated_input_buf,
            compensated_reference_buf,
            mv_field_buf,
            pyramid_input,
            pyramid_reference,
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

        if self.params.prefilter.is_gpu_internal() {
            self.run_prefilter_for_slot(slot);
        }

        self.build_pyramids_for_slot(slot as u32);

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
    /// when MC is disabled or `pyramid_levels == 1`.
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

        if let (Some(pyr_ref), Some(ref_buf)) = (
            self.pyramid_reference.as_ref(),
            self.reference_buf.as_ref(),
        ) {
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

    fn advance_ring(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        self.ring_head += 1;
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }
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

        if let Some(reference_buf) = self.reference_buf.clone() {
            let ref_src = reference_buf
                .clone()
                .offset_start((last_slot as u64) * bytes_per_slot);
            self.copy_frame_into_slot(&reference_buf, &ref_src, next_slot);
        }

        // Keep the motion-estimation pyramid for the duplicated slot
        // in lockstep so a subsequent denoise sees a valid pyramid for
        // every ring slot it visits.
        self.build_pyramids_for_slot(next_slot as u32);

        self.ring_head += 1;
    }

    /// Queue denoise kernels for the current window and kick off an
    /// async readback. Returns a [`Pending`] whose `wait()` produces the
    /// denoised frame.
    ///
    /// Output handles are double-buffered (`outputs: [Handle; 2]`), so
    /// the caller may keep up to `self.outputs.len()` (= 2) `Pending`s
    /// in flight at once — frame N+1's kernels overlap frame N's
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

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        self.run_denoise_kernels(slot)?;

        // Call `read_async` eagerly so the GPU-side copy is queued before
        // the caller dispatches the next frame's kernels. Erasing the
        // future's inferred `&self` lifetime to `'static` is sound:
        // `read_async` internally returns a `DynFut` that moves all owned
        // data into its state machine and holds Arc-shared device refs,
        // so no actual borrow of `self.client` outlives this call.
        let handle = self.outputs[slot].clone();
        let read_future: Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send + '_>> =
            Box::pin(self.client.read_async(vec![handle]));
        let fut: ReadFuture = unsafe {
            std::mem::transmute::<
                Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send + '_>>,
                ReadFuture,
            >(read_future)
        };

        let pixels = (self.width * self.height) as usize;
        Ok(Some(Pending {
            fut,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            pixels,
            _marker: PhantomData,
        }))
    }

    /// Synchronous convenience wrapper: submits + immediately waits.
    /// Prefer [`Self::denoise_submit`] when the caller can hold one frame
    /// in flight, letting frame N+1's kernels overlap with frame N's
    /// readback.
    ///
    /// Returns `Ok(None)` if not enough frames have been pushed yet.
    /// On success returns `Ok(Some(&[f32]))` borrowing a reusable
    /// internal buffer — copy it out (e.g. `to_vec()`) if you need to
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

        // Partial window: pad with duplicates of the last frame so the
        // temporal neighbourhood is complete, then emit one denoised frame.
        if temporal_radius > 0 && self.frames_loaded < total_frames {
            while self.frames_loaded < total_frames {
                self.duplicate_last_frame();
                self.frames_loaded += 1;
            }
            if let Some(pending) = self.denoise_submit()? {
                pending.wait_into(&mut self.output_scratch)?;
                sink(self.output_scratch.as_slice());
            }
        }

        // Trailing `temporal_radius` frames with shrinking future context,
        // each padded by duplicating the most recent frame.
        for _ in 0..temporal_radius {
            self.duplicate_last_frame();
            if let Some(pending) = self.denoise_submit()? {
                pending.wait_into(&mut self.output_scratch)?;
                sink(self.output_scratch.as_slice());
            }
        }

        Ok(())
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
}
