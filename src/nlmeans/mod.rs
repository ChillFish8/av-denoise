pub mod kernels;
pub mod prefilter;

#[cfg(test)]
mod tests;

use cubecl::prelude::*;
use cubecl::server::Handle;

use self::kernels::{
    gpu_copy,
    gpu_zero_buffers,
    nlm_accumulate,
    nlm_dist_2d_weight,
    nlm_dist_2d_weight_ref,
    nlm_distance,
    nlm_distance_pair,
    nlm_distance_pair_ref,
    nlm_distance_ref,
    nlm_finish,
    nlm_fused_pair_accumulate,
    nlm_fused_pair_accumulate_ref,
    nlm_horizontal_sum,
    nlm_horizontal_sum_pair,
    nlm_vertical_weight,
    nlm_vweight_pair_accumulate,
};
pub use self::prefilter::PrefilterMode;
use self::prefilter::{PrefilterCtx, run_prefilter};

pub const BLOCK_X: u32 = 32;
pub const BLOCK_Y: u32 = 8;

const NLM_NORM: f32 = 255.0 * 255.0;
const NLM_LEGACY: f32 = 3.0;

/// Patch radius threshold: use the separable path above this value.
const SEPARABLE_THRESHOLD: u32 = 2;

/// Maximum 1D grid size for GPU dispatch (WebGPU/Vulkan limit).
const MAX_GRID_1D: u32 = 65535;

/// Block size for 1D utility kernels (copy, zero).
const BLOCK_1D: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    /// Single luminance channel. Distance scaled by 3.0.
    Luma,
    /// Two chroma channels (U, V). Distance scaled by 1.5.
    Chroma,
    /// Three channels (Y, U, V). Unscaled sum of squared differences.
    Yuv,
}

impl ChannelMode {
    /// Number of meaningful channels participating in distance/output.
    pub fn count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 3,
        }
    }

    /// Channels-per-pixel in GPU storage. Padded up to the next supported
    /// vectorization factor so kernels can use coalesced `Line<f32>` reads
    /// (backends only support power-of-two line sizes; YUV pads to 4).
    pub fn storage_count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NlmParams {
    /// Temporal radius. 0 = spatial only, d > 0 uses 2*d+1 frames.
    pub temporal_radius: u32,
    /// Search window half-size. Search window is (2*a+1)^2. Default: 2.
    pub search_radius: u32,
    /// Patch comparison half-size. Patch is (2*s+1)^2. Default: 4, range [0, 8].
    pub patch_radius: u32,
    /// Filtering strength. Higher = more smoothing. Default: 1.2.
    pub strength: f32,
    /// Self-weight multiplier. Default: 1.0. Set to 0 for pure NLM.
    pub self_weight: f32,
    /// Which channels to process.
    pub channels: ChannelMode,
    /// Reference clip source used for patch-distance / weight
    /// computation. Default: `None`. When set, weights are derived
    /// from a prefiltered or externally-supplied clip while pixel
    /// accumulation continues to read the original input.
    pub prefilter: PrefilterMode,
}

impl Default for NlmParams {
    fn default() -> Self {
        Self {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Yuv,
            prefilter: PrefilterMode::None,
        }
    }
}

impl NlmParams {
    pub fn h2_inv_norm(&self) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);
        NLM_NORM / (NLM_LEGACY * self.strength * self.strength * s_size as f32)
    }

    fn total_frames(&self) -> u32 {
        1 + 2 * self.temporal_radius
    }
}

/// Stateful NLMeans denoiser. Maintains a ring of frames in
/// `input_buf`; each `push_frame` uploads one frame, each `denoise`
/// processes the current center frame using its temporal neighbourhood.
pub struct NlmDenoiser<R: Runtime> {
    client: ComputeClient<R>,
    params: NlmParams,
    width: u32,
    height: u32,

    /// Monotonic count of frames pushed; `% total_frames` is the next
    /// physical slot in `input_buf` to overwrite.
    ring_head: usize,
    /// Frames loaded so far, capped at `total_frames`.
    frames_loaded: usize,

    /// `[total_frames * height * width * stored_ch]` ring buffer.
    input_buf: Handle,
    /// Reference ring buffer with the same shape as `input_buf`. Used
    /// only when `params.prefilter != None`; supplies the distance
    /// signal for the `_ref` kernel variants.
    reference_buf: Option<Handle>,
    /// CPU scratch for YUV-→4-lane repacking. Empty when no padding needed.
    padding_scratch: Vec<f32>,
    /// `[pixels * stored_ch]` weighted-pixel accumulator.
    accum: Handle,
    /// `[pixels]` total weight per pixel.
    weight_sum: Handle,
    /// `[pixels]` max neighbour weight per pixel.
    max_weight: Handle,
    /// `[pixels]` weight scratch used by the symmetric (k=0) path.
    weight_buf: Handle,
    /// `[pixels]` raw fwd distance (separable path).
    raw_fwd: Handle,
    /// `[pixels]` raw bwd distance (separable path).
    raw_bwd: Handle,
    /// `[pixels]` hsum intermediate, fwd direction (separable path).
    tmp_hsum: Handle,
    /// `[pixels]` hsum intermediate, bwd direction (separable path).
    tmp_hsum_bwd: Handle,
    /// `[pixels * stored_ch]` denoised output.
    output: Handle,
    /// CPU scratch reused for the readback Vec — avoids a per-frame alloc.
    output_scratch: Vec<f32>,

    h2_inv_norm: f32,
    use_separable: bool,
    use_reference: bool,
}

/// Derived sizes plus dispatch shape for the per-frame work, bundled so
/// the dispatch helpers don't carry long parallel argument lists.
struct LaunchCtx {
    total_frame_data: usize,
    frame_size: usize,
    pixels: usize,
    cube_count: CubeCount,
    cube_dim: CubeDim,
}

impl<R: Runtime> NlmDenoiser<R> {
    pub fn new(client: &ComputeClient<R>, params: NlmParams, width: u32, height: u32) -> Self {
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
        let output = client.empty(frame_bytes);

        let h2_inv_norm = params.h2_inv_norm();
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let use_reference = params.prefilter.needs_reference_buf();
        let output_scratch_cap = pixels * params.channels.count() as usize;

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
            output,
            output_scratch: Vec::with_capacity(output_scratch_cap),
            h2_inv_norm,
            use_separable,
            use_reference,
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

        self.advance_ring();
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
        self.advance_ring();
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

        gpu_copy::launch::<R>(
            &self.client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(src, frame_size as usize, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&dst_handle, frame_size as usize, 1) },
            frame_size,
            total_threads,
        )
        .expect("gpu_copy launch failed");
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

        self.ring_head += 1;
    }

    /// Try to denoise the current center frame.
    ///
    /// Returns `Ok(None)` if not enough frames have been pushed yet.
    /// On success returns `Ok(Some(&[f32]))` borrowing a reusable
    /// internal buffer — copy it out (e.g. `to_vec()`) if you need to
    /// hold the data across another `denoise`/`flush`/`push_frame` call.
    pub fn denoise(&mut self) -> Result<Option<&[f32]>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        self.run_denoise_kernels()?;
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
            self.run_denoise_kernels()?;
            sink(self.output_scratch.as_slice());
        }

        // Trailing `temporal_radius` frames with shrinking future context,
        // each padded by duplicating the most recent frame.
        for _ in 0..temporal_radius {
            self.duplicate_last_frame();
            self.run_denoise_kernels()?;
            sink(self.output_scratch.as_slice());
        }

        Ok(())
    }

    /// Physical slot of logical frame 0 (oldest frame in the window).
    /// Defined only once a full window has been pushed.
    fn ring_start(&self) -> u32 {
        let total_frames = self.params.total_frames() as usize;
        (self.ring_head % total_frames) as u32
    }

    /// Resolve a logical frame index in `[0, total_frames)` to its
    /// physical slot inside `input_buf`.
    fn phys_frame(&self, logical: i32) -> u32 {
        let total_frames = self.params.total_frames() as i32;
        let wrapped = logical.rem_euclid(total_frames);
        ((self.ring_start() as i32 + wrapped).rem_euclid(total_frames)) as u32
    }

    fn input_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        let stored_ch = self.params.channels.storage_count() as usize;
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.input_buf, ctx.total_frame_data, stored_ch) }
    }

    fn reference_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        let stored_ch = self.params.channels.storage_count() as usize;
        let buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist when use_reference is set");
        unsafe { ArrayArg::from_raw_parts::<f32>(buf, ctx.total_frame_data, stored_ch) }
    }

    fn accum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        let stored_ch = self.params.channels.storage_count() as usize;
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, ctx.frame_size, stored_ch) }
    }

    fn output_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        let stored_ch = self.params.channels.storage_count() as usize;
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.output, ctx.frame_size, stored_ch) }
    }

    fn weight_sum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, ctx.pixels, 1) }
    }

    fn max_weight_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, ctx.pixels, 1) }
    }

    fn weight_buf_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_buf, ctx.pixels, 1) }
    }

    fn raw_fwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.raw_fwd, ctx.pixels, 1) }
    }

    fn raw_bwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.raw_bwd, ctx.pixels, 1) }
    }

    fn tmp_hsum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.tmp_hsum, ctx.pixels, 1) }
    }

    fn tmp_hsum_bwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<f32>(&self.tmp_hsum_bwd, ctx.pixels, 1) }
    }

    /// Temporal (k≠0) fused-path step: one launch that computes both
    /// weights in registers and applies the +q / −q contributions.
    fn dispatch_fused_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);
        // The backward distance compares against a different neighbour
        // depending on whether the temporal offset is zero. For k=0 the
        // pair collapses to a symmetric (+q, +q) self-comparison; for
        // k≠0 the true (+q, −q) cross-frame comparison applies.
        let (bwd_shift_x, bwd_shift_y) = if q_k == 0 { (q_x, q_y) } else { (-q_x, -q_y) };

        if self.use_reference {
            nlm_fused_pair_accumulate_ref::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.input_arg(ctx),
                self.reference_arg(ctx),
                self.accum_arg(ctx),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_fwd),
                ScalarArg::new(frame_bwd),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                ScalarArg::new(bwd_shift_x),
                ScalarArg::new(bwd_shift_y),
                ScalarArg::new(self.h2_inv_norm),
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            )?;
        } else {
            nlm_fused_pair_accumulate::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.input_arg(ctx),
                self.accum_arg(ctx),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_fwd),
                ScalarArg::new(frame_bwd),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                ScalarArg::new(bwd_shift_x),
                ScalarArg::new(bwd_shift_y),
                ScalarArg::new(self.h2_inv_norm),
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            )?;
        }
        Ok(())
    }

    /// Temporal (k≠0) separable-path step: distance_pair →
    /// horizontal_sum_pair → fused vweight+accumulate. The fused
    /// terminal kernel consumes both hsum buffers, so no global weight
    /// buffer is written.
    fn dispatch_separable_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);

        if self.use_reference {
            nlm_distance_pair_ref::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.reference_arg(ctx),
                self.raw_fwd_arg(ctx),
                self.raw_bwd_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_fwd),
                ScalarArg::new(frame_bwd),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                self.width,
                self.height,
                channels,
            )?;
        } else {
            nlm_distance_pair::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.input_arg(ctx),
                self.raw_fwd_arg(ctx),
                self.raw_bwd_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_fwd),
                ScalarArg::new(frame_bwd),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                self.width,
                self.height,
                channels,
            )?;
        }

        nlm_horizontal_sum_pair::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.raw_fwd_arg(ctx),
            self.raw_bwd_arg(ctx),
            self.tmp_hsum_arg(ctx),
            self.tmp_hsum_bwd_arg(ctx),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        nlm_vweight_pair_accumulate::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.tmp_hsum_arg(ctx),
            self.tmp_hsum_bwd_arg(ctx),
            self.input_arg(ctx),
            self.accum_arg(ctx),
            self.weight_sum_arg(ctx),
            self.max_weight_arg(ctx),
            ScalarArg::new(frame_fwd),
            ScalarArg::new(frame_bwd),
            ScalarArg::new(q_x),
            ScalarArg::new(q_y),
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;
        Ok(())
    }

    /// Spatial (k=0) fused-path step: single-tile weight + accumulate.
    /// Cheaper than the paired fused kernel here because the weight
    /// map is symmetric, so a single tile is enough.
    fn dispatch_fused_iter_k0(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);

        if self.use_reference {
            nlm_dist_2d_weight_ref::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.reference_arg(ctx),
                self.weight_buf_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_t),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                ScalarArg::new(self.h2_inv_norm),
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            )?;
        } else {
            nlm_dist_2d_weight::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.input_arg(ctx),
                self.weight_buf_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_t),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                ScalarArg::new(self.h2_inv_norm),
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            )?;
        }

        nlm_accumulate::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.input_arg(ctx),
            self.accum_arg(ctx),
            self.weight_sum_arg(ctx),
            self.weight_buf_arg(ctx),
            self.weight_buf_arg(ctx),
            self.max_weight_arg(ctx),
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_t),
            ScalarArg::new(q_x),
            ScalarArg::new(q_y),
            self.width,
            self.height,
        )?;
        Ok(())
    }

    /// Spatial (k=0) separable-path step: distance → hsum → vweight
    /// (single buffer) → accumulate. Symmetric weight map, so one
    /// buffer is reused for both forward and backward lookups.
    fn dispatch_separable_iter_k0(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);

        if self.use_reference {
            nlm_distance_ref::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.reference_arg(ctx),
                self.raw_fwd_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_t),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                self.width,
                self.height,
                channels,
            )?;
        } else {
            nlm_distance::launch::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.input_arg(ctx),
                self.raw_fwd_arg(ctx),
                ScalarArg::new(frame_t),
                ScalarArg::new(frame_t),
                ScalarArg::new(q_x),
                ScalarArg::new(q_y),
                self.width,
                self.height,
                channels,
            )?;
        }

        nlm_horizontal_sum::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.raw_fwd_arg(ctx),
            self.tmp_hsum_arg(ctx),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        nlm_vertical_weight::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.tmp_hsum_arg(ctx),
            self.weight_buf_arg(ctx),
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        nlm_accumulate::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.input_arg(ctx),
            self.accum_arg(ctx),
            self.weight_sum_arg(ctx),
            self.weight_buf_arg(ctx),
            self.weight_buf_arg(ctx),
            self.max_weight_arg(ctx),
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_t),
            ScalarArg::new(q_x),
            ScalarArg::new(q_y),
            self.width,
            self.height,
        )?;
        Ok(())
    }

    fn zero_accumulators(&self, ctx: &LaunchCtx) -> Result<(), anyhow::Error> {
        let grid = (ctx.frame_size as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        gpu_zero_buffers::launch::<R>(
            &self.client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, ctx.frame_size, 1) },
            self.weight_sum_arg(ctx),
            self.max_weight_arg(ctx),
            ctx.frame_size as u32,
            ctx.pixels as u32,
            total_threads,
        )?;
        Ok(())
    }

    fn run_finish(&self, ctx: &LaunchCtx, center_t: u32) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        nlm_finish::launch::<R>(
            &self.client,
            ctx.cube_count.clone(),
            ctx.cube_dim,
            self.input_arg(ctx),
            self.output_arg(ctx),
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.accum,
                    ctx.frame_size,
                    self.params.channels.storage_count() as usize,
                )
            },
            self.weight_sum_arg(ctx),
            self.max_weight_arg(ctx),
            ScalarArg::new(self.phys_frame(center_t as i32)),
            ScalarArg::new(self.params.self_weight),
            self.width,
            self.height,
            channels,
        )?;
        Ok(())
    }

    fn read_output_into_scratch(&mut self, pixels: usize) {
        let channels = self.params.channels.count() as usize;
        let stored_ch = self.params.channels.storage_count() as usize;
        let bytes = self.client.read_one(self.output.clone());
        let data = f32::from_bytes(&bytes);

        let out = &mut self.output_scratch;
        out.clear();
        if channels == stored_ch {
            out.extend_from_slice(data);
        } else {
            // Strip the padding lane (YUV: 4 stored → 3 logical) row by
            // row into the contiguous output Vec.
            out.reserve(pixels * channels);
            for pixel in 0..pixels {
                let src = pixel * stored_ch;
                out.extend_from_slice(&data[src..src + channels]);
            }
        }
    }

    fn run_denoise_kernels(&mut self) -> Result<(), anyhow::Error> {
        let width = self.width;
        let height = self.height;
        let stored_ch = self.params.channels.storage_count();
        let temporal_radius = self.params.temporal_radius;
        let search_radius = self.params.search_radius as i32;
        let total_frames = self.params.total_frames();
        let pixels = (width * height) as usize;
        let frame_size = pixels * stored_ch as usize;

        let ctx = LaunchCtx {
            total_frame_data: frame_size * total_frames as usize,
            frame_size,
            pixels,
            cube_count: CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y)),
            cube_dim: CubeDim::new_2d(BLOCK_X, BLOCK_Y),
        };

        self.zero_accumulators(&ctx)?;

        let center_t = temporal_radius;
        let window_side = 2 * search_radius + 1;
        let window_area = window_side * window_side;

        // Visit only the negative-`linear` half of the search window;
        // every iteration applies both +q and −q via the paired
        // accumulate, so the omitted half is implicitly covered.
        let k_start = -(temporal_radius as i32);
        for q_k in k_start..=0 {
            for q_y in -search_radius..=search_radius {
                for q_x in -search_radius..=search_radius {
                    let linear = q_k * window_area + q_y * window_side + q_x;
                    if linear >= 0 {
                        continue;
                    }

                    // The k=0 paths use the single-tile weight kernel
                    // because the weight map is symmetric in q and a
                    // paired tile would carry duplicate content at
                    // shifted origins. The k≠0 paths fuse the weight
                    // computation with the accumulate inside a single
                    // kernel using a register-resident weight pair.
                    if q_k == 0 {
                        if self.use_separable {
                            self.dispatch_separable_iter_k0(&ctx, center_t, q_x, q_y)?;
                        } else {
                            self.dispatch_fused_iter_k0(&ctx, center_t, q_x, q_y)?;
                        }
                    } else if self.use_separable {
                        self.dispatch_separable_iter(&ctx, center_t, q_x, q_y, q_k)?;
                    } else {
                        self.dispatch_fused_iter(&ctx, center_t, q_x, q_y, q_k)?;
                    }
                }
            }
        }

        self.run_finish(&ctx, center_t)?;
        self.read_output_into_scratch(pixels);
        Ok(())
    }
}

pub fn normalize_u8(input: &[u8]) -> Vec<f32> {
    input.iter().map(|&v| v as f32 / 255.0).collect()
}

pub fn denormalize_u8(input: &[f32]) -> Vec<u8> {
    input
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

pub fn normalize_u16(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&v| v as f32 / 65535.0).collect()
}

pub fn denormalize_u16(input: &[f32]) -> Vec<u16> {
    input
        .iter()
        .map(|&v| (v * 65535.0).round().clamp(0.0, 65535.0) as u16)
        .collect()
}
