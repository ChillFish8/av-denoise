pub mod kernels;

#[cfg(test)]
mod tests;

use cubecl::prelude::*;
use cubecl::server::Handle;

use self::kernels::{
    gpu_copy,
    gpu_zero_buffers,
    nlm_accumulate,
    nlm_dist_2d_weight,
    nlm_dist_2d_weight_pair,
    nlm_distance,
    nlm_distance_pair,
    nlm_finish,
    nlm_horizontal_sum,
    nlm_vertical_weight,
};

pub const BLOCK_X: u32 = 32;
pub const BLOCK_Y: u32 = 8;

const NLM_NORM: f32 = 255.0 * 255.0;
const NLM_LEGACY: f32 = 3.0;

/// Patch radius threshold: use separable filter above this value.
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
    /// vectorization factor so kernels can use coalesced `Line<f32>` reads.
    /// (Backends only support power-of-two line sizes; YUV pads to 4 with
    /// the extra lane held at 0.)
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

/// Stateful NLMeans denoiser with GPU-side frame assembly.
///
/// Maintains a ring buffer of frames on the GPU. As frames are pushed
/// in, the denoiser processes the center frame using its temporal
/// neighborhood.
pub struct NlmDenoiser<R: Runtime> {
    client: ComputeClient<R>,
    params: NlmParams,
    width: u32,
    height: u32,

    /// Monotonic count of frames pushed; `% total_frames` gives the
    /// physical slot of the *next* write into `input_buf`.
    ring_head: usize,
    /// Number of frames loaded so far (capped at total_frames).
    frames_loaded: usize,

    /// Contiguous GPU buffer used as a ring of frames. Layout:
    /// `[total_frames * height * width * channels]`. Each `push_frame`
    /// uploads a single frame and `gpu_copy`'s it into slot
    /// `ring_head % total_frames`, so the per-denoise reassembly cost is
    /// zero — kernels read directly from this buffer using physical
    /// frame indices resolved on the host.
    input_buf: Handle,
    /// CPU scratch reused across pushes when channel storage requires
    /// padding (e.g. YUV → 4 lanes). Sized for one frame.
    padding_scratch: Vec<f32>,
    /// Accumulated weighted pixel sums: [pixels * channels].
    accum: Handle,
    /// Total weight per pixel: [pixels].
    weight_sum: Handle,
    /// Max neighbor weight per pixel: [pixels].
    max_weight: Handle,
    /// Forward weight scratch: [pixels].
    weight_fwd: Handle,
    /// Final weight buffer for the spatial path / backward weight in the
    /// temporal path. Accumulate's `weights_bwd` always points here.
    dist_a: Handle,
    /// Raw fwd distance (separable path): [pixels].
    raw_fwd: Handle,
    /// Raw bwd distance (separable path): [pixels].
    raw_bwd: Handle,
    /// Hsum intermediate (separable path): [pixels].
    tmp_hsum: Handle,
    /// Denoised output: [pixels * channels].
    output: Handle,
    /// CPU scratch reused for the denoised frame returned from
    /// `denoise()`/`flush()`. Avoids a fresh `Vec<f32>` per frame.
    output_scratch: Vec<f32>,

    h2_inv_norm: f32,
    use_separable: bool,
}

impl<R: Runtime> NlmDenoiser<R> {
    pub fn new(
        client: &ComputeClient<R>,
        params: NlmParams,
        width: u32,
        height: u32,
    ) -> Self {
        let stored_ch = params.channels.storage_count();
        let total_frames = params.total_frames();
        let pixels = (width * height) as usize;

        let input_buf =
            client.empty(pixels * stored_ch as usize * total_frames as usize * size_of::<f32>());
        let padding_scratch = if params.channels.count() != stored_ch {
            vec![0.0f32; pixels * stored_ch as usize]
        } else {
            Vec::new()
        };
        let accum = client.empty(pixels * stored_ch as usize * size_of::<f32>());
        let weight_sum = client.empty(pixels * size_of::<f32>());
        let max_weight = client.empty(pixels * size_of::<f32>());
        let weight_fwd = client.empty(pixels * size_of::<f32>());
        let dist_a = client.empty(pixels * size_of::<f32>());
        let raw_fwd = client.empty(pixels * size_of::<f32>());
        let raw_bwd = client.empty(pixels * size_of::<f32>());
        let tmp_hsum = client.empty(pixels * size_of::<f32>());
        let output = client.empty(pixels * stored_ch as usize * size_of::<f32>());

        let h2_inv_norm = params.h2_inv_norm();
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;
        let output_scratch_cap = pixels * params.channels.count() as usize;

        Self {
            client: client.clone(),
            params,
            width,
            height,
            ring_head: 0,
            frames_loaded: 0,
            input_buf,
            padding_scratch,
            accum,
            weight_sum,
            max_weight,
            weight_fwd,
            dist_a,
            raw_fwd,
            raw_bwd,
            tmp_hsum,
            output,
            output_scratch: Vec::with_capacity(output_scratch_cap),
            h2_inv_norm,
            use_separable,
        }
    }

    /// Push a new frame into the ring buffer.
    /// The frame data must be `width * height * channels` f32 values
    /// normalized to [0, 1]. When the channel mode requires padding
    /// (YUV → 4 lanes), the data is repacked into a reused CPU scratch
    /// buffer before upload. The frame is uploaded once and copied into
    /// the next physical slot of `input_buf` — no per-denoise
    /// reassembly is needed.
    pub fn push_frame(&mut self, frame: &[f32]) {
        let ch = self.params.channels.count() as usize;
        let stored_ch = self.params.channels.storage_count() as usize;
        let pixels = self.width as usize * self.height as usize;
        let expected = pixels * ch;

        assert_eq!(
            frame.len(),
            expected,
            "frame size mismatch: expected {expected}, got {}",
            frame.len()
        );

        let total_frames = self.params.total_frames() as usize;
        let slot = self.ring_head % total_frames;
        let staging = if ch == stored_ch {
            self.client.create_from_slice(f32::as_bytes(frame))
        } else {
            // Reuse padding_scratch to avoid a per-push Vec alloc.
            for i in 0..pixels {
                let dst = i * stored_ch;
                let src = i * ch;
                self.padding_scratch[dst..dst + ch]
                    .copy_from_slice(&frame[src..src + ch]);
            }
            self.client.create_from_slice(f32::as_bytes(&self.padding_scratch))
        };

        self.copy_frame_into_slot(&staging, slot);

        self.ring_head += 1;
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }
    }

    /// GPU→GPU copy of one frame from `src` into `input_buf` at the
    /// given physical slot.
    fn copy_frame_into_slot(&self, src: &Handle, slot: usize) {
        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let byte_offset =
            (slot as u64) * (frame_size as u64) * (size_of::<f32>() as u64);
        let dst_handle = self.input_buf.clone().offset_start(byte_offset);

        let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        gpu_copy::launch::<R>(
            &self.client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(src, frame_size as usize, 1) },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&dst_handle, frame_size as usize, 1)
            },
            frame_size,
            total_threads,
        )
        .expect("gpu_copy launch failed");
    }

    /// Duplicate the most recently pushed frame into the next ring slot.
    /// Used at end-of-stream (`flush`) to keep the window full while
    /// future context shrinks.
    fn duplicate_last_frame(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        let last_slot = (self.ring_head - 1) % total_frames;
        let next_slot = self.ring_head % total_frames;

        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let byte = (frame_size as u64) * (size_of::<f32>() as u64);
        let src = self
            .input_buf
            .clone()
            .offset_start((last_slot as u64) * byte);

        // copy_frame_into_slot does the dispatch on a sub-view of input_buf.
        // src here is a sub-view into the same buffer — copy is well-defined
        // because slots do not overlap.
        self.copy_frame_into_slot(&src, next_slot);

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

        // Need all frames in the temporal window.
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        self.run_denoise_kernels()?;
        Ok(Some(self.output_scratch.as_slice()))
    }

    /// Flush remaining frames at end-of-stream.
    /// For the last `d` frames where future context is incomplete,
    /// the temporal window is clamped by duplicating the last frame.
    /// The callback `sink` is invoked once per produced frame; the
    /// borrowed slice is only valid for that call.
    pub fn flush(
        &mut self,
        mut sink: impl FnMut(&[f32]),
    ) -> Result<(), anyhow::Error> {
        let d = self.params.temporal_radius as usize;
        let total_frames = self.params.total_frames() as usize;

        // If we never had enough frames for a full window,
        // process what we have by duplicating the last frame.
        if d > 0 && self.frames_loaded < total_frames {
            while self.frames_loaded < total_frames {
                self.duplicate_last_frame();
                self.frames_loaded += 1;
            }

            self.run_denoise_kernels()?;
            sink(self.output_scratch.as_slice());
        }

        // Process the remaining d frames with shrinking future context.
        for _ in 0..d {
            self.duplicate_last_frame();
            self.run_denoise_kernels()?;
            sink(self.output_scratch.as_slice());
        }

        Ok(())
    }

    /// Physical slot of logical frame 0 (oldest frame in the window).
    /// Defined only after a full window has been pushed (`frames_loaded
    /// >= total_frames`).
    fn ring_start(&self) -> u32 {
        let total_frames = self.params.total_frames() as usize;
        (self.ring_head % total_frames) as u32
    }

    /// Resolve a logical frame index in `[0, total_frames)` to its
    /// physical slot inside `input_buf`.
    fn phys_frame(&self, logical: i32) -> u32 {
        let total_frames = self.params.total_frames() as i32;
        let l = logical.rem_euclid(total_frames);
        ((self.ring_start() as i32 + l).rem_euclid(total_frames)) as u32
    }

    /// Compute distance weights using the fused 2D box filter kernel.
    fn compute_weights_fused(
        &self,
        output: &Handle,
        t: u32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        let ch = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let frame_t = self.phys_frame(t as i32);
        let frame_q = self.phys_frame(t as i32 + k);
        nlm_dist_2d_weight::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.input_buf,
                    total_frame_data,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(output, pixels, 1) },
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_q),
            ScalarArg::new(i),
            ScalarArg::new(j),
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            ch,
            total_frames,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;
        Ok(())
    }

    /// Compute distance weights using the separable 3-pass approach.
    /// Flow: distance → raw_fwd → hsum → tmp_hsum → vsum+weight → output.
    fn compute_weights_separable(
        &self,
        output: &Handle,
        t: u32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        let ch = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let frame_t = self.phys_frame(t as i32);
        let frame_q = self.phys_frame(t as i32 + k);
        // Pass 1: Per-pixel squared distance (vectorized input).
        nlm_distance::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.input_buf,
                    total_frame_data,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.raw_fwd, pixels, 1) },
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_q),
            ScalarArg::new(i),
            ScalarArg::new(j),
            self.width,
            self.height,
            ch,
            total_frames,
        )?;

        self.run_separable_box(&self.raw_fwd, output, pixels, cube_count, cube_dim)
    }

    /// Paired separable distance + box filter for a temporal +q/-q pair.
    /// Writes fwd weights to `out_fwd` and bwd weights to `out_bwd`,
    /// each computed via dist→hsum→vsum sharing the `tmp_hsum` scratch.
    fn compute_weights_separable_pair(
        &self,
        out_fwd: &Handle,
        out_bwd: &Handle,
        t: u32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        let ch = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let frame_t = self.phys_frame(t as i32);
        let frame_fwd = self.phys_frame(t as i32 + k);
        let frame_bwd = self.phys_frame(t as i32 - k);
        // Pass 1: Per-pixel squared distance for both fwd and bwd.
        nlm_distance_pair::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.input_buf,
                    total_frame_data,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.raw_fwd, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.raw_bwd, pixels, 1) },
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_fwd),
            ScalarArg::new(frame_bwd),
            ScalarArg::new(i),
            ScalarArg::new(j),
            self.width,
            self.height,
            ch,
            total_frames,
        )?;

        // Box filter each independently. tmp_hsum is overwritten per pass.
        self.run_separable_box(&self.raw_fwd, out_fwd, pixels, cube_count, cube_dim)?;
        self.run_separable_box(&self.raw_bwd, out_bwd, pixels, cube_count, cube_dim)?;
        Ok(())
    }

    /// Apply the 2D separable box filter (hsum then vsum+Welsch weight)
    /// from `raw` (raw distances) into `out` (final weights), using
    /// `self.tmp_hsum` as the intermediate.
    fn run_separable_box(
        &self,
        raw: &Handle,
        out: &Handle,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        // Horizontal box filter: raw → tmp_hsum.
        nlm_horizontal_sum::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts::<f32>(raw, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.tmp_hsum, pixels, 1) },
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        // Vertical box filter + Welsch weight: tmp_hsum → out.
        nlm_vertical_weight::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.tmp_hsum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(out, pixels, 1) },
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;
        Ok(())
    }

    /// Paired fused distance + 2D box filter for a temporal +q/-q pair.
    fn compute_weights_fused_pair(
        &self,
        out_fwd: &Handle,
        out_bwd: &Handle,
        t: u32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        let ch = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let frame_t = self.phys_frame(t as i32);
        let frame_fwd = self.phys_frame(t as i32 + k);
        let frame_bwd = self.phys_frame(t as i32 - k);
        nlm_dist_2d_weight_pair::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.input_buf,
                    total_frame_data,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(out_fwd, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(out_bwd, pixels, 1) },
            ScalarArg::new(frame_t),
            ScalarArg::new(frame_fwd),
            ScalarArg::new(frame_bwd),
            ScalarArg::new(i),
            ScalarArg::new(j),
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            ch,
            total_frames,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;
        Ok(())
    }

    /// Compute distance weights, dispatching to fused or separable path.
    fn compute_weights(
        &self,
        output: &Handle,
        t: u32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        if self.use_separable {
            self.compute_weights_separable(
                output,
                t,
                i,
                j,
                k,
                total_frames,
                total_frame_data,
                pixels,
                cube_count,
                cube_dim,
            )
        } else {
            self.compute_weights_fused(
                output,
                t,
                i,
                j,
                k,
                total_frames,
                total_frame_data,
                pixels,
                cube_count,
                cube_dim,
            )
        }
    }

    /// Run all NLM kernels on the current frame, writing the result
    /// into `self.output_scratch` (resized as needed). The slice can be
    /// read back via `self.output_scratch.as_slice()`.
    fn run_denoise_kernels(&mut self) -> Result<(), anyhow::Error> {
        let w = self.width;
        let h = self.height;
        let ch = self.params.channels.count();
        let stored_ch = self.params.channels.storage_count();
        let d = self.params.temporal_radius;
        let a = self.params.search_radius as i32;
        let total_frames = self.params.total_frames();
        let pixels = (w * h) as usize;
        let frame_size = pixels * stored_ch as usize;
        let total_frame_data = frame_size * total_frames as usize;

        // input_buf is already laid out as a ring with the latest frame
        // in slot (ring_head - 1) % total_frames; no per-denoise copy.

        // Fused zero: accum + weight_sum + max_weight.
        // frame_size >= pixels (stored_ch >= 1), so the bound is just frame_size.
        let zero_grid = (frame_size as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let zero_threads = zero_grid * BLOCK_1D;

        gpu_zero_buffers::launch::<R>(
            &self.client,
            CubeCount::new_1d(zero_grid),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, frame_size, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1) },
            frame_size as u32,
            pixels as u32,
            zero_threads,
        )?;

        // 2D dispatch dimensions.
        let grid_x = w.div_ceil(BLOCK_X);
        let grid_y = h.div_ceil(BLOCK_Y);
        let cube_count = CubeCount::new_2d(grid_x, grid_y);
        let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        // t: center frame index.
        let t = d;
        let spt_side = 2 * a + 1;
        let spt_area = spt_side * spt_side;

        // Spatio-temporal loop with symmetry exploitation.
        let k_start = -(d as i32);
        for k in k_start..=0 {
            for j in -a..=a {
                for i in -a..=a {
                    let linear = k * spt_area + j * spt_side + i;
                    if linear >= 0 {
                        continue;
                    }

                    // Compute weights. For temporal pairs (k != 0) use
                    // the paired kernel that produces both fwd and bwd
                    // weights in a single dispatch (saving launch
                    // overhead and, for the fused path, sharing the
                    // central patch read).
                    let fwd_weights = if k != 0 {
                        if self.use_separable {
                            self.compute_weights_separable_pair(
                                &self.weight_fwd,
                                &self.dist_a,
                                t,
                                i,
                                j,
                                k,
                                total_frames,
                                total_frame_data,
                                pixels,
                                &cube_count,
                                cube_dim,
                            )?;
                        } else {
                            self.compute_weights_fused_pair(
                                &self.weight_fwd,
                                &self.dist_a,
                                t,
                                i,
                                j,
                                k,
                                total_frames,
                                total_frame_data,
                                pixels,
                                &cube_count,
                                cube_dim,
                            )?;
                        }

                        &self.weight_fwd
                    } else {
                        // Spatial only: a single weight buffer is used
                        // for both directions (symmetric in q).
                        self.compute_weights(
                            &self.dist_a,
                            t,
                            i,
                            j,
                            k,
                            total_frames,
                            total_frame_data,
                            pixels,
                            &cube_count,
                            cube_dim,
                        )?;
                        &self.dist_a
                    };

                    let frame_fwd = self.phys_frame(t as i32 + k);
                    let frame_bwd = self.phys_frame(t as i32 - k);
                    nlm_accumulate::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(
                                &self.input_buf,
                                total_frame_data,
                                stored_ch as usize,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(
                                &self.accum,
                                frame_size,
                                stored_ch as usize,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1)
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(fwd_weights, pixels, 1)
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(&self.dist_a, pixels, 1)
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1)
                        },
                        ScalarArg::new(frame_fwd),
                        ScalarArg::new(frame_bwd),
                        ScalarArg::new(i),
                        ScalarArg::new(j),
                        w,
                        h,
                        ch,
                        total_frames,
                    )?;
                }
            }
        }

        // Final normalization.
        nlm_finish::launch::<R>(
            &self.client,
            cube_count,
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &self.input_buf,
                    total_frame_data,
                    stored_ch as usize,
                )
            },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&self.output, frame_size, stored_ch as usize)
            },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&self.accum, frame_size, stored_ch as usize)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1) },
            ScalarArg::new(self.phys_frame(t as i32)),
            ScalarArg::new(self.params.self_weight),
            w,
            h,
            ch,
            total_frames,
        )?;

        // Read back the denoised frame, stripping padding lanes if any.
        // Writes into the reusable `output_scratch` Vec — no per-call
        // heap alloc.
        let bytes = self.client.read_one(self.output.clone());
        let data = f32::from_bytes(&bytes);

        let out = &mut self.output_scratch;
        out.clear();
        if ch == stored_ch {
            out.extend_from_slice(data);
        } else {
            out.reserve(pixels * ch as usize);
            for i in 0..pixels {
                let src = i * stored_ch as usize;
                out.extend_from_slice(&data[src..src + ch as usize]);
            }
        }
        Ok(())
    }
}

// --- Normalization utilities ---

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

