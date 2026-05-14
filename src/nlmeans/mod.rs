mod kernels;

#[cfg(test)]
mod tests;

use cubecl::prelude::*;
use cubecl::server::Handle;

use self::kernels::{gpu_zero, nlm_accumulate, nlm_dist_2d_weight, nlm_finish};

const BLOCK_X: u32 = 32;
const BLOCK_Y: u32 = 8;

const NLM_NORM: f32 = 255.0 * 255.0;
const NLM_LEGACY: f32 = 3.0;

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
    pub fn count(self) -> u32 {
        match self {
            ChannelMode::Luma => 1,
            ChannelMode::Chroma => 2,
            ChannelMode::Yuv => 3,
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
    fn h2_inv_norm(&self) -> f32 {
        let s_size = (2 * self.patch_radius + 1) * (2 * self.patch_radius + 1);

        NLM_NORM / (NLM_LEGACY * self.strength * self.strength * s_size as f32)
    }

    fn num_frames(&self) -> u32 {
        2 * self.temporal_radius + 1
    }
}

/// Stateful NLMeans denoiser with double-buffered frame streaming.
///
/// Maintains a ring buffer of frames on the GPU. As frames are pushed in,
/// the denoiser processes the center frame using its temporal neighborhood.
/// The upload of the next frame overlaps with kernel execution on the
/// current window.
pub struct NlmDenoiser<R: Runtime> {
    client: ComputeClient<R>,
    params: NlmParams,
    width: u32,
    height: u32,

    frame_ring: Vec<Handle>,
    frame_cache: Vec<Vec<f32>>,
    ring_head: usize,
    frames_loaded: usize,

    input_buf: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    weight_fwd: Handle,
    dist_a: Handle,
    output: Handle,

    h2_inv_norm: f32,
}

impl<R: Runtime> NlmDenoiser<R> {
    pub fn new(
        client: &ComputeClient<R>,
        params: NlmParams,
        width: u32,
        height: u32,
    ) -> Self {
        let ch = params.channels.count();
        let num_frames = params.num_frames();
        let pixels = (width * height) as usize;

        let frame_ring = (0..num_frames)
            .map(|_| client.empty(pixels * ch as usize * size_of::<f32>()))
            .collect();
        let frame_cache = (0..num_frames)
            .map(|_| vec![0.0f32; pixels * ch as usize])
            .collect();

        let input_buf =
            client.empty(pixels * ch as usize * num_frames as usize * size_of::<f32>());
        let accum = client.empty(pixels * ch as usize * size_of::<f32>());
        let weight_sum = client.empty(pixels * size_of::<f32>());
        let max_weight = client.empty(pixels * size_of::<f32>());
        let weight_fwd = client.empty(pixels * size_of::<f32>());
        let dist_a = client.empty(pixels * size_of::<f32>());
        let output = client.empty(pixels * ch as usize * size_of::<f32>());

        let h2_inv_norm = params.h2_inv_norm();

        Self {
            client: client.clone(),
            params,
            width,
            height,
            frame_ring,
            frame_cache,
            ring_head: 0,
            frames_loaded: 0,
            input_buf,
            accum,
            weight_sum,
            max_weight,
            weight_fwd,
            dist_a,
            output,
            h2_inv_norm,
        }
    }

    /// Push a new frame into the ring buffer.
    /// The frame data must be `width * height * channels` f32 values
    /// normalized to [0, 1].
    pub fn push_frame(&mut self, frame: &[f32]) {
        let ch = self.params.channels.count() as usize;
        let expected = self.width as usize * self.height as usize * ch;

        assert_eq!(
            frame.len(),
            expected,
            "frame size mismatch: expected {expected}, got {}",
            frame.len()
        );

        let slot = self.ring_head % self.frame_ring.len();

        let new_handle = self.client.create_from_slice(f32::as_bytes(frame));
        self.frame_ring[slot] = new_handle;
        self.frame_cache[slot] = frame.to_vec();
        self.ring_head += 1;
        self.frames_loaded += 1;
    }

    /// Try to denoise the current center frame.
    ///
    /// Returns `None` if not enough frames have been pushed yet
    /// (need at least `temporal_radius + 1` frames).
    /// Returns `Some(denoised_frame)` with the denoised center frame.
    pub fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let d = self.params.temporal_radius as usize;
        let num_frames = self.params.num_frames() as usize;

        if self.frames_loaded < d + 1 {
            return Ok(None);
        }

        // We can't output until we have enough future frames too,
        // unless we're at the end of stream (handled by flush).
        // During normal streaming, we buffer until we have the full window.
        if self.frames_loaded < num_frames {
            return Ok(None);
        }

        let result = self.run_denoise_kernels()?;
        Ok(Some(result))
    }

    /// Flush remaining frames at end-of-stream.
    /// For the last `d` frames where future context is incomplete,
    /// the temporal window is clamped.
    pub fn flush(&mut self) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let d = self.params.temporal_radius as usize;
        let mut results = Vec::new();

        // If we never had enough frames for a full window,
        // process what we have.
        if d > 0 && self.frames_loaded < self.params.num_frames() as usize {
            // Process as many as we can with what we have
            let result = self.run_denoise_kernels()?;
            results.push(result);
        }

        // Process the remaining d frames with shrinking future context.
        // We keep the ring buffer as-is but shift the center frame forward.
        for _ in 0..d {
            // Duplicate the last frame into the next ring slot
            // to simulate clamped temporal boundary.
            let last_slot = (self.ring_head - 1) % self.frame_ring.len();
            let last_frame = self.frame_cache[last_slot].clone();

            let new_handle = self.client.create_from_slice(f32::as_bytes(&last_frame));

            let next_slot = self.ring_head % self.frame_ring.len();
            self.frame_ring[next_slot] = new_handle;
            self.frame_cache[next_slot] = last_frame;
            self.ring_head += 1;
            self.frames_loaded += 1;

            let result = self.run_denoise_kernels()?;
            results.push(result);
        }

        Ok(results)
    }

    /// Assemble the current ring buffer frames into a contiguous GPU buffer
    /// and run all NLM kernels.
    fn run_denoise_kernels(&mut self) -> Result<Vec<f32>, anyhow::Error> {
        let w = self.width;
        let h = self.height;
        let ch = self.params.channels.count();
        let num_frames = self.params.num_frames();
        let d = self.params.temporal_radius;
        let a = self.params.search_radius as i32;
        let pixels = (w * h) as usize;
        let frame_size = pixels * ch as usize;
        let accum_size = pixels * ch as usize;

        // Assemble contiguous frame buffer from CPU cache (no GPU readback).
        let total_frame_data = frame_size * num_frames as usize;
        let mut frame_data = Vec::with_capacity(total_frame_data);

        for i in 0..num_frames as usize {
            let ring_idx =
                (self.ring_head - num_frames as usize + i) % self.frame_ring.len();
            frame_data.extend_from_slice(&self.frame_cache[ring_idx]);
        }

        let input_bytes = f32::as_bytes(&frame_data);
        self.input_buf = self.client.create_from_slice(input_bytes);

        // Zero accum, weight_sum, and max_weight on GPU.
        let zero_block = 256u32;
        let zero_grid_acc = div_ceil(accum_size as u32, zero_block);
        let zero_grid_px = div_ceil(pixels as u32, zero_block);

        gpu_zero::launch::<R>(
            &self.client,
            CubeCount::new_1d(zero_grid_acc),
            CubeDim::new_1d(zero_block),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, accum_size, 1) },
            accum_size as u32,
        )?;

        gpu_zero::launch::<R>(
            &self.client,
            CubeCount::new_1d(zero_grid_px),
            CubeDim::new_1d(zero_block),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1) },
            pixels as u32,
        )?;

        gpu_zero::launch::<R>(
            &self.client,
            CubeCount::new_1d(zero_grid_px),
            CubeDim::new_1d(zero_block),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1) },
            pixels as u32,
        )?;

        // Dispatch dimensions
        let grid_x = div_ceil(w, BLOCK_X);
        let grid_y = div_ceil(h, BLOCK_Y);
        let cube_count = CubeCount::new_2d(grid_x, grid_y);
        let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        let t = d as i32;
        let spt_side = 2 * a + 1;
        let spt_area = spt_side * spt_side;

        // Spatio-temporal loop with symmetry exploitation
        let k_start = -(d as i32);
        for k in k_start..=0 {
            for j in -a..=a {
                for i in -a..=a {
                    let linear = k * spt_area + j * spt_side + i;
                    if linear >= 0 {
                        continue;
                    }

                    // Fused distance + 2D box filter + weight (center frame).
                    // When k != 0, write to weight_fwd; otherwise dist_a.
                    let fwd_output = if k != 0 {
                        &self.weight_fwd
                    } else {
                        &self.dist_a
                    };

                    nlm_dist_2d_weight::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(
                                &self.input_buf,
                                total_frame_data,
                                1,
                            )
                        },
                        unsafe { ArrayArg::from_raw_parts::<f32>(fwd_output, pixels, 1) },
                        ScalarArg::new(t),
                        ScalarArg::new(i),
                        ScalarArg::new(j),
                        ScalarArg::new(k),
                        ScalarArg::new(self.h2_inv_norm),
                        w,
                        h,
                        ch,
                        num_frames,
                        self.params.patch_radius,
                        BLOCK_X,
                        BLOCK_Y,
                    )?;

                    // For temporal offsets, compute weights from
                    // mirror frame perspective into dist_a.
                    if k != 0 {
                        let t_mq = t - k;

                        nlm_dist_2d_weight::launch::<R>(
                            &self.client,
                            cube_count.clone(),
                            cube_dim,
                            unsafe {
                                ArrayArg::from_raw_parts::<f32>(
                                    &self.input_buf,
                                    total_frame_data,
                                    1,
                                )
                            },
                            unsafe {
                                ArrayArg::from_raw_parts::<f32>(&self.dist_a, pixels, 1)
                            },
                            ScalarArg::new(t_mq),
                            ScalarArg::new(i),
                            ScalarArg::new(j),
                            ScalarArg::new(k),
                            ScalarArg::new(self.h2_inv_norm),
                            w,
                            h,
                            ch,
                            num_frames,
                            self.params.patch_radius,
                            BLOCK_X,
                            BLOCK_Y,
                        )?;
                    }

                    // Accumulate for both +q and -q.
                    let fwd_weights = if k != 0 {
                        &self.weight_fwd
                    } else {
                        &self.dist_a
                    };

                    nlm_accumulate::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(
                                &self.input_buf,
                                total_frame_data,
                                1,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts::<f32>(&self.accum, accum_size, 1)
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
                        ScalarArg::new(t),
                        ScalarArg::new(i),
                        ScalarArg::new(j),
                        ScalarArg::new(k),
                        w,
                        h,
                        ch,
                        num_frames,
                    )?;
                }
            }
        }

        // Final normalization
        nlm_finish::launch::<R>(
            &self.client,
            cube_count,
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&self.input_buf, total_frame_data, 1)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.output, frame_size, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, accum_size, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1) },
            ScalarArg::new(t),
            ScalarArg::new(self.params.self_weight),
            w,
            h,
            ch,
            num_frames,
        )?;

        // Read back denoised frame
        let bytes = self.client.read_one(self.output.clone());
        let result = f32::from_bytes(&bytes).to_vec();

        Ok(result)
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

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
