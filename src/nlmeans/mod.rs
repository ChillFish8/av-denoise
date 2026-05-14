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
    nlm_distance,
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

    /// Per-frame GPU handles for the ring buffer.
    frame_ring: Vec<Handle>,
    ring_head: usize,
    frames_loaded: usize,

    /// Contiguous GPU buffer holding all frames for the current window.
    /// Layout: [total_frames * height * width * channels].
    input_buf: Handle,
    /// Accumulated weighted pixel sums: [pixels * channels].
    accum: Handle,
    /// Total weight per pixel: [pixels].
    weight_sum: Handle,
    /// Max neighbor weight per pixel: [pixels].
    max_weight: Handle,
    /// Forward weight scratch: [pixels].
    weight_fwd: Handle,
    /// Primary distance/weight scratch: [pixels].
    dist_a: Handle,
    /// Secondary scratch for separable intermediate: [pixels].
    dist_b: Handle,
    /// Denoised output: [pixels * channels].
    output: Handle,

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
        let ch = params.channels.count();
        let total_frames = params.total_frames();
        let pixels = (width * height) as usize;

        // Ring buffer holds enough frames for one full temporal window.
        let ring_size = total_frames as usize;
        let frame_ring = (0..ring_size)
            .map(|_| client.empty(pixels * ch as usize * size_of::<f32>()))
            .collect();

        let input_buf =
            client.empty(pixels * ch as usize * total_frames as usize * size_of::<f32>());
        let accum = client.empty(pixels * ch as usize * size_of::<f32>());
        let weight_sum = client.empty(pixels * size_of::<f32>());
        let max_weight = client.empty(pixels * size_of::<f32>());
        let weight_fwd = client.empty(pixels * size_of::<f32>());
        let dist_a = client.empty(pixels * size_of::<f32>());
        let dist_b = client.empty(pixels * size_of::<f32>());
        let output = client.empty(pixels * ch as usize * size_of::<f32>());

        let h2_inv_norm = params.h2_inv_norm();
        let use_separable = params.patch_radius > SEPARABLE_THRESHOLD;

        Self {
            client: client.clone(),
            params,
            width,
            height,
            frame_ring,
            ring_head: 0,
            frames_loaded: 0,
            input_buf,
            accum,
            weight_sum,
            max_weight,
            weight_fwd,
            dist_a,
            dist_b,
            output,
            h2_inv_norm,
            use_separable,
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
        self.ring_head += 1;
        self.frames_loaded += 1;
    }

    /// Try to denoise the current center frame.
    ///
    /// Returns `None` if not enough frames have been pushed yet.
    /// Returns `Some(denoised_frame)` with the denoised center frame.
    pub fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;

        // Need all frames in the temporal window.
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        let result = self.run_denoise_kernels()?;
        Ok(Some(result))
    }

    /// Flush remaining frames at end-of-stream.
    /// For the last `d` frames where future context is incomplete,
    /// the temporal window is clamped by duplicating the last frame.
    pub fn flush(&mut self) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let d = self.params.temporal_radius as usize;
        let total_frames = self.params.total_frames() as usize;
        let mut all_results = Vec::new();

        // If we never had enough frames for a full window,
        // process what we have by duplicating the last frame.
        if d > 0 && self.frames_loaded < total_frames {
            while self.frames_loaded < total_frames {
                let last_slot = (self.ring_head - 1) % self.frame_ring.len();
                let next_slot = self.ring_head % self.frame_ring.len();
                self.frame_ring[next_slot] = self.frame_ring[last_slot].clone();
                self.ring_head += 1;
                self.frames_loaded += 1;
            }

            let result = self.run_denoise_kernels()?;
            all_results.push(result);
        }

        // Process the remaining d frames with shrinking future context.
        for _ in 0..d {
            let last_slot = (self.ring_head - 1) % self.frame_ring.len();
            let next_slot = self.ring_head % self.frame_ring.len();
            self.frame_ring[next_slot] = self.frame_ring[last_slot].clone();
            self.ring_head += 1;
            self.frames_loaded += 1;

            let result = self.run_denoise_kernels()?;
            all_results.push(result);
        }

        Ok(all_results)
    }

    /// Assemble the current ring buffer frames into a contiguous GPU
    /// buffer using GPU-to-GPU copies (no CPU readback).
    /// Uses Handle::offset_start to write at the correct position within
    /// input_buf, avoiding runtime scalar offsets which don't work
    /// correctly on the CubeCL CPU runtime.
    fn assemble_input_buf(&self, total_frames: u32) -> Result<(), anyhow::Error> {
        let ch = self.params.channels.count();
        let frame_size = (self.width * self.height * ch) as u32;
        let grid = div_ceil(frame_size, BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        for i in 0..total_frames as usize {
            let ring_idx =
                (self.ring_head - total_frames as usize + i) % self.frame_ring.len();
            let byte_offset =
                (i as u64) * (frame_size as u64) * (size_of::<f32>() as u64);

            // Create a sub-view of input_buf starting at the right offset.
            let dst_handle = self.input_buf.clone().offset_start(byte_offset);

            gpu_copy::launch::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                unsafe {
                    ArrayArg::from_raw_parts::<f32>(
                        &self.frame_ring[ring_idx],
                        frame_size as usize,
                        1,
                    )
                },
                unsafe {
                    ArrayArg::from_raw_parts::<f32>(&dst_handle, frame_size as usize, 1)
                },
                frame_size,
                total_threads,
            )?;
        }

        Ok(())
    }

    /// Compute distance weights using the fused 2D box filter kernel.
    fn compute_weights_fused(
        &self,
        output: &Handle,
        t: i32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        nlm_dist_2d_weight::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&self.input_buf, total_frame_data, 1)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(output, pixels, 1) },
            ScalarArg::new(t),
            ScalarArg::new(i),
            ScalarArg::new(j),
            ScalarArg::new(k),
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            self.params.channels.count(),
            total_frames,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;
        Ok(())
    }

    /// Compute distance weights using the separable 3-pass approach.
    fn compute_weights_separable(
        &self,
        output: &Handle,
        t: i32,
        i: i32,
        j: i32,
        k: i32,
        total_frames: u32,
        total_frame_data: usize,
        pixels: usize,
        cube_count: &CubeCount,
        cube_dim: CubeDim,
    ) -> Result<(), anyhow::Error> {
        // Pass 1: Per-pixel squared distance.
        nlm_distance::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&self.input_buf, total_frame_data, 1)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.dist_b, pixels, 1) },
            ScalarArg::new(t),
            ScalarArg::new(i),
            ScalarArg::new(j),
            ScalarArg::new(k),
            self.width,
            self.height,
            self.params.channels.count(),
            total_frames,
        )?;

        // Pass 2: Horizontal box filter.
        nlm_horizontal_sum::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.dist_b, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(output, pixels, 1) },
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        // Pass 3: Vertical box filter + Welsch weight.
        nlm_vertical_weight::launch::<R>(
            &self.client,
            cube_count.clone(),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts::<f32>(output, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.dist_b, pixels, 1) },
            ScalarArg::new(self.h2_inv_norm),
            self.width,
            self.height,
            self.params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )?;

        // Copy result from dist_b back to output.
        let copy_grid = div_ceil(pixels as u32, BLOCK_1D).min(MAX_GRID_1D);
        let copy_threads = copy_grid * BLOCK_1D;
        gpu_copy::launch::<R>(
            &self.client,
            CubeCount::new_1d(copy_grid),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.dist_b, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(output, pixels, 1) },
            pixels as u32,
            copy_threads,
        )?;
        Ok(())
    }

    /// Compute distance weights, dispatching to fused or separable path.
    fn compute_weights(
        &self,
        output: &Handle,
        t: i32,
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

    /// Run all NLM kernels on the current frame.
    /// Returns the denoised center frame.
    fn run_denoise_kernels(&self) -> Result<Vec<f32>, anyhow::Error> {
        let w = self.width;
        let h = self.height;
        let ch = self.params.channels.count();
        let d = self.params.temporal_radius;
        let a = self.params.search_radius as i32;
        let total_frames = self.params.total_frames();
        let pixels = (w * h) as usize;
        let frame_size = pixels * ch as usize;
        let total_frame_data = frame_size * total_frames as usize;

        // Assemble contiguous frame buffer on GPU.
        self.assemble_input_buf(total_frames)?;

        // Fused zero: accum + weight_sum + max_weight.
        let max_len = if frame_size > pixels {
            frame_size
        } else {
            pixels
        };
        let zero_grid = div_ceil(max_len as u32, BLOCK_1D).min(MAX_GRID_1D);
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
        let grid_x = div_ceil(w, BLOCK_X);
        let grid_y = div_ceil(h, BLOCK_Y);
        let cube_count = CubeCount::new_2d(grid_x, grid_y);
        let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        // t: center frame index.
        let t = d as i32;
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

                    // Compute weights from center frame perspective.
                    let fwd_output = if k != 0 {
                        &self.weight_fwd
                    } else {
                        &self.dist_a
                    };

                    self.compute_weights(
                        fwd_output,
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

                    // For temporal offsets, compute weights from
                    // mirror frame perspective into dist_a.
                    if k != 0 {
                        let t_mq = t - k;

                        self.compute_weights(
                            &self.dist_a,
                            t_mq,
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
                            ArrayArg::from_raw_parts::<f32>(&self.accum, frame_size, 1)
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
                ArrayArg::from_raw_parts::<f32>(&self.input_buf, total_frame_data, 1)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.output, frame_size, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.accum, frame_size, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&self.max_weight, pixels, 1) },
            ScalarArg::new(t),
            ScalarArg::new(self.params.self_weight),
            w,
            h,
            ch,
            total_frames,
        )?;

        // Read back the denoised frame.
        let bytes = self.client.read_one(self.output.clone());
        let data = f32::from_bytes(&bytes);

        Ok(data.to_vec())
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
