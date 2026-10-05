use av_denoise_core::bench_api::collab::kernels::aggregate::{collab_normalise, collab_zero_accum};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, HEIGHT, WIDTH, block_sync, stored_channels};

/// The 65,535 workgroups per dimension GPU limit the library clamps to.
///
/// The library's own constant is crate-private, so a bench target spells it out.
const MAX_GRID_1D: u32 = 65_535;

/// Divides the fixed-point accumulators back out to a finished 1080p frame plane.
///
/// Cost scales with `stored_ch`, since one accumulator slot per channel is read for every pixel.
pub struct CollabNormaliseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

#[derive(Clone)]
pub struct CollabNormaliseInput {
    pub accum: Handle,
    pub wsum: Handle,
    pub output: Handle,
}

impl<R: Runtime> Benchmark for CollabNormaliseBench<R> {
    type Input = CollabNormaliseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;

        // Shaped like a real pass, with a few dozen contributions per pixel, each already scaled
        // into the accumulator's fixed point.
        let accum_data: Vec<i32> = (0..pixels * stored_ch)
            .map(|index| (index % 97) as i32 * 8192)
            .collect();
        let wsum_data: Vec<i32> = (0..pixels)
            .map(|index| ((index % 31) + 20) as i32 * 8192)
            .collect();

        let accum_bytes = i32::as_bytes(&accum_data);
        let accum = self.client.create_from_slice(accum_bytes);
        let wsum_bytes = i32::as_bytes(&wsum_data);
        let wsum = self.client.create_from_slice(wsum_bytes);
        let output = self.client.empty(pixels * stored_ch * size_of::<f32>());

        CollabNormaliseInput { accum, wsum, output }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let cubes_x = WIDTH.div_ceil(BLOCK_X);
        let cubes_y = HEIGHT.div_ceil(BLOCK_Y);

        unsafe {
            collab_normalise::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(cubes_x, cubes_y),
                CubeDim::new_2d(BLOCK_X, BLOCK_Y),
                stored_ch,
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(args.output.clone(), pixels * stored_ch),
                // One frame's region, since this bench measures a single frame's normalisation.
                0u32,
                WIDTH,
                HEIGHT,
                self.channels,
                stored_ch as u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_normalise_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize, self.channels as usize]]
    }
}

/// Clears both accumulators, which runs once before every filter pass.
pub struct CollabZeroAccumBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for CollabZeroAccumBench<R> {
    type Input = CollabNormaliseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let accum = self.client.empty(pixels * stored_ch * size_of::<i32>());
        let wsum = self.client.empty(pixels * size_of::<i32>());
        let output = self.client.empty(size_of::<f32>());

        CollabNormaliseInput { accum, wsum, output }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let dim = 256u32;

        // `collab_zero_accum` strides, so the clamped grid still reaches every slot.
        let grid = ((pixels * stored_ch) as u32).div_ceil(dim).min(MAX_GRID_1D);

        unsafe {
            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(dim),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                // A single-frame region, as a single-frame caller passes.
                0u32,
                pixels as u32,
                stored_ch as u32,
                grid * dim,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_zero_accum_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize, self.channels as usize]]
    }
}
