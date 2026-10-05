use av_denoise_core::bench_api::kernels::gpu_zero_buffers;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, COPY_GRID_1D, HEIGHT, WIDTH, block_sync, shapes_with_channels, stored_channels};

#[derive(Clone)]
pub struct ZeroInput {
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
}

pub struct ZeroBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for ZeroBench<R> {
    type Input = ZeroInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let accum = self.client.empty(pixels * stored_ch * size_of::<f32>());
        let weight_sum = self.client.empty(pixels * size_of::<f32>());
        let max_weight = self.client.empty(pixels * size_of::<f32>());

        ZeroInput {
            accum,
            weight_sum,
            max_weight,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let total_threads = COPY_GRID_1D * BLOCK_1D;

        unsafe {
            gpu_zero_buffers::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(COPY_GRID_1D),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                (pixels * stored_ch) as u32,
                pixels as u32,
                total_threads,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("gpu_zero_buffers_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
