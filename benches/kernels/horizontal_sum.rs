use av_denoise::nlmeans::kernels::nlm_horizontal_sum;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, H, PATCH_RADIUS, W, block_sync, cube_count_2d, cube_dim_2d, map_err};

#[derive(Clone)]
pub struct HSumInput {
    pub input: Handle,
    pub output: Handle,
}

pub struct HSumBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for HSumBench<R> {
    type Input = HSumInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let data = vec![0.5f32; pixels];
        let input = self.client.create_from_slice(f32::as_bytes(&data));
        let output = self.client.empty(pixels * size_of::<f32>());
        HSumInput { input, output }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        nlm_horizontal_sum::launch::<R>(
            &self.client,
            cube_count_2d(),
            cube_dim_2d(),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.input, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.output, pixels, 1) },
            W,
            H,
            PATCH_RADIUS,
            BLOCK_X,
            BLOCK_Y,
        )
        .map_err(map_err)
    }

    fn name(&self) -> String {
        "horizontal_sum_1080p".to_string()
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize]]
    }
}
