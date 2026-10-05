use av_denoise_core::bench_api::kernels::nlm_horizontal_sum;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, HEIGHT, PATCH_RADIUS, WIDTH, block_sync, cube_count_2d, cube_dim_2d};

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
        let pixels = (WIDTH * HEIGHT) as usize;
        let data = vec![0.5f32; pixels];
        let data_bytes = f32::as_bytes(&data);
        let input = self.client.create_from_slice(data_bytes);
        let output = self.client.empty(pixels * size_of::<f32>());

        HSumInput { input, output }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_horizontal_sum::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(args.input.clone(), pixels),
                ArrayArg::from_raw_parts(args.output.clone(), pixels),
                WIDTH,
                HEIGHT,
                PATCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "horizontal_sum_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize]]
    }
}
