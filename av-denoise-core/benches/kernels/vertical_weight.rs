use av_denoise_core::bench_api::DERIVED_WORDS;
use av_denoise_core::bench_api::kernels::nlm_vertical_weight;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::horizontal_sum::HSumInput;
use super::{
    BLOCK_X,
    BLOCK_Y,
    HEIGHT,
    PATCH_RADIUS,
    SEARCH_RADIUS,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    window_noise,
};

/// The row sums and weights, the derived buffer and a spatial table of zeros.
#[derive(Clone)]
pub struct VWeightInput {
    sums: HSumInput,
    noise: Handle,
    offsets: Handle,
    offsets_len: usize,
}

pub struct VWeightBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for VWeightBench<R> {
    type Input = VWeightInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let data = vec![0.5f32; pixels];
        let data_bytes = f32::as_bytes(&data);
        let input = self.client.create_from_slice(data_bytes);
        let output = self.client.empty(pixels * size_of::<f32>());
        let sums = HSumInput { input, output };

        let noise = window_noise(&self.client);
        let side = (2 * SEARCH_RADIUS + 1) as usize;
        let offsets_host = vec![0.0f32; side * side];
        let offsets_bytes = f32::as_bytes(&offsets_host);
        let offsets = self.client.create_from_slice(offsets_bytes);

        VWeightInput {
            sums,
            noise,
            offsets,
            offsets_len: offsets_host.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_vertical_weight::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(args.sums.input.clone(), pixels),
                ArrayArg::from_raw_parts(args.sums.output.clone(), pixels),
                ArrayArg::from_raw_parts(args.noise.clone(), DERIVED_WORDS as usize),
                ArrayArg::from_raw_parts(args.offsets.clone(), args.offsets_len),
                0u32,
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
        "vertical_weight_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize]]
    }
}
