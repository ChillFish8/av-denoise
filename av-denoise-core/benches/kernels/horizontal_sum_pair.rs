use av_denoise::nlmeans::kernels::nlm_horizontal_sum_pair;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, H, PATCH_RADIUS, W, block_sync, cube_count_2d, cube_dim_2d};

#[derive(Clone)]
pub struct HSumPairInput {
    input_fwd: Handle,
    input_bwd: Handle,
    output_fwd: Handle,
    output_bwd: Handle,
}

pub struct HSumPairBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for HSumPairBench<R> {
    type Input = HSumPairInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let data = vec![0.5f32; pixels];
        let input_fwd = self.client.create_from_slice(f32::as_bytes(&data));
        let input_bwd = self.client.create_from_slice(f32::as_bytes(&data));
        let output_fwd = self.client.empty(pixels * size_of::<f32>());
        let output_bwd = self.client.empty(pixels * size_of::<f32>());
        HSumPairInput {
            input_fwd,
            input_bwd,
            output_fwd,
            output_bwd,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        unsafe {
            nlm_horizontal_sum_pair::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                ArrayArg::from_raw_parts(args.input_fwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.input_bwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.output_fwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.output_bwd.clone(), pixels),
                W,
                H,
                PATCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "horizontal_sum_pair_1080p".to_string()
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize]]
    }
}
