use av_denoise::nlmeans::kernels::nlm_vertical_weight;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::horizontal_sum::HSumInput;
use super::{
    BLOCK_X,
    BLOCK_Y,
    H,
    PATCH_RADIUS,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    h2_inv_norm,
    map_err,
};

pub struct VWeightBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for VWeightBench<R> {
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
        unsafe {
            nlm_vertical_weight::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                unsafe { ArrayArg::from_raw_parts(args.input.clone(), pixels) },
                unsafe { ArrayArg::from_raw_parts(args.output.clone(), pixels) },
                h2_inv_norm(),
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
        "vertical_weight_1080p".to_string()
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize]]
    }
}
