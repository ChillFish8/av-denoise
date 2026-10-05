use av_denoise_core::bench_api::kernels::nlm_bilateral;
use av_denoise_core::bench_api::prefilter::bilateral_radius;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{
    BILATERAL_SIGMA_R,
    BILATERAL_SIGMA_S,
    BLOCK_X,
    BLOCK_Y,
    HEIGHT,
    InputOutput,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_channels,
    stored_channels,
};

pub struct BilateralBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for BilateralBench<R> {
    type Input = InputOutput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let output = self.client.empty(pixels * stored_ch * size_of::<f32>());

        InputOutput {
            input,
            output,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let radius = bilateral_radius(BILATERAL_SIGMA_S);
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_bilateral::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.output.clone(), pixels * stored_ch),
                0u32,
                1.0 / (2.0 * BILATERAL_SIGMA_S * BILATERAL_SIGMA_S),
                1.0 / (2.0 * BILATERAL_SIGMA_R * BILATERAL_SIGMA_R),
                WIDTH,
                HEIGHT,
                self.channels,
                radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("bilateral_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
