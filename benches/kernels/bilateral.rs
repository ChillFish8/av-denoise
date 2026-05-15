use av_denoise::nlmeans::kernels::nlm_bilateral;
use av_denoise::nlmeans::prefilter::bilateral_radius;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{
    BILATERAL_SIGMA_R,
    BILATERAL_SIGMA_S,
    BLOCK_X,
    BLOCK_Y,
    H,
    InputOutput,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    map_err,
    shapes_with_ch,
    stored_channels,
};

pub struct BilateralBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for BilateralBench<R> {
    type Input = InputOutput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let output = self.client.empty(pixels * stored * size_of::<f32>());
        InputOutput {
            input,
            output,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let radius = bilateral_radius(BILATERAL_SIGMA_S);
        nlm_bilateral::launch::<R>(
            &self.client,
            cube_count_2d(),
            cube_dim_2d(),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.input, args.frame_len, stored) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.output, pixels * stored, stored) },
            ScalarArg::new(0u32),
            ScalarArg::new(1.0 / (2.0 * BILATERAL_SIGMA_S * BILATERAL_SIGMA_S)),
            ScalarArg::new(1.0 / (2.0 * BILATERAL_SIGMA_R * BILATERAL_SIGMA_R)),
            W,
            H,
            self.ch,
            stored as u32,
            radius,
            BLOCK_X,
            BLOCK_Y,
        )
        .map_err(map_err)
    }

    fn name(&self) -> String {
        format!("bilateral_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
