use av_denoise::nlmeans::kernels::nlm_dist_2d_weight_ref;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{
    BLOCK_X,
    BLOCK_Y,
    H,
    InputOutput,
    PATCH_RADIUS,
    Q_X,
    Q_Y,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    h2_inv_norm,
    make_padded_frame,
    map_err,
    shapes_with_ch,
    stored_channels,
};

pub struct DistWeightRefBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for DistWeightRefBench<R> {
    type Input = InputOutput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let output = self.client.empty(pixels * size_of::<f32>());
        InputOutput {
            input,
            output,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        nlm_dist_2d_weight_ref::launch::<R>(
            &self.client,
            cube_count_2d(),
            cube_dim_2d(),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.input, args.frame_len, stored) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.output, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(0u32),
            ScalarArg::new(Q_X),
            ScalarArg::new(Q_Y),
            ScalarArg::new(h2_inv_norm()),
            W,
            H,
            self.ch,
            PATCH_RADIUS,
            BLOCK_X,
            BLOCK_Y,
        )
        .map_err(map_err)
    }

    fn name(&self) -> String {
        format!("dist_2d_weight_ref_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
