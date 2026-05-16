use av_denoise::nlmeans::kernels::nlm_accumulate;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    H,
    Q_X,
    Q_Y,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_ch,
    stored_channels,
};

#[derive(Clone)]
pub struct AccumulateInput {
    input: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    weights: Handle,
    frame_len: usize,
}

pub struct AccumulateBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for AccumulateBench<R> {
    type Input = AccumulateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let weights_data = vec![0.5f32; pixels];
        let weights = self.client.create_from_slice(f32::as_bytes(&weights_data));
        let accum = self.client.empty(pixels * stored * size_of::<f32>());
        let weight_sum = self.client.empty(pixels * size_of::<f32>());
        let max_weight = self.client.empty(pixels * size_of::<f32>());
        AccumulateInput {
            input,
            accum,
            weight_sum,
            max_weight,
            weights,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_accumulate::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.weights.clone(), pixels),
                ArrayArg::from_raw_parts(args.weights.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                0u32,
                0u32,
                Q_X,
                Q_Y,
                W,
                H,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("accumulate_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
