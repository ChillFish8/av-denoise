use av_denoise_core::bench_api::kernels::nlm_accumulate;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    HEIGHT,
    Q_X,
    Q_Y,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_channels,
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
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for AccumulateBench<R> {
    type Input = AccumulateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);

        let weights_data = vec![0.5f32; pixels];
        let weights_bytes = f32::as_bytes(&weights_data);
        let weights = self.client.create_from_slice(weights_bytes);
        let accum = self.client.empty(pixels * stored_ch * size_of::<f32>());
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
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_accumulate::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.weights.clone(), pixels),
                ArrayArg::from_raw_parts(args.weights.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                0u32,
                0u32,
                Q_X,
                Q_Y,
                WIDTH,
                HEIGHT,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("accumulate_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
