use av_denoise_core::bench_api::kernels::nlm_finish;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    HEIGHT,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_channels,
    stored_channels,
};

#[derive(Clone)]
pub struct FinishInput {
    input: Handle,
    output: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    frame_len: usize,
}

pub struct FinishBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for FinishBench<R> {
    type Input = FinishInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);

        let accum_data = vec![0.25f32; pixels * stored_ch];
        let accum_bytes = f32::as_bytes(&accum_data);
        let accum = self.client.create_from_slice(accum_bytes);

        let weight_sum_data = vec![1.0f32; pixels];
        let weight_sum_bytes = f32::as_bytes(&weight_sum_data);
        let weight_sum = self.client.create_from_slice(weight_sum_bytes);

        let max_weight_data = vec![0.8f32; pixels];
        let max_weight_bytes = f32::as_bytes(&max_weight_data);
        let max_weight = self.client.create_from_slice(max_weight_bytes);

        let output = self.client.empty(pixels * stored_ch * size_of::<f32>());

        FinishInput {
            input,
            output,
            accum,
            weight_sum,
            max_weight,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_finish::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.output.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                0u32,
                0u32,
                1.0f32,
                WIDTH,
                HEIGHT,
                self.channels,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("finish_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
