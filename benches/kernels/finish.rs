use av_denoise::nlmeans::kernels::nlm_finish;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    H,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    map_err,
    shapes_with_ch,
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
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FinishBench<R> {
    type Input = FinishInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let accum_data = vec![0.25f32; pixels * stored];
        let accum = self.client.create_from_slice(f32::as_bytes(&accum_data));
        let ws_data = vec![1.0f32; pixels];
        let weight_sum = self.client.create_from_slice(f32::as_bytes(&ws_data));
        let mw_data = vec![0.8f32; pixels];
        let max_weight = self.client.create_from_slice(f32::as_bytes(&mw_data));
        let output = self.client.empty(pixels * stored * size_of::<f32>());
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
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_finish::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                unsafe { ArrayArg::from_raw_parts(args.input.clone(), args.frame_len) },
                unsafe { ArrayArg::from_raw_parts(args.output.clone(), pixels * stored) },
                unsafe { ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored) },
                unsafe { ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels) },
                unsafe { ArrayArg::from_raw_parts(args.max_weight.clone(), pixels) },
                0u32,
                1.0f32,
                W,
                H,
                self.ch,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("finish_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
