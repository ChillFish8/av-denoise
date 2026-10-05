use av_denoise_core::bench_api::kernels::nlm_vweight_pair_accumulate;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_X,
    BLOCK_Y,
    HEIGHT,
    PATCH_RADIUS,
    Q_X,
    Q_Y,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    h2_inv_norm,
    make_padded_frame,
    shapes_with_channels,
    stored_channels,
};

#[derive(Clone)]
pub struct VWeightPairAccInput {
    hsum_fwd: Handle,
    hsum_bwd: Handle,
    input: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    confidence_dummy: Handle,
    frame_len: usize,
}

pub struct VWeightPairAccBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for VWeightPairAccBench<R> {
    type Input = VWeightPairAccInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let hsum = vec![0.5f32; pixels];
        let hsum_bytes = f32::as_bytes(&hsum);
        let hsum_fwd = self.client.create_from_slice(hsum_bytes);
        let hsum_bwd = self.client.create_from_slice(hsum_bytes);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let accum = self.client.empty(pixels * stored_ch * size_of::<f32>());
        let weight_sum = self.client.empty(pixels * size_of::<f32>());
        let max_weight = self.client.empty(pixels * size_of::<f32>());
        let confidence_dummy = self.client.empty(size_of::<f32>());

        VWeightPairAccInput {
            hsum_fwd,
            hsum_bwd,
            input,
            accum,
            weight_sum,
            max_weight,
            confidence_dummy,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();
        let inv_norm = h2_inv_norm();

        unsafe {
            nlm_vweight_pair_accumulate::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.hsum_fwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.hsum_bwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                false,
                0u32,
                0u32,
                Q_X,
                Q_Y,
                inv_norm,
                0.0f32,
                WIDTH,
                HEIGHT,
                PATCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
                1u32,
                1u32,
                1u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("vweight_pair_accumulate_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
