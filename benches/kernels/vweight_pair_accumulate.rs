use av_denoise::nlmeans::kernels::nlm_vweight_pair_accumulate;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_X,
    BLOCK_Y,
    H,
    PATCH_RADIUS,
    Q_X,
    Q_Y,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    h2_inv_norm,
    make_padded_frame,
    shapes_with_ch,
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
    weight_sq_sum_dummy: Handle,
    confidence_dummy: Handle,
    frame_len: usize,
}

pub struct VWeightPairAccBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for VWeightPairAccBench<R> {
    type Input = VWeightPairAccInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let hsum = vec![0.5f32; pixels];
        let hsum_fwd = self.client.create_from_slice(f32::as_bytes(&hsum));
        let hsum_bwd = self.client.create_from_slice(f32::as_bytes(&hsum));
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let accum = self.client.empty(pixels * stored * size_of::<f32>());
        let weight_sum = self.client.empty(pixels * size_of::<f32>());
        let max_weight = self.client.empty(pixels * size_of::<f32>());
        let weight_sq_sum_dummy = self.client.empty(size_of::<f32>());
        let confidence_dummy = self.client.empty(size_of::<f32>());
        VWeightPairAccInput {
            hsum_fwd,
            hsum_bwd,
            input,
            accum,
            weight_sum,
            max_weight,
            weight_sq_sum_dummy,
            confidence_dummy,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_vweight_pair_accumulate::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.hsum_fwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.hsum_bwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.weight_sq_sum_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                false,
                false,
                0u32,
                0u32,
                Q_X,
                Q_Y,
                h2_inv_norm(),
                0.0f32,
                W,
                H,
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
        format!("vweight_pair_accumulate_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
