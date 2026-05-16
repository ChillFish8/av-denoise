use av_denoise::nlmeans::kernels::nlm_fused_pair_accumulate_ref;
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
pub struct FusedPairRefInput {
    input: Handle,
    reference: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    frame_len: usize,
}

pub struct FusedPairRefBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FusedPairRefBench<R> {
    type Input = FusedPairRefInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let reference = self.client.create_from_slice(f32::as_bytes(&frame));
        let accum = self.client.empty(pixels * stored * size_of::<f32>());
        let weight_sum = self.client.empty(pixels * size_of::<f32>());
        let max_weight = self.client.empty(pixels * size_of::<f32>());
        FusedPairRefInput {
            input,
            reference,
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
            nlm_fused_pair_accumulate_ref::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.reference.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                0u32,
                0u32,
                0u32,
                Q_X,
                Q_Y,
                -Q_X,
                -Q_Y,
                h2_inv_norm(),
                W,
                H,
                self.ch,
                PATCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("fused_pair_accumulate_ref_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
