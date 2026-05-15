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
    map_err,
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
        VWeightPairAccInput {
            hsum_fwd,
            hsum_bwd,
            input,
            accum,
            weight_sum,
            max_weight,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        nlm_vweight_pair_accumulate::launch::<R>(
            &self.client,
            cube_count_2d(),
            cube_dim_2d(),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.hsum_fwd, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.hsum_bwd, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.input, args.frame_len, stored) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.accum, pixels * stored, stored) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.max_weight, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(0u32),
            ScalarArg::new(Q_X),
            ScalarArg::new(Q_Y),
            ScalarArg::new(h2_inv_norm()),
            W,
            H,
            PATCH_RADIUS,
            BLOCK_X,
            BLOCK_Y,
        )
        .map_err(map_err)
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
