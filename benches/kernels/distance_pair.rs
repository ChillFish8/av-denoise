use av_denoise::nlmeans::kernels::nlm_distance_pair;
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
    map_err,
    shapes_with_ch,
    stored_channels,
};

#[derive(Clone)]
pub struct DistancePairInput {
    pub input: Handle,
    pub dist_fwd: Handle,
    pub dist_bwd: Handle,
    pub frame_len: usize,
}

pub struct DistancePairBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for DistancePairBench<R> {
    type Input = DistancePairInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let frame = make_padded_frame(W, H, self.ch);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let dist_fwd = self.client.empty(pixels * size_of::<f32>());
        let dist_bwd = self.client.empty(pixels * size_of::<f32>());
        DistancePairInput {
            input,
            dist_fwd,
            dist_bwd,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        nlm_distance_pair::launch::<R>(
            &self.client,
            cube_count_2d(),
            cube_dim_2d(),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.input, args.frame_len, stored) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.dist_fwd, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.dist_bwd, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(0u32),
            ScalarArg::new(0u32),
            ScalarArg::new(Q_X),
            ScalarArg::new(Q_Y),
            W,
            H,
            self.ch,
        )
        .map_err(map_err)
    }

    fn name(&self) -> String {
        format!("distance_pair_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
