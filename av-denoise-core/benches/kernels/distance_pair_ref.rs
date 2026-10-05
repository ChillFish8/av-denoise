use av_denoise_core::bench_api::kernels::nlm_distance_pair_ref;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::distance_pair::DistancePairInput;
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

pub struct DistancePairRefBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for DistancePairRefBench<R> {
    type Input = DistancePairInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
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
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels) as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_distance_pair_ref::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.dist_fwd.clone(), pixels),
                ArrayArg::from_raw_parts(args.dist_bwd.clone(), pixels),
                0u32,
                0u32,
                0u32,
                Q_X,
                Q_Y,
                WIDTH,
                HEIGHT,
                self.channels,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("distance_pair_ref_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
