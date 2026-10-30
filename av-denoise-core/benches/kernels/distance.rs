use av_denoise_core::bench_api::kernels::nlm_distance;
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
    make_synthetic_frame,
    shapes_with_channels,
};

#[derive(Clone)]
pub struct DistanceInput {
    pub input: Handle,
    pub dist: Handle,
    pub frame_len: usize,
}

pub struct DistanceBench {
    pub client: Client,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl Benchmark for DistanceBench {
    type Input = DistanceInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let frame = make_synthetic_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let dist = self.client.empty(pixels * size_of::<f32>());

        DistanceInput {
            input,
            dist,
            frame_len: frame.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (WIDTH * HEIGHT) as usize;
        let channels = self.channels as usize;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_distance::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                channels,
                BufferArg::from_raw_parts(args.input.clone(), args.frame_len),
                BufferArg::from_raw_parts(args.dist.clone(), pixels),
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
        format!("distance_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
