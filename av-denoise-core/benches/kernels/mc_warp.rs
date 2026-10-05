use av_denoise_core::bench_api::kernels::motion::nlm_mc_warp;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_padded_frame, shapes_with_channels, stored_channels};

const FINE_STEP: u32 = 8;

/// Warps a neighbour into spatial alignment with the centre using the motion field.
///
/// Cost is roughly one packed `Vector<f32, N>` read and store per output pixel.
pub struct WarpBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

#[derive(Clone)]
pub struct WarpInput {
    pub src: Handle,
    pub dst: Handle,
    pub mv_field: Handle,
    pub frame_len: usize,
    pub mv_len: usize,
}

impl<R: Runtime> Benchmark for WarpBench<R> {
    type Input = WarpInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let src = self.client.create_from_slice(frame_bytes);
        let dst = self.client.empty(frame.len() * size_of::<f32>());

        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let mv_len = (blocks_x * blocks_y * 2) as usize;
        let mv_field = self.client.empty(mv_len * size_of::<i32>());

        WarpInput {
            src,
            dst,
            mv_field,
            frame_len: frame.len(),
            mv_len,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let block_x = 16u32;
        let block_y = 16u32;
        let cubes_x = WIDTH.div_ceil(block_x);
        let cubes_y = HEIGHT.div_ceil(block_y);
        let grid = CubeCount::new_2d(cubes_x, cubes_y);
        let dim = CubeDim::new_2d(block_x, block_y);
        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);

        unsafe {
            nlm_mc_warp::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored_ch,
                ArrayArg::from_raw_parts(args.src.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.dst.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), args.mv_len),
                0u32,
                0u32,
                FINE_STEP,
                blocks_x,
                blocks_y,
                WIDTH,
                HEIGHT,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("mc_warp_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
