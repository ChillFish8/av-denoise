use av_denoise_core::nlmeans::kernels::motion::nlm_mc_warp;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

const FINE_STEP: u32 = 8;

/// Apply the MV field to warp a neighbour into spatial alignment with
/// the centre. Cost is roughly one packed `Vector<f32, N>` read +
/// store per output pixel.
pub struct WarpBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
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
        let frame = make_padded_frame(W, H, self.ch);
        let src = self.client.create_from_slice(f32::as_bytes(&frame));
        let dst = self.client.empty(frame.len() * size_of::<f32>());

        let blocks_x = W.div_ceil(FINE_STEP);
        let blocks_y = H.div_ceil(FINE_STEP);
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
        let stored = stored_channels(self.ch) as usize;
        let block_x = 16u32;
        let block_y = 16u32;
        let grid = CubeCount::new_2d(W.div_ceil(block_x), H.div_ceil(block_y));
        let dim = CubeDim::new_2d(block_x, block_y);
        let blocks_x = W.div_ceil(FINE_STEP);
        let blocks_y = H.div_ceil(FINE_STEP);

        unsafe {
            nlm_mc_warp::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored,
                ArrayArg::from_raw_parts(args.src.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.dst.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), args.mv_len),
                0u32,
                0u32,
                FINE_STEP,
                blocks_x,
                blocks_y,
                W,
                H,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("mc_warp_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
