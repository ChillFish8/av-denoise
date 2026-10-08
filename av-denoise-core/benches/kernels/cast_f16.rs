use av_denoise_core::bench_api::MAX_GRID_1D;
use av_denoise_core::bench_api::kernels::gpu_cast_f16;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, block_sync, make_padded_frame, stored_channels};

const RING_SLOTS: usize = 5;
const TARGET_SLOT: u32 = 2;

#[derive(Clone)]
pub struct CastInput {
    src: Handle,
    dst: Handle,
}

pub struct CastF16Bench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> CastF16Bench<R> {
    fn frame_len(&self) -> usize {
        let stored_ch = stored_channels(self.channels) as usize;
        (self.width * self.height) as usize * stored_ch
    }
}

impl<R: Runtime> Benchmark for CastF16Bench<R> {
    type Input = CastInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame_len = self.frame_len();
        let frame = make_padded_frame(self.width, self.height, self.channels);
        let ring = frame.repeat(RING_SLOTS);
        let ring_bytes = f32::as_bytes(&ring);
        let src = self.client.create_from_slice(ring_bytes);
        let dst_bytes = frame_len * RING_SLOTS * size_of::<half::f16>();
        let dst = self.client.empty(dst_bytes);

        CastInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let frame_len = self.frame_len();
        let ring_len = frame_len * RING_SLOTS;
        let grid = (frame_len as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        let offset = TARGET_SLOT * frame_len as u32;

        unsafe {
            gpu_cast_f16::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.src.clone(), ring_len),
                ArrayArg::from_raw_parts(args.dst.clone(), ring_len),
                offset,
                frame_len as u32,
                total_threads,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        let rows = self.height;
        format!("gpu_cast_f16_{rows}p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![
            self.width as usize,
            self.height as usize,
            self.channels as usize,
        ]]
    }
}
