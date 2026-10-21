use av_denoise_core::bench_api::MAX_GRID_1D;
use av_denoise_core::bench_api::kernels::gpu_cast_codes;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, HEIGHT, WIDTH, block_sync, make_padded_frame, stored_channels};

const RING_SLOTS: usize = 5;
const TARGET_SLOT: u32 = 2;

#[derive(Clone)]
pub struct CastCodesInput {
    src: Handle,
    dst: Handle,
}

pub struct CastCodesBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    /// The significant bits per code, 8 or 10.
    pub depth: u32,
}

impl<R: Runtime> CastCodesBench<R> {
    fn frame_len(&self) -> usize {
        let stored_ch = stored_channels(self.channels) as usize;
        (WIDTH * HEIGHT) as usize * stored_ch
    }
}

impl<R: Runtime> Benchmark for CastCodesBench<R> {
    type Input = CastCodesInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame_len = self.frame_len();
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let ring = frame.repeat(RING_SLOTS);
        let ring_bytes = f32::as_bytes(&ring);
        let src = self.client.create_from_slice(ring_bytes);
        let dst_bytes = frame_len * RING_SLOTS * size_of::<u16>();
        let dst = self.client.empty(dst_bytes);

        CastCodesInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let frame_len = self.frame_len();
        let ring_len = frame_len * RING_SLOTS;
        let grid = (frame_len as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        let offset = TARGET_SLOT * frame_len as u32;
        let max_code = ((1u32 << self.depth) - 1) as f32;
        let cube_count = CubeCount::new_1d(grid);
        let cube_dim = CubeDim::new_1d(BLOCK_1D);

        unsafe {
            let src_arg = ArrayArg::from_raw_parts(args.src, ring_len);
            let dst_arg = ArrayArg::from_raw_parts(args.dst, ring_len);

            gpu_cast_codes::launch_unchecked::<u16, R>(
                &self.client,
                cube_count,
                cube_dim,
                src_arg,
                dst_arg,
                max_code,
                offset,
                frame_len as u32,
                total_threads,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        let suffix = if self.depth == 8 { "" } else { "_10bit" };
        format!("gpu_cast_codes/u16/1080p_{}{suffix}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize, self.channels as usize]]
    }
}
