use av_denoise::nlmeans::kernels::gpu_copy;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_1D,
    COPY_GRID_1D,
    H,
    W,
    block_sync,
    make_padded_frame,
    map_err,
    shapes_with_ch,
    stored_channels,
};

#[derive(Clone)]
pub struct CopyInput {
    src: Handle,
    dst: Handle,
}

pub struct CopyBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for CopyBench<R> {
    type Input = CopyInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_padded_frame(W, H, self.ch);
        let src = self.client.create_from_slice(f32::as_bytes(&frame));
        let dst = self.client.empty(frame.len() * size_of::<f32>());
        CopyInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let len = (W * H) as usize * stored_channels(self.ch) as usize;
        let total_threads = COPY_GRID_1D * BLOCK_1D;
        gpu_copy::launch::<R>(
            &self.client,
            CubeCount::new_1d(COPY_GRID_1D),
            CubeDim::new_1d(BLOCK_1D),
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.src, len, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&args.dst, len, 1) },
            len as u32,
            total_threads,
        )
        .map_err(map_err)
    }

    fn name(&self) -> String {
        format!("gpu_copy_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
