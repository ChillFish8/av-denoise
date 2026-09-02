use av_denoise_core::nlmeans::kernels::gpu_pack_wire;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, COPY_GRID_1D, H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

#[derive(Clone)]
pub struct PackWireInput {
    src: Handle,
    dst: Handle,
}

pub struct PackWireBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
    /// Wire bytes per sample, so the bench covers both codecs.
    pub bytes_per_sample: u32,
    pub depth_name: &'static str,
}

impl<R: Runtime> PackWireBench<R> {
    fn samples(&self) -> u32 {
        W * H * self.ch
    }

    fn words(&self) -> u32 {
        self.samples().div_ceil(4 / self.bytes_per_sample)
    }
}

impl<R: Runtime> Benchmark for PackWireBench<R> {
    type Input = PackWireInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_padded_frame(W, H, self.ch);
        let src = self.client.create_from_slice(f32::as_bytes(&frame));
        let dst = self.client.empty(self.words() as usize * size_of::<u32>());
        PackWireInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = W * H;
        let stored_ch = stored_channels(self.ch);
        let samples_per_word = 4 / self.bytes_per_sample;
        let total_threads = COPY_GRID_1D * BLOCK_1D;

        unsafe {
            gpu_pack_wire::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(COPY_GRID_1D),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.src.clone(), (pixels * stored_ch) as usize),
                ArrayArg::from_raw_parts(args.dst.clone(), self.words() as usize),
                255.0f32,
                pixels,
                self.ch,
                stored_ch,
                self.ch,
                false,
                samples_per_word,
                self.words(),
                total_threads,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("gpu_pack_wire_1080p_{}_{}", self.depth_name, self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
