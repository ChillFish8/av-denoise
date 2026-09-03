use av_denoise_core::Depth;
use av_denoise_core::nlmeans::kernels::gpu_unpack_wire;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, COPY_GRID_1D, H, W, block_sync, shapes_with_ch, stored_channels};

#[derive(Clone)]
pub struct UnpackWireInput {
    src: Handle,
    dst: Handle,
}

pub struct UnpackWireBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
    /// Covers both wire codecs, 4 samples per word at 8-bit and 2 above it.
    pub depth: Depth,
}

impl<R: Runtime> UnpackWireBench<R> {
    fn samples(&self) -> u32 {
        W * H * self.ch
    }

    fn words(&self) -> u32 {
        self.samples().div_ceil(self.depth.wire_pack().samples_per_word())
    }

    fn elements(&self) -> u32 {
        W * H * stored_channels(self.ch)
    }

    /// Wire bytes whose samples spread over the depth's whole range, so
    /// the run measures a realistic spread of values rather than every
    /// lane in a wave decoding to the same one.
    fn wire(&self) -> Vec<u8> {
        let bytes = self.depth.bytes_per_sample();
        let mask = (1u32 << self.depth.bits()) - 1;
        let mut wire = vec![0u8; self.words() as usize * size_of::<u32>()];

        // An xorshift rather than a multiply, so the samples are not all
        // even and the fill reaches every value the depth can express.
        for (i, chunk) in wire.chunks_exact_mut(bytes).enumerate() {
            let mut hash = i as u32 ^ 0x9E37_79B9;
            hash ^= hash << 13;
            hash ^= hash >> 17;
            hash ^= hash << 5;
            chunk.copy_from_slice(&(hash & mask).to_le_bytes()[..bytes]);
        }

        wire
    }
}

impl<R: Runtime> Benchmark for UnpackWireBench<R> {
    type Input = UnpackWireInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let wire = self.wire();
        let src = self.client.create_from_slice(&wire);
        let dst = self.client.empty(self.elements() as usize * size_of::<f32>());
        UnpackWireInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = W * H;
        let stored_ch = stored_channels(self.ch);
        // One `Depth` yields both, so the bench cannot pair a scale with
        // the wrong lane count the way hand-written literals could.
        let pack = self.depth.wire_pack();
        let total_threads = COPY_GRID_1D * BLOCK_1D;

        unsafe {
            gpu_unpack_wire::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(COPY_GRID_1D),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.src.clone(), self.words() as usize),
                ArrayArg::from_raw_parts(args.dst.clone(), self.elements() as usize),
                pack.max(),
                0u32,
                pixels,
                self.ch,
                stored_ch,
                pack.samples_per_word(),
                self.elements(),
                total_threads,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("gpu_unpack_wire_1080p_{:?}_{}", self.depth, self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
