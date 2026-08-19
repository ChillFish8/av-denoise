use av_denoise::collab::kernels::aggregate::{collab_normalise, collab_zero_accum};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, H, W, block_sync, stored_channels};

/// Divides the fixed-point accumulators the filters scattered into back
/// out to a finished 1080p frame plane, for each channel mode. Cost
/// scales with `stored_ch`, since one accumulator slot per channel is
/// read for every pixel.
pub struct CollabNormaliseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

#[derive(Clone)]
pub struct CollabNormaliseInput {
    pub accum: Handle,
    pub wsum: Handle,
    pub output: Handle,
}

impl<R: Runtime> Benchmark for CollabNormaliseBench<R> {
    type Input = CollabNormaliseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored = stored_channels(self.ch) as usize;
        let pixels = (W * H) as usize;

        // Shaped like a real pass: a few dozen contributions per pixel,
        // each already scaled into the accumulator's fixed point.
        let accum_data: Vec<i32> = (0..pixels * stored).map(|i| (i % 97) as i32 * 8192).collect();
        let wsum_data: Vec<i32> = (0..pixels).map(|i| ((i % 31) + 20) as i32 * 8192).collect();

        CollabNormaliseInput {
            accum: self.client.create_from_slice(i32::as_bytes(&accum_data)),
            wsum: self.client.create_from_slice(i32::as_bytes(&wsum_data)),
            output: self.client.empty(pixels * stored * size_of::<f32>()),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored = stored_channels(self.ch) as usize;
        let pixels = (W * H) as usize;

        unsafe {
            collab_normalise::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(W.div_ceil(BLOCK_X), H.div_ceil(BLOCK_Y)),
                CubeDim::new_2d(BLOCK_X, BLOCK_Y),
                stored,
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(args.output.clone(), pixels * stored),
                // Single-frame region, the same `frame_offset = 0` every
                // shipped single-frame caller passes (see
                // `src/collab/pipeline.rs`); this bench measures one
                // frame's worth of normalisation, not a cross-frame ring.
                0u32,
                W,
                H,
                self.ch,
                stored as u32,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_normalise_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, self.ch as usize]]
    }
}

/// Clears both accumulators, which runs once before each of the two
/// filter passes.
pub struct CollabZeroAccumBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for CollabZeroAccumBench<R> {
    type Input = CollabNormaliseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored = stored_channels(self.ch) as usize;
        let pixels = (W * H) as usize;
        CollabNormaliseInput {
            accum: self.client.empty(pixels * stored * size_of::<i32>()),
            wsum: self.client.empty(pixels * size_of::<i32>()),
            output: self.client.empty(size_of::<f32>()),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored = stored_channels(self.ch) as usize;
        let pixels = (W * H) as usize;
        let dim = 256u32;

        unsafe {
            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(((pixels * stored) as u32).div_ceil(dim)),
                CubeDim::new_1d(dim),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                // Single-frame region, the same `frame_offset = 0` every
                // shipped single-frame caller passes.
                0u32,
                pixels as u32,
                stored as u32,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_zero_accum_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, self.ch as usize]]
    }
}
