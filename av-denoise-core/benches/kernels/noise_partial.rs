use av_denoise_core::bench_api::kernels::{nlm_noise_partial, nlm_noise_reduce};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_1D,
    BLOCK_X,
    BLOCK_Y,
    HEIGHT,
    WIDTH,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_channels,
};

/// Logical channel count for the bench's YUV storage frame.
const NOISE_CHANNELS: u32 = 3;
/// Padded storage width for YUV, padded up to a vec4 lane.
const NOISE_STORED_CH: u32 = 4;

/// Both stages of the Immerkær noise estimate, run back to back on one 1080p YUV frame.
///
/// `nlm_noise_partial` reduces every `BLOCK_X × BLOCK_Y` cube to one partial per channel lane,
/// then `nlm_noise_reduce` folds every partial into the frame-level total.
pub struct NoisePartialBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct NoiseInput {
    pub input: Handle,
    pub partials: Handle,
    pub results: Handle,
}

fn partials_len() -> usize {
    (WIDTH.div_ceil(BLOCK_X) * HEIGHT.div_ceil(BLOCK_Y) * 4) as usize
}

impl<R: Runtime> Benchmark for NoisePartialBench<R> {
    type Input = NoiseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_padded_frame(WIDTH, HEIGHT, NOISE_CHANNELS);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let partial_lanes = partials_len();
        let partials = self.client.empty(partial_lanes * size_of::<f32>());
        let results = self.client.empty(4 * size_of::<f32>());

        NoiseInput {
            input,
            partials,
            results,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let total_input = (WIDTH * HEIGHT * NOISE_STORED_CH) as usize;
        let partial_lanes = partials_len();
        let partial_count = (partial_lanes / 4) as u32;
        let cube_count = cube_count_2d();
        let cube_dim = cube_dim_2d();

        unsafe {
            nlm_noise_partial::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                NOISE_STORED_CH as usize,
                ArrayArg::from_raw_parts(args.input.clone(), total_input),
                ArrayArg::from_raw_parts(args.partials.clone(), partial_lanes),
                0u32,
                WIDTH,
                HEIGHT,
                NOISE_CHANNELS,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        unsafe {
            nlm_noise_reduce::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.partials.clone(), partial_lanes),
                ArrayArg::from_raw_parts(args.results.clone(), 4),
                0u32,
                partial_count,
                BLOCK_1D,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "noise_estimate_1080p_yuv".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(NOISE_CHANNELS)
    }
}
