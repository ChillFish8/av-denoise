use av_denoise::nlmeans::kernels::{nlm_noise_partial, nlm_noise_reduce};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_1D,
    BLOCK_X,
    BLOCK_Y,
    H,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    make_padded_frame,
    shapes_with_ch,
};

/// Logical channel count for the bench's YUV storage frame.
const NOISE_CHANNELS: u32 = 3;
/// Padded storage width for YUV (padded up to a vec4 lane).
const NOISE_STORED_CH: u32 = 4;

/// Both stages of the Immerkær noise estimate, dispatched back-to-back
/// against a single 1080p YUV frame. `nlm_noise_partial` reduces every
/// `BLOCK_X × BLOCK_Y` cube down to one partial per channel lane, then
/// `nlm_noise_reduce` folds every partial into the frame-level total.
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
    (W.div_ceil(BLOCK_X) * H.div_ceil(BLOCK_Y) * 4) as usize
}

impl<R: Runtime> Benchmark for NoisePartialBench<R> {
    type Input = NoiseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_padded_frame(W, H, NOISE_CHANNELS);
        let input = self.client.create_from_slice(f32::as_bytes(&frame));
        let partials = self.client.empty(partials_len() * size_of::<f32>());
        let results = self.client.empty(4 * size_of::<f32>());
        NoiseInput {
            input,
            partials,
            results,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let total_input = (W * H * NOISE_STORED_CH) as usize;
        let n_partials = partials_len();
        let num_partials = (n_partials / 4) as u32;

        unsafe {
            nlm_noise_partial::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                NOISE_STORED_CH as usize,
                ArrayArg::from_raw_parts(args.input.clone(), total_input),
                ArrayArg::from_raw_parts(args.partials.clone(), n_partials),
                0u32,
                W,
                H,
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
                ArrayArg::from_raw_parts(args.partials.clone(), n_partials),
                ArrayArg::from_raw_parts(args.results.clone(), 4),
                0u32,
                num_partials,
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
        shapes_with_ch(NOISE_CHANNELS)
    }
}
