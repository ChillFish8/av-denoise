use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, nlm_mc_block_match_coarse};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

// The library's default Mvtools block geometry, so the bench reflects the cost of motion
// compensation with no overrides.
const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
const SEARCH_RADIUS: u32 = 4;

/// The hierarchical coarse pass on the half-resolution luma pyramid level.
///
/// One cube per coarse block runs a SAD search over a `(2·r + 1)²` window, and each block's
/// vector is scaled up to seed the fine pass.
pub struct BlockMatchCoarseBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct CoarseInput {
    pub centre: Handle,
    pub neighbour: Handle,
    pub mv_field: Handle,
}

impl<R: Runtime> Benchmark for BlockMatchCoarseBench<R> {
    type Input = CoarseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let coarse_width = WIDTH / 2;
        let coarse_height = HEIGHT / 2;
        let centre_frame = make_synthetic_frame(coarse_width, coarse_height, 1);
        let neighbour_frame = make_synthetic_frame(coarse_width, coarse_height, 1);
        let centre_bytes = f32::as_bytes(&centre_frame);
        let centre = self.client.create_from_slice(centre_bytes);
        let neighbour_bytes = f32::as_bytes(&neighbour_frame);
        let neighbour = self.client.create_from_slice(neighbour_bytes);

        let fine_blocks_x = WIDTH.div_ceil(FINE_STEP);
        let fine_blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let mv_field = self
            .client
            .empty((fine_blocks_x * fine_blocks_y * 2) as usize * size_of::<i32>());

        CoarseInput {
            centre,
            neighbour,
            mv_field,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let coarse_width = WIDTH / 2;
        let coarse_height = HEIGHT / 2;
        let coarse_blksize = FINE_BLKSIZE / 2;
        let coarse_step = FINE_STEP / 2;
        let coarse_scale = 2u32;
        let fine_blocks_x = WIDTH.div_ceil(FINE_STEP);
        let fine_blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let coarse_blocks_x = coarse_width.div_ceil(coarse_step);
        let coarse_blocks_y = coarse_height.div_ceil(coarse_step);

        let level_len = (coarse_width * coarse_height) as usize;
        let mv_len = (fine_blocks_x * fine_blocks_y * 2) as usize;
        let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            nlm_mc_block_match_coarse::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), mv_len),
                coarse_width,
                coarse_height,
                coarse_blksize,
                coarse_step,
                SEARCH_RADIUS,
                coarse_scale,
                fine_blocks_x,
                fine_blocks_y,
                FINE_STEP,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "mc_block_match_coarse_540p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
