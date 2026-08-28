use av_denoise::nlmeans::kernels::motion::nlm_mc_block_match_coarse;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

/// Default MVTools-style block-matcher tuning at the coarse pyramid
/// level (`/2`). Mirrors the dispatcher's defaults so the bench
/// reflects steady-state cost when MC is enabled with no overrides.
const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
const SEARCH_RADIUS: u32 = 4;

/// Hierarchical coarse pass: one cube per coarse block, SAD search
/// over a `(2·r + 1)²` window on the `/2` luma pyramid level. Per-block
/// MV result is up-scaled to seed the fine pass.
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
        let coarse_w = W / 2;
        let coarse_h = H / 2;
        let centre_frame = make_synthetic_frame(coarse_w, coarse_h, 1);
        let neighbour_frame = make_synthetic_frame(coarse_w, coarse_h, 1);
        let centre = self.client.create_from_slice(f32::as_bytes(&centre_frame));
        let neighbour = self.client.create_from_slice(f32::as_bytes(&neighbour_frame));

        let fine_blocks_x = W.div_ceil(FINE_STEP);
        let fine_blocks_y = H.div_ceil(FINE_STEP);
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
        let coarse_w = W / 2;
        let coarse_h = H / 2;
        let coarse_blksize = FINE_BLKSIZE / 2;
        let coarse_step = FINE_STEP / 2;
        let coarse_scale = 2u32;
        let fine_blocks_x = W.div_ceil(FINE_STEP);
        let fine_blocks_y = H.div_ceil(FINE_STEP);
        let coarse_blocks_x = coarse_w.div_ceil(coarse_step);
        let coarse_blocks_y = coarse_h.div_ceil(coarse_step);

        let level_len = (coarse_w * coarse_h) as usize;
        let mv_len = (fine_blocks_x * fine_blocks_y * 2) as usize;
        let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            nlm_mc_block_match_coarse::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), mv_len),
                coarse_w,
                coarse_h,
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
        shapes_with_ch(1)
    }
}
