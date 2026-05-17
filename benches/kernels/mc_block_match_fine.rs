use av_denoise::nlmeans::kernels::motion::nlm_mc_block_match_fine;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
const SEARCH_RADIUS: u32 = 4;

/// Fine refinement pass at full resolution. Reads no seed (the bench
/// uses `use_seed = 0` so the kernel cost is bounded purely by the
/// `(2·r + 1)²` search window per block — fair across runs even if
/// the coarse bench tuning changes).
pub struct BlockMatchFineBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct FineInput {
    pub centre: Handle,
    pub neighbour: Handle,
    pub mv_field: Handle,
}

impl<R: Runtime> Benchmark for BlockMatchFineBench<R> {
    type Input = FineInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let centre_frame = make_synthetic_frame(W, H, 1);
        let neighbour_frame = make_synthetic_frame(W, H, 1);
        let centre = self.client.create_from_slice(f32::as_bytes(&centre_frame));
        let neighbour = self.client.create_from_slice(f32::as_bytes(&neighbour_frame));

        let blocks_x = W.div_ceil(FINE_STEP);
        let blocks_y = H.div_ceil(FINE_STEP);
        let mv_field = self
            .client
            .empty((blocks_x * blocks_y * 2) as usize * size_of::<i32>());

        FineInput {
            centre,
            neighbour,
            mv_field,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let level_len = (W * H) as usize;
        let blocks_x = W.div_ceil(FINE_STEP);
        let blocks_y = H.div_ceil(FINE_STEP);
        let mv_len = (blocks_x * blocks_y * 2) as usize;

        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            nlm_mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), mv_len),
                W,
                H,
                FINE_BLKSIZE,
                FINE_STEP,
                SEARCH_RADIUS,
                0u32, // use_seed = 0; bench worst-case without coarse seed
                blocks_x,
                blocks_y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "mc_block_match_fine_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(1)
    }
}
