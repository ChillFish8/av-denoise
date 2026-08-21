use av_denoise::nlmeans::kernels::motion::nlm_mc_block_match_fine;
use av_denoise::nlmeans::motion::{DEFAULT_BLKSIZE, DEFAULT_OVERLAP};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

const CONF_STEP: u32 = DEFAULT_BLKSIZE - DEFAULT_OVERLAP;
const SEARCH_RADIUS: u32 = 0;

// Mirrors `motion::confidence::THSAD_PIXEL` / `thsad`, duplicated here
// since those host helpers are crate-internal and unreachable from the
// bench binary.
const THSAD_PIXEL: f32 = 0.02;

/// The no-MC confidence pass. A single-candidate SAD (`search_radius =
/// 0`, no seed) at the library's default block geometry, the cost
/// profile of confidence weighting when motion compensation itself is
/// off. Cheaper than [`super::mc_block_match_fine::BlockMatchFineBench`],
/// which sweeps a real `(2·4 + 1)²` search window.
pub struct McConfidenceBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct ConfidenceInput {
    pub centre: Handle,
    pub neighbour: Handle,
    pub mv_scratch: Handle,
    pub confidence: Handle,
}

impl<R: Runtime> Benchmark for McConfidenceBench<R> {
    type Input = ConfidenceInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let centre_frame = make_synthetic_frame(W, H, 1);
        let neighbour_frame = make_synthetic_frame(W, H, 1);
        let centre = self.client.create_from_slice(f32::as_bytes(&centre_frame));
        let neighbour = self.client.create_from_slice(f32::as_bytes(&neighbour_frame));

        let blocks_x = W.div_ceil(CONF_STEP);
        let blocks_y = H.div_ceil(CONF_STEP);
        let mv_scratch = self
            .client
            .empty((blocks_x * blocks_y * 2) as usize * size_of::<i32>());
        let confidence = self
            .client
            .empty((blocks_x * blocks_y) as usize * size_of::<f32>());

        ConfidenceInput {
            centre,
            neighbour,
            mv_scratch,
            confidence,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let level_len = (W * H) as usize;
        let blocks_x = W.div_ceil(CONF_STEP);
        let blocks_y = H.div_ceil(CONF_STEP);
        let mv_len = (blocks_x * blocks_y * 2) as usize;
        let conf_len = (blocks_x * blocks_y) as usize;
        let thsad = (DEFAULT_BLKSIZE * DEFAULT_BLKSIZE) as f32 * THSAD_PIXEL;

        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            nlm_mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_scratch.clone(), mv_len),
                ArrayArg::from_raw_parts(args.confidence.clone(), conf_len),
                true, // this bench measures the confidence write itself
                0.0,
                thsad,
                W,
                H,
                DEFAULT_BLKSIZE,
                CONF_STEP,
                SEARCH_RADIUS,
                0u32, // use_seed = 0 (no coarse pass in the no-MC path)
                blocks_x,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "mc_confidence_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(1)
    }
}
