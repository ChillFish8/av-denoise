use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, nlm_mc_block_match_fine};
use av_denoise_core::bench_api::motion::{DEFAULT_BLKSIZE, DEFAULT_OVERLAP};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

const CONF_STEP: u32 = DEFAULT_BLKSIZE - DEFAULT_OVERLAP;
const SEARCH_RADIUS: u32 = 0;

/// The library's per-pixel SAD threshold.
///
/// The library's own constant and `thsad` helper are crate-private, so a bench target spells it out.
const THSAD_PIXEL: f32 = 0.02;

/// The confidence pass without motion compensation.
///
/// It scores a single candidate with no seed and `search_radius = 0` at the library's default
/// block geometry, which is the cost of confidence weighting when motion compensation is off.
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
        let centre_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let neighbour_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let centre_bytes = f32::as_bytes(&centre_frame);
        let centre = self.client.create_from_slice(centre_bytes);
        let neighbour_bytes = f32::as_bytes(&neighbour_frame);
        let neighbour = self.client.create_from_slice(neighbour_bytes);

        let blocks_x = WIDTH.div_ceil(CONF_STEP);
        let blocks_y = HEIGHT.div_ceil(CONF_STEP);
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
        let level_len = (WIDTH * HEIGHT) as usize;
        let blocks_x = WIDTH.div_ceil(CONF_STEP);
        let blocks_y = HEIGHT.div_ceil(CONF_STEP);
        let mv_len = (blocks_x * blocks_y * 2) as usize;
        let conf_len = (blocks_x * blocks_y) as usize;
        let thsad = (DEFAULT_BLKSIZE * DEFAULT_BLKSIZE) as f32 * THSAD_PIXEL;

        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            nlm_mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_scratch.clone(), mv_len),
                ArrayArg::from_raw_parts(args.confidence.clone(), conf_len),
                true, // The confidence write is what this bench measures.
                0.0,
                thsad,
                WIDTH,
                HEIGHT,
                DEFAULT_BLKSIZE,
                CONF_STEP,
                SEARCH_RADIUS,
                0u32, // `use_seed = 0`, since there is no coarse pass without motion compensation.
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
        shapes_with_channels(1)
    }
}
