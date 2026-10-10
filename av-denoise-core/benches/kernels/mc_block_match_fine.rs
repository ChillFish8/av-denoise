use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, nlm_mc_block_match_fine};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
const SEARCH_RADIUS: u32 = 4;

/// The library's per-pixel SAD threshold.
///
/// The library's own constant and `thsad` helper are crate-private, so a bench target spells it out.
const THSAD_PIXEL: f32 = 0.02;

/// The fine refinement pass at full resolution.
///
/// It reads no seed, so the cost is bounded by the `(2·r + 1)²` search window per block and stays
/// fair even if the coarse bench's tuning changes. `sad_noise_floor` is 0.0 because no noise
/// estimate is available at this level, and `thsad` uses the library's default `thsad_scale` of 1.0.
pub struct BlockMatchFineBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct FineInput {
    pub centre: Handle,
    pub neighbour: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
}

impl<R: Runtime> Benchmark for BlockMatchFineBench<R> {
    type Input = FineInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let centre_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let neighbour_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let centre_bytes = f32::as_bytes(&centre_frame);
        let centre = self.client.create_from_slice(centre_bytes);
        let neighbour_bytes = f32::as_bytes(&neighbour_frame);
        let neighbour = self.client.create_from_slice(neighbour_bytes);

        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let mv_field = self
            .client
            .empty((blocks_x * blocks_y * 2) as usize * size_of::<i32>());
        let confidence = self
            .client
            .empty((blocks_x * blocks_y) as usize * size_of::<f32>());

        FineInput {
            centre,
            neighbour,
            mv_field,
            confidence,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let level_len = (WIDTH * HEIGHT) as usize;
        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let mv_len = (blocks_x * blocks_y * 2) as usize;
        let conf_len = (blocks_x * blocks_y) as usize;
        let thsad = (FINE_BLKSIZE * FINE_BLKSIZE) as f32 * THSAD_PIXEL;

        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            nlm_mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.centre.clone(), level_len),
                ArrayArg::from_raw_parts(args.neighbour.clone(), level_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), mv_len),
                ArrayArg::from_raw_parts(args.confidence.clone(), conf_len),
                true, // Includes the confidence write, the full production cost.
                0.0,
                thsad,
                WIDTH,
                HEIGHT,
                FINE_BLKSIZE,
                FINE_STEP,
                SEARCH_RADIUS,
                0u32, // `use_seed = 0`, the worst case without a coarse seed.
                blocks_x,
                1,
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
        shapes_with_channels(1)
    }
}
