use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, nlm_mc_block_match_fine};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

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
    /// Neighbours matched in one launch, each against the centre frame.
    pub neighbours: u32,
}

#[derive(Clone)]
pub struct FineInput {
    /// The centre frame in slot 0 and each neighbour in the slots after it.
    pub pyramid: Handle,
    /// One entry per neighbour, naming slots 1 onwards.
    pub neighbour_slots: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
}

impl<R: Runtime> Benchmark for BlockMatchFineBench<R> {
    type Input = FineInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let mut frames = make_synthetic_frame(WIDTH, HEIGHT, 1);

        for _ in 0..self.neighbours {
            let neighbour_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
            frames.extend_from_slice(&neighbour_frame);
        }

        let frame_bytes = f32::as_bytes(&frames);
        let pyramid = self.client.create_from_slice(frame_bytes);
        let table: Vec<u32> = (1..=self.neighbours).collect();
        let table_bytes = u32::as_bytes(&table);
        let neighbour_slots = self.client.create_from_slice(table_bytes);

        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let blocks = (blocks_x * blocks_y) as usize;
        let neighbours = self.neighbours as usize;
        let mv_field = self.client.empty(neighbours * blocks * 2 * size_of::<i32>());
        let confidence = self.client.empty(neighbours * blocks * size_of::<f32>());

        FineInput {
            pyramid,
            neighbour_slots,
            mv_field,
            confidence,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let level_len = (WIDTH * HEIGHT) as usize;
        let neighbours = self.neighbours as usize;
        let pyramid_len = (neighbours + 1) * level_len;
        let blocks_x = WIDTH.div_ceil(FINE_STEP);
        let blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let conf_stride = blocks_x * blocks_y;
        let mv_stride = 2 * conf_stride;
        let mv_len = neighbours * mv_stride as usize;
        let conf_len = neighbours * conf_stride as usize;
        let thsad = (FINE_BLKSIZE * FINE_BLKSIZE) as f32 * THSAD_PIXEL;

        let grid = CubeCount::new_3d(blocks_x, blocks_y, self.neighbours);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            let pyramid_arg = ArrayArg::from_raw_parts(args.pyramid, pyramid_len);
            let neighbour_slots_arg = ArrayArg::from_raw_parts(args.neighbour_slots, neighbours);
            let mv_field_arg = ArrayArg::from_raw_parts(args.mv_field, mv_len);
            let confidence_arg = ArrayArg::from_raw_parts(args.confidence, conf_len);

            nlm_mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                pyramid_arg,
                neighbour_slots_arg,
                mv_field_arg,
                confidence_arg,
                0u32,
                true, // Includes the confidence write, the full production cost.
                0.0,
                thsad,
                WIDTH,
                HEIGHT,
                level_len as u32,
                FINE_BLKSIZE,
                FINE_STEP,
                SEARCH_RADIUS,
                0u32, // `use_seed = 0`, the worst case without a coarse seed.
                blocks_x,
                mv_stride,
                conf_stride,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        neighbour_bench_name("mc_block_match_fine_1080p_luma", self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
