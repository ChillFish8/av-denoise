use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, mc_block_match_coarse};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

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
    /// Neighbours matched in one launch, each against the centre frame.
    pub neighbours: u32,
}

#[derive(Clone)]
pub struct CoarseInput {
    /// The centre frame in slot 0 and each neighbour in the slots after it.
    pub pyramid: Handle,
    /// One entry per neighbour, naming slots 1 onwards.
    pub neighbour_slots: Handle,
    pub mv_field: Handle,
}

impl<R: Runtime> Benchmark for BlockMatchCoarseBench<R> {
    type Input = CoarseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let coarse_width = WIDTH / 2;
        let coarse_height = HEIGHT / 2;
        let mut frames = make_synthetic_frame(coarse_width, coarse_height, 1);

        for _ in 0..self.neighbours {
            let neighbour_frame = make_synthetic_frame(coarse_width, coarse_height, 1);
            frames.extend_from_slice(&neighbour_frame);
        }

        let frame_bytes = f32::as_bytes(&frames);
        let pyramid = self.client.create_from_slice(frame_bytes);
        let table: Vec<u32> = (1..=self.neighbours).collect();
        let table_bytes = u32::as_bytes(&table);
        let neighbour_slots = self.client.create_from_slice(table_bytes);

        let fine_blocks_x = WIDTH.div_ceil(FINE_STEP);
        let fine_blocks_y = HEIGHT.div_ceil(FINE_STEP);
        let mv_stride = (fine_blocks_x * fine_blocks_y * 2) as usize;
        let mv_len = self.neighbours as usize * mv_stride;
        let mv_field = self.client.empty(mv_len * size_of::<i32>());

        CoarseInput {
            pyramid,
            neighbour_slots,
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
        let pyramid_len = (self.neighbours as usize + 1) * level_len;
        let slots_len = self.neighbours as usize;
        let mv_stride = fine_blocks_x * fine_blocks_y * 2;
        let mv_len = self.neighbours as usize * mv_stride as usize;
        let grid = CubeCount::new_3d(coarse_blocks_x, coarse_blocks_y, self.neighbours);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            let pyramid_arg = ArrayArg::from_raw_parts(args.pyramid, pyramid_len);
            let neighbour_slots_arg = ArrayArg::from_raw_parts(args.neighbour_slots, slots_len);
            let mv_field_arg = ArrayArg::from_raw_parts(args.mv_field, mv_len);

            mc_block_match_coarse::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                pyramid_arg,
                neighbour_slots_arg,
                mv_field_arg,
                0u32,
                coarse_width,
                coarse_height,
                0,
                level_len as u32,
                coarse_blksize,
                coarse_step,
                SEARCH_RADIUS,
                coarse_scale,
                fine_blocks_x,
                fine_blocks_y,
                FINE_STEP,
                mv_stride,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        neighbour_bench_name("mc_block_match_coarse_540p_luma", self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
