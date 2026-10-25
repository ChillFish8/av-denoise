use av_denoise_core::bench_api::kernels::motion::{BLOCK_MATCH_THREADS, mc_block_match_fine};
use av_denoise_core::bench_api::motion::{DEFAULT_BLKSIZE, DEFAULT_OVERLAP};
use av_denoise_core::bench_api::{DERIVED_CONFIDENCE_SAD_FLOOR, DERIVED_WORDS, DerivedValues};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

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
    /// Neighbours scored in one launch, each against the centre frame.
    pub neighbours: u32,
}

#[derive(Clone)]
pub struct ConfidenceInput {
    /// The centre frame in slot 0 and each neighbour in the slots after it.
    pub pyramid: Handle,
    /// One entry per neighbour, naming slots 1 onwards.
    pub neighbour_slots: Handle,
    pub mv_scratch: Handle,
    pub confidence: Handle,
    /// The derived noise values, with SAD floors of 0.
    pub noise: Handle,
}

impl<R: Runtime> Benchmark for McConfidenceBench<R> {
    type Input = ConfidenceInput;
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

        let blocks_x = WIDTH.div_ceil(CONF_STEP);
        let blocks_y = HEIGHT.div_ceil(CONF_STEP);
        let blocks = (blocks_x * blocks_y) as usize;
        let neighbours = self.neighbours as usize;
        let mv_scratch = self.client.empty(neighbours * blocks * 2 * size_of::<i32>());
        let confidence = self.client.empty(neighbours * blocks * size_of::<f32>());

        let noise_words = DerivedValues::default().to_words();
        let noise_bytes = f32::as_bytes(&noise_words);
        let noise = self.client.create_from_slice(noise_bytes);

        ConfidenceInput {
            pyramid,
            neighbour_slots,
            mv_scratch,
            confidence,
            noise,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let level_len = (WIDTH * HEIGHT) as usize;
        let neighbours = self.neighbours as usize;
        let pyramid_len = (neighbours + 1) * level_len;
        let blocks_x = WIDTH.div_ceil(CONF_STEP);
        let blocks_y = HEIGHT.div_ceil(CONF_STEP);
        let conf_stride = blocks_x * blocks_y;
        let mv_stride = 2 * conf_stride;
        let mv_len = neighbours * mv_stride as usize;
        let conf_len = neighbours * conf_stride as usize;
        let thsad = (DEFAULT_BLKSIZE * DEFAULT_BLKSIZE) as f32 * THSAD_PIXEL;

        let grid = CubeCount::new_3d(blocks_x, blocks_y, self.neighbours);
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            let pyramid_arg = ArrayArg::from_raw_parts(args.pyramid, pyramid_len);
            let neighbour_slots_arg = ArrayArg::from_raw_parts(args.neighbour_slots, neighbours);
            let mv_scratch_arg = ArrayArg::from_raw_parts(args.mv_scratch, mv_len);
            let confidence_arg = ArrayArg::from_raw_parts(args.confidence, conf_len);
            let noise_arg = ArrayArg::from_raw_parts(args.noise, DERIVED_WORDS as usize);

            mc_block_match_fine::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                pyramid_arg,
                neighbour_slots_arg,
                mv_scratch_arg,
                confidence_arg,
                0u32,
                true, // The confidence write is what this bench measures.
                noise_arg,
                DERIVED_CONFIDENCE_SAD_FLOOR,
                thsad,
                WIDTH,
                HEIGHT,
                level_len as u32,
                DEFAULT_BLKSIZE,
                CONF_STEP,
                SEARCH_RADIUS,
                0u32, // `use_seed = 0`, since there is no coarse pass without motion compensation.
                blocks_x,
                mv_stride,
                conf_stride,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        neighbour_bench_name("mc_confidence_1080p_luma", self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
