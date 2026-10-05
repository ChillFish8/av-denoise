use av_denoise_core::bench_api::kernels::nlm_temporal_noise_stats;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_padded_frame, shapes_with_channels};

/// Logical channel count for the bench's YUV storage frame.
const TEMPORAL_CHANNELS: u32 = 3;
/// Padded storage width for YUV, padded up to a vec4 lane.
const TEMPORAL_STORED_CH: u32 = 4;
/// Matches `nlmeans::noise::TEMPORAL_NOISE_BLOCK`.
const TEMPORAL_BLOCK: u32 = 16;

/// The temporal-residual noise-stats kernel over two 1080p YUV ring slots.
///
/// It diffs the slots and reduces every `16 × 16` block into its stats record. `luma_fields`
/// picks which of the kernel's two compiled variants this row times.
pub struct TemporalNoiseStatsBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub luma_fields: bool,
}

#[derive(Clone)]
pub struct TemporalNoiseStatsInput {
    pub input: Handle,
    pub stats: Handle,
}

fn blocks() -> (u32, u32) {
    (WIDTH.div_ceil(TEMPORAL_BLOCK), HEIGHT.div_ceil(TEMPORAL_BLOCK))
}

/// The stats buffer length, one `nlmeans::noise::temporal_stats_record_len` record per block.
///
/// A record is a sum and a sum of squares per stored channel, one lag-1 total, and four quarter
/// records of nine fields each.
fn stats_len() -> usize {
    let (blocks_x, blocks_y) = blocks();
    (blocks_x * blocks_y * (2 * TEMPORAL_STORED_CH + 37)) as usize
}

impl<R: Runtime> Benchmark for TemporalNoiseStatsBench<R> {
    type Input = TemporalNoiseStatsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Two ring slots, a frame and a slightly perturbed copy, so the diff the kernel reduces is
        // not zero everywhere.
        let frame = make_padded_frame(WIDTH, HEIGHT, TEMPORAL_CHANNELS);
        let mut ring = frame.clone();
        let perturbed = frame.iter().map(|&sample| (sample + 0.01).clamp(0.0, 1.0));
        ring.extend(perturbed);

        let ring_bytes = f32::as_bytes(&ring);
        let input = self.client.create_from_slice(ring_bytes);
        let total_stats = stats_len();
        let stats = self.client.empty(total_stats * size_of::<f32>());

        TemporalNoiseStatsInput { input, stats }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let total_input = (2 * WIDTH * HEIGHT * TEMPORAL_STORED_CH) as usize;
        let (blocks_x, blocks_y) = blocks();
        let total_stats = stats_len();

        unsafe {
            nlm_temporal_noise_stats::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(blocks_x, blocks_y),
                CubeDim::new_2d(TEMPORAL_BLOCK, TEMPORAL_BLOCK),
                TEMPORAL_STORED_CH as usize,
                ArrayArg::from_raw_parts(args.input.clone(), total_input),
                ArrayArg::from_raw_parts(args.stats.clone(), total_stats),
                1u32,
                0u32,
                WIDTH,
                HEIGHT,
                TEMPORAL_STORED_CH,
                TEMPORAL_BLOCK,
                self.luma_fields,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        if self.luma_fields {
            "temporal_noise_stats_1080p_yuv_luma_fields_on".to_string()
        } else {
            "temporal_noise_stats_1080p_yuv".to_string()
        }
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(TEMPORAL_CHANNELS)
    }
}
