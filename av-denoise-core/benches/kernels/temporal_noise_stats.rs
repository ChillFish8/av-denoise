use av_denoise_core::nlmeans::kernels::nlm_temporal_noise_stats;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_padded_frame, shapes_with_ch};

/// Logical channel count for the bench's YUV storage frame.
const TEMPORAL_CHANNELS: u32 = 3;
/// Padded storage width for YUV (padded up to a vec4 lane).
const TEMPORAL_STORED_CH: u32 = 4;
/// Matches `nlmeans::noise::TEMPORAL_NOISE_BLOCK`.
const TEMPORAL_BLOCK: u32 = 16;

/// The temporal-residual noise-stats kernel, diffing two 1080p YUV
/// ring slots against each other and reducing every `16 × 16` block
/// into its stats record.
///
/// `luma_fields` picks which of the kernel's two compiled variants this
/// row times, matching `nlm_temporal_noise_stats`'s own flag.
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
    (W.div_ceil(TEMPORAL_BLOCK), H.div_ceil(TEMPORAL_BLOCK))
}

/// Matches `nlmeans::noise::temporal_stats_record_len`: a sum and a sum
/// of squares per stored channel, one lag-1 total, and four luma-only
/// fields.
fn stats_len() -> usize {
    let (bx, by) = blocks();
    (bx * by * (2 * TEMPORAL_STORED_CH + 5)) as usize
}

impl<R: Runtime> Benchmark for TemporalNoiseStatsBench<R> {
    type Input = TemporalNoiseStatsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Two ring slots: a frame and a slightly perturbed copy, so the
        // diff the kernel reduces isn't degenerately zero everywhere.
        let frame = make_padded_frame(W, H, TEMPORAL_CHANNELS);
        let mut ring = frame.clone();
        ring.extend(frame.iter().map(|&v| (v + 0.01).clamp(0.0, 1.0)));
        let input = self.client.create_from_slice(f32::as_bytes(&ring));
        let stats = self.client.empty(stats_len() * size_of::<f32>());
        TemporalNoiseStatsInput { input, stats }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let total_input = (2 * W * H * TEMPORAL_STORED_CH) as usize;
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
                W,
                H,
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
        shapes_with_ch(TEMPORAL_CHANNELS)
    }
}
