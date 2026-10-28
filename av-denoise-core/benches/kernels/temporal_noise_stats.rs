use av_denoise_core::bench_api::noise_kernels::nlm_temporal_noise_stats;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_padded_frame, shapes_with_channels};

/// Logical channel count for the bench's YUV storage frame.
const TEMPORAL_CHANNELS: u32 = 3;
/// Padded storage width for YUV, padded up to a vec4 lane.
const TEMPORAL_STORED_CH: u32 = 4;
/// Matches `noise::TEMPORAL_NOISE_BLOCK`.
const TEMPORAL_BLOCK: u32 = 16;
/// The motion grid step of the compensated row.
const MOTION_STEP: u32 = 8;
/// The whole-pixel vector every motion block of the compensated row holds.
const MOTION_VECTOR: [i32; 2] = [3, -2];

/// The temporal-residual noise-stats kernel over two 1080p YUV ring slots.
///
/// It diffs the slots and reduces every `16 × 16` block into its stats record. `luma_fields`
/// and `compensated` pick which of the kernel's compiled variants this row times. The compensated
/// row follows a uniform [MOTION_VECTOR] field on a [MOTION_STEP] grid.
pub struct TemporalNoiseStatsBench {
    pub client: Client,
    pub luma_fields: bool,
    pub compensated: bool,
}

#[derive(Clone)]
pub struct TemporalNoiseStatsInput {
    pub input: Handle,
    pub stats: Handle,
    pub mv_field: Handle,
}

fn blocks() -> (u32, u32) {
    (WIDTH.div_ceil(TEMPORAL_BLOCK), HEIGHT.div_ceil(TEMPORAL_BLOCK))
}

fn motion_blocks() -> (u32, u32) {
    (WIDTH.div_ceil(MOTION_STEP), HEIGHT.div_ceil(MOTION_STEP))
}

fn mv_len() -> usize {
    let (blocks_x, blocks_y) = motion_blocks();
    (blocks_x * blocks_y * 2) as usize
}

/// The stats buffer length, one `noise::temporal_stats_record_len` record per block.
///
/// A record is a sum and a sum of squares per stored channel, the lag-1, texture and moved lanes,
/// and four quarter records of nine fields each.
fn stats_len() -> usize {
    let (blocks_x, blocks_y) = blocks();
    (blocks_x * blocks_y * (2 * TEMPORAL_STORED_CH + 39)) as usize
}

/// The slot whose pixel `(x, y)` is the frame's pixel `(x, y) + MOTION_VECTOR` plus 0.01.
///
/// Positions past the frame edge clamp to it.
fn shifted_slot(frame: &[f32]) -> Vec<f32> {
    let stored_ch = TEMPORAL_STORED_CH as usize;
    let max_x = WIDTH as i32 - 1;
    let max_y = HEIGHT as i32 - 1;
    let mut shifted = Vec::with_capacity(frame.len());
    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let source_x = (x + MOTION_VECTOR[0]).clamp(0, max_x) as usize;
            let source_y = (y + MOTION_VECTOR[1]).clamp(0, max_y) as usize;
            let start = (source_y * WIDTH as usize + source_x) * stored_ch;
            let pixel = &frame[start..start + stored_ch];
            let offset_pixel = pixel.iter().map(|&sample| sample + 0.01);
            shifted.extend(offset_pixel);
        }
    }

    shifted
}

impl Benchmark for TemporalNoiseStatsBench {
    type Input = TemporalNoiseStatsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Two ring slots. The second is the first read along the motion vector plus a small offset,
        // so a block following its vector sees a near-constant residual and the zero vector does not.
        let frame = make_padded_frame(WIDTH, HEIGHT, TEMPORAL_CHANNELS);
        let shifted = shifted_slot(&frame);
        let mut ring = frame;
        ring.extend(shifted);

        let ring_bytes = f32::as_bytes(&ring);
        let input = self.client.create_from_slice(ring_bytes);
        let total_stats = stats_len();
        let stats = self.client.empty(total_stats * size_of::<f32>());

        let total_mv = mv_len();
        let block_count = total_mv / 2;
        let vectors = MOTION_VECTOR.repeat(block_count);
        let vector_bytes = i32::as_bytes(&vectors);
        let mv_field = self.client.create_from_slice(vector_bytes);

        TemporalNoiseStatsInput {
            input,
            stats,
            mv_field,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let total_input = (2 * WIDTH * HEIGHT * TEMPORAL_STORED_CH) as usize;
        let (blocks_x, blocks_y) = blocks();
        let total_stats = stats_len();
        let total_mv = mv_len();
        let (mv_blocks_x, mv_blocks_y) = motion_blocks();

        unsafe {
            nlm_temporal_noise_stats::launch_unchecked(
                &self.client,
                CubeCount::new_2d(blocks_x, blocks_y),
                CubeDim::new_2d(TEMPORAL_BLOCK, TEMPORAL_BLOCK),
                TEMPORAL_STORED_CH as usize,
                BufferArg::from_raw_parts(args.input.clone(), total_input),
                BufferArg::from_raw_parts(args.stats.clone(), total_stats),
                BufferArg::from_raw_parts(args.mv_field.clone(), total_mv),
                1u32,
                0u32,
                0u32,
                WIDTH,
                HEIGHT,
                TEMPORAL_STORED_CH,
                TEMPORAL_BLOCK,
                self.luma_fields,
                self.compensated,
                MOTION_STEP,
                mv_blocks_x,
                mv_blocks_y,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        let luma = if self.luma_fields { "_luma_fields_on" } else { "" };
        let motion = if self.compensated { "_compensated" } else { "" };
        format!("temporal_noise_stats_1080p_yuv{luma}{motion}")
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(TEMPORAL_CHANNELS)
    }
}
