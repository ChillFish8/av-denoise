use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::noise::stats::lower_quartile;
use crate::noise::temporal::{
    BLOCK_MOVED_LANE,
    BLOCK_TEXTURE_LANE,
    COMPENSATION_GAIN_MAX,
    StatsMotion,
    TEMPORAL_NOISE_BLOCK,
    TemporalStatsCtx,
    run_temporal_noise_stats,
    temporal_stats_blocks,
    temporal_stats_buf_bytes,
    temporal_stats_record_len,
    temporal_stats_slot_view,
    texture_lane,
};
use crate::tune::tests::fixtures::{hashed_ring, hashed_unit, noisy_field_over, upload};
use crate::tune::tests::runtime::test_client;

#[test]
fn lower_quartile_odd_count_exact_index() {
    // With 5 values the quartile lands exactly on index 1.
    let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(got, 2.0);
}

#[test]
fn lower_quartile_even_count() {
    // With 4 values the quartile lands at 0.75, between the first two.
    let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0]);
    assert!((got - 1.75).abs() < 1e-6, "expected 1.75, got {got}");
}

#[test]
fn lower_quartile_interpolates_at_a_fractional_index() {
    // With 3 values the quartile lands halfway between the first two.
    let got = lower_quartile(&[10.0, 20.0, 30.0]);
    assert!((got - 15.0).abs() < 1e-6, "expected 15.0, got {got}");
}

#[test]
fn lower_quartile_single_element_returns_it() {
    let got = lower_quartile(&[42.0]);
    assert_eq!(got, 42.0);
}

/// A whole-pixel motion grid in the per-neighbour layout a pass's `mv_field` holds.
struct MotionGrid {
    vectors: Vec<i32>,
    step: u32,
    blocks_x: u32,
    blocks_y: u32,
}

impl MotionGrid {
    fn from_fn(width: u32, height: u32, step: u32, vector_at: impl Fn(u32, u32) -> [i32; 2]) -> Self {
        let blocks_x = width.div_ceil(step);
        let blocks_y = height.div_ceil(step);

        let mut vectors = Vec::with_capacity((blocks_x * blocks_y * 2) as usize);
        for row in 0..blocks_y {
            for col in 0..blocks_x {
                let vector = vector_at(col, row);
                vectors.extend_from_slice(&vector);
            }
        }

        Self {
            vectors,
            step,
            blocks_x,
            blocks_y,
        }
    }

    fn uniform(width: u32, height: u32, step: u32, vector: [i32; 2]) -> Self {
        Self::from_fn(width, height, step, |_, _| vector)
    }
}

/// A vector far outside any test frame, which zeroes every block that reads it.
const DECOY_VECTOR: [i32; 2] = [1000, 1000];

/// Two frames packed as `pixels * channels`, `previous` in ring slot 0 and `new` in slot 1.
struct StatsFrames {
    width: u32,
    height: u32,
    channels: u32,
    previous: Vec<f32>,
    new: Vec<f32>,
}

impl StatsFrames {
    fn record_len(&self) -> usize {
        temporal_stats_record_len(self.channels) as usize
    }

    fn record<'a>(&self, records: &'a [f32], block_x: u32, block_y: u32) -> &'a [f32] {
        let (blocks_x, _) = temporal_stats_blocks(self.width, self.height);
        let record_len = self.record_len();
        let start = (block_y * blocks_x + block_x) as usize * record_len;

        &records[start..start + record_len]
    }

    fn block_extent(&self, block_x: u32, block_y: u32) -> (u32, u32) {
        let block_width = TEMPORAL_NOISE_BLOCK.min(self.width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = TEMPORAL_NOISE_BLOCK.min(self.height - block_y * TEMPORAL_NOISE_BLOCK);
        (block_width, block_height)
    }
}

/// A motion grid uploaded behind a slice of [DECOY_VECTOR]s, so a launch that ignores `mv_offset`
/// zeroes every record.
struct UploadedField {
    handle: Handle,
    len: usize,
    offset: u32,
}

fn upload_behind_decoys(client: &Client, grid: &MotionGrid) -> UploadedField {
    let block_count = grid.vectors.len() / 2;
    let mut field = DECOY_VECTOR.repeat(block_count);
    let offset = field.len() as u32;
    field.extend_from_slice(&grid.vectors);

    let field_bytes = i32::as_bytes(&field);
    let handle = client.create_from_slice(field_bytes);

    UploadedField {
        handle,
        len: field.len(),
        offset,
    }
}

/// Runs the stats kernel on `frames` and reads slot 1's records back.
fn run_stats(frames: &StatsFrames, motion: Option<&MotionGrid>, luma_fields: bool) -> Vec<f32> {
    let client = test_client();
    let frame_count = 2u32;
    let align = StorageAlign::from_client(&client);

    let mut ring = frames.previous.clone();
    ring.extend_from_slice(&frames.new);
    let input_buf = upload(&client, &ring);

    let stats_bytes =
        temporal_stats_buf_bytes(frames.width, frames.height, frames.channels, frame_count, align);
    let stats_fill = vec![-7.0f32; stats_bytes / size_of::<f32>()];
    let stats_buf = upload(&client, &stats_fill);

    let uploaded = motion.map(|grid| upload_behind_decoys(&client, grid));
    let stats_motion = match (motion, uploaded.as_ref()) {
        (Some(grid), Some(field)) => Some(StatsMotion {
            mv_field: &field.handle,
            mv_len: field.len,
            mv_offset: field.offset,
            step: grid.step,
            blocks_x: grid.blocks_x,
            blocks_y: grid.blocks_y,
        }),
        _ => None,
    };

    let ctx = TemporalStatsCtx {
        width: frames.width,
        height: frames.height,
        channels: frames.channels,
        frame_count,
        slot_new: 1,
        slot_prev: 0,
        input_buf: &input_buf,
        stats_buf: &stats_buf,
        align,
        motion: stats_motion,
    };
    run_temporal_noise_stats(&client, &ctx, luma_fields).expect("the stats queue");

    let slot = temporal_stats_slot_view(
        &stats_buf,
        frames.width,
        frames.height,
        frames.channels,
        frame_count,
        1,
        align,
    );
    let bytes = client.read_one(slot).expect("the stats read back");

    f32::from_bytes(&bytes).to_vec()
}

/// Hashed texture over a `width x height` frame, sampled at `(x + shift_x, y + shift_y)` of a
/// larger field so a shifted copy stays defined at the edges.
fn texture_frame(width: u32, height: u32, shift_x: u32, shift_y: u32) -> Vec<f32> {
    let field_width = width + 16;
    let mut frame = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let index = (y + shift_y) * field_width + x + shift_x;
            let value = hashed_unit(index, 11);
            frame.push(value);
        }
    }

    frame
}

/// Channel 0's per-block sigma for every block with a non-zero record.
fn block_sigmas(frames: &StatsFrames, records: &[f32]) -> Vec<f32> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(frames.width, frames.height);
    let channels = frames.channels as usize;

    let mut sigmas = Vec::new();
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let record = frames.record(records, block_x, block_y);
            if record[channels] == 0.0 {
                continue;
            }

            let (block_width, block_height) = frames.block_extent(block_x, block_y);
            let pixels = (block_width * block_height) as f32;
            let mean = record[0] / pixels;
            let variance = record[channels] / pixels - mean * mean;
            let sigma = variance.max(0.0).sqrt() / std::f32::consts::SQRT_2;
            sigmas.push(sigma);
        }
    }

    sigmas
}

fn median_of(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Channel 0's summed squared residual of one block shifted by `vector`, computed on the host.
fn host_sum_d2(frames: &StatsFrames, block_x: u32, block_y: u32, vector: [i32; 2]) -> f32 {
    let (block_width, block_height) = frames.block_extent(block_x, block_y);
    let stride = frames.channels as usize;

    let mut sum_d2 = 0.0f32;
    for local_y in 0..block_height {
        for local_x in 0..block_width {
            let x = block_x * TEMPORAL_NOISE_BLOCK + local_x;
            let y = block_y * TEMPORAL_NOISE_BLOCK + local_y;
            let prev_x = (x as i32 + vector[0]) as u32;
            let prev_y = (y as i32 + vector[1]) as u32;
            let new_index = (y * frames.width + x) as usize * stride;
            let prev_index = (prev_y * frames.width + prev_x) as usize * stride;
            let residual = frames.new[new_index] - frames.previous[prev_index];
            sum_d2 += residual * residual;
        }
    }

    sum_d2
}

/// Channel 0's residual variance of one block over the pixels whose match by `vector` stays in
/// the frame, or `None` when no match does.
fn host_variance(frames: &StatsFrames, block_x: u32, block_y: u32, vector: [i32; 2]) -> Option<f32> {
    let (block_width, block_height) = frames.block_extent(block_x, block_y);
    let stride = frames.channels as usize;

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u32;
    for local_y in 0..block_height {
        for local_x in 0..block_width {
            let x = block_x * TEMPORAL_NOISE_BLOCK + local_x;
            let y = block_y * TEMPORAL_NOISE_BLOCK + local_y;
            let prev_x = x as i32 + vector[0];
            let prev_y = y as i32 + vector[1];
            let in_frame =
                prev_x >= 0 && prev_x < frames.width as i32 && prev_y >= 0 && prev_y < frames.height as i32;
            if !in_frame {
                continue;
            }

            let new_index = (y * frames.width + x) as usize * stride;
            let prev_index = (prev_y as u32 * frames.width + prev_x as u32) as usize * stride;
            let residual = (frames.new[new_index] - frames.previous[prev_index]) as f64;
            sum += residual;
            sum_sq += residual * residual;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }

    let mean = sum / count as f64;
    let variance = sum_sq / count as f64 - mean * mean;
    Some(variance as f32)
}

/// The share of its zero-vector variance one block keeps along `vector`, or `None` when no match
/// stays in the frame.
fn host_gain(frames: &StatsFrames, block_x: u32, block_y: u32, vector: [i32; 2]) -> Option<f32> {
    let zero_variance = host_variance(frames, block_x, block_y, [0, 0])?;
    let vector_variance = host_variance(frames, block_x, block_y, vector)?;

    Some(vector_variance / zero_variance)
}

#[test]
fn compensated_stats_cancel_a_known_pan() {
    let (width, height) = (128u32, 96u32);
    let sigma = 2.0 / 255.0;

    // `previous` is `new` shifted by `(+3, -2)`, so `new(x, y)` matches `previous(x + 3, y - 2)`.
    let new_clean = texture_frame(width, height, 3, 2);
    let previous_clean = texture_frame(width, height, 0, 4);
    let frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: noisy_field_over(&previous_clean, width, height, sigma, 1),
        new: noisy_field_over(&new_clean, width, height, sigma, 2),
    };

    let grid = MotionGrid::uniform(width, height, 8, [3, -2]);
    let compensated = run_stats(&frames, Some(&grid), true);
    let raw = run_stats(&frames, None, true);

    let compensated_sigma = median_of(block_sigmas(&frames, &compensated));
    let raw_sigma = median_of(block_sigmas(&frames, &raw));

    let relative_error = (compensated_sigma - sigma).abs() / sigma;
    assert!(
        relative_error <= 0.05,
        "compensated sigma {compensated_sigma} vs true {sigma} (rel err {relative_error:.3})"
    );
    assert!(
        raw_sigma > 3.0 * sigma,
        "raw sigma {raw_sigma} should read the pan, not the noise"
    );
}

#[test]
fn zero_vector_compensated_stats_equal_raw_stats() {
    let (width, height) = (45u32, 29u32);

    for channels in [1u32, 2] {
        let ring = hashed_ring(width, height, channels, 2, 5);
        let frame_len = (width * height * channels) as usize;
        let frames = StatsFrames {
            width,
            height,
            channels,
            previous: ring[..frame_len].to_vec(),
            new: ring[frame_len..].to_vec(),
        };
        let grid = MotionGrid::uniform(width, height, 8, [0, 0]);

        for luma_fields in [false, true] {
            let compensated = run_stats(&frames, Some(&grid), luma_fields);
            let raw = run_stats(&frames, None, luma_fields);

            let compensated_bits: Vec<u32> = compensated.iter().map(|value| value.to_bits()).collect();
            let raw_bits: Vec<u32> = raw.iter().map(|value| value.to_bits()).collect();
            assert_eq!(
                compensated_bits, raw_bits,
                "channels {channels}, luma_fields {luma_fields}"
            );

            let moved_lane = 2 * channels as usize + BLOCK_MOVED_LANE as usize;
            let record_len = frames.record_len();
            let moved_flags = raw.chunks(record_len).map(|record| record[moved_lane]);
            for moved in moved_flags {
                assert_eq!(moved, 0.0);
            }
        }
    }
}

#[test]
fn off_frame_displacement_zeroes_the_record() {
    let (width, height) = (96u32, 48u32);

    // `new(x, y)` matches `previous(x + 20, y)`, so every block follows the vector.
    let frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: texture_frame(width, height, 0, 0),
        new: texture_frame(width, height, 20, 0),
    };
    let grid = MotionGrid::uniform(width, height, 8, [20, 0]);
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);

    for luma_fields in [false, true] {
        let records = run_stats(&frames, Some(&grid), luma_fields);

        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let record = frames.record(&records, block_x, block_y);
                let zeroed = record.iter().all(|&value| value == 0.0);
                let expect_zeroed = block_x + 2 >= blocks_x;
                assert_eq!(
                    zeroed, expect_zeroed,
                    "block ({block_x}, {block_y}), luma_fields {luma_fields}: {record:?}"
                );
            }
        }
    }
}

#[test]
fn texture_lane_reads_mean_cell_energy() {
    // The last block column is 8 pixels wide, so it holds 2 by 4 full cells and 10 pairs.
    let (width, height) = (56u32, 32u32);
    let slope = 3.0f32 / 1024.0;
    let ramp: Vec<f32> = (0..width * height)
        .map(|index| (index % width) as f32 * slope)
        .collect();
    let frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: vec![0.5; ramp.len()],
        new: ramp,
    };
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    // Horizontal cell pairs differ by `4 * slope` and vertical pairs by nothing. A full block holds
    // 12 of each, and the ragged block 4 horizontal and 6 vertical.
    let cell_step = 4.0 * slope;
    let full_expected = 12.0 * cell_step * cell_step / 24.0;
    let ragged_expected = 4.0 * cell_step * cell_step / 10.0;
    let texture_lane = 2 + BLOCK_TEXTURE_LANE as usize;

    for luma_fields in [false, true] {
        let records = run_stats(&frames, None, luma_fields);

        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let record = frames.record(&records, block_x, block_y);
                let texture = record[texture_lane];
                let ragged = block_x + 1 == blocks_x;
                let expected = if ragged { ragged_expected } else { full_expected };
                let relative_error = (texture - expected).abs() / expected;
                assert!(
                    relative_error <= 1e-5,
                    "block ({block_x}, {block_y}): texture {texture}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn each_channel_texture_lane_reads_its_own_cell_energy() {
    // Channel 0 ramps across and channel 1 ramps down, so each channel's texture has its own value.
    let (width, height) = (56u32, 32u32);
    let across_slope = 3.0f32 / 1024.0;
    let down_slope = 5.0f32 / 1024.0;
    let mut new = Vec::with_capacity((2 * width * height) as usize);
    for index in 0..width * height {
        let across = (index % width) as f32 * across_slope;
        let down = (index / width) as f32 * down_slope;
        new.push(across);
        new.push(down);
    }

    let frames = StatsFrames {
        width,
        height,
        channels: 2,
        previous: vec![0.5; new.len()],
        new,
    };
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    // A full block holds 12 pairs along each ramp out of 24. The ragged block holds 4 horizontal
    // and 6 vertical pairs out of 10.
    let across_step = 4.0 * across_slope;
    let down_step = 4.0 * down_slope;
    let full_expected = [
        12.0 * across_step * across_step / 24.0,
        12.0 * down_step * down_step / 24.0,
    ];
    let ragged_expected = [
        4.0 * across_step * across_step / 10.0,
        6.0 * down_step * down_step / 10.0,
    ];
    let still = MotionGrid::from_fn(width, height, TEMPORAL_NOISE_BLOCK, |_, _| [0, 0]);

    for motion in [None, Some(&still)] {
        let records = run_stats(&frames, motion, false);

        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let record = frames.record(&records, block_x, block_y);
                let ragged = block_x + 1 == blocks_x;
                let expected = if ragged { ragged_expected } else { full_expected };

                for channel in 0..2u32 {
                    let lane = texture_lane(2, channel) as usize;
                    let texture = record[lane];
                    let wanted = expected[channel as usize];
                    let relative_error = (texture - wanted).abs() / wanted;
                    let what = format!("block ({block_x}, {block_y}) channel {channel}");
                    assert!(
                        relative_error <= 1e-5,
                        "{what}: texture {texture}, expected {wanted}"
                    );
                }
            }
        }
    }
}

#[test]
fn moved_lane_flags_non_zero_vectors() {
    // Five block columns leave the rightmost one even, so no shifted block leaves the frame.
    // `new(x, y)` matches `previous(x + 1, y)`, so every odd column follows its vector.
    let (width, height) = (80u32, 32u32);
    let frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: texture_frame(width, height, 0, 0),
        new: texture_frame(width, height, 1, 0),
    };
    let vector_at = |col: u32, _row: u32| if col.is_multiple_of(2) { [0, 0] } else { [1, 0] };
    let grid = MotionGrid::from_fn(width, height, TEMPORAL_NOISE_BLOCK, vector_at);
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let moved_lane = 2 + BLOCK_MOVED_LANE as usize;

    let records = run_stats(&frames, Some(&grid), true);

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let record = frames.record(&records, block_x, block_y);
            let expected = (block_x % 2) as f32;
            assert_eq!(record[moved_lane], expected, "block ({block_x}, {block_y})");
        }
    }
}

#[test]
fn covering_block_rule_picks_the_origin_block() {
    for step in [4u32, 8, 32] {
        assert_covering_block_rule(step);
    }
}

/// Every noise block reads the vector of the motion block covering its origin, and every other
/// motion block holds the decoy.
///
/// At step 32 each motion block covers some noise block's origin, so no decoy fits.
fn assert_covering_block_rule(step: u32) {
    let (width, height) = (64u32, 64u32);
    let motion_blocks_x = width.div_ceil(step);
    let motion_blocks_y = height.div_ceil(step);
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let covering_col = |block_x: u32| (block_x * TEMPORAL_NOISE_BLOCK / step).min(motion_blocks_x - 1);
    let covering_row = |block_y: u32| (block_y * TEMPORAL_NOISE_BLOCK / step).min(motion_blocks_y - 1);

    // Each motion block's own vector points inwards, so no noise block it covers leaves the frame.
    let motion_vector = |col: u32, row: u32| {
        let reach_x = 1 + ((col + 2 * row) % 3) as i32;
        let reach_y = ((col * row) % 3) as i32;
        let shift_x = if col * step < width / 2 { reach_x } else { -reach_x };
        let shift_y = if row * step < height / 2 {
            reach_y
        } else {
            -reach_y
        };
        [shift_x, shift_y]
    };
    let block_vector = |block_x: u32, block_y: u32| {
        let col = covering_col(block_x);
        let row = covering_row(block_y);
        motion_vector(col, row)
    };

    let ring = hashed_ring(width, height, 1, 2, 13);
    let frame_len = (width * height) as usize;
    let mut frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: ring[..frame_len].to_vec(),
        new: ring[frame_len..].to_vec(),
    };
    plant_block_matches(&mut frames, block_vector);
    frames.new = noisy_field_over(&frames.new, width, height, 0.01, 14);

    let vector_at = |col: u32, row: u32| {
        let origin_col = (0..blocks_x).any(|block_x| covering_col(block_x) == col);
        let origin_row = (0..blocks_y).any(|block_y| covering_row(block_y) == row);
        if origin_col && origin_row {
            motion_vector(col, row)
        } else {
            DECOY_VECTOR
        }
    };
    let grid = MotionGrid::from_fn(width, height, step, vector_at);

    let records = run_stats(&frames, Some(&grid), true);

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let record = frames.record(&records, block_x, block_y);
            let vector = block_vector(block_x, block_y);
            let label = format!("step {step}, block ({block_x}, {block_y})");
            let gain = host_gain(&frames, block_x, block_y, vector).expect("the match stays in the frame");
            assert!(gain <= COMPENSATION_GAIN_MAX, "{label}: gain {gain}");

            let expected = host_sum_d2(&frames, block_x, block_y, vector);
            let relative_error = (record[1] - expected).abs() / expected;
            assert!(
                relative_error <= 1e-4,
                "{label}: sum_d2 {}, expected {expected}",
                record[1]
            );
        }
    }
}

/// Copies each block of `new` into `previous` at its shifted position, so a block matches along
/// its own vector except where a neighbour's copy overlaps it.
fn plant_block_matches(frames: &mut StatsFrames, vector_at: impl Fn(u32, u32) -> [i32; 2]) {
    let (blocks_x, blocks_y) = temporal_stats_blocks(frames.width, frames.height);

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let vector = vector_at(block_x, block_y);
            let (block_width, block_height) = frames.block_extent(block_x, block_y);

            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    let x = block_x * TEMPORAL_NOISE_BLOCK + local_x;
                    let y = block_y * TEMPORAL_NOISE_BLOCK + local_y;
                    let prev_x = (x as i32 + vector[0]) as u32;
                    let prev_y = (y as i32 + vector[1]) as u32;
                    let new_index = (y * frames.width + x) as usize;
                    let prev_index = (prev_y * frames.width + prev_x) as usize;
                    frames.previous[prev_index] = frames.new[new_index];
                }
            }
        }
    }
}

/// On independent noise a forced vector explains nothing, so most blocks keep the zero-vector
/// residual. A block whose gain still falls to the limit by chance follows the vector, as the host
/// rule says.
#[test]
fn noise_fitted_vector_keeps_the_zero_vector_residual() {
    let (width, height) = (64u32, 48u32);
    let sigma = 2.0 / 255.0;
    let clean = vec![0.5f32; (width * height) as usize];
    let frames = StatsFrames {
        width,
        height,
        channels: 1,
        previous: noisy_field_over(&clean, width, height, sigma, 31),
        new: noisy_field_over(&clean, width, height, sigma, 32),
    };
    let vector = [1, 0];
    let grid = MotionGrid::uniform(width, height, 8, vector);
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let moved_lane = 2 + BLOCK_MOVED_LANE as usize;

    for luma_fields in [false, true] {
        let compensated = run_stats(&frames, Some(&grid), luma_fields);
        let raw = run_stats(&frames, None, luma_fields);

        let mut kept = 0;
        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let gain =
                    host_gain(&frames, block_x, block_y, vector).expect("the match stays in the frame");
                let follows = gain <= COMPENSATION_GAIN_MAX;
                let record = frames.record(&compensated, block_x, block_y);
                let label = format!("block ({block_x}, {block_y}) gain {gain}, luma_fields {luma_fields}");
                let expected_moved = if follows { 1.0 } else { 0.0 };
                assert_eq!(record[moved_lane], expected_moved, "{label}");

                if follows {
                    continue;
                }

                let raw_record = frames.record(&raw, block_x, block_y);
                let record_bits: Vec<u32> = record.iter().map(|value| value.to_bits()).collect();
                let raw_bits: Vec<u32> = raw_record.iter().map(|value| value.to_bits()).collect();
                assert_eq!(record_bits, raw_bits, "{label}");
                kept += 1;
            }
        }

        let blocks = blocks_x * blocks_y;
        assert!(
            kept * 4 >= blocks * 3,
            "only {kept} of {blocks} blocks kept the zero vector"
        );
    }
}
