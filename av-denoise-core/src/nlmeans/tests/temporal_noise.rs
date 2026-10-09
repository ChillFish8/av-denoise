use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::noise::{
    NoiseCtx,
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    TemporalStatsCtx,
    aggregate_temporal_noise_stats,
    correlation_factor,
    partials_len,
    read_temporal_stats_slot,
    run_noise_estimate,
    run_temporal_noise_stats,
    sigma_from_abs_sum,
    temporal_stats_blocks,
    temporal_stats_buf_bytes,
    temporal_stats_record_len,
};
use crate::nlmeans::*;

/// Runs the temporal-residual stats kernel on `previous` and `new` as ring slots 0 and 1.
///
/// Both frames are packed as `pixels * stored_ch`, and slot 1's stats are returned. The stats
/// buffer starts full of garbage, so a lane the kernel never writes shows up as a mismatch.
fn run_temporal_stats(
    width: u32,
    height: u32,
    stored_ch: u32,
    previous: &[f32],
    new: &[f32],
    luma_fields: bool,
) -> Vec<f32> {
    run_temporal_stats_prefilled(width, height, stored_ch, previous, new, luma_fields, -7.0)
}

/// [run_temporal_stats] over a stats buffer first filled with `fill`.
fn run_temporal_stats_prefilled(
    width: u32,
    height: u32,
    stored_ch: u32,
    previous: &[f32],
    new: &[f32],
    luma_fields: bool,
    fill: f32,
) -> Vec<f32> {
    let client = make_client();
    let frame_count = 2u32;
    let frame_len = (width * height * stored_ch) as usize;
    assert_eq!(previous.len(), frame_len);
    assert_eq!(new.len(), frame_len);

    let mut ring = vec![0.0f32; frame_len * frame_count as usize];
    ring[..frame_len].copy_from_slice(previous);
    ring[frame_len..].copy_from_slice(new);

    let ring_bytes = f32::as_bytes(&ring);
    let input_buf = client.create_from_slice(ring_bytes);
    let align = test_align();
    let stats_bytes = temporal_stats_buf_bytes(width, height, stored_ch, frame_count, align);
    let stats_fill = vec![fill; stats_bytes / size_of::<f32>()];
    let stats_fill_bytes = f32::as_bytes(&stats_fill);
    let stats_buf = client.create_from_slice(stats_fill_bytes);

    let ctx = TemporalStatsCtx {
        width,
        height,
        stored_ch,
        frame_count,
        slot_new: 1,
        slot_prev: 0,
        input_buf: &input_buf,
        stats_buf: &stats_buf,
        align,
    };
    run_temporal_noise_stats::<R>(&client, &ctx, luma_fields).expect("temporal noise stats dispatch failed");

    read_temporal_stats_slot::<R>(
        &client,
        &stats_buf,
        width,
        height,
        stored_ch,
        frame_count,
        1,
        align,
    )
    .expect("readback failed")
}

/// CPU oracle for `nlm_temporal_noise_stats`, written from scratch rather than from a closed form.
///
/// Only valid for `stored_ch == 1`, because [quarter_fields_host] indexes the frames as single-channel.
fn reference_temporal_stats(
    width: u32,
    height: u32,
    stored_ch: u32,
    previous: &[f32],
    new: &[f32],
) -> Vec<f32> {
    let channel_stride = stored_ch as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let mut out = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let origin_x = block_x * TEMPORAL_NOISE_BLOCK;
            let origin_y = block_y * TEMPORAL_NOISE_BLOCK;
            let block_width = TEMPORAL_NOISE_BLOCK.min(width - origin_x);
            let block_height = TEMPORAL_NOISE_BLOCK.min(height - origin_y);

            let mut sum_d = vec![0.0f32; channel_stride];
            let mut sum_d2 = vec![0.0f32; channel_stride];
            let mut sum_lag = 0.0f32;

            for local_y in 0..block_height {
                let y = origin_y + local_y;
                let mut luma_residual_row = Vec::with_capacity(block_width as usize);
                for local_x in 0..block_width {
                    let x = origin_x + local_x;
                    let idx = (y * width + x) as usize;
                    for channel in 0..channel_stride {
                        let sample = idx * channel_stride + channel;
                        let residual = new[sample] - previous[sample];
                        sum_d[channel] += residual;
                        sum_d2[channel] += residual * residual;
                        if channel == 0 {
                            luma_residual_row.push(residual);
                        }
                    }
                }

                for local_x in 0..(block_width as usize).saturating_sub(1) {
                    sum_lag += luma_residual_row[local_x] * luma_residual_row[local_x + 1];
                }
            }

            let block_index = (block_y * blocks_x + block_x) as usize;
            let base = block_index * record_len;
            out[base..base + channel_stride].copy_from_slice(&sum_d);
            out[base + channel_stride..base + 2 * channel_stride].copy_from_slice(&sum_d2);
            out[base + 2 * channel_stride] = sum_lag;

            for quarter in 0..TEMPORAL_QUARTERS {
                let fields = quarter_fields_host(new, previous, width, height, block_x, block_y, quarter);
                let quarter_start = base + quarter_offset(stored_ch, quarter);
                let quarter_end = quarter_start + TEMPORAL_QUARTER_FIELDS as usize;
                out[quarter_start..quarter_end].copy_from_slice(&fields);
            }
        }
    }

    out
}

/// Where quarter `quarter`'s record starts within one block record.
fn quarter_offset(stored_ch: u32, quarter: u32) -> usize {
    (2 * stored_ch + TEMPORAL_QUARTER_BASE + quarter * TEMPORAL_QUARTER_FIELDS) as usize
}

/// The nine fields of one quarter's record, computed on the host from single-channel frames.
///
/// An empty quarter keeps the kernel's min and max seeds.
fn quarter_fields_host(
    new: &[f32],
    previous: &[f32],
    width: u32,
    height: u32,
    block_x: u32,
    block_y: u32,
    quarter: u32,
) -> [f32; 9] {
    let size = TEMPORAL_QUARTER_SIZE;
    let origin_x = block_x * TEMPORAL_NOISE_BLOCK + (quarter % 2) * size;
    let origin_y = block_y * TEMPORAL_NOISE_BLOCK + (quarter / 2) * size;
    let quarter_width = size.min(width.saturating_sub(origin_x));
    let quarter_height = size.min(height.saturating_sub(origin_y));

    let mut sum_d = 0.0f32;
    let mut sum_d2 = 0.0f32;
    let mut luma_sum = 0.0f32;
    let mut luma_min = 1.0e30f32;
    let mut luma_max = -1.0e30f32;
    for y in 0..quarter_height {
        for x in 0..quarter_width {
            let idx = ((origin_y + y) * width + origin_x + x) as usize;
            let residual = new[idx] - previous[idx];
            sum_d += residual;
            sum_d2 += residual * residual;
            luma_sum += new[idx];
            luma_min = luma_min.min(new[idx]);
            luma_max = luma_max.max(new[idx]);
        }
    }

    let full = quarter_width == size && quarter_height == size;
    let flatness = if full {
        let mut cells = [0.0f32; 16];
        for cell_y in 0..4u32 {
            for cell_x in 0..4u32 {
                let mut cell = 0.0f32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let row = origin_y + 2 * cell_y + dy;
                        let column = origin_x + 2 * cell_x + dx;
                        let idx = (row * width + column) as usize;
                        cell += 0.5 * (new[idx] + previous[idx]);
                    }
                }

                cells[(cell_y * 4 + cell_x) as usize] = cell / 4.0;
            }
        }

        let mut energy = 0.0f32;
        for cell_y in 0..4usize {
            for cell_x in 0..4usize {
                let here = cells[cell_y * 4 + cell_x];
                if cell_x + 1 < 4 {
                    let right = cells[cell_y * 4 + cell_x + 1];
                    energy += (here - right) * (here - right);
                }

                if cell_y + 1 < 4 {
                    let below = cells[(cell_y + 1) * 4 + cell_x];
                    energy += (here - below) * (here - below);
                }
            }
        }

        energy / 24.0
    } else {
        3.0e38
    };

    let mean_at = |x: u32, y: u32| {
        let idx = ((origin_y + y) * width + origin_x + x) as usize;
        0.5 * (new[idx] + previous[idx])
    };

    let mut tensor_xx = 0.0f32;
    let mut tensor_yy = 0.0f32;
    let mut tensor_xy = 0.0f32;
    for y in 0..quarter_height.saturating_sub(1) {
        for x in 0..quarter_width.saturating_sub(1) {
            let top_left = mean_at(x, y);
            let top_right = mean_at(x + 1, y);
            let bottom_left = mean_at(x, y + 1);
            let bottom_right = mean_at(x + 1, y + 1);
            let grad_x = 0.5 * ((top_right + bottom_right) - (top_left + bottom_left));
            let grad_y = 0.5 * ((bottom_left + bottom_right) - (top_left + top_right));
            tensor_xx += grad_x * grad_x;
            tensor_yy += grad_y * grad_y;
            tensor_xy += grad_x * grad_y;
        }
    }

    let mut fields = [0.0f32; 9];
    fields[QUARTER_SUM_D as usize] = sum_d;
    fields[QUARTER_SUM_D2 as usize] = sum_d2;
    fields[QUARTER_LUMA_SUM as usize] = luma_sum;
    fields[QUARTER_FLATNESS as usize] = flatness;
    fields[QUARTER_LUMA_MIN as usize] = luma_min;
    fields[QUARTER_LUMA_MAX as usize] = luma_max;
    fields[QUARTER_TENSOR_XX as usize] = tensor_xx;
    fields[QUARTER_TENSOR_YY as usize] = tensor_yy;
    fields[QUARTER_TENSOR_XY as usize] = tensor_xy;
    fields
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, message: &str) {
    assert_eq!(actual.len(), expected.len(), "{message}: length mismatch");
    for (i, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "{message}: index {i}: got {actual_value}, expected {expected_value} (tol {tolerance})"
        );
    }
}

/// The frame grids into exactly four full blocks, so every block's sums have a closed form.
#[test]
fn kernel_uniform_diff_exact_sums() {
    let width = 32;
    let height = 32;
    let difference = 0.1f32;
    let previous = vec![0.0f32; (width * height) as usize];
    let new = vec![difference; (width * height) as usize];

    let block_pixels = 256.0f32;
    let lag_pairs = 240.0f32;
    let expected_block = [
        block_pixels * difference,
        block_pixels * difference * difference,
        lag_pairs * difference * difference,
    ];

    let got = run_temporal_stats(width, height, 1, &previous, &new, true);
    let oracle = reference_temporal_stats(width, height, 1, &previous, &new);

    let record_len = temporal_stats_record_len(1) as usize;
    for block in 0..4 {
        let record = &got[block * record_len..block * record_len + 3];
        let label = format!("block {block}");
        assert_close(record, &expected_block, 1e-4, &label);
    }

    assert_close(&got, &oracle, 1e-4, "kernel vs CPU oracle");
}

/// Checked against the CPU oracle rather than a closed form, covering per-pixel variation the uniform
/// test cannot reach.
#[test]
fn kernel_gradient_diff_exact_sums() {
    let width = 48;
    let height = 32;
    let mut previous = vec![0.0f32; (width * height) as usize];
    let mut new = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            new[(y * width + x) as usize] = x as f32;
        }
    }

    // `previous` stays zero so the residual equals `new`.
    previous.fill(0.0);

    let got = run_temporal_stats(width, height, 1, &previous, &new, true);
    let oracle = reference_temporal_stats(width, height, 1, &previous, &new);
    assert_close(&got, &oracle, 1e-2, "kernel vs CPU oracle (gradient diff)");
}

/// A 33x17 frame leaves a last column one pixel wide and a last row one pixel tall.
///
/// A uniform difference still has a closed form per block, using each block's truncated counts.
#[test]
fn kernel_ragged_block_dims() {
    let width = 33;
    let height = 17;
    let difference = 0.2f32;
    let previous = vec![0.0f32; (width * height) as usize];
    let new = vec![difference; (width * height) as usize];

    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    assert_eq!((blocks_x, blocks_y), (3, 2));

    let got = run_temporal_stats(width, height, 1, &previous, &new, true);
    let oracle = reference_temporal_stats(width, height, 1, &previous, &new);
    // `luma_sum` reduces through a shared-memory tree on the device but a sequential loop in the
    // oracle. Summing 256 copies of a value f32 cannot hold exactly in a different order needs more
    // slack than the other lanes.
    assert_close(&got, &oracle, 5e-4, "kernel vs CPU oracle (ragged)");

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_width = TEMPORAL_NOISE_BLOCK.min(width - block_x * TEMPORAL_NOISE_BLOCK);
            let block_height = TEMPORAL_NOISE_BLOCK.min(height - block_y * TEMPORAL_NOISE_BLOCK);
            let block_pixels = (block_width * block_height) as f32;
            let lag_pairs = (block_height * block_width.saturating_sub(1)) as f32;
            let expected = [
                block_pixels * difference,
                block_pixels * difference * difference,
                lag_pairs * difference * difference,
            ];

            let block = (block_y * blocks_x + block_x) as usize;
            let record_len = temporal_stats_record_len(1) as usize;
            let record = &got[block * record_len..block * record_len + 3];
            // Looser than the oracle comparison, because the kernel accumulates up to 256 copies of a
            // value f32 cannot hold exactly while the closed form multiplies once.
            let label = format!("ragged block ({block_x},{block_y}), dims {block_width}x{block_height}");
            assert_close(record, &expected, 5e-3, &label);
        }
    }
}

fn assert_relative(actual: f32, expected: f32, tolerance: f32, message: &str) {
    let rel_err = (actual - expected).abs() / expected.abs().max(1e-8);
    assert!(
        rel_err <= tolerance,
        "{message}: got {actual}, expected {expected} (rel err {rel_err})"
    );
}

/// A textured frame with a small, pixel-varying residual.
fn textured_pair(width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let mut new = vec![0.0f32; (width * height) as usize];
    let mut previous = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let value = 0.2 + 0.6 * ((x * 7 + y * 13) % 17) as f32 / 17.0;
            new[idx] = value;
            let offset = 0.02 * (((x + y) % 3) as f32 - 1.0);
            previous[idx] = (value + offset).clamp(0.0, 1.0);
        }
    }

    (previous, new)
}

fn assert_field_close(actual: f32, expected: f32, message: &str) {
    let tolerance = 1e-4 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{message}: got {actual}, expected {expected} (tol {tolerance})"
    );
}

/// Checks every quarter record of a single-channel run against the host mirror.
///
/// Returns how many quarters carried real flatness.
fn assert_quarters_match_mirror(width: u32, height: u32, previous: &[f32], new: &[f32]) -> usize {
    let stored_ch = 1u32;
    let got = run_temporal_stats(width, height, stored_ch, previous, new, true);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);

    let mut real_flatness = 0;
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_index = (block_y * blocks_x + block_x) as usize;
            let record = &got[block_index * record_len..(block_index + 1) * record_len];

            for quarter in 0..TEMPORAL_QUARTERS {
                let expected = quarter_fields_host(new, previous, width, height, block_x, block_y, quarter);
                let offset = quarter_offset(stored_ch, quarter);
                let fields = &record[offset..offset + TEMPORAL_QUARTER_FIELDS as usize];
                let label = format!("{width}x{height} block ({block_x},{block_y}) quarter {quarter}");

                let exact_fields = [
                    QUARTER_SUM_D,
                    QUARTER_SUM_D2,
                    QUARTER_LUMA_SUM,
                    QUARTER_TENSOR_XX,
                    QUARTER_TENSOR_YY,
                    QUARTER_TENSOR_XY,
                ];
                for field in exact_fields {
                    let index = field as usize;
                    let field_label = format!("{label} field {field}");
                    assert_field_close(fields[index], expected[index], &field_label);
                }

                let min_index = QUARTER_LUMA_MIN as usize;
                let max_index = QUARTER_LUMA_MAX as usize;
                assert_eq!(fields[min_index], expected[min_index], "{label}: luma_min");
                assert_eq!(fields[max_index], expected[max_index], "{label}: luma_max");

                let flatness_index = QUARTER_FLATNESS as usize;
                let expected_flatness = expected[flatness_index];
                let got_flatness = fields[flatness_index];
                if expected_flatness == 3.0e38 {
                    assert_eq!(got_flatness, 3.0e38, "{label}: flatness sentinel");
                } else {
                    let flatness_label = format!("{label} flatness");
                    assert_relative(got_flatness, expected_flatness, 1e-4, &flatness_label);
                    real_flatness += 1;
                }
            }
        }
    }

    // The scalar lanes stay exactly as the 16x16 oracle computes them.
    let oracle = reference_temporal_stats(width, height, stored_ch, previous, new);
    for block in 0..(blocks_x * blocks_y) as usize {
        let got_record = &got[block * record_len..block * record_len + 3];
        let oracle_record = &oracle[block * record_len..block * record_len + 3];
        let label = format!("block {block} sum_d/sum_d2/lag");
        assert_close(got_record, oracle_record, 1e-4, &label);
    }

    real_flatness
}

#[test]
fn quarter_records_match_the_host_mirror_on_texture() {
    let width = 48u32;
    let height = 32u32;
    let (previous, new) = textured_pair(width, height);

    let real_flatness = assert_quarters_match_mirror(width, height, &previous, &new);

    assert_eq!(real_flatness, 24, "every quarter of six full blocks is full");
}

#[test]
fn quarter_records_match_the_host_mirror_on_flat_noise() {
    let size = 32u32;
    let previous = noisy_copy(size, 0.5, 0.01, 1);
    let new = noisy_copy(size, 0.5, 0.01, 2);

    let real_flatness = assert_quarters_match_mirror(size, size, &previous, &new);

    assert_eq!(real_flatness, 16);
}

/// A 45x29 frame leaves the last column 13 pixels wide and the last row 13 pixels tall, so each
/// ragged block holds full and partial quarters.
#[test]
fn quarter_records_match_the_host_mirror_on_ragged_edges() {
    let width = 45u32;
    let height = 29u32;
    let (previous, new) = textured_pair(width, height);

    let real_flatness = assert_quarters_match_mirror(width, height, &previous, &new);

    // Block (0,0) has 4 full quarters, block (1,0) 4, block (2,0) 2, block (0,1) 2, block (1,1) 2
    // and block (2,1) 1.
    assert_eq!(real_flatness, 15);
}

/// A 40x24 frame leaves the last column and row exactly 8 pixels, so some quarters lie entirely
/// outside the frame.
#[test]
fn quarter_records_match_the_host_mirror_with_empty_quarters() {
    let width = 40u32;
    let height = 24u32;
    let (previous, new) = textured_pair(width, height);

    let real_flatness = assert_quarters_match_mirror(width, height, &previous, &new);

    assert_eq!(real_flatness, 4 + 4 + 2 + 2 + 2 + 1);
}

#[test]
fn flat_noisy_block_reads_low_flatness_in_every_quarter() {
    let size = 16u32;
    let sigma = 0.01f32;
    let stored_ch = 1u32;
    let previous = noisy_copy(size, 0.5, sigma, 1);
    let new = noisy_copy(size, 0.5, sigma, 2);

    let got = run_temporal_stats(size, size, stored_ch, &previous, &new, true);

    for quarter in 0..TEMPORAL_QUARTERS {
        let offset = quarter_offset(stored_ch, quarter);
        let flatness = got[offset + QUARTER_FLATNESS as usize];
        assert!(
            flatness < 0.5 * sigma * sigma,
            "quarter {quarter}: expected low flatness on pure noise, got {flatness}"
        );
    }
}

/// The off run starts from a buffer of garbage, so a zero quarter lane was written by the kernel.
#[test]
fn luma_fields_off_leaves_the_quarter_lanes_zero_and_the_rest_unchanged() {
    let width = 45u32;
    let height = 29u32;
    let stored_ch = 1u32;
    let (previous, new) = textured_pair(width, height);

    let with_luma = run_temporal_stats(width, height, stored_ch, &previous, &new, true);
    let without_luma = run_temporal_stats_prefilled(width, height, stored_ch, &previous, &new, false, 123.0);

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let scalar_len = 2 * stored_ch as usize + 1;
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);

    for block in 0..(blocks_x * blocks_y) as usize {
        let base = block * record_len;
        let with_record = &with_luma[base..base + scalar_len];
        let without_record = &without_luma[base..base + scalar_len];
        assert_eq!(
            with_record, without_record,
            "block {block}: sum_d/sum_d2/lag must not depend on luma_fields"
        );

        let quarter_lanes = &without_luma[base + scalar_len..base + record_len];
        assert_eq!(quarter_lanes.len(), 36);
        assert!(
            quarter_lanes.iter().all(|&value| value == 0.0),
            "block {block}: quarter lanes must read 0 with luma_fields off, got {quarter_lanes:?}"
        );
    }
}

#[test]
fn white_noise_pair_recovers_known_sigma() {
    let size = 256;
    let true_sigma = 8.0 / 255.0;

    let previous = noisy_copy(size, 0.5, true_sigma, 1);
    let new = noisy_copy(size, 0.5, true_sigma, 2);

    let records = run_temporal_stats(size, size, 1, &previous, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, size, size)
        .expect("a static white-noise pair should clear the static-block floor");

    let rel_err = (sample.sigma[0] - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.10,
        "estimated sigma {} vs true {true_sigma} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
}

/// The temporal measurement ignores spatial correlation, so the sigma stays within 10%.
///
/// The correlation reading must still register the blur, which Immerkær reads low.
#[test]
fn correlated_noise_pair_recovers_marginal_sigma_and_rho() {
    let width = 256;
    let height = 256;
    let sigma_marginal = 8.0 / 255.0;
    // The blur scales the variance by 0.375, the sum of its squared taps, so the sigma before the
    // blur is raised to land the blurred field on the target.
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();

    let previous = correlated_noisy_frame(width, height, 0.5, sigma_pre, 11);
    let new = correlated_noisy_frame(width, height, 0.5, sigma_pre, 12);

    let records = run_temporal_stats(width, height, 1, &previous, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, width, height)
        .expect("a static correlated-noise pair should clear the static-block floor");

    let rel_err = (sample.sigma[0] - sigma_marginal).abs() / sigma_marginal;
    assert!(
        rel_err <= 0.10,
        "estimated sigma {} vs marginal truth {sigma_marginal} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
    assert!(
        sample.rho > 0.4,
        "expected rho > 0.4 for horizontally-blurred grain, got {}",
        sample.rho
    );
}

/// A ramp shifted by a few pixels stands in for motion, and its steady per-pixel offset fails the
/// static check almost everywhere.
#[test]
fn moving_content_pair_mostly_non_static() {
    let width = 64;
    let height = 64;
    let shift = 4u32;

    let mut previous = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            previous[(y * width + x) as usize] = x as f32 / width as f32;
        }
    }

    let mut new = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x + shift).min(width - 1);
            new[(y * width + x) as usize] = previous[(y * width + source_x) as usize];
        }
    }

    let records = run_temporal_stats(width, height, 1, &previous, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, width, height);
    let static_fraction = sample.map(|sample| sample.static_fraction).unwrap_or(0.0);

    assert!(
        static_fraction < 0.5,
        "expected mostly non-static blocks under a content shift, got static_fraction={static_fraction}"
    );
}

/// Immerkær alone reads several times lower on the same correlated grain.
///
/// The blur's lag-1 correlation is two thirds, from the same taps that give the 0.375 variance scale.
#[test]
fn hq_temporal_folds_correlated_grain_above_immerkaer_alone() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let sigma_marginal = 8.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let base = 0.5f32;

    let frame_count = 14;
    let frames: Vec<Vec<f32>> = (0..frame_count)
        .map(|i| correlated_noisy_frame(width, height, base, sigma_pre, 100 + i as u32))
        .collect();

    // What Immerkær alone reads on this content, computed directly rather than through the denoiser.
    let immerkaer_only = {
        let input_bytes = f32::as_bytes(&frames[0]);
        let partials_bytes = partials_len(width, height) * size_of::<f32>();
        let input_buf = client.create_from_slice(input_bytes);
        let partials_buf = client.empty(partials_bytes);
        let results_buf = client.empty(4 * size_of::<f32>());
        let ctx = NoiseCtx {
            width,
            height,
            channels: 1,
            stored_ch: 1,
            frame_count: 1,
            frame: 0,
            slot: 0,
            input_buf: &input_buf,
            partials_buf: &partials_buf,
            results_buf: &results_buf,
        };
        run_noise_estimate::<R>(&client, &ctx).expect("immerkaer dispatch failed");

        let bytes = client.read_one(results_buf).expect("immerkaer readback failed");
        let data = f32::from_bytes(&bytes);
        sigma_from_abs_sum(data[0], width, height)
    };

    assert!(
        immerkaer_only < sigma_marginal * 0.5,
        "expected Immerkær alone to read well below the marginal truth {sigma_marginal} \
         on correlated grain, got {immerkaer_only}"
    );

    let params = NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    for frame in &frames {
        denoiser.push_frame(frame);
        let _ = denoiser.denoise().unwrap();
    }

    let folded = denoiser
        .noise_estimator
        .current()
        .expect("estimator should hold a value after several full-window submits")[0];

    let expected = sigma_marginal * correlation_factor(2.0 / 3.0);
    let rel_err = (folded - expected).abs() / expected;
    assert!(
        rel_err <= 0.25,
        "folded sigma {folded} vs corrected truth {expected} (rel err {rel_err:.3}), \
         versus Immerkær-alone {immerkaer_only}"
    );
}

/// On grain with true rho 2/3, blending from 0 with `EMA_ALPHA = 0.2` would read about 0.13 after
/// the first sample, an 80% relative error. The 25% tolerance separates seeding from blending.
#[test]
fn rho_smoothed_seeds_from_first_sample_not_from_zero() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let sigma_marginal = 8.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let base = 0.5f32;
    let true_rho = 2.0 / 3.0;

    let params = NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    let mut first_seeded_rho = None;
    for i in 0..(width.min(height)) {
        let frame = correlated_noisy_frame(width, height, base, sigma_pre, 200 + i);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
        if let Some(rho) = denoiser.rho_smoothed {
            first_seeded_rho = Some(rho);
            break;
        }
    }

    let first_seeded_rho =
        first_seeded_rho.expect("estimator should hold a value after enough full-window submits");
    let rel_err = (first_seeded_rho - true_rho).abs() / true_rho;
    assert!(
        rel_err <= 0.25,
        "rho_smoothed after its first update was {first_seeded_rho} vs true rho {true_rho} \
         (rel err {rel_err:.3}); a from-zero blend would read close to {}",
        0.2 * true_rho
    );
}

/// The top half is static flat content and the bottom half is fine texture panning a few pixels, both
/// under the same noise.
///
/// The panning residual averages near zero over a block, so the mean check alone lets nearly the
/// whole frame through. Only the top half is really noise, so the aggregation must reject most of
/// the bottom half.
#[test]
fn moving_texture_pair_near_zero_mean_residual_recovers_true_sigma() {
    let width = 128;
    let height = 128;
    let true_sigma = 2.0 / 255.0;
    let texture_sigma_pre = 24.0 / 255.0 / 0.375f32.sqrt();
    let shift = 3u32;
    let base = 0.5f32;

    // One texture field for the bottom half, independent of the measurement noise added below.
    let raw_texture = correlated_noisy_frame(width, height / 2, base, texture_sigma_pre, 1);

    let mut clean_previous = vec![base; (width * height) as usize];
    let mut clean_new = vec![base; (width * height) as usize];
    for y in 0..(height / 2) {
        for x in 0..width {
            let previous_value = raw_texture[(y * width + x) as usize];
            let source_x = (x + shift).min(width - 1);
            let new_value = raw_texture[(y * width + source_x) as usize];
            let out_y = height / 2 + y;
            clean_previous[(out_y * width + x) as usize] = previous_value;
            clean_new[(out_y * width + x) as usize] = new_value;
        }
    }

    let previous = noisy_field_over(&clean_previous, width, height, true_sigma, 11);
    let new = noisy_field_over(&clean_new, width, height, true_sigma, 12);

    let records = run_temporal_stats(width, height, 1, &previous, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, width, height)
        .expect("the static top half should clear STATIC_FRACTION_MIN on its own");

    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let total_blocks = blocks_x * blocks_y;
    let top_half_blocks = total_blocks / 2;
    assert!(
        (sample.static_fraction * total_blocks as f32) <= top_half_blocks as f32 * 1.5,
        "expected static_fraction to stay close to the true static top half ({} of {total_blocks} \
         blocks), got {}",
        top_half_blocks,
        sample.static_fraction
    );

    let rel_err = (sample.sigma[0] - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.25,
        "estimated sigma {} vs true static-half sigma {true_sigma} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
}
