use super::helpers::*;
use crate::nlmeans::noise::{
    NoiseCurve,
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    accepted_static_blocks,
    aggregate_temporal_noise_stats,
    build_noise_curve,
    temporal_noise_reading,
    temporal_stats_record_len,
};
use crate::nlmeans::*;

/// One synthetic 8x8 quarter, as the stats kernel would record it.
#[derive(Clone, Copy)]
pub(super) struct SyntheticQuarter {
    pub(super) sigma: f32,
    pub(super) mean_residual: f32,
    pub(super) mean_luma: f32,
    pub(super) flatness: f32,
    pub(super) luma_min: f32,
    pub(super) luma_max: f32,
    pub(super) tensor_xx: f32,
    pub(super) tensor_yy: f32,
    pub(super) tensor_xy: f32,
}

/// A static, flat, unclipped quarter at `luma` with `sigma`.
pub(super) fn quarter_at(luma: f32, sigma: f32) -> SyntheticQuarter {
    SyntheticQuarter {
        sigma,
        mean_residual: 0.0,
        mean_luma: luma,
        flatness: 0.0,
        luma_min: luma - 0.02,
        luma_max: luma + 0.02,
        tensor_xx: 0.0,
        tensor_yy: 0.0,
        tensor_xy: 0.0,
    }
}

pub(super) fn quarters_at(luma: f32, sigma: f32, count: usize) -> Vec<SyntheticQuarter> {
    vec![quarter_at(luma, sigma); count]
}

/// Writes `quarter` into a luma-only block record as a quarter of `pixels` pixels.
///
/// The block's scalar lanes gain the quarter's sums, so they stay the sums of its quarters.
pub(super) fn write_quarter(
    record: &mut [f32],
    quarter_index: usize,
    quarter: &SyntheticQuarter,
    pixels: f32,
) {
    let stored_ch = 1u32;
    let variance = 2.0 * quarter.sigma * quarter.sigma;
    let mean = quarter.mean_residual;
    let sum_d = mean * pixels;
    let sum_d2 = (variance + mean * mean) * pixels;
    record[0] += sum_d;
    record[1] += sum_d2;

    let quarter_offset = quarter_index as u32 * TEMPORAL_QUARTER_FIELDS;
    let base = (2 * stored_ch + TEMPORAL_QUARTER_BASE + quarter_offset) as usize;
    record[base + QUARTER_SUM_D as usize] = sum_d;
    record[base + QUARTER_SUM_D2 as usize] = sum_d2;
    record[base + QUARTER_LUMA_SUM as usize] = quarter.mean_luma * pixels;
    record[base + QUARTER_FLATNESS as usize] = quarter.flatness;
    record[base + QUARTER_LUMA_MIN as usize] = quarter.luma_min;
    record[base + QUARTER_LUMA_MAX as usize] = quarter.luma_max;
    record[base + QUARTER_TENSOR_XX as usize] = quarter.tensor_xx;
    record[base + QUARTER_TENSOR_YY as usize] = quarter.tensor_yy;
    record[base + QUARTER_TENSOR_XY as usize] = quarter.tensor_xy;
}

/// Records for a single row of full 16x16 blocks, luma only (`stored_ch` 1).
///
/// Every four quarters form one block, in top-left, top-right,
/// bottom-left, bottom-right order.
pub(super) fn synthetic_records(quarters: &[SyntheticQuarter]) -> Vec<f32> {
    assert_eq!(quarters.len() % 4, 0, "quarters must fill whole blocks");

    let quarter_pixels = 64.0f32;
    let record_len = temporal_stats_record_len(1) as usize;
    let block_count = quarters.len() / 4;
    let mut records = vec![0.0f32; block_count * record_len];

    for (block_index, block_quarters) in quarters.chunks(4).enumerate() {
        let record = &mut records[block_index * record_len..(block_index + 1) * record_len];

        for (quarter_index, quarter) in block_quarters.iter().enumerate() {
            write_quarter(record, quarter_index, quarter, quarter_pixels);
        }
    }

    records
}

/// A single row of full 16x16 blocks, one block per four quarters.
pub(super) fn frame_dims(quarter_count: usize) -> (u32, u32) {
    let block_count = quarter_count / 4;
    (16 * block_count as u32, 16)
}

/// The curve a single-row frame of `quarters` produces.
fn curve_for(quarters: &[SyntheticQuarter]) -> Option<NoiseCurve> {
    let stored_ch = 1;
    let channels = 1;
    let (width, height) = frame_dims(quarters.len());
    let records = synthetic_records(quarters);

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("a trusted selection");
    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("a trusted sample");
    build_noise_curve(&records, stored_ch, &accepted, sample.sigma[0])
}

/// Gives each block the residual means `+m`, `+m`, `-m`, `-m`, so every
/// block's mean stays 0 while its 16x16 sigma reads above its quarters'.
fn with_cancelling_means(quarters: &mut [SyntheticQuarter], mean: f32) {
    for (index, quarter) in quarters.iter_mut().enumerate() {
        quarter.mean_residual = if index % 4 < 2 { mean } else { -mean };
    }
}

#[test]
fn reading_sample_equals_the_scalar_aggregation() {
    let stored_ch = 1;
    let channels = 1;

    let good_sigmas_255 = [2.0f32, 3.0, 4.0, 3.5, 2.5];
    let good_quarters: Vec<SyntheticQuarter> = good_sigmas_255
        .iter()
        .flat_map(|&sigma_255| quarters_at(0.2, sigma_255 / 255.0, 4))
        .collect();

    let mut records = synthetic_records(&good_quarters);

    // Two blocks with a large mean residual, so the static gate rejects
    // them before either function ever reads their sigma.
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let n = 256.0f32;
    let mean0_bad = 10.0 / 255.0;
    for _ in 0..2 {
        let mut bad = vec![0.0f32; record_len];
        bad[0] = n * mean0_bad;
        records.extend(bad);
    }

    let (width, height) = frame_dims(good_quarters.len() + 8);

    let sample_scalar = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height);
    let reading = temporal_noise_reading(&records, channels, stored_ch, width, height, true, None);

    assert!(sample_scalar.is_some());
    assert_eq!(reading.sample, sample_scalar);
}

#[test]
fn curve_uses_the_median_per_bin_and_normalises() {
    let mut quarters = quarters_at(0.15, 0.02, 160);
    quarters.extend(quarters_at(0.35, 0.01, 160));
    quarters.extend(quarters_at(0.6, 0.005, 160));

    let curve = curve_for(&quarters).expect("three populated bins");

    let tol = 1e-6;
    assert!((curve.ratios[2] - 2.0).abs() < tol, "{:?}", curve.ratios);
    assert!((curve.ratios[5] - 1.0).abs() < tol, "{:?}", curve.ratios);
    assert!((curve.ratios[9] - 0.5).abs() < tol, "{:?}", curve.ratios);

    // The flat low and high ends, held at the nearest populated bin.
    assert!((curve.ratios[0] - 2.0).abs() < tol);
    assert!((curve.ratios[1] - 2.0).abs() < tol);
    assert!((curve.ratios[15] - 0.5).abs() < tol);

    // Linear interpolation between the populated bins.
    let expected = 2.0 + (1.0 - 2.0) / 3.0;
    assert!(
        (curve.ratios[3] - expected).abs() < tol,
        "{} vs {expected}",
        curve.ratios[3]
    );
}

#[test]
fn each_quarter_is_binned_by_its_own_mean_luma() {
    // Every block mixes one quarter from each bin, so binning by the
    // block's mean luma would put them all in one bin.
    let mut quarters = Vec::new();
    for _ in 0..40 {
        quarters.push(quarter_at(0.15, 0.02));
        quarters.push(quarter_at(0.35, 0.01));
        quarters.push(quarter_at(0.6, 0.005));
        quarters.push(quarter_at(0.9, 0.01));
    }

    let curve = curve_for(&quarters).expect("four populated bins");

    let tol = 1e-6;
    assert!((curve.ratios[2] - 2.0).abs() < tol, "{:?}", curve.ratios);
    assert!((curve.ratios[5] - 1.0).abs() < tol, "{:?}", curve.ratios);
    assert!((curve.ratios[9] - 0.5).abs() < tol, "{:?}", curve.ratios);
    assert!((curve.ratios[14] - 1.0).abs() < tol, "{:?}", curve.ratios);
}

#[test]
fn a_bin_with_fewer_than_32_quarters_is_empty() {
    let base = |third_bin: usize| {
        let mut quarters = quarters_at(0.15, 0.02, 40);
        quarters.extend(quarters_at(0.35, 0.02, 40));
        quarters.extend(quarters_at(0.6, 0.02, third_bin));
        // Pads the frame to whole blocks with a bin that never fills.
        let padding = 4 - third_bin % 4;
        quarters.extend(quarters_at(0.95, 0.02, padding));
        quarters
    };

    let short = base(31);
    assert!(
        curve_for(&short).is_none(),
        "only 2 of the 3 bins reach 32 quarters"
    );

    let enough = base(32);
    assert!(curve_for(&enough).is_some(), "32 quarters fill the third bin");
}

#[test]
fn fewer_than_three_bins_gives_none() {
    let mut quarters = quarters_at(0.15, 0.02, 40);
    quarters.extend(quarters_at(0.35, 0.02, 40));

    assert!(
        curve_for(&quarters).is_none(),
        "only 2 bins populated, below the minimum of 3"
    );
}

/// Two full bins plus a third bin that reaches 32 quarters only if all
/// five `odd_one` quarters count. Each odd quarter shares its block with
/// three plain ones, so its parent block is still accepted.
fn frame_with_odd_quarters(odd_one: SyntheticQuarter) -> Vec<SyntheticQuarter> {
    let mut quarters = quarters_at(0.15, 0.02, 40);
    quarters.extend(quarters_at(0.35, 0.02, 40));
    for index in 0..36 {
        let replaced = index % 4 == 3 && index < 20;
        let quarter = if replaced { odd_one } else { quarter_at(0.6, 0.02) };
        quarters.push(quarter);
    }

    quarters
}

/// Asserts the frame from [frame_with_odd_quarters] forms a curve with
/// plain quarters, and none with `gated` ones, which shows the gate kept
/// them out.
fn assert_gate_keeps_the_bin_empty(gated: SyntheticQuarter) {
    let plain = frame_with_odd_quarters(quarter_at(0.6, 0.02));
    assert!(
        curve_for(&plain).is_some(),
        "the ungated frame should form a curve"
    );

    let with_gated = frame_with_odd_quarters(gated);
    assert!(
        curve_for(&with_gated).is_none(),
        "the gated quarters must not let the third bin reach the 32-quarter minimum"
    );
}

#[test]
fn the_flat_and_clip_gates_reject_their_quarters() {
    // Above 0.5 * sigma_quarter_median^2 = 0.5 * 0.02^2 = 0.0002.
    let textured = SyntheticQuarter {
        flatness: 1.0,
        ..quarter_at(0.6, 0.02)
    };
    assert_gate_keeps_the_bin_empty(textured);

    let clipped_low = SyntheticQuarter {
        luma_min: 2.0 / 255.0,
        ..quarter_at(0.6, 0.02)
    };
    assert_gate_keeps_the_bin_empty(clipped_low);

    let clipped_high = SyntheticQuarter {
        luma_max: 253.0 / 255.0,
        ..quarter_at(0.6, 0.02)
    };
    assert_gate_keeps_the_bin_empty(clipped_high);
}

#[test]
fn a_ragged_quarter_is_rejected_by_the_flat_gate() {
    let ragged = SyntheticQuarter {
        flatness: 3.0e38,
        ..quarter_at(0.6, 0.02)
    };
    assert_gate_keeps_the_bin_empty(ragged);
}

#[test]
fn the_quarter_static_gate_passes_a_mean_between_the_block_and_quarter_gates() {
    let slow = SyntheticQuarter {
        mean_residual: 2.5 / 255.0,
        ..quarter_at(0.6, 0.02)
    };
    let quarters = frame_with_odd_quarters(slow);

    assert!(
        curve_for(&quarters).is_some(),
        "a 2.5 code mean is static for a 64-pixel quarter"
    );
}

#[test]
fn the_quarter_static_gate_rejects_a_moving_quarter() {
    let moving = SyntheticQuarter {
        mean_residual: 3.5 / 255.0,
        ..quarter_at(0.6, 0.02)
    };
    assert_gate_keeps_the_bin_empty(moving);
}

#[test]
fn the_rho_sigma_gate_rejects_a_noiseless_quarter() {
    let noiseless = quarter_at(0.6, 0.0);
    assert_gate_keeps_the_bin_empty(noiseless);
}

#[test]
fn quarters_of_a_rejected_block_never_count() {
    let build = |third_sigma: f32| {
        let mut quarters = quarters_at(0.15, 0.01, 40);
        quarters.extend(quarters_at(0.35, 0.01, 40));
        quarters.extend(quarters_at(0.6, third_sigma, 40));
        quarters
    };

    // Below 5 times the blocks' lower quartile, so these blocks stay.
    let kept = build(0.04);
    assert!(curve_for(&kept).is_some());

    // Above it, so the outlier check rejects the whole block, even
    // though each quarter alone would pass the quarter gates.
    let rejected = build(0.2);
    assert!(curve_for(&rejected).is_none());
}

#[test]
fn the_curve_normalises_by_the_quarter_median() {
    let sigma = 0.01;
    let mut quarters = quarters_at(0.15, sigma, 40);
    quarters.extend(quarters_at(0.35, sigma, 40));
    quarters.extend(quarters_at(0.6, sigma, 40));
    with_cancelling_means(&mut quarters, 0.005);

    // The 16x16 median reads above the quarters' own sigma, so
    // normalising by it would put every ratio below 1.
    let stored_ch = 1;
    let channels = 1;
    let (width, height) = frame_dims(quarters.len());
    let records = synthetic_records(&quarters);
    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("every block is static");
    assert!(sample.sigma[0] > sigma * 1.02, "{}", sample.sigma[0]);

    let curve = curve_for(&quarters).expect("three populated bins");

    assert_eq!(curve.ratios, [1.0; NOISE_CURVE_BINS]);
}

#[test]
fn the_flat_gate_uses_the_block_median() {
    // The quarters read sigma 0.01, which would set the flat limit at
    // 5.0e-5. Their blocks read about 0.0106, which sets it at 5.625e-5.
    let build = |flatness: f32| {
        let mut quarters = quarters_at(0.15, 0.01, 40);
        quarters.extend(quarters_at(0.35, 0.01, 40));
        for _ in 0..40 {
            let quarter = SyntheticQuarter {
                flatness,
                ..quarter_at(0.6, 0.01)
            };
            quarters.push(quarter);
        }

        with_cancelling_means(&mut quarters, 0.005);
        quarters
    };

    let between = build(5.3e-5);
    assert!(
        curve_for(&between).is_some(),
        "above the quarter limit but below the block limit"
    );

    let textured = build(5.8e-5);
    assert!(curve_for(&textured).is_none(), "above the block limit");
}

#[test]
fn a_ragged_parent_reads_its_partial_quarters_with_their_own_pixel_count() {
    // Each row holds a full block and a 12 pixel wide ragged one. The
    // ragged block's right quarters are 4x8, so they hold 32 pixels.
    let rows = 32u32;
    let width = 28;
    let height = 16 * rows;
    let stored_ch = 1;
    let channels = 1;
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let mut records = vec![0.0f32; 2 * rows as usize * record_len];

    // The full blocks fill three bins at sigma 0.01. Every other quarter
    // reads 0.02, and textured or ragged ones never reach a bin.
    let textured = SyntheticQuarter {
        flatness: 1.0,
        ..quarter_at(0.5, 0.02)
    };
    let ragged = SyntheticQuarter {
        flatness: 3.0e38,
        ..quarter_at(0.5, 0.02)
    };
    let full_block = [
        quarter_at(0.15, 0.01),
        quarter_at(0.35, 0.01),
        quarter_at(0.6, 0.01),
        textured,
    ];
    let ragged_block = [textured, ragged, textured, ragged];
    let ragged_pixels = [64.0, 32.0, 64.0, 32.0];

    for row in 0..rows as usize {
        let full_start = 2 * row * record_len;
        let ragged_start = full_start + record_len;

        let full_record = &mut records[full_start..full_start + record_len];
        for (quarter_index, quarter) in full_block.iter().enumerate() {
            write_quarter(full_record, quarter_index, quarter, 64.0);
        }

        let ragged_record = &mut records[ragged_start..ragged_start + record_len];
        for (quarter_index, quarter) in ragged_block.iter().enumerate() {
            write_quarter(
                ragged_record,
                quarter_index,
                quarter,
                ragged_pixels[quarter_index],
            );
        }
    }

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("a trusted selection");
    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("a trusted sample");
    let curve =
        build_noise_curve(&records, stored_ch, &accepted, sample.sigma[0]).expect("three populated bins");

    // 96 quarters read 0.01 and 160 read 0.02, so the quarter median is
    // 0.02 only when the 64 partial quarters read their sigma over 32
    // pixels. Over 64 they would read 0.0141, and skipped the median
    // would fall to 0.015.
    for ratio in curve.ratios {
        assert!((ratio - 0.5).abs() < 1e-5, "{:?}", curve.ratios);
    }
}

#[test]
fn letterbox_bars_do_not_reach_the_curve() {
    let mut base = quarters_at(0.15, 0.02, 160);
    base.extend(quarters_at(0.35, 0.01, 160));
    base.extend(quarters_at(0.6, 0.005, 160));

    let base_curve = curve_for(&base).expect("three populated bins");

    let mut with_bars = base.clone();
    with_bars.extend(quarters_at(0.0, 0.0, 800));

    let bar_curve = curve_for(&with_bars).expect("three populated bins");

    assert_eq!(bar_curve, base_curve);
}

#[test]
fn no_passing_quarter_gives_none() {
    // Every block's mean cancels to 0, so each block is accepted, but
    // every quarter alone fails the static gate.
    let mut quarters = quarters_at(0.15, 0.01, 40);
    quarters.extend(quarters_at(0.35, 0.01, 40));
    quarters.extend(quarters_at(0.6, 0.01, 40));
    with_cancelling_means(&mut quarters, 3.0 / 255.0);

    assert!(curve_for(&quarters).is_none());
}

/// A brightness ramp with static noise in each band, so a curve forms
/// with at least three populated bins.
pub(super) fn ramp_frame(width: u32, height: u32, seed: u32) -> Vec<f32> {
    let band_height = height / 3;
    let mut clean = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        let luma = match y / band_height {
            0 => 0.15,
            1 => 0.45,
            _ => 0.8,
        };
        for x in 0..width {
            clean[(y * width + x) as usize] = luma;
        }
    }
    noisy_field_over(&clean, width, height, 0.02, seed)
}

#[test]
fn reset_stream_state_clears_the_curve() {
    let client = make_client();
    let width = 320;
    let height = 240;

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
            windowed_noise_estimation: false,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.set_luma_noise_fields(true);

    let mut curve_seen = false;
    for i in 0..12u32 {
        let frame = ramp_frame(width, height, 100 + i);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
        if denoiser.current_noise_curve().is_some() {
            curve_seen = true;
            break;
        }
    }
    assert!(curve_seen, "expected a curve to form over the brightness ramp");

    denoiser.reset_stream_state();
    assert!(denoiser.current_noise_curve().is_none());
}
