use super::helpers::*;
use crate::nlmeans::noise::{
    TEMPORAL_LUMA_FLATNESS,
    TEMPORAL_LUMA_MAX,
    TEMPORAL_LUMA_MIN,
    TEMPORAL_LUMA_SUM,
    accepted_static_blocks,
    aggregate_temporal_noise_stats,
    build_noise_curve,
    temporal_noise_reading,
    temporal_stats_record_len,
};
use crate::nlmeans::*;

/// One synthetic 16x16 block, as the stats kernel would record it for a static, noisy block.
#[derive(Clone)]
struct SyntheticBlock {
    sigma: f32,
    mean_luma: f32,
    flatness: f32,
    luma_min: f32,
    luma_max: f32,
}

/// Records for a frame of `blocks_x` by `blocks_y` full 16x16 blocks, luma only (`stored_ch` 1).
fn synthetic_records(blocks: &[SyntheticBlock]) -> Vec<f32> {
    let pixels = 256.0f32;
    let record_len = temporal_stats_record_len(1) as usize;
    let mut records = vec![0.0f32; blocks.len() * record_len];
    for (index, block) in blocks.iter().enumerate() {
        let rec = &mut records[index * record_len..(index + 1) * record_len];
        let variance = 2.0 * block.sigma * block.sigma;
        rec[0] = 0.0;
        rec[1] = variance * pixels;
        rec[2] = 0.0;
        rec[2 + TEMPORAL_LUMA_SUM as usize] = block.mean_luma * pixels;
        rec[2 + TEMPORAL_LUMA_FLATNESS as usize] = block.flatness;
        rec[2 + TEMPORAL_LUMA_MIN as usize] = block.luma_min;
        rec[2 + TEMPORAL_LUMA_MAX as usize] = block.luma_max;
    }
    records
}

/// A flat, unclipped block repeated `count` times at `luma` with `sigma`.
fn blocks_at(luma: f32, sigma: f32, count: usize) -> Vec<SyntheticBlock> {
    (0..count)
        .map(|_| SyntheticBlock {
            sigma,
            mean_luma: luma,
            flatness: 0.0,
            luma_min: luma - 0.02,
            luma_max: luma + 0.02,
        })
        .collect()
}

/// A single row of `total_blocks` 16x16 blocks, wide enough to hold them
/// without any of them being ragged.
fn frame_dims(total_blocks: usize) -> (u32, u32) {
    (16 * total_blocks as u32, 16)
}

#[test]
fn reading_sample_equals_the_scalar_aggregation() {
    let stored_ch = 1;
    let channels = 1;

    let good_sigmas_255 = [2.0f32, 3.0, 4.0, 3.5, 2.5];
    let good_blocks: Vec<SyntheticBlock> = good_sigmas_255
        .iter()
        .map(|&sigma_255| SyntheticBlock {
            sigma: sigma_255 / 255.0,
            mean_luma: 0.2,
            flatness: 0.0,
            luma_min: 0.18,
            luma_max: 0.22,
        })
        .collect();

    let mut records = synthetic_records(&good_blocks);

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

    let (width, height) = frame_dims(good_blocks.len() + 2);

    let sample_scalar = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height);
    let reading = temporal_noise_reading(&records, channels, stored_ch, width, height, true);

    assert_eq!(reading.sample, sample_scalar);
}

#[test]
fn curve_uses_the_median_per_bin_and_normalises() {
    let stored_ch = 1;
    let channels = 1;
    let sigma_median = 0.01;

    let mut blocks = blocks_at(0.15, 0.02, 40);
    blocks.extend(blocks_at(0.35, 0.01, 40));
    blocks.extend(blocks_at(0.6, 0.005, 40));

    let (width, height) = frame_dims(blocks.len());
    let records = synthetic_records(&blocks);

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("every block is static");
    let curve =
        build_noise_curve(&records, stored_ch, &accepted, sigma_median).expect("three populated bins");

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
fn a_bin_with_fewer_than_32_blocks_is_empty() {
    let stored_ch = 1;
    let channels = 1;

    let mut blocks = blocks_at(0.15, 0.02, 31);
    blocks.extend(blocks_at(0.35, 0.02, 40));
    blocks.extend(blocks_at(0.6, 0.02, 40));

    let (width, height) = frame_dims(blocks.len());
    let records = synthetic_records(&blocks);

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("every block is static");
    let curve = build_noise_curve(&records, stored_ch, &accepted, 0.02);

    assert!(curve.is_none(), "only 2 of the 3 bins reach the 32-block minimum");
}

#[test]
fn fewer_than_three_bins_gives_none() {
    let stored_ch = 1;
    let channels = 1;

    let mut blocks = blocks_at(0.15, 0.02, 40);
    blocks.extend(blocks_at(0.35, 0.02, 40));

    let (width, height) = frame_dims(blocks.len());
    let records = synthetic_records(&blocks);

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("every block is static");
    let curve = build_noise_curve(&records, stored_ch, &accepted, 0.02);

    assert!(curve.is_none(), "only 2 bins populated, below the minimum of 3");
}

/// Builds a frame with two full 16x16-block bins plus a third bin that
/// reaches 32 blocks only if `gated` blocks, which one of the extra
/// gates should reject, are counted. Asserts the curve never forms,
/// which shows the third bin stayed empty.
fn assert_gate_keeps_the_bin_empty(gated: impl Fn() -> SyntheticBlock) {
    let stored_ch = 1;
    let channels = 1;

    let mut blocks = blocks_at(0.15, 0.02, 40);
    blocks.extend(blocks_at(0.35, 0.02, 40));
    blocks.extend(blocks_at(0.6, 0.02, 31));
    for _ in 0..5 {
        blocks.push(gated());
    }

    let (width, height) = frame_dims(blocks.len());
    let records = synthetic_records(&blocks);

    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("every block is static");
    let curve = build_noise_curve(&records, stored_ch, &accepted, 0.01);

    assert!(
        curve.is_none(),
        "the gated blocks must not let the third bin reach the 32-block minimum"
    );
}

#[test]
fn the_flat_and_clip_gates_reject_their_blocks() {
    // Above 0.5 * sigma_median^2 = 0.5 * 0.01^2 = 0.00005.
    assert_gate_keeps_the_bin_empty(|| SyntheticBlock {
        sigma: 0.02,
        mean_luma: 0.6,
        flatness: 1.0,
        luma_min: 0.58,
        luma_max: 0.62,
    });

    assert_gate_keeps_the_bin_empty(|| SyntheticBlock {
        sigma: 0.02,
        mean_luma: 0.6,
        flatness: 0.0,
        luma_min: 2.0 / 255.0,
        luma_max: 0.62,
    });

    assert_gate_keeps_the_bin_empty(|| SyntheticBlock {
        sigma: 0.02,
        mean_luma: 0.6,
        flatness: 0.0,
        luma_min: 0.58,
        luma_max: 253.0 / 255.0,
    });
}

#[test]
fn a_ragged_block_is_rejected_by_the_flat_gate() {
    assert_gate_keeps_the_bin_empty(|| SyntheticBlock {
        sigma: 0.02,
        mean_luma: 0.6,
        flatness: 3.0e38,
        luma_min: 0.58,
        luma_max: 0.62,
    });
}

#[test]
fn letterbox_bars_do_not_reach_the_curve() {
    let stored_ch = 1;
    let channels = 1;
    let sigma_median = 0.01;

    let mut base = blocks_at(0.15, 0.02, 40);
    base.extend(blocks_at(0.35, 0.01, 40));
    base.extend(blocks_at(0.6, 0.005, 40));

    let (width, height) = frame_dims(base.len());
    let base_records = synthetic_records(&base);
    let base_accepted = accepted_static_blocks(&base_records, channels, stored_ch, width, height)
        .expect("every base block is static");
    let base_curve = build_noise_curve(&base_records, stored_ch, &base_accepted, sigma_median)
        .expect("three populated bins");

    let mut with_bars = base.clone();
    with_bars.extend(blocks_at(0.0, 0.0, 200));

    let (width_bars, height_bars) = frame_dims(with_bars.len());
    let bar_records = synthetic_records(&with_bars);
    let bar_accepted = accepted_static_blocks(&bar_records, channels, stored_ch, width_bars, height_bars)
        .expect("the noisy blocks alone clear the static-fraction floor");
    let bar_curve = build_noise_curve(&bar_records, stored_ch, &bar_accepted, sigma_median)
        .expect("three populated bins");

    assert_eq!(bar_curve, base_curve);
}

#[test]
fn a_non_positive_median_gives_none() {
    let stored_ch = 1;
    let channels = 1;

    let mut blocks = blocks_at(0.15, 0.02, 40);
    blocks.extend(blocks_at(0.35, 0.02, 40));
    blocks.extend(blocks_at(0.6, 0.02, 40));

    let (width, height) = frame_dims(blocks.len());
    let records = synthetic_records(&blocks);
    let accepted =
        accepted_static_blocks(&records, channels, stored_ch, width, height).expect("every block is static");

    assert!(build_noise_curve(&records, stored_ch, &accepted, 0.0).is_none());
    assert!(build_noise_curve(&records, stored_ch, &accepted, -0.01).is_none());
}

/// A brightness ramp with static noise in each band, so a curve forms
/// with at least three populated bins.
fn ramp_frame(width: u32, height: u32, seed: u32) -> Vec<f32> {
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
