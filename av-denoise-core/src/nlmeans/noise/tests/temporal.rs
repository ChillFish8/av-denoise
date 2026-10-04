use crate::nlmeans::noise::temporal::{
    RHO_SIGMA_GATE,
    STATIC_GATE,
    TEMPORAL_NOISE_BLOCK,
    aggregate_temporal_noise_stats,
    temporal_stats_blocks,
    temporal_stats_record_len,
    temporal_stats_slot_len,
};

/// The two ratios bracketing the outlier check's calibrated boundary.
///
/// They are literals so the tests check the calibration rather than the constant against itself.
/// A static frame with a real noise spread up to fourfold survives and trimming starts at
/// fivefold, so a factor of 5.0 leaves room above real spread and catches texture. Changing
/// `SIGMA_OUTLIER_FACTOR` means updating both.
const OUTLIER_FACTOR_SURVIVES_RATIO: f32 = 4.99;
const OUTLIER_FACTOR_REJECTS_RATIO: f32 = 5.01;

/// One single-channel block record with the given scalar lanes and every quarter lane at 0.
fn scalar_record(sum_d: f32, sum_d2: f32, sum_lag: f32) -> Vec<f32> {
    let record_len = temporal_stats_record_len(1) as usize;
    let mut record = vec![0.0f32; record_len];
    record[0] = sum_d;
    record[1] = sum_d2;
    record[2] = sum_lag;
    record
}

/// One full block's record with a zero mean residual, so it always clears [STATIC_GATE].
fn zero_mean_block_record(sigma_255: f32, rho: f32) -> Vec<f32> {
    let pixel_count = 256.0f32;
    let n_pairs = 240.0f32;
    let sigma = sigma_255 / 255.0;
    let variance = 2.0 * sigma * sigma;
    let sum_d2 = pixel_count * variance;
    let sum_lag = n_pairs * rho * variance;
    scalar_record(0.0, sum_d2, sum_lag)
}

/// A 48x48 frame of nine blocks, eight identical and a ninth at `ninth_ratio` times their sigma.
///
/// The lower quartile of nine lands on the third smallest, which stays one of the eight identical
/// blocks while the ninth sorts above them.
fn outlier_factor_boundary_records(
    background_sigma_255: f32,
    background_rho: f32,
    ninth_ratio: f32,
) -> Vec<f32> {
    let mut records = Vec::new();
    for _ in 0..8 {
        records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
    }

    let ninth_sigma_255 = background_sigma_255 * ninth_ratio;
    records.extend_from_slice(&zero_mean_block_record(ninth_sigma_255, 0.0));
    records
}

#[test]
fn temporal_stats_blocks_and_slot_len() {
    assert_eq!(temporal_stats_blocks(32, 16), (2, 1));
    assert_eq!(temporal_stats_blocks(33, 17), (3, 2)); // ragged on both axes
    assert_eq!(temporal_stats_record_len(1), 39);
    assert_eq!(temporal_stats_record_len(4), 45);
    assert_eq!(temporal_stats_slot_len(32, 16, 1), 78); // 2 blocks x record_len 39
}

#[test]
fn aggregate_only_static_block_contributes_to_sigma_and_rho() {
    let width = 32;
    let height = 16;
    let stored_ch = 1;
    let channels = 1;
    let pixel_count = 256.0f32;
    let n_pairs = 240.0f32;

    let sigma_target = 4.0 / 255.0;
    let variance = 2.0 * sigma_target * sigma_target;
    let rho_target = 0.5f32;
    let sum_d2_static = pixel_count * variance;
    let sum_lag_static = n_pairs * rho_target * variance;

    let mean0_bad = 10.0 / 255.0;
    let sum_d_bad = pixel_count * mean0_bad;

    let static_block = scalar_record(0.0, sum_d2_static, sum_lag_static);
    let moving_block = scalar_record(sum_d_bad, 0.0, 0.0);
    let records = [static_block, moving_block].concat();

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("one of two blocks passes the static gate, above the 5% floor");

    assert!((sample.static_fraction - 0.5).abs() < 1e-6);
    assert!(
        (sample.sigma[0] - sigma_target).abs() < 1e-4,
        "sigma {} vs target {sigma_target}",
        sample.sigma[0]
    );
    assert!(
        (sample.rho - rho_target).abs() < 1e-4,
        "rho {} vs target {rho_target}",
        sample.rho
    );
}

/// Three letterbox-style zero blocks are over a quarter of eight, so an ungated lower quartile
/// would land on zero.
#[test]
fn aggregate_excludes_perfectly_static_blocks_from_the_population() {
    let width = 16 * 8;
    let height = 16;
    let stored_ch = 1;
    let channels = 1;
    let pixel_count = 256.0f32;
    let n_pairs = 240.0f32;
    let rho_target = 0.5f32;

    let sigmas_255 = [0.0f32, 0.0, 0.0, 2.0, 2.5, 3.0, 3.5, 4.0];

    let mut records = Vec::new();
    for sigma_255 in sigmas_255.iter() {
        let sigma = sigma_255 / 255.0;
        let variance = 2.0 * sigma * sigma;
        let sum_d2 = pixel_count * variance;
        let sum_lag = n_pairs * rho_target * variance;
        let record = scalar_record(0.0, sum_d2, sum_lag);
        records.extend(record);
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("five blocks clear the gate");

    assert!(
        (sample.static_fraction - 0.625).abs() < 1e-6,
        "expected 5 of 8 blocks counted, got {}",
        sample.static_fraction
    );
    assert!(
        (sample.sigma[0] - 3.0 / 255.0).abs() < 1e-4,
        "expected median sigma 3/255 over [2,2.5,3,3.5,4], got {}",
        sample.sigma[0]
    );
    assert!(
        (sample.sigma_low[0] - 2.5 / 255.0).abs() < 1e-4,
        "expected lower-quartile sigma 2.5/255, not a zero dragged down by the static blocks, \
         got {}",
        sample.sigma_low[0]
    );
}

#[test]
fn aggregate_median_over_multiple_static_blocks() {
    let width = 16 * 5;
    let height = 16;
    let stored_ch = 1;
    let channels = 1;
    let pixel_count = 256.0f32;
    let n_pairs = 240.0f32;

    // Block 0 sits below RHO_SIGMA_GATE, so it never enters the population.
    let sigmas_255 = [0.1f32, 2.0, 3.0, 4.0, 5.0];
    let rhos = [0.0f32, 0.1, 0.3, 0.5, 0.7];

    let mut records = Vec::new();
    for (sigma_255, rho) in sigmas_255.iter().zip(rhos.iter()) {
        let sigma = sigma_255 / 255.0;
        let variance = 2.0 * sigma * sigma;
        let sum_d2 = pixel_count * variance;
        let sum_lag = n_pairs * rho * variance;
        let record = scalar_record(0.0, sum_d2, sum_lag);
        records.extend(record);
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("all five blocks are static");

    assert!((sample.static_fraction - 0.8).abs() < 1e-6);
    assert!(
        (sample.sigma[0] - 3.5 / 255.0).abs() < 1e-4,
        "expected median sigma 3.5/255 (middle of [2,3,4,5]), got {}",
        sample.sigma[0]
    );
    assert!(
        (sample.sigma_low[0] - 2.75 / 255.0).abs() < 1e-4,
        "expected lower-quartile sigma 2.75/255 (index 0.25*3=0.75 of [2,3,4,5]), got {}",
        sample.sigma_low[0]
    );
    assert!(
        (sample.rho - 0.4).abs() < 1e-4,
        "expected median rho 0.4 over the four blocks clearing the rho gate, got {}",
        sample.rho
    );
}

/// The one static block carries valid noise, which isolates the static-fraction floor.
#[test]
fn aggregate_below_static_floor_returns_none() {
    let width = 16 * 5;
    let height = 16 * 5;
    let stored_ch = 1;
    let channels = 1;
    let pixel_count = 256.0f32;

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let mut records = vec![0.0f32; 25 * record_len];
    let mean0_bad = 10.0 / 255.0;
    for block in 1..25 {
        records[block * record_len] = pixel_count * mean0_bad;
    }

    let sigma = 4.0 / 255.0;
    let variance = 2.0 * sigma * sigma;
    records[1] = pixel_count * variance;
    records[2] = 240.0 * 0.5 * variance;

    assert!(
        aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
        "1 of 25 static blocks (4%) should fall back below the 5% floor"
    );
}

/// Every block passes the static check, so this reaches the no-measurable-noise path.
#[test]
fn aggregate_zeroed_slot_returns_none() {
    let width = 32;
    let height = 32;
    let stored_ch = 1;
    let channels = 1;
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let records = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];

    assert!(
        aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
        "a zero-filled duplicate slot's stats must fall back to Immerkær, not report sigma=0"
    );
}

/// YUV storage puts three channels in four lanes, and the padding lane must never affect the
/// result.
#[test]
fn aggregate_multi_channel_layout_reads_correct_offsets() {
    let width = 16;
    let height = 16;
    let stored_ch = 4;
    let channels = 3;
    let pixel_count = 256.0f32;
    let n_pairs = 240.0f32;

    let sigmas_255 = [2.0f32, 4.0, 6.0];
    let rho_target = 0.6f32;

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let mut record = vec![0.0f32; record_len];
    let mut var0 = 0.0f32;
    for (channel, sigma_255) in sigmas_255.iter().enumerate() {
        let sigma = sigma_255 / 255.0;
        let variance = 2.0 * sigma * sigma;
        record[stored_ch as usize + channel] = pixel_count * variance;
        if channel == 0 {
            var0 = variance;
        }
    }

    record[2 * stored_ch as usize] = n_pairs * rho_target * var0;

    let sample = aggregate_temporal_noise_stats(&record, channels, stored_ch, width, height)
        .expect("the single block is static with measurable channel-0 noise");

    for (channel, sigma_255) in sigmas_255.iter().enumerate() {
        let expected = sigma_255 / 255.0;
        assert!(
            (sample.sigma[channel] - expected).abs() < 1e-4,
            "channel {channel}: expected {expected}, got {}",
            sample.sigma[channel]
        );
    }

    assert!((sample.rho - rho_target).abs() < 1e-4);
}

/// Ten of sixteen blocks stand in for panning texture, with a zero mean, a tenfold sigma and a
/// white-noise rho, so a correlation check cannot catch them.
#[test]
fn aggregate_rejects_majority_zero_mean_texture_outliers() {
    let width = 64;
    let height = 64;
    let stored_ch = 1;
    let channels = 1;

    let background_sigma_255 = 2.0f32;
    let background_rho = 0.1f32;
    let texture_sigma_255 = 20.0f32;

    let mut records = Vec::new();
    for _ in 0..6 {
        records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
    }

    for _ in 0..10 {
        records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("the static minority clears STATIC_FRACTION_MIN on its own");

    let expected_sigma = background_sigma_255 / 255.0;
    assert!(
        (sample.sigma[0] - expected_sigma).abs() < 1e-4,
        "expected the outlier gate to isolate the real noise floor {expected_sigma}, got {}",
        sample.sigma[0]
    );
    assert!(
        (sample.static_fraction - 6.0 / 16.0).abs() < 1e-4,
        "expected only the 6 background blocks to survive both gates, got static_fraction={}",
        sample.static_fraction
    );
    assert!(
        (sample.rho - background_rho).abs() < 1e-4,
        "expected rho to come from the surviving background blocks only, got {}",
        sample.rho
    );
}

/// Every block is static, with a threefold real noise spread like a dark region beside a bright one.
#[test]
fn aggregate_keeps_genuinely_static_blocks_despite_spatial_sigma_spread() {
    let width = 64;
    let height = 64;
    let stored_ch = 1;
    let channels = 1;

    let low_sigma_255 = 2.0f32;
    let high_sigma_255 = 6.0f32; // 3x low_sigma_255, real spatial spread.
    let rho = 0.1f32;

    let mut records = Vec::new();
    for _ in 0..8 {
        records.extend_from_slice(&zero_mean_block_record(low_sigma_255, rho));
    }

    for _ in 0..8 {
        records.extend_from_slice(&zero_mean_block_record(high_sigma_255, rho));
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("every block is static");

    assert!(
        (sample.static_fraction - 1.0).abs() < 1e-6,
        "a real 3x spatial sigma spread must not trip the outlier gate, got static_fraction={}",
        sample.static_fraction
    );
}

#[test]
fn aggregate_outlier_factor_survives_just_under_threshold() {
    let width = 48;
    let height = 48;
    let stored_ch = 1;
    let channels = 1;
    let background_sigma_255 = 2.0f32;
    let background_rho = 0.1f32;

    let records = outlier_factor_boundary_records(
        background_sigma_255,
        background_rho,
        OUTLIER_FACTOR_SURVIVES_RATIO,
    );

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("all 9 blocks clear the static-fraction floor");

    assert!(
        (sample.static_fraction - 1.0).abs() < 1e-6,
        "a 9th block at {OUTLIER_FACTOR_SURVIVES_RATIO}x the reference must survive, \
         got static_fraction={}",
        sample.static_fraction
    );
}

#[test]
fn aggregate_outlier_factor_rejects_just_over_threshold() {
    let width = 48;
    let height = 48;
    let stored_ch = 1;
    let channels = 1;
    let background_sigma_255 = 2.0f32;
    let background_rho = 0.1f32;

    let records =
        outlier_factor_boundary_records(background_sigma_255, background_rho, OUTLIER_FACTOR_REJECTS_RATIO);

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("the 8 background blocks alone still clear the static-fraction floor");

    assert!(
        (sample.static_fraction - 8.0 / 9.0).abs() < 1e-6,
        "a 9th block at {OUTLIER_FACTOR_REJECTS_RATIO}x the reference must be rejected, \
         got static_fraction={}",
        sample.static_fraction
    );
}

/// 26 of 100 blocks are zero letterbox bars and 74 carry static noise across five close levels.
///
/// The spread stands in for real sampling variance, which lets the population pass the anchor
/// check. The 74 split as 15, 15, 15, 15 and 14, so the median lands exactly on the third level.
#[test]
fn aggregate_returns_correct_sigma_with_letterbox_zero_population() {
    let width = 160;
    let height = 160;
    let stored_ch = 1;
    let channels = 1;

    let background_rho = 0.2f32;
    let sigma_levels_255 = [3.8f32, 3.9, 4.0, 4.1, 4.2];

    let mut records = Vec::new();
    for _ in 0..26 {
        records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
    }

    for i in 0..74 {
        let sigma_255 = sigma_levels_255[i % sigma_levels_255.len()];
        records.extend_from_slice(&zero_mean_block_record(sigma_255, background_rho));
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("74 of 100 blocks carry real static noise, far above the 5% floor");

    let expected_sigma = 4.0 / 255.0;
    assert!(
        (sample.sigma[0] - expected_sigma).abs() < 1e-4,
        "expected the letterbox bars to leave the real noise floor near {expected_sigma} intact, got {}",
        sample.sigma[0]
    );
    assert!(
        (sample.rho - background_rho).abs() < 1e-4,
        "expected rho to come from the real-noise blocks, got {}",
        sample.rho
    );
}

/// 26 zero letterbox blocks beside 74 texture blocks at one repeated sigma, which set and pass
/// their own ceiling.
#[test]
fn aggregate_returns_none_when_the_only_above_gate_population_is_texture() {
    let width = 160;
    let height = 160;
    let stored_ch = 1;
    let channels = 1;

    let texture_sigma_255 = 20.0f32;

    let mut records = Vec::new();
    for _ in 0..26 {
        records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
    }

    for _ in 0..74 {
        records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
    }

    assert!(
        aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
        "a homogeneous above-gate population with no genuine low anchor must fall back to \
         None rather than report the texture level as sigma"
    );
}

#[test]
fn aggregate_rejects_texture_outliers_with_zero_population_present() {
    let width = 64;
    let height = 80; // 4 x 5 TEMPORAL_NOISE_BLOCK grid, 20 blocks.
    let stored_ch = 1;
    let channels = 1;

    let background_sigma_255 = 2.0f32;
    let background_rho = 0.1f32;
    let texture_sigma_255 = 20.0f32;

    let mut records = Vec::new();
    for _ in 0..6 {
        records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
    }

    for _ in 0..10 {
        records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
    }

    for _ in 0..4 {
        records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("the static minority clears STATIC_FRACTION_MIN on its own");

    let expected_sigma = background_sigma_255 / 255.0;
    assert!(
        (sample.sigma[0] - expected_sigma).abs() < 1e-4,
        "expected the outlier gate to isolate the real noise floor {expected_sigma} despite \
         the zero population, got {}",
        sample.sigma[0]
    );
    assert!(
        (sample.static_fraction - 6.0 / 20.0).abs() < 1e-4,
        "expected only the 6 background blocks to survive, the zero blocks carry no \
         measurable noise, got static_fraction={}",
        sample.static_fraction
    );
    assert!(
        (sample.rho - background_rho).abs() < 1e-4,
        "expected rho to come from the surviving background blocks only, got {}",
        sample.rho
    );
}

#[test]
fn aggregate_keeps_static_spread_with_zero_population_present() {
    let width = 64;
    let height = 80; // 4 x 5 TEMPORAL_NOISE_BLOCK grid, 20 blocks.
    let stored_ch = 1;
    let channels = 1;

    let low_sigma_255 = 2.0f32;
    let high_sigma_255 = 6.0f32; // 3x low_sigma_255, real spatial spread.
    let rho = 0.1f32;

    let mut records = Vec::new();
    for _ in 0..8 {
        records.extend_from_slice(&zero_mean_block_record(low_sigma_255, rho));
    }

    for _ in 0..8 {
        records.extend_from_slice(&zero_mean_block_record(high_sigma_255, rho));
    }

    for _ in 0..4 {
        records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
    }

    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("every non-zero block is static");

    assert!(
        (sample.static_fraction - 0.8).abs() < 1e-6,
        "a real 3x spatial sigma spread plus a zero population must not trip the outlier \
         gate, and the 4 zero blocks carry no measurable noise, got static_fraction={}",
        sample.static_fraction
    );
}

/// A block's rho can read above 1, because the lag-1 total averages over adjacent pairs while the
/// variance averages over every pixel. Each row here follows the pattern that maximises that ratio.
#[test]
fn aggregate_rho_estimate_stays_within_unit_range() {
    let width = TEMPORAL_NOISE_BLOCK;
    let height = TEMPORAL_NOISE_BLOCK;
    let stored_ch = 1;
    let channels = 1;

    let scale = 0.007f32;
    let row: Vec<f32> = (1..=width)
        .map(|i| scale * (i as f32 * std::f32::consts::PI / (width + 1) as f32).sin())
        .collect();

    let pixel_count = (width * height) as f32;
    let sum_d: f32 = row.iter().sum::<f32>() * height as f32;
    let sum_d2: f32 = row.iter().map(|value| value * value).sum::<f32>() * height as f32;
    let sum_lag: f32 = row.windows(2).map(|pair| pair[0] * pair[1]).sum::<f32>() * height as f32;

    // Both gates must pass so the assertion below tests the clamp rather than a gate miss.
    assert!(
        (sum_d / pixel_count).abs() < STATIC_GATE,
        "construction must clear the static gate"
    );

    let variance = sum_d2 / pixel_count - (sum_d / pixel_count) * (sum_d / pixel_count);
    assert!(
        (variance.sqrt() / std::f32::consts::SQRT_2) > RHO_SIGMA_GATE,
        "construction must clear the rho-sample sigma gate"
    );

    let records = scalar_record(sum_d, sum_d2, sum_lag);
    let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
        .expect("the single block clears both gates");

    assert!(
        (0.0..=1.0).contains(&sample.rho),
        "the mismatched-denominator estimate must stay within [0, 1], got {}",
        sample.rho
    );
}
