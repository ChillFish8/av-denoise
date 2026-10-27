use crate::noise::correlation::correlation_factor;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::estimator::EMA_ALPHA;
use crate::noise::host::HostNoise;
use crate::noise::strength_map::QuarterClasses;
use crate::noise::temporal::{TemporalNoiseReading, TemporalNoiseSample};

fn sample_with_rho(rho: f32) -> TemporalNoiseSample {
    TemporalNoiseSample {
        sigma: [0.02, 0.0, 0.0],
        sigma_low: [0.015, 0.0, 0.0],
        rho,
        static_fraction: 0.5,
    }
}

fn curve_of(ratio: f32) -> NoiseCurve {
    NoiseCurve {
        ratios: [ratio; NOISE_CURVE_BINS],
        sigma_quarter_median: 0.01,
    }
}

fn lender_with_curve(ratio: f32) -> TemporalNoiseReading {
    let sample = sample_with_rho(0.5);
    let classes = QuarterClasses::from_classes(1, 1, vec![None]);

    TemporalNoiseReading {
        sample: Some(sample),
        curve: Some(curve_of(ratio)),
        classes: Some(classes),
    }
}

#[test]
fn a_fold_with_a_temporal_sample_takes_the_larger_boosted_sigma() {
    let mut host = HostNoise::default();
    let sample = sample_with_rho(0.5);
    let fold = host.fold([0.01, 0.0, 0.0], Some(sample), [0.008, 0.0, 0.0], 1, 1.0);
    let factor = correlation_factor(0.5);
    let temporal_only = host.sigmas_temporal_only(1);
    let low_unboosted = host.sigmas_low_unboosted(1);

    assert_eq!(fold.median[0], 0.02 * factor);
    assert_eq!(fold.low[0], 0.015 * factor);
    assert_eq!(temporal_only[0], 0.02);
    assert_eq!(low_unboosted[0], 0.015);
}

#[test]
fn a_fold_without_a_sample_keeps_the_immerkaer_sigmas() {
    let mut host = HostNoise::default();
    let fold = host.fold([0.01, 0.0, 0.0], None, [0.008, 0.0, 0.0], 1, 1.0);
    let sigmas = host.sigmas(1);

    assert_eq!(fold.median[0], 0.01);
    assert_eq!(fold.low[0], 0.008);
    assert_eq!(sigmas, [0.01, 0.0, 0.0]);
    assert_eq!(host.rho, None);
}

#[test]
fn the_first_temporal_sample_seeds_rho_directly() {
    let mut host = HostNoise::default();
    let first = sample_with_rho(0.5);
    host.fold([0.01, 0.0, 0.0], Some(first), [0.008, 0.0, 0.0], 1, 1.0);
    let seeded = host.rho;

    let second = sample_with_rho(0.3);
    host.fold([0.01, 0.0, 0.0], Some(second), [0.008, 0.0, 0.0], 1, 1.0);
    let blended = host.rho;
    let expected = EMA_ALPHA * 0.3 + (1.0 - EMA_ALPHA) * 0.5;

    assert_eq!(seeded, Some(0.5));
    assert_eq!(blended, Some(expected));
}

#[test]
fn the_temporal_chain_only_updates_with_a_sample() {
    let mut host = HostNoise::default();
    host.fold([0.01, 0.0, 0.0], None, [0.008, 0.0, 0.0], 1, 1.0);
    let before = host.temporal_only.current().is_some();

    let sample = sample_with_rho(0.5);
    host.fold([0.01, 0.0, 0.0], Some(sample), [0.008, 0.0, 0.0], 1, 1.0);
    let seeded = host.sigmas_temporal_only(1);

    host.fold([0.05, 0.0, 0.0], None, [0.04, 0.0, 0.0], 1, 1.0);
    let after = host.sigmas_temporal_only(1);

    assert!(!before);
    assert_eq!(seeded[0], 0.02);
    assert_eq!(after, seeded);
}

#[test]
fn the_temporal_reading_falls_back_to_the_low_unboosted_chain() {
    let mut host = HostNoise::default();
    host.fold([0.01, 0.0, 0.0], None, [0.008, 0.0, 0.0], 1, 1.0);
    let temporal_only = host.sigmas_temporal_only(1);
    let low_unboosted = host.sigmas_low_unboosted(1);

    assert_eq!(temporal_only, [0.008, 0.0, 0.0]);
    assert_eq!(temporal_only, low_unboosted);
}

#[test]
fn sigma_scale_applies_before_smoothing() {
    let mut host = HostNoise::default();
    let first = host.fold([0.01, 0.0, 0.0], None, [0.008, 0.0, 0.0], 1, 2.0);

    let sample = TemporalNoiseSample {
        sigma: [0.001, 0.0, 0.0],
        sigma_low: [0.001, 0.0, 0.0],
        rho: 0.0,
        static_fraction: 0.5,
    };
    let second = host.fold([0.03, 0.0, 0.0], Some(sample), [0.02, 0.0, 0.0], 1, 2.0);
    let temporal_only = host.sigmas_temporal_only(1);

    let expected_median = EMA_ALPHA * 0.06 + (1.0 - EMA_ALPHA) * 0.02;
    let expected_low = EMA_ALPHA * 0.04 + (1.0 - EMA_ALPHA) * 0.016;

    assert_eq!(first.median[0], 0.02);
    assert_eq!(first.low[0], 0.016);
    assert_eq!(second.median[0], expected_median);
    assert_eq!(second.low[0], expected_low);
    assert_eq!(temporal_only[0], 0.002);
}

#[test]
fn a_fold_only_reads_the_active_channels() {
    let mut host = HostNoise::default();
    let fold = host.fold([0.01, 0.02, 0.03], None, [0.008, 0.009, 0.01], 2, 1.0);
    let sigmas = host.sigmas(2);

    assert_eq!(fold.median, [0.01, 0.02, 0.0]);
    assert_eq!(fold.low, [0.008, 0.009, 0.0]);
    assert_eq!(sigmas, [0.01, 0.02, 0.0]);
}

#[test]
fn adopting_a_lender_without_a_sample_keeps_the_curve() {
    let mut host = HostNoise::default();
    let lender = lender_with_curve(1.5);
    host.adopt(&lender);

    let empty = TemporalNoiseReading::empty();
    host.adopt(&empty);

    let unsampled = TemporalNoiseReading {
        sample: None,
        curve: Some(curve_of(2.0)),
        classes: None,
    };
    host.adopt(&unsampled);

    let expected_curve = curve_of(1.5);
    let classes_kept = host.classes.is_some();

    assert_eq!(host.curve, Some(expected_curve));
    assert!(classes_kept);
}

#[test]
fn adopting_a_lender_with_a_sample_takes_its_curve_and_classes() {
    let mut host = HostNoise::default();
    let first = lender_with_curve(1.5);
    host.adopt(&first);

    let sample = sample_with_rho(0.5);
    let curveless = TemporalNoiseReading {
        sample: Some(sample),
        curve: None,
        classes: None,
    };
    host.adopt(&curveless);

    let classes_cleared = host.classes.is_none();

    assert_eq!(host.curve, None);
    assert!(classes_cleared);
}

#[test]
fn a_reset_clears_every_chain_and_the_curve() {
    let mut host = HostNoise::default();
    let sample = sample_with_rho(0.5);
    host.fold([0.01, 0.0, 0.0], Some(sample), [0.008, 0.0, 0.0], 1, 1.0);
    let lender = lender_with_curve(1.5);
    host.adopt(&lender);

    host.reset();

    let median_cleared = host.median.current().is_none();
    let low_cleared = host.low.current().is_none();
    let low_unboosted_cleared = host.low_unboosted.current().is_none();
    let temporal_only_cleared = host.temporal_only.current().is_none();
    let classes_cleared = host.classes.is_none();
    let sigmas = host.sigmas_temporal_only(1);

    assert!(median_cleared);
    assert!(low_cleared);
    assert!(low_unboosted_cleared);
    assert!(temporal_only_cleared);
    assert!(classes_cleared);
    assert_eq!(host.rho, None);
    assert_eq!(host.curve, None);
    assert_eq!(sigmas, [0.0; 3]);
}
