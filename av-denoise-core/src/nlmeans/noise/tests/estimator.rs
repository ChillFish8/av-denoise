use crate::nlmeans::noise::estimator::{NoiseEstimator, SIGMA_FLOOR};

#[test]
fn first_sample_initializes_state() {
    let mut estimator = NoiseEstimator::default();
    let smoothed = estimator.update(&[0.05, 0.02]);
    assert_eq!(smoothed, &[0.05, 0.02]);
}

#[test]
fn update_converges_toward_changed_level() {
    let mut estimator = NoiseEstimator::default();
    estimator.update(&[0.02]);

    let mut last = 0.0;
    for _ in 0..200 {
        last = estimator.update(&[0.10])[0];
    }

    assert!(
        (last - 0.10).abs() < 1e-4,
        "expected convergence close to 0.10, got {last}"
    );
}

#[test]
fn update_floors_near_zero_samples() {
    let mut estimator = NoiseEstimator::default();
    let smoothed = estimator.update(&[0.0]);
    assert_eq!(smoothed[0], SIGMA_FLOOR);

    let smoothed = estimator.update(&[0.0]);
    assert_eq!(smoothed[0], SIGMA_FLOOR);
}

#[test]
fn reset_clears_state() {
    let mut estimator = NoiseEstimator::default();
    estimator.update(&[0.10]);
    estimator.reset();

    let smoothed = estimator.update(&[0.02]);
    assert_eq!(smoothed, &[0.02]);
}
