use super::{Aggregated, Setup, cross_frame_setup, run_fused};
use crate::collab::tests::helpers::noisy_flat_field;
use crate::nlmeans::NOISE_CURVE_BINS;

/// The side of the square frame the stepped-curve test filters.
const STEP_FRAME_SIDE: u32 = 64;

/// Columns either side of the brightness seam left out of each half's measurement.
const SEAM_MARGIN: u32 = 8;

/// A threshold low enough that scaling it still changes which noise coefficients survive.
const CURVE_LAMBDA: f32 = 1.0;

/// A curve at `2.0` over the darker half of the luma range and `0.5` over the brighter half.
pub(super) fn stepped_curve() -> [f32; NOISE_CURVE_BINS] {
    let mut curve = [2.0f32; NOISE_CURVE_BINS];
    curve[NOISE_CURVE_BINS / 2..].fill(0.5);

    curve
}

pub(super) fn assert_identical(label: &str, got: &Aggregated, want: &Aggregated) {
    assert_eq!(
        got.group_weight, want.group_weight,
        "{label}: group weights differ"
    );
    assert_eq!(got.accum, want.accum, "{label}: accumulated values differ");
    assert_eq!(got.wsum, want.wsum, "{label}: weight sums differ");
    assert!(
        want.group_weight.iter().any(|weight| *weight > 0.0),
        "{label}: nothing aggregated, so agreeing proves nothing",
    );
}

fn run_with_curve(curve: Option<[f32; NOISE_CURVE_BINS]>) -> Aggregated {
    let mut setup = cross_frame_setup(64, 64, 2);
    setup.noise_curve = curve;
    run_fused(&setup)
}

/// Asserts every pixel in columns `x_start..x_end` of a single-frame run matches exactly.
pub(super) fn assert_columns_identical(
    label: &str,
    got: &Aggregated,
    want: &Aggregated,
    side: u32,
    x_start: u32,
    x_end: u32,
) {
    for y in 0..side {
        for x in x_start..x_end {
            let idx = (y * side + x) as usize;
            assert_eq!(
                got.accum[idx], want.accum[idx],
                "{label}: accumulated value differs at ({x}, {y})",
            );
            assert_eq!(
                got.wsum[idx], want.wsum[idx],
                "{label}: weight sum differs at ({x}, {y})",
            );
        }
    }
}

pub(super) fn columns_differ(
    first: &Aggregated,
    second: &Aggregated,
    side: u32,
    x_start: u32,
    x_end: u32,
) -> bool {
    for y in 0..side {
        for x in x_start..x_end {
            let idx = (y * side + x) as usize;
            if first.accum[idx] != second.accum[idx] {
                return true;
            }
        }
    }

    false
}

fn run_spatial_with_lambda(frame: &[f32], side: u32, lambda_ht: f32) -> Aggregated {
    let mut setup = Setup::spatial_only(frame.to_vec(), side, side);
    setup.lambda_ht = lambda_ht;
    run_fused(&setup)
}

#[test]
fn a_flat_unit_curve_changes_nothing() {
    let plain = run_with_curve(None);
    let unit = run_with_curve(Some([1.0; NOISE_CURVE_BINS]));

    assert_identical("unit curve", &unit, &plain);
}

#[test]
fn a_flat_curve_equals_scaling_lambda() {
    let flat = run_with_curve(Some([1.5; NOISE_CURVE_BINS]));

    let mut scaled = cross_frame_setup(64, 64, 2);
    scaled.lambda_ht *= 1.5;
    let want = run_fused(&scaled);

    assert_identical("flat 1.5 curve", &flat, &want);
}

#[test]
fn a_stepped_curve_thresholds_each_brightness_by_its_own_noise() {
    let side = STEP_FRAME_SIDE;
    let half = side / 2;
    let noise = noisy_flat_field(side, side, 0.0, 0.02);
    let mut frame = Vec::with_capacity(noise.len());
    for (idx, sample) in noise.iter().enumerate() {
        let x = idx as u32 % side;
        let base = if x < half { 0.2 } else { 0.8 };
        frame.push(base + sample);
    }

    let mut curved_setup = Setup::spatial_only(frame.clone(), side, side);
    curved_setup.lambda_ht = CURVE_LAMBDA;
    let curve = stepped_curve();
    curved_setup.noise_curve = Some(curve);
    let curved = run_fused(&curved_setup);

    let plain = run_spatial_with_lambda(&frame, side, CURVE_LAMBDA);
    let dark_scaled = run_spatial_with_lambda(&frame, side, CURVE_LAMBDA * 2.0);
    let bright_scaled = run_spatial_with_lambda(&frame, side, CURVE_LAMBDA * 0.5);

    let left_end = half - SEAM_MARGIN;
    let right_start = half + SEAM_MARGIN;
    assert_columns_identical("dark half", &curved, &dark_scaled, side, 0, left_end);
    assert_columns_identical("bright half", &curved, &bright_scaled, side, right_start, side);

    let dark_changed = columns_differ(&dark_scaled, &plain, side, 0, left_end);
    let bright_changed = columns_differ(&bright_scaled, &plain, side, right_start, side);
    assert!(dark_changed, "doubling lambda changed nothing in the dark half");
    assert!(
        bright_changed,
        "halving lambda changed nothing in the bright half"
    );
}

#[test]
fn the_curve_is_sampled_at_bin_centres() {
    let side = STEP_FRAME_SIDE;
    let frame = noisy_flat_field(side, side, 0.25, 0.02);

    let mut curve = [0.33f32; NOISE_CURVE_BINS];
    curve[3] = 2.0;
    curve[4] = 2.0;

    let mut curved_setup = Setup::spatial_only(frame.clone(), side, side);
    curved_setup.lambda_ht = CURVE_LAMBDA;
    curved_setup.noise_curve = Some(curve);
    let curved = run_fused(&curved_setup);

    let plain = run_spatial_with_lambda(&frame, side, CURVE_LAMBDA);
    let scaled = run_spatial_with_lambda(&frame, side, CURVE_LAMBDA * 2.0);
    assert_ne!(scaled.accum, plain.accum, "doubling lambda changed nothing");

    assert_identical("bin centre", &curved, &scaled);
}

#[test]
fn the_curve_is_clamped() {
    let above = run_with_curve(Some([10.0; NOISE_CURVE_BINS]));
    let ceiling = run_with_curve(Some([3.0; NOISE_CURVE_BINS]));
    assert_identical("above the ceiling", &above, &ceiling);

    let below = run_with_curve(Some([0.01; NOISE_CURVE_BINS]));
    let floor = run_with_curve(Some([0.33; NOISE_CURVE_BINS]));
    assert_identical("below the floor", &below, &floor);

    let plain = run_with_curve(None);
    assert_ne!(ceiling.accum, plain.accum, "the ceiling curve changed nothing");
    assert_ne!(floor.accum, plain.accum, "the floor curve changed nothing");
}
