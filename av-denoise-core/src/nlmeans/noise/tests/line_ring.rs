use crate::nlmeans::noise::line_ring::{LineInput, dilate, line_quarters};
use crate::nlmeans::noise::{NOISE_CURVE_BINS, NoiseCurve};

const SIGMA: f32 = 0.02;

/// A flat curve predicting [SIGMA] at every luma.
fn flat_curve() -> NoiseCurve {
    NoiseCurve {
        ratios: [1.0; NOISE_CURVE_BINS],
        sigma_quarter_median: SIGMA,
    }
}

/// A quarter whose RMS gradient is `rms_gradient` and whose luma range is `range_codes` 8-bit codes.
fn quarter(rms_gradient: f32, range_codes: f32) -> Option<LineInput> {
    let input = LineInput {
        luma: 0.3,
        luma_range: range_codes / 255.0,
        tensor_trace: rms_gradient * rms_gradient * 49.0,
    };
    Some(input)
}

fn is_line(input: Option<LineInput>) -> bool {
    let curve = flat_curve();
    let lines = line_quarters(&[input], &curve);
    lines[0]
}

#[test]
fn a_strong_contrasty_quarter_is_a_line() {
    assert!(is_line(quarter(5.0 * SIGMA, 64.0)));
}

#[test]
fn the_gradient_cut_sits_at_three_sigma() {
    assert!(is_line(quarter(3.05 * SIGMA, 64.0)));
    assert!(!is_line(quarter(2.95 * SIGMA, 64.0)));
}

#[test]
fn the_contrast_floor_sits_at_32_codes() {
    assert!(is_line(quarter(5.0 * SIGMA, 32.5)));
    assert!(!is_line(quarter(5.0 * SIGMA, 31.5)));
}

#[test]
fn a_partial_quarter_is_never_a_line() {
    assert!(!is_line(None));
}

#[test]
fn a_zero_radius_keeps_only_the_marked_quarters() {
    let mut marked = vec![false; 5 * 5];
    marked[2 * 5 + 2] = true;

    let grown = dilate(&marked, 5, 5, 0);

    assert_eq!(grown, marked);
}

#[test]
fn a_radius_of_one_grows_a_three_by_three_square() {
    let mut marked = vec![false; 5 * 5];
    marked[2 * 5 + 2] = true;

    let grown = dilate(&marked, 5, 5, 1);

    for row in 0..5 {
        for col in 0..5 {
            let inside = (1..=3).contains(&row) && (1..=3).contains(&col);
            assert_eq!(grown[row * 5 + col], inside, "row {row} col {col}");
        }
    }
}

#[test]
fn a_ring_at_a_corner_is_clamped_to_the_grid() {
    let mut marked = vec![false; 4 * 3];
    marked[0] = true;

    let grown = dilate(&marked, 4, 3, 2);

    for row in 0..3 {
        for col in 0..4 {
            let inside = row <= 2 && col <= 2;
            assert_eq!(grown[row * 4 + col], inside, "row {row} col {col}");
        }
    }
}

#[test]
fn a_radius_past_the_grid_fills_it() {
    let mut marked = vec![false; 3 * 2];
    marked[4] = true;

    let grown = dilate(&marked, 3, 2, 8);

    assert!(grown.iter().all(|&inside| inside));
}

#[test]
fn nothing_marked_grows_nothing() {
    let marked = vec![false; 4 * 4];

    let grown = dilate(&marked, 4, 4, 2);

    assert!(grown.iter().all(|&inside| !inside));
}
