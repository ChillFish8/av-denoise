use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::repeat::{SeedScores, block_offsets, fft, kept_lags, repeat_score};
use crate::nl4d::grain::seeds::FGSE_SEEDS;

const NINETY_ONE_DAYS: [i32; AR_COEFFS] = [
    2, -7, 2, 15, 11, -4, 2, -10, 14, -5, -4, -12, 14, -6, 3, -2, 2, 81, 35, 5, 8, 6, -13, 88,
];

#[test]
fn block_offsets_match_the_decoder() {
    let first_row = block_offsets(7391, 0, 4);
    let second_row = block_offsets(7391, 1, 4);

    assert_eq!(first_row, [(35, 23), (21, 31), (15, 19), (11, 29)]);
    assert_eq!(second_row, [(21, 19), (15, 13), (11, 27), (25, 33)]);
}

#[test]
fn fft_of_an_impulse_is_flat() {
    let mut real = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut imaginary = [0.0; 8];

    fft(&mut real, &mut imaginary, false);

    for (real_value, imaginary_value) in real.iter().zip(imaginary) {
        assert!((real_value - 1.0).abs() < 1e-9);
        assert!(imaginary_value.abs() < 1e-9);
    }
}

#[test]
fn fft_of_a_shifted_impulse_uses_the_negative_exponent() {
    let mut real = [0.0, 1.0, 0.0, 0.0];
    let mut imaginary = [0.0; 4];

    fft(&mut real, &mut imaginary, false);

    let expected_real = [1.0, 0.0, -1.0, 0.0];
    let expected_imaginary = [0.0, -1.0, 0.0, 1.0];
    for index in 0..4 {
        assert!((real[index] - expected_real[index]).abs() < 1e-9);
        assert!((imaginary[index] - expected_imaginary[index]).abs() < 1e-9);
    }
}

#[test]
fn fft_then_inverse_returns_the_input() {
    let input: Vec<f64> = (0..16).map(|index| ((index * 7 + 3) % 11) as f64 - 4.5).collect();
    let mut real = input.clone();
    let mut imaginary = vec![0.0; 16];

    fft(&mut real, &mut imaginary, false);
    fft(&mut real, &mut imaginary, true);

    for (restored, original) in real.iter().zip(&input) {
        assert!((restored - original).abs() < 1e-9);
    }
    for value in &imaginary {
        assert!(value.abs() < 1e-9);
    }
}

#[test]
fn a_repeating_seed_scores_above_a_calm_one() {
    let kept = kept_lags();
    let calm = repeat_score(&NINETY_ONE_DAYS, 8, 24239, &kept);
    let repeating = repeat_score(&NINETY_ONE_DAYS, 8, 14889, &kept);

    assert!((calm - 0.0741).abs() < 0.005, "got {calm}");
    assert!((repeating - 0.229).abs() < 0.01, "got {repeating}");
}

#[test]
fn the_calmest_seed_is_a_listed_seed_with_a_low_score() {
    let mut scores = SeedScores::new();

    let seed = scores
        .calmest(&NINETY_ONE_DAYS, 8, &[])
        .expect("the list is not exhausted");

    assert!([64774, 60537].contains(&seed), "got {seed}");
}

#[test]
fn taken_seeds_are_skipped() {
    let mut scores = SeedScores::new();

    let first = scores.calmest(&NINETY_ONE_DAYS, 8, &[]).expect("a seed");
    let second = scores.calmest(&NINETY_ONE_DAYS, 8, &[first]).expect("a seed");
    let again = scores.calmest(&NINETY_ONE_DAYS, 8, &[]).expect("a seed");

    assert_ne!(first, second);
    assert_eq!(first, again);
    assert!(FGSE_SEEDS.contains(&second));
}

#[test]
fn an_exhausted_list_gives_no_seed() {
    let mut scores = SeedScores::new();

    assert_eq!(scores.calmest(&NINETY_ONE_DAYS, 8, &FGSE_SEEDS), None);
}
