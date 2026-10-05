use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, LAGS};
use crate::nl4d::grain::gaussian::GAUSSIAN_SEQUENCE;
use crate::nl4d::grain::template::{Av1Random, luma_template, template_seed, template_stats};

const SCENE_EIGHT: [i32; AR_COEFFS] = [
    0, 0, -4, 6, 1, -1, 0, -1, -1, 4, -19, -7, 0, -2, -3, 3, -21, 61, 15, 1, -2, 5, -22, 71,
];

/// The sum and sum of squares of the samples.
fn sums(samples: &[i32]) -> (i64, i64) {
    let sum = samples.iter().map(|&sample| sample as i64).sum();
    let sum_sq = samples
        .iter()
        .map(|&sample| (sample as i64) * (sample as i64))
        .sum();

    (sum, sum_sq)
}

#[test]
fn gaussian_sequence_matches_the_specification() {
    let samples: Vec<i32> = GAUSSIAN_SEQUENCE.iter().map(|&sample| sample as i32).collect();
    let (sum, sum_sq) = sums(&samples);

    assert_eq!(GAUSSIAN_SEQUENCE.len(), 2048);
    assert_eq!(sum, 1120);
    assert_eq!(sum_sq, 535_876_128);
    assert_eq!(GAUSSIAN_SEQUENCE[..4], [56, 568, -180, 172]);
    assert_eq!(GAUSSIAN_SEQUENCE[2044..], [288, 944, 428, -484]);
}

#[test]
fn lag_and_offset_tables_have_the_expected_order() {
    assert_eq!(LAGS[0], (0, 0));
    assert_eq!(LAGS[6], (0, 6));
    assert_eq!(LAGS[7], (1, -6));
    assert_eq!(LAGS[45], (3, 6));
    assert_eq!(AR_OFFSETS[0], (-3, -3));
    assert_eq!(AR_OFFSETS[17], (-1, 0));
    assert_eq!(AR_OFFSETS[23], (0, -1));
}

#[test]
fn lfsr_matches_the_specification() {
    let mut random = Av1Random::new(1000);
    let draws: Vec<u32> = (0..8).map(|_| random.next(11)).collect();

    assert_eq!(draws, [1039, 519, 259, 129, 1088, 1568, 1808, 904]);
}

#[test]
fn template_matches_the_reference_port() {
    let template = luma_template(&SCENE_EIGHT, 7, 1000);
    let (sum, sum_sq) = sums(&template);
    let row = |y: usize, x: usize| template[y * 82 + x..y * 82 + x + 8].to_vec();
    let top_row = row(3, 0);
    let middle_row = row(40, 30);
    let bottom_row = row(72, 70);

    assert_eq!(template.len(), 73 * 82);
    assert_eq!(sum, -2598);
    assert_eq!(sum_sq, 11_079_620);
    assert_eq!(top_row, [-29, 27, -56, -20, 7, 58, 21, 47]);
    assert_eq!(middle_row, [-15, -67, -39, -9, 16, 45, 48, -1]);
    assert_eq!(bottom_row, [10, 4, 44, 60, -48, -10, 7, 37]);
}

#[test]
fn zero_weights_give_the_raw_gaussian_template() {
    let template = luma_template(&[0; AR_COEFFS], 6, 1000);
    let (sum, sum_sq) = sums(&template);

    assert_eq!(sum, 398);
    assert_eq!(sum_sq, 6_119_742);
    assert_eq!(template[..8], [-22, 17, 50, -18, -18, -44, 27, 6]);
}

#[test]
fn template_stats_match_the_reference_port() {
    let (sigma, median) = template_stats(&SCENE_EIGHT, 7);
    let (flat_sigma, flat_median) = template_stats(&[0; AR_COEFFS], 6);

    assert!((sigma - 44.770_018).abs() < 1e-4, "sigma {sigma}");
    assert!((median - 43.094_649).abs() < 1e-4, "median {median}");
    assert!((flat_sigma - 31.909_499).abs() < 1e-4, "flat sigma {flat_sigma}");
    assert!(
        (flat_median - 31.788_979).abs() < 1e-4,
        "flat median {flat_median}"
    );
}

#[test]
fn template_seeds_stay_sixteen_bit() {
    let first_seed = template_seed(0);
    let later_seed = template_seed(23);

    assert_eq!(first_seed, 1000);
    assert_eq!(later_seed, ((1000u32 + 7919 * 23) & 0xFFFF) as u16);
}
