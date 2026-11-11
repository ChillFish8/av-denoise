use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, CELL, LAGS, TEMPLATE_SEEDS};
use crate::nl4d::grain::gaussian::GAUSSIAN_SEQUENCE;
use crate::nl4d::grain::template::{
    Av1Random,
    ChromaPlane,
    chroma_template,
    chroma_template_stats,
    luma_template,
    median_of,
    sample_std,
    template_seed,
    template_stats,
};

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

fn raw_chroma_draws(seed: u16, count: usize) -> Vec<i32> {
    let mut random = Av1Random::new(seed);
    let mut draws = Vec::with_capacity(count);
    for _ in 0..count {
        let index = random.next(11) as usize;
        let gaussian = GAUSSIAN_SEQUENCE[index] as i32;
        draws.push((gaussian + 8) >> 4);
    }

    draws
}

#[test]
fn zero_weight_chroma_templates_are_the_raw_gaussian_draws() {
    for (plane, mask) in [(ChromaPlane::Cb, 0xb524), (ChromaPlane::Cr, 0x49d8)] {
        let template = chroma_template(&[0; AR_COEFFS], 6, 1000, plane);
        let expected = raw_chroma_draws(1000 ^ mask, 38 * 44);

        assert_eq!(plane.seed_mask(), mask);
        assert_eq!(template, expected);
    }
}

#[test]
fn chroma_templates_have_the_420_size() {
    let cb_template = chroma_template(&SCENE_EIGHT, 7, 1000, ChromaPlane::Cb);
    let cr_template = chroma_template(&SCENE_EIGHT, 7, 1000, ChromaPlane::Cr);

    assert_eq!(cb_template.len(), 38 * 44);
    assert_eq!(cr_template.len(), 38 * 44);
}

#[test]
fn one_chroma_weight_filters_from_the_left() {
    let mut coeffs = [0; AR_COEFFS];
    coeffs[23] = 64;
    let template = chroma_template(&coeffs, 6, 1000, ChromaPlane::Cb);
    let raw = raw_chroma_draws(1000 ^ 0xb524, 38 * 44);
    let at = |values: &[i32], y: usize, x: usize| values[y * 44 + x];

    let first_total = 64 * at(&raw, 3, 2);
    let first_rounded = (first_total + 32) >> 6;
    let first_sum = at(&raw, 3, 3) + first_rounded;
    let first = first_sum.clamp(-128, 127);
    let second_total = 64 * first;
    let second_rounded = (second_total + 32) >> 6;
    let second_sum = at(&raw, 3, 4) + second_rounded;
    let second = second_sum.clamp(-128, 127);

    assert_eq!(at(&template, 3, 2), at(&raw, 3, 2));
    assert_eq!(at(&template, 3, 3), first);
    assert_eq!(at(&template, 3, 4), second);
    assert_eq!(at(&template, 3, 41), at(&raw, 3, 41));
}

#[test]
fn cb_and_cr_templates_differ() {
    let cb_template = chroma_template(&SCENE_EIGHT, 7, 1000, ChromaPlane::Cb);
    let cr_template = chroma_template(&SCENE_EIGHT, 7, 1000, ChromaPlane::Cr);

    assert_ne!(cb_template, cr_template);
}

#[test]
fn chroma_template_stats_grow_with_correlation() {
    let (sigma, median) = chroma_template_stats(&SCENE_EIGHT, 7, ChromaPlane::Cb);
    let (flat_sigma, flat_median) = chroma_template_stats(&[0; AR_COEFFS], 6, ChromaPlane::Cb);

    assert!(sigma.is_finite() && sigma > 0.0, "sigma {sigma}");
    assert!(median.is_finite() && median > 0.0, "median {median}");
    assert!(
        flat_sigma.is_finite() && flat_sigma > 0.0,
        "flat sigma {flat_sigma}"
    );
    assert!(
        flat_median.is_finite() && flat_median > 0.0,
        "flat median {flat_median}"
    );
    assert!(sigma > flat_sigma, "sigma {sigma} flat sigma {flat_sigma}");
}

/// The columns of a 4:2:0 chroma template.
const CHROMA_COLS: usize = 44;
/// The first row and column of a chroma template's interior.
const CHROMA_INTERIOR_START: usize = 6;
/// The side length of a chroma template's interior.
const CHROMA_INTERIOR_SIZE: usize = 32;
const BLOCK_SIZE: usize = CELL as usize;
const BLOCKS_PER_SIDE: usize = CHROMA_INTERIOR_SIZE / BLOCK_SIZE;

/// The sample std of every 8x8 block of a chroma template's 32x32 interior.
fn interior_block_stds(template: &[i32]) -> Vec<f64> {
    let mut stds = Vec::new();
    for block_y in 0..BLOCKS_PER_SIDE {
        for block_x in 0..BLOCKS_PER_SIDE {
            let mut block = Vec::with_capacity(BLOCK_SIZE * BLOCK_SIZE);
            for y in 0..BLOCK_SIZE {
                let row = CHROMA_INTERIOR_START + block_y * BLOCK_SIZE + y;
                let col = CHROMA_INTERIOR_START + block_x * BLOCK_SIZE;
                let row_start = row * CHROMA_COLS + col;
                let values = template[row_start..row_start + BLOCK_SIZE].iter();
                block.extend(values.map(|&value| value as f64));
            }

            let block_std = sample_std(&block);
            stds.push(block_std);
        }
    }

    stds
}

#[test]
fn chroma_template_stats_use_8x8_blocks() {
    let mut block_stds = Vec::new();
    for index in 0..TEMPLATE_SEEDS {
        let seed = template_seed(index);
        let template = chroma_template(&SCENE_EIGHT, 7, seed, ChromaPlane::Cr);
        let stds = interior_block_stds(&template);
        block_stds.extend(stds);
    }

    let expected = median_of(&mut block_stds);
    let (_, median) = chroma_template_stats(&SCENE_EIGHT, 7, ChromaPlane::Cr);

    assert_eq!(median, expected);
}
