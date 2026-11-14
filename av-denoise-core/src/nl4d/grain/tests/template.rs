use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, CELL, LAGS, STD_MAX, TEMPLATE_SEEDS};
use crate::nl4d::grain::fit::code_step;
use crate::nl4d::grain::gaussian::GAUSSIAN_SEQUENCE;
use crate::nl4d::grain::template::{
    Av1Random,
    ChromaPlane,
    TemplatePlane,
    chroma_template,
    curve_cells,
    even_offsets,
    high_pass_curve,
    high_pass_std,
    luma_template,
    median_high_pass,
    median_of,
    seed_sigma,
    template_seed,
    unit_cells,
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

/// The columns of a luma template.
const LUMA_COLS: usize = 82;
/// The first row and column of a luma template's interior.
const LUMA_INTERIOR_START: usize = 9;
/// The side length of a luma template's interior.
const LUMA_INTERIOR_SIZE: usize = 64;

/// The population std of every sample in the square interiors of `templates`.
fn pooled_interior_std(templates: &[Vec<i32>], cols: usize, start: usize, size: usize) -> f64 {
    let mut samples = Vec::new();
    for template in templates {
        for row in start..start + size {
            let row_start = row * cols + start;
            let values = template[row_start..row_start + size].iter();
            samples.extend(values.map(|&value| value as f64));
        }
    }

    let count = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / count;
    let squares: f64 = samples.iter().map(|value| (value - mean) * (value - mean)).sum();

    (squares / count).sqrt()
}

/// The pooled luma interior std of the calibration seeds' templates.
fn calibration_luma_std(coeffs: &[i32; AR_COEFFS], ar_shift: u32) -> f64 {
    let templates: Vec<Vec<i32>> = (0..TEMPLATE_SEEDS)
        .map(|index| {
            let seed = template_seed(index);
            luma_template(coeffs, ar_shift, seed)
        })
        .collect();

    pooled_interior_std(&templates, LUMA_COLS, LUMA_INTERIOR_START, LUMA_INTERIOR_SIZE)
}

#[test]
fn calibration_templates_match_the_reference_port() {
    let sigma = calibration_luma_std(&SCENE_EIGHT, 7);
    let flat_sigma = calibration_luma_std(&[0; AR_COEFFS], 6);

    assert!((sigma - 44.770_018).abs() < 1e-4, "sigma {sigma}");
    assert!((flat_sigma - 31.909_499).abs() < 1e-4, "flat sigma {flat_sigma}");
}

#[test]
fn seed_sigma_keeps_its_recorded_value() {
    let sigma = seed_sigma(&SCENE_EIGHT, 7, TemplatePlane::Luma, 1000);
    let flat_sigma = seed_sigma(&[0; AR_COEFFS], 6, TemplatePlane::Luma, 1000);

    assert!((sigma - 44.128_652).abs() < 1e-4, "sigma {sigma}");
    assert!((flat_sigma - 31.928_815).abs() < 1e-4, "flat sigma {flat_sigma}");
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

/// The mean [seed_sigma] of the first four calibration seeds.
fn mean_seed_sigma(coeffs: &[i32; AR_COEFFS], ar_shift: u32, plane: TemplatePlane) -> f64 {
    let total: f64 = (0..4)
        .map(|index| {
            let seed = template_seed(index);
            seed_sigma(coeffs, ar_shift, plane, seed)
        })
        .sum();

    total / 4.0
}

#[test]
fn chroma_seed_sigma_grows_with_correlation() {
    let cb = TemplatePlane::Chroma(ChromaPlane::Cb);
    let sigma = mean_seed_sigma(&SCENE_EIGHT, 7, cb);
    let flat_sigma = mean_seed_sigma(&[0; AR_COEFFS], 6, cb);

    assert!(sigma.is_finite() && sigma > 0.0, "sigma {sigma}");
    assert!(
        flat_sigma.is_finite() && flat_sigma > 0.0,
        "flat sigma {flat_sigma}"
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

/// The high-pass std of every 8x8 block of a chroma template's 32x32 interior.
fn interior_high_pass_stds(template: &[i32]) -> Vec<f64> {
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

            let std = high_pass_std(&block);
            stds.push(std);
        }
    }

    stds
}

#[test]
fn unit_cells_are_the_8x8_blocks_of_the_chroma_interior() {
    let cr = TemplatePlane::Chroma(ChromaPlane::Cr);
    let mut stds = Vec::new();
    let mut templates = Vec::new();
    for index in 0..TEMPLATE_SEEDS {
        let seed = template_seed(index);
        let template = chroma_template(&SCENE_EIGHT, 7, seed, ChromaPlane::Cr);
        let interior_stds = interior_high_pass_stds(&template);
        stds.extend(interior_stds);
        templates.push(template);
    }

    let sigma = pooled_interior_std(
        &templates,
        CHROMA_COLS,
        CHROMA_INTERIOR_START,
        CHROMA_INTERIOR_SIZE,
    );
    let expected = median_of(&mut stds) / sigma;
    let cells = unit_cells(&SCENE_EIGHT, 7, cr);
    let offsets = even_offsets(cells.len());
    let median = median_high_pass(&cells, 1.0, &offsets, None);

    assert_eq!(
        cells.len(),
        TEMPLATE_SEEDS as usize * BLOCKS_PER_SIDE * BLOCKS_PER_SIDE
    );
    assert!((median / expected - 1.0).abs() < 1e-12, "{median} vs {expected}");
}

#[test]
fn a_constant_cell_has_no_high_pass() {
    let cell = [3.0f64; 64];

    assert_eq!(high_pass_std(&cell), 0.0);
}

/// Offsets spread evenly over one code but in reverse cell order and a quarter step along.
fn reversed_offsets(count: usize) -> Vec<f64> {
    let spacing = 1.0 / count as f64;

    (0..count)
        .map(|index| ((count - 1 - index) as f64 + 0.25) * spacing)
        .collect()
}

#[test]
fn high_pass_curve_recovers_template_grain() {
    let step = code_step(8);
    let textures = [([0; AR_COEFFS], 6), (SCENE_EIGHT, 7)];
    let planes = [TemplatePlane::Luma, TemplatePlane::Chroma(ChromaPlane::Cb)];
    for (coeffs, shift) in textures {
        for plane in planes {
            let curve = high_pass_curve(&coeffs, shift, plane, Some(step));
            let cells = curve_cells(&coeffs, shift, plane);
            let offsets = reversed_offsets(cells.len());
            for codes in [0.3, 0.6, 1.5] {
                let std = codes * step;
                let median = median_high_pass(&cells, std, &offsets, Some(step));
                let recovered = curve.field_std(median);

                let error = (recovered / std - 1.0).abs();
                assert!(error < 0.03, "{plane:?} {shift} {codes}: {recovered} vs {std}");
            }
        }
    }
}

#[test]
fn high_pass_curve_maps_zero_to_zero_and_clamps_above() {
    let step = code_step(8);
    let curve = high_pass_curve(&SCENE_EIGHT, 7, TemplatePlane::Luma, Some(step));

    let zero = curve.field_std(0.0);
    let top = curve.field_std(1.0e3);
    let higher = curve.field_std(1.0e6);
    let strongest_reading = curve.field_std(STD_MAX as f64);

    assert_eq!(zero, 0.0);
    assert!(top.is_finite() && top > 0.0, "{top}");
    assert_eq!(top, higher);
    assert!(strongest_reading < top, "{strongest_reading} vs {top}");
}

#[test]
fn high_pass_curve_never_falls() {
    let step = code_step(8);
    let curve = high_pass_curve(&SCENE_EIGHT, 7, TemplatePlane::Luma, Some(step));

    let mut previous = 0.0;
    for index in 1..400 {
        let median = index as f64 * 0.01 * step;
        let field_std = curve.field_std(median);

        assert!(
            field_std.is_finite() && field_std >= previous,
            "{median}: {field_std}"
        );
        previous = field_std;
    }
}

#[test]
fn no_bit_depth_gives_a_straight_curve() {
    let curve = high_pass_curve(&SCENE_EIGHT, 7, TemplatePlane::Luma, None);
    let cells = unit_cells(&SCENE_EIGHT, 7, TemplatePlane::Luma);
    let offsets = even_offsets(cells.len());
    let unit_median = median_high_pass(&cells, 1.0, &offsets, None);

    for median in [0.1 / 255.0, 0.5 / 255.0, 4.0 / 255.0] {
        let field_std = curve.field_std(median);
        let expected = median / unit_median;

        assert!(
            (field_std / expected - 1.0).abs() < 1e-9,
            "{field_std} vs {expected}"
        );
    }
}

#[test]
fn ten_bit_rounding_barely_moves_strong_grain() {
    let ten_bit = code_step(10);
    let white = [0; AR_COEFFS];
    let rounded = high_pass_curve(&white, 6, TemplatePlane::Luma, Some(ten_bit));
    let float = high_pass_curve(&white, 6, TemplatePlane::Luma, None);
    let median = 0.8 / 255.0;
    let unrounded_median = (median * median - ten_bit * ten_bit / 12.0).sqrt();

    let rounded_std = rounded.field_std(median);
    let expected = float.field_std(unrounded_median);
    let float_std = float.field_std(median);

    assert!(
        (rounded_std / expected - 1.0).abs() < 0.01,
        "{rounded_std} vs {expected}"
    );
    assert!(float_std / rounded_std < 1.03, "{float_std} vs {rounded_std}");
}
