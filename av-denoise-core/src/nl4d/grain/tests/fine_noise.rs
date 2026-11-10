use super::synthetic::{
    ar_field,
    cell_mean_removed_record,
    gaussian_field,
    set_kept_count,
    set_source_count,
    table_chunk,
};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    AUTOCOV_LEN,
    BUCKETS_PER_GROUP,
    CELL,
    HIST_LEN,
    STD_BUCKETS,
};
use crate::nl4d::grain::fit::{
    ar_spectrum,
    bucket_edges,
    bucket_of,
    floor_fraction,
    undo_mean_removal,
    yule_walker,
};
use crate::nl4d::grain::segment::{FittedEntry, SceneGrain, fit_scenes, take_out_fine_noise};
use crate::nl4d::grain::table::{FitOptions, build_table};
use crate::nl4d::grain::template::{sample_std, template_stats};

const NTSC: (u64, u64) = (24_000, 1_001);
const SIZE: usize = 512;
/// The std of each synthetic grain part, 3 codes.
const PART_STD: f64 = 3.0 / 255.0;
/// The largest peak over mean a white grain's reduced spectrum may reach. Flat is 1.
const WHITE_PEAK_BOUND: f64 = 2.0;

/// The table [mixed_field] gave before the fine noise option existed.
const MIXED_TABLE: &str = concat!(
    "filmgrn1\n",
    "E 0 9801458 1 7391 1\n",
    "\tp 3 9 0 11 0 1 128 192 256 128 192 256\n",
    "\tsY 8  72 241 88 241 104 241 120 241 136 241 152 241 168 241 184 241\n",
    "\tsCb 2 0 0 255 0\n",
    "\tsCr 2 0 0 255 0\n",
    "\tcY -5 -7 -7 21 1 -2 -3 -7 -7 -4 48 9 -2 -4 -5 -3 14 98 39 14 9 24 48 110\n",
    "\tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n",
    "\tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n",
);

/// Weights of a separable AR filter, `rho` both left and up.
fn strong_weights(rho: f64) -> [f64; AR_COEFFS] {
    let mut weights = [0.0f64; AR_COEFFS];
    let pairs = [((0, -1), rho), ((-1, 0), rho), ((-1, -1), -rho * rho)];
    for (offset, value) in pairs {
        let index = AR_OFFSETS
            .iter()
            .position(|&candidate| candidate == offset)
            .expect("a lag-3 offset");
        weights[index] = value;
    }

    weights
}

fn scaled_to(field: Vec<f64>, std: f64) -> Vec<f64> {
    let current = sample_std(&field);
    let gain = std / current;

    field.into_iter().map(|value| value * gain).collect()
}

fn textured_field(seed: u64) -> Vec<f64> {
    let weights = strong_weights(0.7);
    let field = ar_field(&weights, SIZE, SIZE, seed);

    scaled_to(field, PART_STD)
}

fn white_field(seed: u64) -> Vec<f64> {
    let field = gaussian_field(SIZE, SIZE, seed);

    scaled_to(field, PART_STD)
}

fn sum_of(first: &[f64], second: &[f64]) -> Vec<f64> {
    first
        .iter()
        .zip(second)
        .map(|(first_value, second_value)| first_value + second_value)
        .collect()
}

fn variance_of(field: &[f64]) -> f64 {
    let std = sample_std(field);

    std * std
}

/// Textured and white grain of equal variance summed, with each part's variance.
fn mixed_field() -> (Vec<f64>, f64, f64) {
    let textured = textured_field(11);
    let white = white_field(12);
    let textured_variance = variance_of(&textured);
    let white_variance = variance_of(&white);
    let field = sum_of(&textured, &white);

    (field, textured_variance, white_variance)
}

/// The std of every mean-removed 8x8 block the measure kernel reads.
fn block_stds(field: &[f64]) -> Vec<f64> {
    let cell = CELL as usize;
    let mut stds = Vec::new();
    for cell_top in (0..SIZE - 2 * cell).step_by(cell) {
        for cell_left in (cell..SIZE - 2 * cell).step_by(cell) {
            let mut block = Vec::with_capacity(cell * cell);
            for y in cell_top..cell_top + cell {
                let row = &field[y * SIZE + cell_left..y * SIZE + cell_left + cell];
                block.extend_from_slice(row);
            }

            let block_std = sample_std(&block);
            stds.push(block_std);
        }
    }

    stds
}

/// A chunk measured from `field` in luma bins 4 to 11, its record in the median block's group.
///
/// With a `kept_ratio`, every block also leaves kept grain of that share of its std.
fn chunk_of(field: &[f64], kept_ratio: Option<f64>) -> GrainChunk {
    let edges = bucket_edges();
    let mut counts = vec![0u32; STD_BUCKETS];
    let mut kept_counts = vec![0u32; STD_BUCKETS];
    let mut stds = block_stds(field);
    for &std in &stds {
        let bucket = bucket_of(std as f32, &edges);
        counts[bucket] += 1;

        if let Some(ratio) = kept_ratio {
            let kept_std = std * ratio;
            let kept_bucket = bucket_of(kept_std as f32, &edges);
            kept_counts[kept_bucket] += 1;
        }
    }

    let mut chunk = table_chunk();
    for bin in 4..12 {
        for (bucket, &count) in counts.iter().enumerate() {
            set_source_count(&mut chunk, bin * STD_BUCKETS + bucket, count);
        }

        for (bucket, &count) in kept_counts.iter().enumerate() {
            set_kept_count(&mut chunk, bin * STD_BUCKETS + bucket, count);
        }
    }

    stds.sort_by(f64::total_cmp);
    let median = stds[stds.len() / 2];
    let group = bucket_of(median as f32, &edges) / BUCKETS_PER_GROUP;
    let record = cell_mean_removed_record(field, SIZE, SIZE);
    let start = 2 * HIST_LEN + group * AUTOCOV_LEN;
    let sums = &mut chunk.frame_tables[start..start + AUTOCOV_LEN];
    for (sum, &extra) in sums.iter_mut().zip(&record) {
        *sum += extra as f32;
    }

    chunk
}

fn scenes_of(field: &[f64]) -> Vec<SceneGrain> {
    scenes_with_kept(field, None)
}

fn scenes_with_kept(field: &[f64], kept_ratio: Option<f64>) -> Vec<SceneGrain> {
    let chunk = chunk_of(field, kept_ratio);
    let scene = SceneGrain {
        first_frame: 0,
        chunks: vec![chunk],
        chroma_chunks: Vec::new(),
    };

    vec![scene]
}

/// The AR weights fitted to `field`'s measured record.
fn weights_of(field: &[f64]) -> [f64; AR_COEFFS] {
    let record = cell_mean_removed_record(field, SIZE, SIZE);
    let corrected = undo_mean_removal(&record);

    yule_walker(&corrected).expect("a solvable record")
}

fn floor_of(field: &[f64]) -> f64 {
    let weights = weights_of(field);

    floor_fraction(&weights)
}

fn fit_with(field: &[f64], fine_noise: f64) -> FittedEntry {
    fit_with_kept(field, fine_noise, None)
}

fn fit_with_kept(field: &[f64], fine_noise: f64, kept_ratio: Option<f64>) -> FittedEntry {
    let scenes = scenes_with_kept(field, kept_ratio);
    let options = FitOptions { fine_noise };
    let scenes: Vec<&SceneGrain> = scenes.iter().collect();
    let mut entries = fit_scenes(&scenes, &options);

    assert_eq!(entries.len(), 1);
    entries.remove(0)
}

/// The whole-field variance each scaling point of `entry` synthesises.
fn fitted_variances(entry: &FittedEntry) -> Vec<f64> {
    let (sigma_template, _) = template_stats(&entry.ar_coeffs, entry.ar_shift);
    let scale = (1u32 << entry.scaling_shift) as f64;
    let mut variances = Vec::with_capacity(entry.points.len());
    for &(_, value) in &entry.points {
        let sigma = value as f64 * sigma_template / (255.0 * scale);
        variances.push(sigma * sigma);
    }

    variances
}

/// The entry's quantised weights as real values.
fn real_weights(entry: &FittedEntry) -> [f64; AR_COEFFS] {
    let scale = (1u32 << entry.ar_shift) as f64;

    entry.ar_coeffs.map(|coeff| coeff as f64 / scale)
}

/// The weights' model spectrum, scaled to a mean of 1.
fn normalised_spectrum(weights: &[f64; AR_COEFFS]) -> Vec<f64> {
    let spectrum = ar_spectrum(weights);
    let total: f64 = spectrum.iter().sum();
    let mean = total / spectrum.len() as f64;

    spectrum.into_iter().map(|power| power / mean).collect()
}

/// The largest gap, in the entry's quantisation steps, between its weights and `reference`.
fn steps_from(entry: &FittedEntry, reference: &[f64; AR_COEFFS]) -> i32 {
    let scale = (1u32 << entry.ar_shift) as f64;
    let mut largest = 0;
    for (&fitted, &weight) in entry.ar_coeffs.iter().zip(reference) {
        let expected = (weight * scale).round() as i32;
        let gap = (fitted - expected).abs();
        largest = largest.max(gap);
    }

    largest
}

#[test]
fn zero_fine_noise_leaves_the_table_unchanged() {
    let (field, _, _) = mixed_field();
    let scenes = scenes_of(&field);
    let options = FitOptions::default();

    let text = build_table(&scenes, NTSC, &options);

    assert_eq!(options.fine_noise, 0.0);
    assert_eq!(text, MIXED_TABLE);
}

#[test]
fn white_noise_alone_reads_as_all_floor() {
    let field = white_field(21);

    let floor = floor_of(&field);

    assert!((floor - 1.0).abs() < 0.05, "floor {floor}");
}

#[test]
fn textured_grain_alone_has_almost_no_floor() {
    let field = textured_field(31);

    let floor = floor_of(&field);

    assert!(floor < 0.1, "floor {floor}");
}

#[test]
fn fine_noise_takes_its_share_of_the_white_part() {
    let (field, textured_variance, white_variance) = mixed_field();
    let expected = textured_variance + 0.5 * white_variance;

    let entry = fit_with(&field, 0.5);
    let variances = fitted_variances(&entry);

    for variance in variances {
        let error = (variance - expected).abs() / expected;
        assert!(error < 0.1, "variance {variance}, expected {expected}");
    }
}

#[test]
fn fine_noise_fits_the_texture_plus_what_is_left_of_the_white() {
    let (field, _, _) = mixed_field();
    let textured = textured_field(11);
    let white = white_field(12);
    let half_white: Vec<f64> = white.iter().map(|value| value * 0.5f64.sqrt()).collect();
    let reference_field = sum_of(&textured, &half_white);
    let reference_weights = weights_of(&reference_field);
    let reference = normalised_spectrum(&reference_weights);

    let entry = fit_with(&field, 0.5);
    let table_weights = real_weights(&entry);
    let fitted = normalised_spectrum(&table_weights);

    let mut total = 0.0;
    for (&fitted_power, &reference_power) in fitted.iter().zip(&reference) {
        total += (fitted_power - reference_power).abs() / reference_power;
    }

    let difference = total / reference.len() as f64;
    assert!(difference < 0.1, "mean relative difference {difference}");
}

#[test]
fn fine_noise_moves_the_weights_toward_the_texture() {
    let (field, _, _) = mixed_field();
    let textured = textured_field(11);
    let reference = weights_of(&textured);

    let mut distances = Vec::new();
    for fine_noise in [0.0, 0.5, 1.0] {
        let entry = fit_with(&field, fine_noise);
        let distance = steps_from(&entry, &reference);
        distances.push(distance);
    }

    assert!(distances[1] < distances[0], "distances {distances:?}");
    assert!(distances[2] < distances[1], "distances {distances:?}");
}

#[test]
fn fine_noise_never_drives_a_bin_to_zero() {
    let (field, _, _) = mixed_field();

    let entry = fit_with(&field, 1.0);
    let kept_entry = fit_with_kept(&field, 1.0, Some(0.8));

    assert!(
        entry.points.iter().all(|&(_, value)| value > 0),
        "{:?}",
        entry.points
    );
    assert!(
        kept_entry.points.iter().all(|&(_, value)| value > 0),
        "{:?}",
        kept_entry.points
    );
}

/// The peak over the mean of the spectrum solved after `take_out` reduces white grain's record.
fn reduced_white_peak(take_out: impl Fn(&mut [f64], f64)) -> f64 {
    let field = white_field(21);
    let record = cell_mean_removed_record(&field, SIZE, SIZE);
    let mut corrected = undo_mean_removal(&record);
    let measured = yule_walker(&corrected).expect("a solvable record");
    let floor = floor_fraction(&measured);

    take_out(&mut corrected, floor);
    let reduced = yule_walker(&corrected).expect("a solvable reduced record");
    let spectrum = normalised_spectrum(&reduced);

    spectrum.into_iter().fold(0.0, f64::max)
}

#[test]
fn the_full_share_keeps_the_reduced_white_spectrum_bounded() {
    let capped = reduced_white_peak(|corrected, floor| take_out_fine_noise(corrected, floor, 1.0));
    let uncapped = reduced_white_peak(|corrected, floor| corrected[0] -= floor * corrected[0]);

    assert!(capped < WHITE_PEAK_BOUND, "capped peak {capped}");
    assert!(uncapped > WHITE_PEAK_BOUND, "uncapped peak {uncapped}");
}

#[test]
fn pure_white_grain_at_the_full_share_keeps_a_finite_table() {
    let field = white_field(21);

    let entry = fit_with(&field, 1.0);
    let variances = fitted_variances(&entry);

    assert!(
        variances.iter().all(|variance| variance.is_finite()),
        "{variances:?}"
    );
}

#[test]
fn build_table_clamps_the_fine_noise_share() {
    let (field, _, _) = mixed_field();
    let scenes = scenes_of(&field);
    let none = FitOptions { fine_noise: 0.0 };
    let full = FitOptions { fine_noise: 1.0 };
    let none_table = build_table(&scenes, NTSC, &none);
    let full_table = build_table(&scenes, NTSC, &full);

    for (share, expected) in [(-0.5, &none_table), (f64::NAN, &none_table), (1.5, &full_table)] {
        let options = FitOptions { fine_noise: share };

        let table = build_table(&scenes, NTSC, &options);

        assert_eq!(&table, expected, "share {share}");
    }

    assert_ne!(none_table, full_table);
}
