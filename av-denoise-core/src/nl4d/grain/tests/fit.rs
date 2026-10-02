use super::synthetic::{ar_field, autocov_of, gaussian_field};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, HIST_LEN, LAG_COUNT, STD_BUCKETS, STD_MAX, STD_MIN};
use crate::nl4d::grain::fit::{
    bucket_edges,
    bucket_of,
    hist_median,
    quantise_ar,
    scaling_points,
    yule_walker,
};
use crate::nl4d::grain::template::{median_of, sample_std, template_stats};

fn set_weight(weights: &mut [f64; AR_COEFFS], offset: (i32, i32), value: f64) {
    let index = AR_OFFSETS
        .iter()
        .position(|&candidate| candidate == offset)
        .expect("offset exists");
    weights[index] = value;
}

fn scene_eight_weights() -> [f64; AR_COEFFS] {
    let mut weights = [0.0f64; AR_COEFFS];
    set_weight(&mut weights, (0, -1), 0.555);
    set_weight(&mut weights, (-1, 0), 0.477);
    set_weight(&mut weights, (0, -2), -0.172);
    set_weight(&mut weights, (-2, 0), -0.148);
    set_weight(&mut weights, (-1, -1), -0.164);
    set_weight(&mut weights, (-1, 1), 0.117);
    weights
}

#[test]
fn bucket_edges_are_log_spaced_between_the_limits() {
    let edges = bucket_edges();
    let first_ratio = edges[1] / edges[0];
    let last_ratio = edges[STD_BUCKETS] / edges[STD_BUCKETS - 1];

    assert_eq!(edges.len(), STD_BUCKETS + 1);
    assert!((edges[0] - STD_MIN).abs() < 1e-9);
    assert!((edges[STD_BUCKETS] - STD_MAX).abs() < 1e-7);
    assert!((first_ratio - last_ratio).abs() < 1e-4);
}

#[test]
fn bucket_of_clamps_to_the_last_bucket() {
    let edges = bucket_edges();

    assert_eq!(bucket_of(STD_MIN, &edges), 0);
    assert_eq!(bucket_of(STD_MAX * 4.0, &edges), STD_BUCKETS - 1);
    assert_eq!(bucket_of(edges[10] * 1.0001, &edges), 10);
}

#[test]
fn histogram_median_is_within_one_bucket_of_the_exact_median() {
    let edges = bucket_edges();
    let field = gaussian_field(4000, 1, 7);
    let mut values: Vec<f64> = field.iter().map(|sample| (2.0 + sample.abs()) / 255.0).collect();
    let mut counts = vec![0u32; STD_BUCKETS];
    for &value in &values {
        let bucket = bucket_of(value as f32, &edges);
        counts[bucket] += 1;
    }

    let exact = median_of(&mut values);
    let estimate = hist_median(&counts, &edges).expect("non-empty histogram");
    let bucket_ratio = (edges[1] / edges[0]) as f64;

    assert!(
        estimate / exact < bucket_ratio && exact / estimate < bucket_ratio,
        "{estimate} vs {exact}"
    );
}

#[test]
fn histogram_median_of_an_empty_histogram_is_none() {
    let edges = bucket_edges();
    let counts = vec![0u32; STD_BUCKETS];

    assert_eq!(hist_median(&counts, &edges), None);
}

#[test]
fn yule_walker_recovers_known_weights() {
    let weights = scene_eight_weights();
    let field = ar_field(&weights, 400, 400, 11);
    let autocov = autocov_of(&field, 400, 400);
    let solved = yule_walker(&autocov).expect("a well-conditioned system");

    for (index, (&got, &want)) in solved.iter().zip(weights.iter()).enumerate() {
        assert!((got - want).abs() < 0.01, "weight {index}: {got} vs {want}");
    }
}

#[test]
fn yule_walker_rejects_an_empty_record() {
    let autocov = vec![0.0f64; LAG_COUNT + 1];

    assert!(yule_walker(&autocov).is_none());
}

#[test]
fn quantise_picks_the_finest_shift_that_fits() {
    let mut weights = [0.0f64; AR_COEFFS];
    weights[23] = 0.555;
    let (coeffs, shift) = quantise_ar(&weights);

    assert_eq!(shift, 7);
    assert_eq!(coeffs[23], 71);

    weights[23] = 0.2;
    let (fine, fine_shift) = quantise_ar(&weights);

    assert_eq!(fine_shift, 9);
    assert_eq!(fine[23], 102);
}

#[test]
fn calibration_recovers_a_known_sigma() {
    let weights = scene_eight_weights();
    let field = ar_field(&weights, 512, 512, 23);
    let true_sigma = sample_std(&field);
    let mut block_stds = Vec::new();
    for block_y in 0..64 {
        for block_x in 0..64 {
            let mut block = Vec::with_capacity(64);
            for row in 0..8 {
                let start = (block_y * 8 + row) * 512 + block_x * 8;
                block.extend_from_slice(&field[start..start + 8]);
            }

            block_stds.push(sample_std(&block));
        }
    }

    let measured = median_of(&mut block_stds);
    let (quantised, shift) = quantise_ar(&weights);
    let (sigma_template, template_median) = template_stats(&quantised, shift);
    let estimate = measured * sigma_template / template_median;

    assert!(
        (estimate / true_sigma - 1.0).abs() < 0.03,
        "{estimate} vs {true_sigma}"
    );
}

#[test]
fn scaling_points_use_the_finest_shift_under_255() {
    let points = [(24u8, 1.0f64 / 255.0), (120u8, 2.0f64 / 255.0)];
    let (scaled, shift) = scaling_points(&points, 40.0);

    assert_eq!(shift, 11);
    assert_eq!(scaled[0], (24, 51));
    assert_eq!(scaled[1], (120, 102));
}

#[test]
fn chunks_merge_by_adding() {
    let mut first = GrainChunk::empty();
    first.frames = 3;
    first.source_hist[5] = 2;
    first.autocov[0] = 1.5;
    first.pixels = 10.0;

    let mut second = GrainChunk::empty();
    second.frames = 4;
    second.source_hist[5] = 1;
    second.kept_hist[2] = 7;
    second.autocov[0] = 0.5;
    second.pixels = 6.0;

    first.merge(&second);

    assert_eq!(first.frames, 7);
    assert_eq!(first.source_hist[5], 3);
    assert_eq!(first.kept_hist[2], 7);
    assert_eq!(first.autocov[0], 2.0);
    assert_eq!(first.pixels, 16.0);
    assert_eq!(first.source_hist.len(), HIST_LEN);
    assert_eq!(first.source_blocks(), 3);
}
