use super::synthetic::{ar_field, autocov_of, cell_mean_removed_record, gaussian_field};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    HIST_LEN,
    LAG_COUNT,
    LAGS,
    STD_BUCKETS,
    STD_MAX,
    STD_MIN,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::fit::{
    bucket_edges,
    bucket_of,
    hist_median,
    measured_lane,
    quantise_ar,
    scaling_points,
    undo_mean_removal,
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
    let lowest = bucket_of(STD_MIN, &edges);
    let far_above = bucket_of(STD_MAX * 4.0, &edges);
    let just_past_edge = bucket_of(edges[10] * 1.0001, &edges);

    assert_eq!(lowest, 0);
    assert_eq!(far_above, STD_BUCKETS - 1);
    assert_eq!(just_past_edge, 10);
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
    let median = hist_median(&counts, &edges);

    assert_eq!(median, None);
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
    let solved = yule_walker(&autocov);

    assert!(solved.is_none());
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

            let block_std = sample_std(&block);
            block_stds.push(block_std);
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
    first.autocov[LAG_COUNT] = 1.5;
    first.pixels[1] = 10.0;

    let mut second = GrainChunk::empty();
    second.frames = 4;
    second.source_hist[5] = 1;
    second.kept_hist[2] = 7;
    second.autocov[LAG_COUNT] = 0.5;
    second.pixels[1] = 6.0;

    first.merge(&second);

    assert_eq!(first.frames, 7);
    assert_eq!(first.source_hist[5], 3);
    assert_eq!(first.kept_hist[2], 7);
    assert_eq!(first.group_autocov(1)[0], 2.0);
    assert_eq!(first.pixels[1], 16.0);
    assert_eq!(first.pixels.len(), STRENGTH_GROUPS);
    assert_eq!(first.source_hist.len(), HIST_LEN);
    assert_eq!(first.source_blocks(), 3);
}

/// The record's lag sums divided by its zero lag.
fn correlation_of(record: &[f64]) -> Vec<f64> {
    record[..LAG_COUNT].iter().map(|&sum| sum / record[0]).collect()
}

#[test]
fn measured_lanes_follow_the_lag_order() {
    for (lane, &(dy, dx)) in LAGS.iter().enumerate() {
        let forward = measured_lane(dy, dx);
        let mirrored = measured_lane(-dy, -dx);

        assert_eq!(forward, Some(lane));
        assert_eq!(mirrored, Some(lane));
    }

    let too_far_down = measured_lane(4, 0);
    let too_far_right = measured_lane(0, 7);

    assert_eq!(too_far_down, None);
    assert_eq!(too_far_right, None);
}

#[test]
fn mean_correction_gives_white_grain_zero_weights() {
    let field = gaussian_field(600, 600, 21);
    let record = cell_mean_removed_record(&field, 600, 600);
    let corrected = undo_mean_removal(&record);

    let raw_weights = yule_walker(&record).expect("a well-conditioned system");
    let corrected_weights = yule_walker(&corrected).expect("a well-conditioned system");
    let raw_sum: f64 = raw_weights.iter().sum();
    let corrected_sum: f64 = corrected_weights.iter().sum();

    assert!(raw_sum < -0.15, "raw sum {raw_sum}");
    assert!(corrected_sum.abs() < 0.03, "corrected sum {corrected_sum}");
}

#[test]
fn mean_correction_recovers_an_ar_fields_correlation() {
    let size = 1016;
    let mut weights = [0.0f64; AR_COEFFS];
    set_weight(&mut weights, (0, -1), 0.3);
    set_weight(&mut weights, (-1, 0), 0.3);
    let field = ar_field(&weights, size, size, 3);
    let full = autocov_of(&field, size, size);
    let record = cell_mean_removed_record(&field, size, size);
    let corrected = undo_mean_removal(&record);

    let truth = correlation_of(&full);
    let measured = correlation_of(&record);
    let recovered = correlation_of(&corrected);
    for lane in 0..LAG_COUNT {
        let error = (recovered[lane] - truth[lane]).abs();
        assert!(
            error < 0.02,
            "lane {lane}: {} vs {}",
            recovered[lane],
            truth[lane]
        );
    }

    let full_weights = yule_walker(&full).expect("a well-conditioned system");
    let corrected_weights = yule_walker(&corrected).expect("a well-conditioned system");
    for (index, (&got, &want)) in corrected_weights.iter().zip(&full_weights).enumerate() {
        assert!((got - want).abs() < 0.02, "weight {index}: {got} vs {want}");
    }

    assert!(truth[1] - measured[1] > 0.03, "{} vs {}", measured[1], truth[1]);
}
