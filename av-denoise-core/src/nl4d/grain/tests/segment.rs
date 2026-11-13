use super::synthetic::{
    add_group_record,
    ar_field,
    autocov_of,
    chroma_chunk,
    chroma_options,
    chunk_at,
    chunk_in_bucket,
    clear_pixels,
    fit_measured,
    gaussian_field,
    grain_record,
    keep_frame_bins,
    set_frame_count,
    set_source_count,
    table_chunk,
};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    BUCKETS_PER_GROUP,
    CHUNK_FRAMES,
    LUMA_BINS,
    MAX_POINTS,
    STD_BUCKETS,
};
use crate::nl4d::grain::fit::{
    bucket_edges,
    bucket_of,
    code_step,
    hist_median,
    quantise_ar,
    undo_mean_removal,
    yule_walker,
};
use crate::nl4d::grain::segment::{FittedEntry, SceneGrain, fit_scenes, pooled_record, segment_scene};
use crate::nl4d::grain::table::FitOptions;
use crate::nl4d::grain::template::{ChromaPlane, TemplatePlane, high_pass_curve, seed_sigma};

/// A source bucket whose median doubled lands in the first bucket of a strength group.
const EDGE_BUCKET: usize = 37;

/// Fits the scenes with chroma on and every measured part of the grain kept.
fn fit_with_chroma(scenes: &[SceneGrain]) -> Vec<FittedEntry> {
    let options = chroma_options();
    let scenes: Vec<&SceneGrain> = scenes.iter().collect();

    fit_scenes(&scenes, &options)
}

/// Fits one chunk as its own scene, with its curves rounded at `bit_depth`, and gives its first
/// frame's entry.
fn fit_chunk(chunk: GrainChunk, bit_depth: Option<u32>) -> FittedEntry {
    let scene = scene_of(0, vec![chunk]);
    let options = match bit_depth {
        Some(depth) => FitOptions::default().with_bit_depth(depth),
        None => FitOptions::default(),
    };

    let mut entries = fit_scenes(&[&scene], &options);
    assert_eq!(entries.len(), CHUNK_FRAMES as usize);

    entries.remove(0)
}

/// The entry of `entries` that starts at `frame`.
fn entry_at(entries: &[FittedEntry], frame: u64) -> &FittedEntry {
    entries
        .iter()
        .find(|entry| entry.first_frame == frame)
        .expect("an entry at the frame")
}

/// The whole-field std each point decodes to at the entry's own seed, with half a scaling step.
///
/// `plane` is `None` for luma, else the chroma plane whose points are read.
fn decoded_stds(entry: &FittedEntry, plane: Option<ChromaPlane>) -> Vec<(f64, f64)> {
    let (coeffs, points, template_plane) = match plane {
        None => (&entry.ar_coeffs, &entry.points, TemplatePlane::Luma),
        Some(chroma_plane) => {
            let planes = entry.chroma.as_ref().expect("both chroma planes");
            let plane_entry = &planes[chroma_plane.index()];
            let template_plane = TemplatePlane::Chroma(chroma_plane);
            (&plane_entry.ar_coeffs, &plane_entry.points, template_plane)
        },
    };

    let sigma_template = seed_sigma(coeffs, entry.ar_shift, template_plane, entry.seed);
    let scale = (1u32 << entry.scaling_shift) as f64;
    let step = sigma_template / (255.0 * scale);

    points
        .iter()
        .map(|&(_, value)| (value as f64 * step, 0.5 * step))
        .collect()
}

/// Whether two entries' points decode to the same strength, within their rounding.
fn same_strength(first: &FittedEntry, second: &FittedEntry, plane: Option<ChromaPlane>) -> bool {
    let first_stds = decoded_stds(first, plane);
    let second_stds = decoded_stds(second, plane);
    if first_stds.len() != second_stds.len() {
        return false;
    }

    first_stds
        .iter()
        .zip(&second_stds)
        .all(|(&(first_std, first_half), &(second_std, second_half))| {
            let gap = (first_std - second_std).abs();
            gap <= first_half + second_half + 1e-12
        })
}

/// Each luma point's scaling value over two to the scaling shift.
fn effective_points(entry: &FittedEntry) -> Vec<f64> {
    let scale = (1u32 << entry.scaling_shift) as f64;

    entry
        .points
        .iter()
        .map(|&(_, value)| value as f64 / scale)
        .collect()
}

/// The quantised weights of all the grain measured in `chunk`, before any fine noise comes off.
fn measured_weights(chunk: &GrainChunk) -> ([i32; AR_COEFFS], u32) {
    let stats = chunk.summed_table();
    let edges = bucket_edges();
    let record = pooled_record(&stats, &edges).expect("a pooled record");
    let corrected = undo_mean_removal(&record);
    let weights = yule_walker(&corrected).expect("solvable weights");

    quantise_ar(&weights)
}

/// The median of a bucket's histogram, holding every block in `bucket`.
fn bucket_median(bucket: usize) -> f64 {
    let edges = bucket_edges();
    let mut counts = vec![0u32; STD_BUCKETS];
    counts[bucket] = 300;

    hist_median(&counts, &edges).expect("a populated bucket")
}

/// The position of the left neighbour's weight.
fn left_lane() -> usize {
    AR_OFFSETS
        .iter()
        .position(|&offset| offset == (0, -1))
        .expect("left offset")
}

/// The autocovariance record of a lag-3 AR field with only a left weight of `weight`.
fn left_record(weight: f64) -> Vec<f64> {
    let left = left_lane();
    let mut weights = [0.0f64; AR_COEFFS];
    weights[left] = weight;
    let field = ar_field(&weights, 300, 300, 5);

    autocov_of(&field, 300, 300)
}

/// Sets the first frame's one-frame counts at `bucket` in luma bins 4 to 11.
fn fill_frame_bucket(chunk: &mut GrainChunk, bucket: usize) {
    for bin in 4..12 {
        set_frame_count(chunk, bin * STD_BUCKETS + bucket, 300);
    }
}

/// A chunk whose source and one-frame blocks sit at 2 codes in luma bins 4 to 11, with `record` as
/// its texture.
fn chunk_with_record(record: &[f64]) -> GrainChunk {
    let edges = bucket_edges();
    let bucket = bucket_of(2.0 / 255.0, &edges);
    let mut chunk = chunk_in_bucket(bucket, 300);
    fill_frame_bucket(&mut chunk, bucket);
    add_group_record(&mut chunk, bucket / BUCKETS_PER_GROUP, record);

    chunk
}

/// A chunk with an AR record and 1500 blocks at `std_codes` in only two luma bins.
///
/// It holds enough blocks to start a segment, but too few populated bins for its own strength.
fn narrow_chunk(std_codes: f32) -> GrainChunk {
    let edges = bucket_edges();
    let bucket = bucket_of(std_codes / 255.0, &edges);
    let mut chunk = table_chunk();

    for bin in 4..6 {
        set_source_count(&mut chunk, bin * STD_BUCKETS + bucket, 1500);
    }

    let record = grain_record();
    add_group_record(&mut chunk, bucket / BUCKETS_PER_GROUP, &record);

    chunk
}

fn scene_of(first_frame: u64, chunks: Vec<GrainChunk>) -> SceneGrain {
    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks: Vec::new(),
    }
}

/// The record of noise box-blurred over 9x9 pixels and scaled by `gain`, correlated past lag 3.
fn coarse_record(gain: f64) -> Vec<f64> {
    let size = 300;
    let radius = 4;
    let padded = size + 2 * radius;
    let noise = gaussian_field(padded, padded, 9);
    let window = ((2 * radius + 1) * (2 * radius + 1)) as f64;
    let mut field = vec![0.0f64; size * size];

    for y in 0..size {
        for x in 0..size {
            let mut total = 0.0;
            for window_y in y..=y + 2 * radius {
                let row = &noise[window_y * padded + x..window_y * padded + x + 2 * radius + 1];
                total += row.iter().sum::<f64>();
            }

            field[y * size + x] = gain * total / window;
        }
    }

    autocov_of(&field, size, size)
}

/// The overall source median of a chunk built by [chunk_in_bucket].
fn median_of_chunk(chunk: &GrainChunk) -> f64 {
    let table = chunk.summed_table();
    let mut merged = vec![0u32; STD_BUCKETS];
    for (index, &count) in table.source_hist.iter().enumerate() {
        merged[index % STD_BUCKETS] += count;
    }

    let edges = bucket_edges();
    hist_median(&merged, &edges).expect("accepted blocks")
}

/// A chunk at [EDGE_BUCKET] and the strength group holding twice its median.
fn edge_chunk() -> (GrainChunk, usize) {
    let edges = bucket_edges();
    let mut chunk = chunk_in_bucket(EDGE_BUCKET, 300);
    fill_frame_bucket(&mut chunk, EDGE_BUCKET);
    let band_high = 2.0 * median_of_chunk(&chunk);
    let edge_bucket = bucket_of(band_high as f32, &edges);

    assert_eq!(edge_bucket % BUCKETS_PER_GROUP, 0);

    (chunk, edge_bucket / BUCKETS_PER_GROUP)
}

#[test]
fn steady_scene_is_one_segment() {
    let first = chunk_at(2.0, 300, None);
    let second = chunk_at(2.05, 300, None);
    let third = chunk_at(1.98, 300, None);
    let chunks = vec![first, second, third];
    let scene = scene_of(100, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].first_frame, 100);
    assert_eq!(segments[0].last_frame, 100 + 3 * CHUNK_FRAMES as u64 - 1);
}

#[test]
fn drift_splits_on_a_chunk_boundary() {
    let first = chunk_at(2.0, 300, None);
    let second = chunk_at(2.0, 300, None);
    let drifted = chunk_at(2.8, 300, None);
    let chunks = vec![first, second, drifted];
    let scene = scene_of(0, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[1].first_frame, 2 * CHUNK_FRAMES as u64);
}

#[test]
fn a_thin_chunk_joins_the_open_segment() {
    let full = chunk_at(2.0, 300, None);
    let thin = chunk_at(4.0, 10, None);
    let chunks = vec![full, thin];
    let scene = scene_of(0, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 1);
}

#[test]
fn a_fitted_entry_has_points_and_weights() {
    let chunk = chunk_at(2.0, 300, None);
    let scene = scene_of(0, vec![chunk]);
    let scenes = vec![scene];

    let entries = fit_measured(&scenes);

    assert_eq!(entries.len(), CHUNK_FRAMES as usize);
    assert_eq!(entries[0].points.len(), 8);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

/// The chunks around the thin one read different one-frame strengths, so one segment over all three
/// gives frames 24 and 72 the same strength.
#[test]
fn a_thin_chunk_never_gets_its_own_entry() {
    let first_chunk = chunk_at(1.0, 300, None);
    let before_thin = chunk_at(3.0, 300, Some(3.0));
    let thin = table_chunk();
    let after_thin = chunk_at(3.0, 300, Some(4.0));
    let second_chunks = vec![before_thin, thin, after_thin];
    let first_scene = scene_of(0, vec![first_chunk]);
    let second_scene = scene_of(24, second_chunks);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_measured(&scenes);
    let first = entry_at(&entries, 0);
    let before = entry_at(&entries, 24);
    let after = entry_at(&entries, 24 + 2 * CHUNK_FRAMES as u64);
    let last = entries.last().expect("entries");

    assert_eq!(entries.len(), 4 * CHUNK_FRAMES as usize);
    assert_eq!(last.last_frame, 24 + 3 * CHUNK_FRAMES as u64 - 1);
    assert!(same_strength(before, after, None));
    assert!(!same_strength(first, before, None));
}

#[test]
fn missing_texture_borrows_from_the_nearest_segment() {
    let mut no_texture = chunk_at(2.0, 300, None);
    clear_pixels(&mut no_texture);
    let textured = chunk_at(2.0, 300, None);
    let first_scene = scene_of(0, vec![textured]);
    let second_scene = scene_of(24, vec![no_texture]);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_measured(&scenes);
    let borrower = entry_at(&entries, 24);

    assert_eq!(entries.len(), 2 * CHUNK_FRAMES as usize);
    assert_eq!(borrower.ar_coeffs, entries[0].ar_coeffs);
}

#[test]
fn sparse_scene_borrows_from_a_neighbouring_scene() {
    let dense = chunk_at(2.0, 300, None);
    let sparse = table_chunk();
    let first_scene = scene_of(0, vec![dense]);
    let second_scene = scene_of(24, vec![sparse]);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_measured(&scenes);
    let borrower = entry_at(&entries, 24);

    assert_eq!(entries.len(), 2 * CHUNK_FRAMES as usize);
    assert!(same_strength(borrower, &entries[0], None));
}

#[test]
fn sparse_segment_with_no_donor_gets_no_entry() {
    let sparse = table_chunk();
    let scene = scene_of(0, vec![sparse]);
    let scenes = vec![scene];

    let entries = fit_measured(&scenes);

    assert!(entries.is_empty());
}

#[test]
fn an_outlier_group_leaves_the_texture_unchanged() {
    let edges = bucket_edges();
    let grain_only = chunk_at(2.0, 300, None);
    let grain_group = bucket_of(2.0 / 255.0, &edges) / BUCKETS_PER_GROUP;
    let outlier_bucket = bucket_of(8.0 / 255.0, &edges);
    let coarse = coarse_record(40.0);

    let mut with_outlier = grain_only.clone();
    set_source_count(&mut with_outlier, 5 * STD_BUCKETS + outlier_bucket, 10);
    add_group_record(&mut with_outlier, outlier_bucket / BUCKETS_PER_GROUP, &coarse);

    let mut mixed = grain_only.clone();
    add_group_record(&mut mixed, grain_group, &coarse);

    let clean_scene = scene_of(0, vec![grain_only]);
    let outlier_scene = scene_of(0, vec![with_outlier]);
    let mixed_scene = scene_of(0, vec![mixed]);
    let clean_entries = fit_measured(&[clean_scene]);
    let outlier_entries = fit_measured(&[outlier_scene]);
    let mixed_entries = fit_measured(&[mixed_scene]);
    let grain = grain_record();

    assert!(coarse[0] > 10.0 * grain[0]);
    assert!(clean_entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
    assert_eq!(outlier_entries[0].ar_coeffs, clean_entries[0].ar_coeffs);
    assert_ne!(mixed_entries[0].ar_coeffs, clean_entries[0].ar_coeffs);
}

#[test]
fn a_group_overlapping_the_band_edge_gives_the_texture() {
    let (mut chunk, edge_group) = edge_chunk();
    let record = grain_record();
    add_group_record(&mut chunk, edge_group, &record);

    let scene = scene_of(0, vec![chunk]);
    let entries = fit_measured(&[scene]);

    assert_eq!(entries.len(), CHUNK_FRAMES as usize);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

#[test]
fn a_group_past_the_band_gives_no_texture() {
    let (mut chunk, edge_group) = edge_chunk();
    let record = grain_record();
    add_group_record(&mut chunk, edge_group + 1, &record);

    let scene = scene_of(0, vec![chunk]);
    let entries = fit_measured(&[scene]);

    assert!(entries.is_empty());
}

#[test]
fn a_sparse_segment_borrows_from_its_own_scene_first() {
    let dense = chunk_at(2.0, 300, None);
    let first_narrow = narrow_chunk(3.0);
    let second_narrow = narrow_chunk(4.0);
    let next_scene_chunk = chunk_at(1.0, 300, None);
    let own_scene = vec![dense, first_narrow, second_narrow];
    let first_scene = scene_of(0, own_scene);
    let second_scene = scene_of(3 * CHUNK_FRAMES as u64, vec![next_scene_chunk]);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_measured(&scenes);
    let second_narrow = entry_at(&entries, 2 * CHUNK_FRAMES as u64);
    let next_scene = entry_at(&entries, 3 * CHUNK_FRAMES as u64);

    assert_eq!(entries.len(), 4 * CHUNK_FRAMES as usize);
    assert!(same_strength(second_narrow, &entries[0], None));
    assert!(!same_strength(second_narrow, next_scene, None));
}

#[test]
fn a_chunk_in_every_luma_bin_thins_to_the_point_limit() {
    let edges = bucket_edges();
    let bucket = bucket_of(2.0 / 255.0, &edges);
    let mut chunk = chunk_at(2.0, 300, None);
    for bin in 0..LUMA_BINS {
        set_source_count(&mut chunk, bin * STD_BUCKETS + bucket, 300);
        set_frame_count(&mut chunk, bin * STD_BUCKETS + bucket, 300);
    }

    let scene = scene_of(0, vec![chunk]);
    let entries = fit_measured(&[scene]);
    let points = &entries[0].points;
    let first_luma = points.first().expect("points").0;
    let last_luma = points.last().expect("points").0;

    assert_eq!(points.len(), MAX_POINTS);
    assert!(points.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert_eq!(first_luma, 8);
    assert_eq!(last_luma, 248);
}

#[test]
fn a_segment_without_chroma_borrows_from_its_neighbour() {
    let steady = chunk_at(2.0, 300, None);
    let drifted = chunk_at(2.8, 300, None);
    let settled = chunk_at(2.0, 300, None);
    let mut thin_chroma = drifted.clone();
    keep_frame_bins(&mut thin_chroma, 4..6);
    let steady_chroma = chroma_chunk(&steady, &steady);
    let drifted_chroma = chroma_chunk(&thin_chroma, &thin_chroma);
    let settled_chroma = chroma_chunk(&settled, &settled);
    let scene = SceneGrain {
        first_frame: 0,
        chunks: vec![steady, drifted, settled],
        chroma_chunks: vec![steady_chroma, drifted_chroma, settled_chroma],
    };

    let entries = fit_with_chroma(&[scene]);
    let steady_entry = &entries[0];
    let drifted_entry = entry_at(&entries, CHUNK_FRAMES as u64);
    let steady_planes = steady_entry.chroma.as_ref().expect("both chroma planes");
    let drifted_planes = drifted_entry.chroma.as_ref().expect("both chroma planes");

    assert_eq!(entries.len(), 3 * CHUNK_FRAMES as usize);
    assert_eq!(drifted_entry.ar_shift, steady_entry.ar_shift);
    for plane in [ChromaPlane::Cb, ChromaPlane::Cr] {
        let slot = plane.index();
        assert_eq!(drifted_planes[slot].ar_coeffs, steady_planes[slot].ar_coeffs);
        assert!(
            same_strength(drifted_entry, steady_entry, Some(plane)),
            "{plane:?}"
        );
    }
}

#[test]
fn the_ar_shift_fits_every_plane() {
    let luma_record = left_record(0.15);
    let chroma_record = left_record(0.7);
    let luma = chunk_with_record(&luma_record);
    let chroma_tables = chunk_with_record(&chroma_record);
    let chroma = chroma_chunk(&chroma_tables, &chroma_tables);
    let scenes = vec![SceneGrain {
        first_frame: 0,
        chunks: vec![luma],
        chroma_chunks: vec![chroma],
    }];

    let luma_only = fit_measured(&scenes);
    let with_chroma = fit_with_chroma(&scenes);
    let entry = &with_chroma[0];
    let planes = entry.chroma.as_ref().expect("both chroma planes");
    let left = left_lane();
    let luma_only_left = luma_only[0].ar_coeffs[left];
    let shared_left = entry.ar_coeffs[left];

    assert_eq!(luma_only[0].ar_shift, 9);
    assert_eq!(entry.ar_shift, 7);
    assert!(
        (4 * shared_left - luma_only_left).abs() <= 2,
        "{shared_left} {luma_only_left}"
    );
    for plane in planes {
        let chroma_left = plane.ar_coeffs[left];
        assert!((64..127).contains(&chroma_left), "{chroma_left}");
    }
}

#[test]
fn strong_chroma_lowers_the_scaling_shift() {
    let luma = chunk_at(2.0, 300, None);
    let chroma_tables = chunk_at(2.0, 300, Some(8.0));
    let chroma = chroma_chunk(&chroma_tables, &chroma_tables);
    let scenes = vec![SceneGrain {
        first_frame: 0,
        chunks: vec![luma],
        chroma_chunks: vec![chroma],
    }];

    let luma_only = fit_measured(&scenes);
    let with_chroma = fit_with_chroma(&scenes);
    let entry = &with_chroma[0];
    let planes = entry.chroma.as_ref().expect("both chroma planes");

    assert!(entry.scaling_shift < luma_only[0].scaling_shift);
    for plane in planes {
        let capped = plane.points.iter().any(|&(_, value)| value == 255);
        assert!(!capped || entry.scaling_shift == 8, "{:?}", plane.points);
    }
}

/// The one-frame grain doubles, so each point grows by the ratio of the two bucket medians.
#[test]
fn strength_reads_the_one_frame_histogram() {
    let edges = bucket_edges();
    let weak_bucket = bucket_of(2.0 / 255.0, &edges);
    let strong_bucket = bucket_of(4.0 / 255.0, &edges);
    let weak_median = bucket_median(weak_bucket);
    let strong_median = bucket_median(strong_bucket);
    let growth = strong_median / weak_median;
    let weak_chunk = chunk_at(1.0, 300, Some(2.0));
    let strong_chunk = chunk_at(1.0, 300, Some(4.0));

    let weak = fit_chunk(weak_chunk, None);
    let strong = fit_chunk(strong_chunk, None);
    let weak_points = effective_points(&weak);
    let strong_points = effective_points(&strong);
    let coarsest_shift = weak.scaling_shift.min(strong.scaling_shift);
    let step = 1.0 / (1u32 << coarsest_shift) as f64;

    assert!((growth / 2.0 - 1.0).abs() < 0.11, "{growth}");
    assert_eq!(weak.ar_coeffs, strong.ar_coeffs);
    assert_eq!(weak_points.len(), strong_points.len());
    for (&weak_point, &strong_point) in weak_points.iter().zip(&strong_points) {
        let error = (strong_point - growth * weak_point).abs();
        assert!(error <= step, "{weak_point} {strong_point} {growth}");
    }
}

#[test]
fn the_source_histogram_does_not_set_strength() {
    let low_source = chunk_at(1.0, 300, Some(2.0));
    let high_source = chunk_at(1.5, 300, Some(2.0));

    let low = fit_chunk(low_source, None);
    let high = fit_chunk(high_source, None);

    assert_eq!(low.ar_coeffs, high.ar_coeffs);
    assert_eq!(low.scaling_shift, high.scaling_shift);
    assert_eq!(low.points, high.points);
}

#[test]
fn rounding_comes_off_each_bin_at_eight_bits() {
    let edges = bucket_edges();
    let frame_bucket = bucket_of(0.6 / 255.0, &edges);
    let median = bucket_median(frame_bucket);

    let chunk = chunk_at(1.0, 300, Some(0.6));
    let (coeffs, shift) = measured_weights(&chunk);
    let float_entry = fit_chunk(chunk.clone(), None);
    let eight_bit_entry = fit_chunk(chunk, Some(8));
    let float_curve = high_pass_curve(&coeffs, shift, TemplatePlane::Luma, None);
    let eight_bit = Some(code_step(8));
    let eight_bit_curve = high_pass_curve(&coeffs, shift, TemplatePlane::Luma, eight_bit);
    let float_std = float_curve.field_std(median);
    let eight_bit_std = eight_bit_curve.field_std(median);
    let expected = eight_bit_std / float_std;

    let float_points = effective_points(&float_entry);
    let eight_bit_points = effective_points(&eight_bit_entry);

    assert!(expected < 0.95, "{expected}");
    assert!(!float_points.is_empty());
    assert_eq!(float_points.len(), eight_bit_points.len());
    for (&before, &after) in float_points.iter().zip(&eight_bit_points) {
        let ratio = after / before;
        assert!((ratio / expected - 1.0).abs() < 0.02, "{ratio} vs {expected}");
    }
}

#[test]
fn faint_grain_gives_finite_small_points() {
    let chunk = chunk_at(1.0, 300, Some(0.2));
    let float_entry = fit_chunk(chunk.clone(), None);
    let eight_bit_entry = fit_chunk(chunk, Some(8));
    let float_points = effective_points(&float_entry);
    let eight_bit_points = effective_points(&eight_bit_entry);

    assert_eq!(float_points.len(), eight_bit_points.len());
    for (&before, &after) in float_points.iter().zip(&eight_bit_points) {
        assert!(after.is_finite() && after > 0.0, "{after}");
        assert!(after < before, "{after} vs {before}");
    }
}
