use super::synthetic::{
    add_group_record,
    autocov_of,
    chunk_at,
    chunk_in_bucket,
    gaussian_field,
    grain_record,
};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{BUCKETS_PER_GROUP, CHUNK_FRAMES, LUMA_BINS, MAX_POINTS, STD_BUCKETS};
use crate::nl4d::grain::fit::{bucket_edges, bucket_of, hist_median};
use crate::nl4d::grain::segment::{SceneGrain, fit_scenes, segment_scene};

/// A source bucket whose median doubled lands in the first bucket of a strength group.
const EDGE_BUCKET: usize = 37;

fn sparse_chunk() -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;

    chunk
}

/// A chunk with an AR record and 1500 blocks at `std_codes` in only two luma bins.
///
/// It holds enough blocks to start a segment, but too few populated bins for its own strength.
fn narrow_chunk(std_codes: f32) -> GrainChunk {
    let edges = bucket_edges();
    let bucket = bucket_of(std_codes / 255.0, &edges);
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;

    for bin in 4..6 {
        chunk.source_hist[bin * STD_BUCKETS + bucket] = 1500;
    }

    let record = grain_record();
    add_group_record(&mut chunk, bucket / BUCKETS_PER_GROUP, &record);

    chunk
}

fn scene_of(first_frame: u64, chunks: Vec<GrainChunk>) -> SceneGrain {
    SceneGrain { first_frame, chunks }
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
    let mut merged = vec![0u32; STD_BUCKETS];
    for (index, &count) in chunk.source_hist.iter().enumerate() {
        merged[index % STD_BUCKETS] += count;
    }

    let edges = bucket_edges();
    hist_median(&merged, &edges).expect("accepted blocks")
}

/// A chunk at [EDGE_BUCKET] and the strength group holding twice its median.
fn edge_chunk() -> (GrainChunk, usize) {
    let edges = bucket_edges();
    let chunk = chunk_in_bucket(EDGE_BUCKET, 300);
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

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].points.len(), 8);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

#[test]
fn kept_equal_to_source_gives_zero_strength() {
    let chunk = chunk_at(2.0, 300, Some(2.0));
    let scene = scene_of(0, vec![chunk]);
    let scenes = vec![scene];

    let entries = fit_scenes(&scenes);

    assert!(entries[0].points.iter().all(|&(_, scale)| scale == 0));
}

#[test]
fn kept_above_source_clamps_to_zero() {
    let chunk = chunk_at(2.0, 300, Some(3.0));
    let scene = scene_of(0, vec![chunk]);
    let scenes = vec![scene];

    let entries = fit_scenes(&scenes);

    assert!(entries[0].points.iter().all(|&(_, scale)| scale == 0));
}

#[test]
fn a_thin_chunk_never_gets_its_own_entry() {
    let first_chunk = chunk_at(1.0, 300, None);
    let before_thin = chunk_at(3.0, 300, None);
    let thin = sparse_chunk();
    let after_thin = chunk_at(3.0, 300, None);
    let second_chunks = vec![before_thin, thin, after_thin];
    let first_scene = scene_of(0, vec![first_chunk]);
    let second_scene = scene_of(24, second_chunks);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].first_frame, 24);
    assert_eq!(entries[1].last_frame, 24 + 3 * CHUNK_FRAMES as u64 - 1);
}

#[test]
fn missing_texture_borrows_from_the_nearest_segment() {
    let mut no_texture = chunk_at(2.0, 300, None);
    no_texture.pixels.fill(0.0);
    let textured = chunk_at(2.0, 300, None);
    let first_scene = scene_of(0, vec![textured]);
    let second_scene = scene_of(24, vec![no_texture]);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].ar_coeffs, entries[0].ar_coeffs);
}

#[test]
fn sparse_scene_borrows_from_a_neighbouring_scene() {
    let dense = chunk_at(2.0, 300, None);
    let sparse = sparse_chunk();
    let first_scene = scene_of(0, vec![dense]);
    let second_scene = scene_of(24, vec![sparse]);
    let scenes = vec![first_scene, second_scene];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].points, entries[0].points);
}

#[test]
fn sparse_segment_with_no_donor_gets_no_entry() {
    let sparse = sparse_chunk();
    let scene = scene_of(0, vec![sparse]);
    let scenes = vec![scene];

    let entries = fit_scenes(&scenes);

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
    with_outlier.source_hist[5 * STD_BUCKETS + outlier_bucket] = 10;
    add_group_record(&mut with_outlier, outlier_bucket / BUCKETS_PER_GROUP, &coarse);

    let mut mixed = grain_only.clone();
    add_group_record(&mut mixed, grain_group, &coarse);

    let clean_scene = scene_of(0, vec![grain_only]);
    let outlier_scene = scene_of(0, vec![with_outlier]);
    let mixed_scene = scene_of(0, vec![mixed]);
    let clean_entries = fit_scenes(&[clean_scene]);
    let outlier_entries = fit_scenes(&[outlier_scene]);
    let mixed_entries = fit_scenes(&[mixed_scene]);
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
    let entries = fit_scenes(&[scene]);

    assert_eq!(entries.len(), 1);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

#[test]
fn a_group_past_the_band_gives_no_texture() {
    let (mut chunk, edge_group) = edge_chunk();
    let record = grain_record();
    add_group_record(&mut chunk, edge_group + 1, &record);

    let scene = scene_of(0, vec![chunk]);
    let entries = fit_scenes(&[scene]);

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

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 4);
    assert_eq!(entries[2].first_frame, 2 * CHUNK_FRAMES as u64);
    assert_eq!(entries[2].points, entries[0].points);
    assert_ne!(entries[2].points, entries[3].points);
}

#[test]
fn a_chunk_in_every_luma_bin_thins_to_the_point_limit() {
    let edges = bucket_edges();
    let bucket = bucket_of(2.0 / 255.0, &edges);
    let mut chunk = chunk_at(2.0, 300, None);
    for bin in 0..LUMA_BINS {
        chunk.source_hist[bin * STD_BUCKETS + bucket] = 300;
    }

    let scene = scene_of(0, vec![chunk]);
    let entries = fit_scenes(&[scene]);
    let points = &entries[0].points;
    let first_luma = points.first().expect("points").0;
    let last_luma = points.last().expect("points").0;

    assert_eq!(points.len(), MAX_POINTS);
    assert!(points.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert_eq!(first_luma, 8);
    assert_eq!(last_luma, 248);
}
