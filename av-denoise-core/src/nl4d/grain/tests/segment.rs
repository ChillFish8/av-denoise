use super::synthetic::{
    add_group_record,
    autocov_of,
    chunk_at,
    chunk_in_bucket,
    gaussian_field,
    grain_record,
};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{BUCKETS_PER_GROUP, CHUNK_FRAMES, STD_BUCKETS};
use crate::nl4d::grain::fit::{bucket_edges, bucket_of, hist_median};
use crate::nl4d::grain::segment::{SceneGrain, fit_scenes, segment_scene};

/// A source bucket whose median doubled lands in the first bucket of a strength group.
const EDGE_BUCKET: usize = 37;

fn sparse_chunk() -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;
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
    let chunks = vec![
        chunk_at(2.0, 300, None),
        chunk_at(2.05, 300, None),
        chunk_at(1.98, 300, None),
    ];
    let scene = scene_of(100, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].first_frame, 100);
    assert_eq!(segments[0].last_frame, 100 + 3 * CHUNK_FRAMES as u64 - 1);
}

#[test]
fn drift_splits_on_a_chunk_boundary() {
    let chunks = vec![
        chunk_at(2.0, 300, None),
        chunk_at(2.0, 300, None),
        chunk_at(2.8, 300, None),
    ];
    let scene = scene_of(0, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[1].first_frame, 2 * CHUNK_FRAMES as u64);
}

#[test]
fn a_thin_chunk_joins_the_open_segment() {
    let chunks = vec![chunk_at(2.0, 300, None), chunk_at(4.0, 10, None)];
    let scene = scene_of(0, chunks);

    let segments = segment_scene(&scene);

    assert_eq!(segments.len(), 1);
}

#[test]
fn a_fitted_entry_has_points_and_weights() {
    let scenes = vec![scene_of(0, vec![chunk_at(2.0, 300, None)])];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].points.len(), 8);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

#[test]
fn kept_equal_to_source_gives_zero_strength() {
    let scenes = vec![scene_of(0, vec![chunk_at(2.0, 300, Some(2.0))])];

    let entries = fit_scenes(&scenes);

    assert!(entries[0].points.iter().all(|&(_, scale)| scale == 0));
}

#[test]
fn kept_above_source_clamps_to_zero() {
    let scenes = vec![scene_of(0, vec![chunk_at(2.0, 300, Some(3.0))])];

    let entries = fit_scenes(&scenes);

    assert!(entries[0].points.iter().all(|&(_, scale)| scale == 0));
}

#[test]
fn a_thin_chunk_never_gets_its_own_entry() {
    let second_chunks = vec![chunk_at(3.0, 300, None), sparse_chunk(), chunk_at(3.0, 300, None)];
    let scenes = vec![
        scene_of(0, vec![chunk_at(1.0, 300, None)]),
        scene_of(24, second_chunks),
    ];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].first_frame, 24);
    assert_eq!(entries[1].last_frame, 24 + 3 * CHUNK_FRAMES as u64 - 1);
}

#[test]
fn missing_texture_borrows_from_the_nearest_segment() {
    let mut no_texture = chunk_at(2.0, 300, None);
    no_texture.pixels.fill(0.0);
    let scenes = vec![
        scene_of(0, vec![chunk_at(2.0, 300, None)]),
        scene_of(24, vec![no_texture]),
    ];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].ar_coeffs, entries[0].ar_coeffs);
}

#[test]
fn sparse_scene_borrows_from_a_neighbouring_scene() {
    let scenes = vec![
        scene_of(0, vec![chunk_at(2.0, 300, None)]),
        scene_of(24, vec![sparse_chunk()]),
    ];

    let entries = fit_scenes(&scenes);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].points, entries[0].points);
}

#[test]
fn sparse_segment_with_no_donor_gets_no_entry() {
    let scenes = vec![scene_of(0, vec![sparse_chunk()])];

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

    let clean_entries = fit_scenes(&[scene_of(0, vec![grain_only])]);
    let outlier_entries = fit_scenes(&[scene_of(0, vec![with_outlier])]);
    let mixed_entries = fit_scenes(&[scene_of(0, vec![mixed])]);

    assert!(coarse[0] > 10.0 * grain_record()[0]);
    assert!(clean_entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
    assert_eq!(outlier_entries[0].ar_coeffs, clean_entries[0].ar_coeffs);
    assert_ne!(mixed_entries[0].ar_coeffs, clean_entries[0].ar_coeffs);
}

#[test]
fn a_group_overlapping_the_band_edge_gives_the_texture() {
    let (mut chunk, edge_group) = edge_chunk();
    let record = grain_record();
    add_group_record(&mut chunk, edge_group, &record);

    let entries = fit_scenes(&[scene_of(0, vec![chunk])]);

    assert_eq!(entries.len(), 1);
    assert!(entries[0].ar_coeffs.iter().any(|&coeff| coeff != 0));
}

#[test]
fn a_group_past_the_band_gives_no_texture() {
    let (mut chunk, edge_group) = edge_chunk();
    let record = grain_record();
    add_group_record(&mut chunk, edge_group + 1, &record);

    let entries = fit_scenes(&[scene_of(0, vec![chunk])]);

    assert!(entries.is_empty());
}
