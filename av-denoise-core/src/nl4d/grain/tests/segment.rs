use super::synthetic::chunk_at;
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::CHUNK_FRAMES;
use crate::nl4d::grain::segment::{SceneGrain, fit_scenes, segment_scene};

fn sparse_chunk() -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;
    chunk
}

fn scene_of(first_frame: u64, chunks: Vec<GrainChunk>) -> SceneGrain {
    SceneGrain { first_frame, chunks }
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
    no_texture.pixels = 0.0;
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
