use cubecl::prelude::*;

use crate::motion::global::{
    AFFINE_WORDS,
    MODEL_WORDS,
    ModelRing,
    NO_PAIR,
    compose_gaps,
    run_compose_models,
};
use crate::tune::gpu::read_f32;
use crate::tune::tests::fixtures::upload;
use crate::tune::tests::runtime::test_client;

const RING_DEPTH: u32 = 8;
const RADIUS: usize = 2;
const NEIGHBOURS: usize = 2 * RADIUS;
const TOLERANCE: f64 = 1e-4;

/// A pair model as `a, b, tx, ty` in luma pixels.
#[derive(Debug, Clone, Copy)]
struct Similarity {
    a: f64,
    b: f64,
    tx: f64,
    ty: f64,
}

impl Similarity {
    fn new(degrees: f64, zoom: f64, tx: f64, ty: f64) -> Self {
        let radians = degrees.to_radians();
        Self {
            a: zoom * radians.cos(),
            b: zoom * radians.sin(),
            tx,
            ty,
        }
    }
}

/// A 2x2 matrix and translation, `m00, m01, m10, m11, tx, ty`.
type Affine = [f64; 6];

const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn as_affine(model: Similarity) -> Affine {
    [model.a, -model.b, model.b, model.a, model.tx, model.ty]
}

/// `outer` applied after `inner`.
fn then(inner: Affine, outer: Affine) -> Affine {
    let m00 = outer[0] * inner[0] + outer[1] * inner[2];
    let m01 = outer[0] * inner[1] + outer[1] * inner[3];
    let m10 = outer[2] * inner[0] + outer[3] * inner[2];
    let m11 = outer[2] * inner[1] + outer[3] * inner[3];
    let tx = outer[0] * inner[4] + outer[1] * inner[5] + outer[4];
    let ty = outer[2] * inner[4] + outer[3] * inner[5] + outer[5];

    [m00, m01, m10, m11, tx, ty]
}

fn inverse(affine: Affine) -> Affine {
    let det = affine[0] * affine[3] - affine[1] * affine[2];
    let m00 = affine[3] / det;
    let m01 = -affine[1] / det;
    let m10 = -affine[2] / det;
    let m11 = affine[0] / det;
    let tx = -(m00 * affine[4] + m01 * affine[5]);
    let ty = -(m10 * affine[4] + m11 * affine[5]);

    [m00, m01, m10, m11, tx, ty]
}

/// Rewrites a luma affine in pass pixels, where `x_luma = x_pass * scale` per axis.
fn in_pass_pixels(affine: Affine, scale_x: f64, scale_y: f64) -> Affine {
    [
        affine[0],
        affine[1] * scale_y / scale_x,
        affine[2] * scale_x / scale_y,
        affine[3],
        affine[4] / scale_x,
        affine[5] / scale_y,
    ]
}

/// A ring whose slot for each listed stream frame holds that pair model.
fn ring_with(client: &Client, pairs: &[(u32, Similarity, bool)]) -> ModelRing {
    let mut words = vec![0.0f32; (RING_DEPTH * MODEL_WORDS) as usize];
    for &(frame, model, valid) in pairs {
        let base = ((frame % RING_DEPTH) * MODEL_WORDS) as usize;
        words[base] = model.a as f32;
        words[base + 1] = model.b as f32;
        words[base + 2] = model.tx as f32;
        words[base + 3] = model.ty as f32;
        words[base + 4] = if valid { 1.0 } else { 0.0 };
    }

    let handle = upload(client, &words);
    ModelRing {
        handle,
        depth: RING_DEPTH,
    }
}

/// Composes every neighbour of the window `slot_frames`, centred on logical slot `RADIUS`.
fn compose(
    client: &Client,
    ring: &ModelRing,
    slot_frames: &[u32],
    scale_x: f32,
    scale_y: f32,
) -> Vec<Affine> {
    let gaps = compose_gaps(slot_frames, RADIUS, RADIUS, ring);
    let gap_bytes = u32::as_bytes(&gaps);
    let gaps_handle = client.create_from_slice(gap_bytes);
    let out_len = NEIGHBOURS * AFFINE_WORDS as usize;
    let out = client.empty(out_len * size_of::<f32>());

    run_compose_models(
        client,
        ring,
        &gaps_handle,
        &out,
        NEIGHBOURS as u32,
        RADIUS as u32,
        scale_x,
        scale_y,
    );

    let words = read_f32(client, &out, out_len).expect("composed predictions read back");
    let (predictions, _) = words.as_chunks::<{ AFFINE_WORDS as usize }>();
    predictions
        .iter()
        .map(|prediction| prediction.map(f64::from))
        .collect()
}

fn assert_affine_close(actual: Affine, expected: Affine, label: &str) {
    for word in 0..6 {
        let difference = (actual[word] - expected[word]).abs();
        assert!(
            difference < TOLERANCE,
            "{label}: word {word} is {}, expected {} (all {actual:?} vs {expected:?})",
            actual[word],
            expected[word],
        );
    }
}

fn window_pairs() -> [(u32, Similarity, bool); 4] {
    [
        (9, Similarity::new(2.0, 1.02, 3.0, 1.5), true),
        (10, Similarity::new(-4.0, 0.98, -6.0, 2.0), true),
        (11, Similarity::new(3.0, 1.01, 5.0, -2.0), true),
        (12, Similarity::new(-1.5, 0.99, -3.0, 4.0), true),
    ]
}

#[test]
fn composed_forward_matches_the_product_of_pair_models() {
    let client = test_client();
    let pairs = window_pairs();
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[8, 9, 10, 11, 12], 1.0, 1.0);

    let one_step = as_affine(pairs[2].1);
    let second_step = as_affine(pairs[3].1);
    let two_steps = then(one_step, second_step);

    assert_affine_close(predictions[2], one_step, "+1");
    assert_affine_close(predictions[3], two_steps, "+2");
}

#[test]
fn composed_past_neighbour_is_the_inverse_product() {
    let client = test_client();
    let pairs = window_pairs();
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[8, 9, 10, 11, 12], 1.0, 1.0);

    let older_pair = as_affine(pairs[0].1);
    let newer_pair = as_affine(pairs[1].1);
    let one_step = inverse(newer_pair);
    let older_to_centre = then(older_pair, newer_pair);
    let two_steps = inverse(older_to_centre);

    assert_affine_close(predictions[1], one_step, "-1");
    assert_affine_close(predictions[0], two_steps, "-2");
}

#[test]
fn no_pair_gaps_compose_to_identity() {
    let client = test_client();
    let pairs = window_pairs();
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[10, 10, 10, 10, 10], 1.0, 1.0);

    for (neighbour, prediction) in predictions.into_iter().enumerate() {
        let label = format!("neighbour {neighbour}");
        assert_affine_close(prediction, IDENTITY, &label);
    }
}

#[test]
fn invalid_models_compose_to_identity() {
    let client = test_client();
    let pairs = window_pairs().map(|(frame, model, _)| (frame, model, false));
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[8, 9, 10, 11, 12], 1.0, 1.0);

    for (neighbour, prediction) in predictions.into_iter().enumerate() {
        let label = format!("neighbour {neighbour}");
        assert_affine_close(prediction, IDENTITY, &label);
    }
}

#[test]
fn chroma_scale_at_422_gives_the_anisotropic_affine() {
    let client = test_client();
    let pairs = window_pairs();
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[8, 9, 10, 11, 12], 2.0, 1.0);

    let first_step = as_affine(pairs[2].1);
    let second_step = as_affine(pairs[3].1);
    let forward = then(first_step, second_step);
    let past_pair = as_affine(pairs[1].1);
    let past = inverse(past_pair);
    let forward_expected = in_pass_pixels(forward, 2.0, 1.0);
    let past_expected = in_pass_pixels(past, 2.0, 1.0);

    assert_affine_close(predictions[3], forward_expected, "+2 at 4:2:2");
    assert_affine_close(predictions[1], past_expected, "-1 at 4:2:2");
    assert!((predictions[3][1] - predictions[3][2]).abs() > 1e-3);
}

#[test]
fn chroma_scale_at_420_halves_the_shift() {
    let client = test_client();
    let pairs = window_pairs();
    let ring = ring_with(&client, &pairs);
    let predictions = compose(&client, &ring, &[8, 9, 10, 11, 12], 2.0, 2.0);

    let forward = as_affine(pairs[2].1);
    let forward_expected = in_pass_pixels(forward, 2.0, 2.0);

    assert_affine_close(predictions[2], forward_expected, "+1 at 4:2:0");
}

#[test]
fn compose_gaps_skips_duplicated_frames() {
    let client = test_client();
    let ring = ring_with(&client, &[]);
    let gaps = compose_gaps(&[0, 0, 0, 1, 2], RADIUS, RADIUS, &ring);

    let expected = vec![
        NO_PAIR, NO_PAIR, // -2
        NO_PAIR, NO_PAIR, // -1
        1, NO_PAIR, // +1
        1, 2, // +2
    ];
    assert_eq!(gaps, expected);
}

#[test]
fn compose_gaps_lists_past_gaps_from_the_centre_outward() {
    let client = test_client();
    let ring = ring_with(&client, &[]);
    let gaps = compose_gaps(&[13, 14, 15, 16, 16], RADIUS, RADIUS, &ring);

    let ends_at_14 = 14 % RING_DEPTH;
    let ends_at_15 = 15 % RING_DEPTH;
    let ends_at_16 = 16 % RING_DEPTH;
    let expected = vec![
        ends_at_15, ends_at_14, // -2
        ends_at_15, NO_PAIR, // -1
        ends_at_16, NO_PAIR, // +1
        ends_at_16, NO_PAIR, // +2
    ];
    assert_eq!(gaps, expected);
}

#[test]
fn compose_gaps_wraps_physical_slots() {
    let client = test_client();
    let ring = ring_with(&client, &[]);
    let logical = compose_gaps(&[8, 9, 10, 11, 12], RADIUS, RADIUS, &ring);
    let physical = compose_gaps(&[11, 12, 8, 9, 10], 4, RADIUS, &ring);

    assert_eq!(physical, logical);
}
