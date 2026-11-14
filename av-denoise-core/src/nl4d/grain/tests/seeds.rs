use std::collections::HashSet;

use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::seeds::{FGSE_SEEDS, SeedSigmas, fresh_seed};
use crate::nl4d::grain::template::{ChromaPlane, TemplatePlane, luma_template, seed_sigma};

/// Weight 71 on the left neighbour at shift 7, the rest 0.
fn left_neighbour_coeffs() -> [i32; AR_COEFFS] {
    let mut coeffs = [0; AR_COEFFS];
    coeffs[AR_COEFFS - 1] = 71;

    coeffs
}

/// The population std of the 64x64 luma interior starting at row and column 9.
fn interior_std(template: &[i32], cols: usize) -> f64 {
    let mut values = Vec::new();
    for row in 9..73 {
        for col in 9..73 {
            values.push(template[row * cols + col] as f64);
        }
    }

    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let squares: f64 = values.iter().map(|value| (value - mean) * (value - mean)).sum();

    (squares / count).sqrt()
}

#[test]
fn the_list_holds_135_distinct_seeds_without_zero() {
    let unique: HashSet<u16> = FGSE_SEEDS.iter().copied().collect();

    assert_eq!(unique.len(), 135);
    assert!(!unique.contains(&0));
}

#[test]
fn the_list_starts_in_the_fixed_shuffled_order() {
    assert_eq!(FGSE_SEEDS[..8], [4, 63504, 64435, 65501, 48, 59665, 36, 64240]);
}

#[test]
fn fresh_seeds_cycle_the_list_by_frame() {
    assert_eq!(fresh_seed(0), 4);
    assert_eq!(fresh_seed(1), 63504);
    assert_eq!(fresh_seed(135), 4);
    assert_eq!(fresh_seed(136), 63504);
}

#[test]
fn seed_sigma_is_the_std_of_that_seeds_luma_interior() {
    let coeffs = left_neighbour_coeffs();
    let template = luma_template(&coeffs, 7, 4);
    let expected = interior_std(&template, 82);
    let sigma = seed_sigma(&coeffs, 7, TemplatePlane::Luma, 4);

    assert!((sigma - expected).abs() < 1e-9);
}

#[test]
fn chroma_planes_read_different_templates_for_one_seed() {
    let coeffs = left_neighbour_coeffs();
    let cb = seed_sigma(&coeffs, 7, TemplatePlane::Chroma(ChromaPlane::Cb), 4);
    let cr = seed_sigma(&coeffs, 7, TemplatePlane::Chroma(ChromaPlane::Cr), 4);

    assert_ne!(cb, cr);
}

#[test]
fn the_cache_holds_one_value_per_weights_plane_and_seed() {
    let coeffs = left_neighbour_coeffs();
    let mut sigmas = SeedSigmas::new();
    let first = sigmas.sigma(&coeffs, 7, TemplatePlane::Luma, 4);
    let again = sigmas.sigma(&coeffs, 7, TemplatePlane::Luma, 4);
    sigmas.sigma(&coeffs, 7, TemplatePlane::Luma, 48);

    assert_eq!(first, again);
    assert_eq!(sigmas.len(), 2);
}
