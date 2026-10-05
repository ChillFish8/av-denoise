use crate::nlmeans::dispatch::{
    BILATERAL_RESIDUAL_FRACTION,
    NLM_SPATIAL_RESIDUAL_FRACTION,
    mc_sad_noise_floor_sigma,
};
use crate::nlmeans::motion::neighbour_idx_for_k;
use crate::nlmeans::prefilter::PrefilterMode;

#[test]
fn matches_the_sequential_fill_order() {
    for radius in 1..=8u32 {
        let mut expected = 0u32;
        for k in -(radius as i32)..=(radius as i32) {
            if k == 0 {
                continue;
            }

            let idx = neighbour_idx_for_k(radius, k);
            assert_eq!(idx, expected, "radius={radius} k={k}");
            expected += 1;
        }
    }
}

/// Distinct indices keep one frame's confidence off another frame's temporal weight.
#[test]
fn forward_and_backward_indices_are_distinct_and_in_range() {
    for radius in 1..=8u32 {
        for q_k in -(radius as i32)..0 {
            let forward = neighbour_idx_for_k(radius, q_k);
            let backward = neighbour_idx_for_k(radius, -q_k);
            assert_ne!(forward, backward, "radius={radius} q_k={q_k}");
            assert!(forward < 2 * radius, "radius={radius} q_k={q_k} fwd={forward}");
            assert!(backward < 2 * radius, "radius={radius} q_k={q_k} bwd={backward}");
        }
    }
}

#[test]
fn radius_two_explicit_indices() {
    let back_two = neighbour_idx_for_k(2, -2);
    let back_one = neighbour_idx_for_k(2, -1);
    let forward_one = neighbour_idx_for_k(2, 1);
    let forward_two = neighbour_idx_for_k(2, 2);
    assert_eq!(back_two, 0);
    assert_eq!(back_one, 1);
    assert_eq!(forward_one, 2);
    assert_eq!(forward_two, 3);
}

/// A literal, so a recalibration fails here instead of passing against itself.
#[test]
fn nlm_spatial_residual_fraction_is_calibrated_to_zero() {
    assert_eq!(NLM_SPATIAL_RESIDUAL_FRACTION, 0.0);
}

#[test]
fn bilateral_residual_fraction_is_calibrated_to_zero() {
    assert_eq!(BILATERAL_RESIDUAL_FRACTION, 0.0);
}

#[test]
fn mc_sad_noise_floor_sigma_scales_nlm_spatial_by_the_calibrated_fraction() {
    let raw = 0.02f32;
    let prefilter = PrefilterMode::NlmSpatial { strength_scale: 1.0 };
    let floor = mc_sad_noise_floor_sigma(prefilter, raw);
    assert_eq!(floor, raw * NLM_SPATIAL_RESIDUAL_FRACTION);
}

#[test]
fn mc_sad_noise_floor_sigma_scales_bilateral_by_the_calibrated_fraction() {
    let raw = 0.02f32;
    let prefilter = PrefilterMode::Bilateral {
        sigma_s: 3.0,
        sigma_r: 0.02,
    };
    let floor = mc_sad_noise_floor_sigma(prefilter, raw);
    assert_eq!(floor, raw * BILATERAL_RESIDUAL_FRACTION);
}

#[test]
fn mc_sad_noise_floor_sigma_keeps_raw_sigma_for_none() {
    let raw = 0.02f32;
    let floor = mc_sad_noise_floor_sigma(PrefilterMode::None, raw);
    assert_eq!(floor, raw);
}
