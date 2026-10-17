use super::noise_curve::stepped_curve;
use super::{Setup, chroma_cross_frame_setup, cross_frame_setup, run_fused_variant, unique_frame};
use crate::collab::kernels::fused::CollabVariant;
use crate::tune::SiteCandidate;
use crate::tune::tests::fixtures::{variant_chunk, variant_chunk_tests};

/// One labelled launch the variants are compared on.
type Case = (&'static str, Setup);

/// Asserts each of `variants` aggregated exactly what the baseline did, byte for byte.
///
/// Every variant offers the same candidates with the same arithmetic. A masked turn carries
/// [UNFILLED_DISTANCE](crate::collab::kernels::plane_ops::UNFILLED_DISTANCE), which cannot displace
/// a slot. A difference therefore means a variant let a dead position in or dropped a live one, not
/// that floating point drifted.
fn assert_listed_variants_agree(label: &str, setup: &Setup, variants: &[CollabVariant]) {
    let baseline = run_fused_variant(setup, CollabVariant::BASELINE);
    let aggregated = baseline.group_weight.iter().any(|weight| *weight > 0.0);

    assert!(
        aggregated,
        "{label}: the baseline aggregated nothing, so agreeing proves nothing"
    );

    for &variant in variants {
        if variant == CollabVariant::BASELINE {
            continue;
        }

        let got = run_fused_variant(setup, variant);
        let name = variant.name();

        assert_eq!(
            got.group_weight, baseline.group_weight,
            "{label} {name}: retired different groups"
        );
        assert_eq!(
            got.accum, baseline.accum,
            "{label} {name}: scattered different values"
        );
        assert_eq!(
            got.wsum, baseline.wsum,
            "{label} {name}: scattered different weights"
        );
    }
}

fn assert_case_agrees(case: Case, variants: &[CollabVariant]) {
    let (label, setup) = case;
    assert_listed_variants_agree(label, &setup, variants);
}

/// References along the left and top edges have a clipped rectangle narrower than the unclipped
/// span, which is where the uniform walk takes extra turns. Those turns must score nothing.
fn spatial_case() -> Case {
    let (width, height) = (48u32, 48u32);
    let frame = unique_frame(width, height);
    let setup = Setup::spatial_only(frame, width, height);

    ("spatial", setup)
}

/// [cross_frame_setup] pushes some refine windows off the frame and its confidences straddle `c_min`,
/// the two things that give groups sharing a warp different trip counts under the clipped walk.
fn cross_frame_case() -> Case {
    let setup = cross_frame_setup(64, 64, 2);

    ("cross frame", setup)
}

/// The uniform walk reads a gated motion block the clipped walk never touches, so the `seen_*` slot
/// it leaves must stay empty or it would hide positions a later covering block still owes the search.
fn all_gated_case() -> Case {
    let mut setup = cross_frame_setup(64, 64, 2);
    // Above every planted confidence, so every `seen_*` slot is one the uniform walk wrote.
    setup.c_min = 2.0;

    ("all gated", setup)
}

/// At radius 1 the grid is four volumes of two frames.
fn radius_one_case() -> Case {
    let setup = cross_frame_setup(64, 64, 1);

    ("radius one", setup)
}

fn noise_curve_case() -> Case {
    let mut setup = cross_frame_setup(64, 64, 2);
    let curve = stepped_curve();
    setup.noise_curve = Some(curve);

    ("noise curve", setup)
}

/// [cross_frame_setup] with every motion block of a neighbour carrying one vector, so the refine
/// rectangles covering an anchor coincide.
fn shared_vector_setup(radius: u32) -> Setup {
    let mut setup = cross_frame_setup(64, 64, radius);

    for t in 0..(2 * radius) {
        for block in 0..setup.conf_stride {
            let mv_index = (t * setup.mv_stride + block * 2) as usize;
            setup.mv_field[mv_index] = t as i32 - 1;
            setup.mv_field[mv_index + 1] = 1 - t as i32;
        }
    }

    setup
}

/// [cross_frame_setup] with vectors that push most refine windows past the right and bottom edges by
/// slightly different amounts, so clamping makes rectangles equal or nested.
fn edge_clamped_setup(radius: u32) -> Setup {
    let mut setup = cross_frame_setup(64, 64, radius);

    for t in 0..(2 * radius) {
        for block in 0..setup.conf_stride {
            let mv_index = (t * setup.mv_stride + block * 2) as usize;
            let overshoot_x = (block % 3) as i32;
            let overshoot_y = (block % 2) as i32;
            setup.mv_field[mv_index] = 40 + overshoot_x;
            setup.mv_field[mv_index + 1] = 40 + overshoot_y;
        }
    }

    setup
}

fn shared_vector_case() -> Case {
    let setup = shared_vector_setup(2);

    ("shared vector", setup)
}

fn shared_confident_case() -> Case {
    let mut setup = shared_vector_setup(2);
    setup.confidence.fill(1.0);

    ("shared vector, all confident", setup)
}

fn edge_clamped_case() -> Case {
    let mut setup = edge_clamped_setup(2);
    setup.confidence.fill(1.0);

    ("edge clamped", setup)
}

fn f16_cross_frame_case() -> Case {
    let mut setup = cross_frame_setup(64, 64, 2);
    setup.f16_search = true;

    ("f16 cross frame", setup)
}

fn f16_shared_vector_case() -> Case {
    let mut setup = shared_vector_setup(2);
    setup.f16_search = true;
    setup.confidence.fill(1.0);

    ("f16 shared vector", setup)
}

variant_chunk_tests! {
    assert_case_agrees(spatial_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_on_the_spatial_pass_in_the_first_half => 0,
        every_variant_matches_the_baseline_on_the_spatial_pass_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(cross_frame_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_across_frames_in_the_first_half => 0,
        every_variant_matches_the_baseline_across_frames_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(all_gated_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_when_every_neighbour_is_gated_in_the_first_half => 0,
        every_variant_matches_the_baseline_when_every_neighbour_is_gated_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(radius_one_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_at_radius_one_in_the_first_half => 0,
        every_variant_matches_the_baseline_at_radius_one_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(noise_curve_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_with_a_noise_curve_in_the_first_half => 0,
        every_variant_matches_the_baseline_with_a_noise_curve_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(shared_vector_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_when_covering_blocks_share_a_vector_in_the_first_half => 0,
        every_variant_matches_the_baseline_when_covering_blocks_share_a_vector_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(shared_confident_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_when_every_shared_block_is_confident_in_the_first_half => 0,
        every_variant_matches_the_baseline_when_every_shared_block_is_confident_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(edge_clamped_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_when_the_edge_clamps_rectangles_together_in_the_first_half => 0,
        every_variant_matches_the_baseline_when_the_edge_clamps_rectangles_together_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(f16_cross_frame_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_with_the_f16_search_across_frames_in_the_first_half => 0,
        every_variant_matches_the_baseline_with_the_f16_search_across_frames_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_agrees(f16_shared_vector_case()), 2 chunks,
    [
        every_variant_matches_the_baseline_with_the_f16_search_on_a_shared_vector_in_the_first_half => 0,
        every_variant_matches_the_baseline_with_the_f16_search_on_a_shared_vector_in_the_second_half => 1,
    ]
}

fn f16_wide_storage_setup() -> Setup {
    let mut setup = chroma_cross_frame_setup(64, 64, 2);
    setup.f16_search = true;

    setup
}

#[test]
fn every_variant_matches_the_baseline_with_the_f16_search_on_wide_storage_in_the_first_third() {
    let setup = f16_wide_storage_setup();
    let variants = variant_chunk(0, 3);

    assert_listed_variants_agree("f16 chroma cross frame", &setup, variants);
}

#[test]
fn every_variant_matches_the_baseline_with_the_f16_search_on_wide_storage_in_the_second_third() {
    let setup = f16_wide_storage_setup();
    let variants = variant_chunk(1, 3);

    assert_listed_variants_agree("f16 chroma cross frame", &setup, variants);
}

#[test]
fn every_variant_matches_the_baseline_with_the_f16_search_on_wide_storage_in_the_third_third() {
    let setup = f16_wide_storage_setup();
    let variants = variant_chunk(2, 3);

    assert_listed_variants_agree("f16 chroma cross frame", &setup, variants);
}
