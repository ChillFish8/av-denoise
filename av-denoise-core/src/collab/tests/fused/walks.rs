use super::noise_curve::stepped_curve;
use super::{Setup, cross_frame_setup, run_fused_walk, unique_frame};

/// Asserts the two search walks aggregated the same thing, byte for byte.
///
/// The warp-uniform walk offers the same candidates in the same order with the same arithmetic. Its
/// extra turns carry the `3.0e38` an unfilled slot already holds, which cannot displace a slot. A
/// difference therefore means the masking let a dead position in or dropped a live one, not that
/// floating point drifted.
fn assert_walks_agree(label: &str, setup: &Setup) {
    let clipped = run_fused_walk(setup, Some(false));
    let uniform = run_fused_walk(setup, Some(true));

    assert_eq!(
        clipped.group_weight, uniform.group_weight,
        "{label}: the two search walks retired different groups",
    );
    assert_eq!(
        clipped.accum, uniform.accum,
        "{label}: the two search walks scattered different values",
    );
    assert_eq!(
        clipped.wsum, uniform.wsum,
        "{label}: the two search walks scattered different weights",
    );
    assert!(
        uniform.group_weight.iter().any(|weight| *weight > 0.0),
        "{label}: neither walk aggregated anything, so agreeing proves nothing",
    );
}

/// References along the left and top edges have a clipped rectangle narrower than the unclipped
/// span, which is where the uniform walk takes extra turns. Those turns must score nothing.
#[test]
fn warp_uniform_search_matches_the_clipped_search_on_the_spatial_pass() {
    let (width, height) = (48u32, 48u32);
    let frame = unique_frame(width, height);
    let setup = Setup::spatial_only(frame, width, height);

    assert_walks_agree("spatial", &setup);
}

/// [cross_frame_setup] pushes some refine windows off the frame and its confidences straddle `c_min`,
/// the two things that give groups sharing a warp different trip counts under the clipped walk.
#[test]
fn warp_uniform_search_matches_the_clipped_search_across_frames() {
    let setup = cross_frame_setup(64, 64, 2);

    assert_walks_agree("cross frame", &setup);
}

/// The uniform walk reads a gated motion block the clipped walk never touches, so the `seen_*` slot
/// it leaves must stay empty or it would hide positions a later covering block still owes the search.
#[test]
fn warp_uniform_search_matches_the_clipped_search_when_every_neighbour_is_gated() {
    let mut setup = cross_frame_setup(64, 64, 2);
    // Above every planted confidence, so every `seen_*` slot is one the uniform walk wrote.
    setup.c_min = 2.0;

    assert_walks_agree("all gated", &setup);
}

/// At radius 1 the grid is four volumes of two frames.
#[test]
fn warp_uniform_search_matches_the_clipped_search_at_radius_one() {
    let setup = cross_frame_setup(64, 64, 1);

    assert_walks_agree("radius one", &setup);
}

#[test]
fn warp_uniform_search_matches_the_clipped_search_with_a_noise_curve() {
    let mut setup = cross_frame_setup(64, 64, 2);
    let curve = stepped_curve();
    setup.noise_curve = Some(curve);

    assert_walks_agree("noise curve", &setup);
}
