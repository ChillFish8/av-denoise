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

#[test]
fn warp_uniform_search_matches_the_clipped_search_when_covering_blocks_share_a_vector() {
    let setup = shared_vector_setup(2);

    assert_walks_agree("shared vector", &setup);
}

#[test]
fn warp_uniform_search_matches_the_clipped_search_when_every_shared_block_is_confident() {
    let mut setup = shared_vector_setup(2);
    setup.confidence.fill(1.0);

    assert_walks_agree("shared vector, all confident", &setup);
}

#[test]
fn warp_uniform_search_matches_the_clipped_search_when_the_edge_clamps_rectangles_together() {
    let mut setup = edge_clamped_setup(2);
    setup.confidence.fill(1.0);

    assert_walks_agree("edge clamped", &setup);
}
