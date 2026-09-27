use super::noise_curve::stepped_curve;
use super::{Setup, cross_frame_setup, run_fused_walk, unique_frame};

/// Asserts the two search walks aggregated the same thing, byte for
/// byte.
///
/// Exact equality is the right bar rather than a tolerance. The
/// warp-uniform walk offers the same candidates, in the same order, and
/// scores them with the same arithmetic. Its extra turns carry the
/// `3.0e38` an unfilled slot already holds, which cannot displace a
/// slot, so they change nothing about the group that is retired. A
/// difference here means the masking let a dead position into a group
/// or dropped a live one, not that floating point drifted.
fn assert_walks_agree(label: &str, s: &Setup) {
    let clipped = run_fused_walk(s, Some(false));
    let uniform = run_fused_walk(s, Some(true));

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
        uniform.group_weight.iter().any(|w| *w > 0.0),
        "{label}: neither walk aggregated anything, so agreeing proves nothing",
    );
}

/// The warp-uniform walk is only correct if it is a pure change of
/// schedule, so this pins it against the walk it replaces on the spatial
/// search alone.
///
/// The references along the left and top edges are the ones whose
/// clipped rectangle is narrower than the unclipped span, which is
/// exactly where the uniform walk takes turns the other one does not.
/// Those turns have to score nothing.
#[test]
fn warp_uniform_search_matches_the_clipped_search_on_the_spatial_pass() {
    let (w, h) = (48u32, 48u32);
    let s = Setup::spatial_only(unique_frame(w, h), w, h);

    assert_walks_agree("spatial", &s);
}

/// The same equivalence across the temporal search, which is where the
/// two walks diverge most.
///
/// [cross_frame_setup] is built for this: its motion vectors push some
/// refine windows off the frame so the clip matters, and its confidence
/// field straddles `c_min`, so blocks that one group scores its
/// neighbour skips. Under the clipped walk those are the two things that
/// give groups sharing a warp different trip counts, and under the
/// uniform walk they have to fold into the mask instead without moving
/// a single member.
#[test]
fn warp_uniform_search_matches_the_clipped_search_across_frames() {
    let s = cross_frame_setup(64, 64, 2);

    assert_walks_agree("cross frame", &s);
}

/// A gated neighbour is the one case where the uniform walk reads a
/// motion block the clipped walk never touches, so the `seen_*` slot it
/// leaves behind has to stay empty or it would hide positions a later
/// covering block still owes the search.
#[test]
fn warp_uniform_search_matches_the_clipped_search_when_every_neighbour_is_gated() {
    let mut s = cross_frame_setup(64, 64, 2);
    // Above every confidence the setup plants, so no temporal block ever
    // scores and every `seen_*` slot is one the uniform walk wrote.
    s.c_min = 2.0;

    assert_walks_agree("all gated", &s);
}

/// The same equivalence at radius 1, where the grid is four volumes of two frames.
#[test]
fn warp_uniform_search_matches_the_clipped_search_at_radius_one() {
    let s = cross_frame_setup(64, 64, 1);

    assert_walks_agree("radius one", &s);
}

#[test]
fn warp_uniform_search_matches_the_clipped_search_with_a_noise_curve() {
    let mut s = cross_frame_setup(64, 64, 2);
    s.noise_curve = Some(stepped_curve());

    assert_walks_agree("noise curve", &s);
}
