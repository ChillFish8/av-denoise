use super::{
    assert_matches_recorded,
    cross_frame_setup,
    run_fused,
    three_frame_ring_with_a_planted_match,
    unique_frame,
    Digest,
    Setup,
};
use crate::collab::tests::helpers::noisy_field_over;

/// Content without ties, so nothing about the result depends on how the
/// insert breaks one.
///
/// `make_unique_frame` is built so that any two distinct 8x8 windows
/// differ in most of their 64 pixels.
#[test]
fn fused_reproduces_recorded_output_on_unique_content() {
    let (w, h) = (128u32, 96u32);
    let s = Setup::spatial_only(unique_frame(w, h), w, h);
    assert_matches_recorded(
        "unique content",
        &run_fused(&s),
        &Digest {
            covered: 12288,
            pixel_mean: 0.500102660422,
            pixel_rms: 0.577471243275,
            weight_mean: 1250.000000000,
            probes: [
                0.917905456141,
                0.787302672863,
                0.650475382805,
                0.517024146186,
                0.386674649788,
                0.255359411240,
                0.120508321126,
                0.989773918601,
            ],
        },
    );
}

/// The same content with its ramp turned on its side.
///
/// `make_unique_frame` ramps along x, which makes a one-column shift far
/// costlier than a one-row shift, so every member a group keeps sits in
/// a narrow column band and the search rectangle's x extent never
/// decides anything. Transposing the frame moves that band onto the x
/// axis, so this is the run where the horizontal bounds are
/// load-bearing.
#[test]
fn fused_reproduces_recorded_output_on_a_transposed_ramp() {
    let (w, h) = (128u32, 96u32);
    let source = unique_frame(h, w);
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            frame[(y * w + x) as usize] = source[(x * h + y) as usize];
        }
    }
    let s = Setup::spatial_only(frame, w, h);
    assert_matches_recorded(
        "transposed ramp",
        &run_fused(&s),
        &Digest {
            covered: 12288,
            pixel_mean: 0.500026936557,
            pixel_rms: 0.577366054447,
            weight_mean: 1238.896681776,
            probes: [
                0.075242505755,
                0.724634047477,
                0.367116374354,
                0.014184951782,
                0.659262769363,
                0.307459000618,
                0.951007338131,
                0.589665272066,
            ],
        },
    );
}

/// A width whose reference count is not a multiple of 8 leaves the last
/// cube of each row partly out of range.
///
/// 104 pixels wide gives 25 reference patches at `STEP = 4`, so the
/// fourth cube runs one live group and seven dead ones. Those seven must
/// reach every barrier and write nothing. A dead group that scattered
/// would double the last reference's contribution, which shows up here
/// as a moved pixel rather than needing its own assertion.
#[test]
fn fused_reproduces_recorded_output_when_refs_are_not_a_multiple_of_eight() {
    let (w, h) = (104u32, 96u32);
    let s = Setup::spatial_only(unique_frame(w, h), w, h);
    assert_matches_recorded(
        "ragged reference row",
        &run_fused(&s),
        &Digest {
            covered: 9984,
            pixel_mean: 0.500121022858,
            pixel_rms: 0.577469311374,
            weight_mean: 1240.579711065,
            probes: [
                0.745903455294,
                0.891958951950,
                0.032306798299,
                0.177212221869,
                0.322843606131,
                0.468612211367,
                0.608005691977,
                0.754309082031,
            ],
        },
    );
}

/// The stack transform's shorter ladders, on a search space too small to
/// fill a group.
///
/// At `spatial_radius = 1` a corner reference sees a 2x2 rectangle, so
/// four candidates and a group of four, an edge reference sees 2x3 and
/// also keeps four, and an interior one sees 3x3 and fills to eight.
/// Every wider configuration fills every group to eight, so this is the
/// run where the 2- and 4-member ladders execute at all.
#[test]
fn fused_reproduces_recorded_output_on_a_short_search_space() {
    let (w, h) = (64u32, 64u32);
    let mut s = Setup::spatial_only(unique_frame(w, h), w, h);
    s.spatial_radius = 1;
    assert_matches_recorded(
        "short search space",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500333883408,
            pixel_rms: 0.577536371630,
            weight_mean: 768.518540988,
            probes: [
                0.836218530965,
                0.571090123027,
                0.304096429037,
                0.033846737369,
                0.774293684286,
                0.508250150663,
                0.242787978384,
                0.977499961853,
            ],
        },
    );
}

/// The tie-break path, on content where a great many candidates score
/// the same distance.
///
/// Noise over a flat field has no ramp to separate the candidates, so
/// this is the run where the self-match sentinel and the first-wins
/// insert decide the member set.
#[test]
fn fused_reproduces_recorded_output_on_noise() {
    let (w, h) = (64u32, 64u32);
    let s = Setup::spatial_only(noisy_field_over(w, h, 0.5, 0.05), w, h);
    assert_matches_recorded(
        "noise",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500598531425,
            pixel_rms: 0.500781913699,
            weight_mean: 168.165750156,
            probes: [
                0.473047106911,
                0.525827771943,
                0.505852930189,
                0.501965226326,
                0.472216666744,
                0.498205827272,
                0.493595121410,
                0.515348414403,
            ],
        },
    );
}

/// A non-zero `rho`, where the correlation profile stops being all ones.
///
/// The old filter multiplied the profile into each member's variance
/// before the variance ladder ran. The fused kernel multiplies it in at
/// the threshold instead. The ladder only averages and the profile is a
/// constant factor across the stack axis, so the two orders agree in
/// exact arithmetic, and this is the run that says so on a GPU. Every
/// other run here uses `dct_noise_profile(0.0)`, which is all ones and
/// cannot tell the two orders apart. `0.86` is the shipped table's high
/// end.
#[test]
fn fused_reproduces_recorded_output_under_correlation_shaping() {
    let (w, h) = (64u32, 64u32);
    let mut s = Setup::spatial_only(noisy_field_over(w, h, 0.5, 0.05), w, h);
    s.rho = 0.86;
    assert_matches_recorded(
        "correlation shaping",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500501375321,
            pixel_rms: 0.502136468679,
            weight_mean: 54.170715162,
            probes: [
                0.439152209001,
                0.572103197408,
                0.475816598569,
                0.509686441252,
                0.405882571403,
                0.498373582524,
                0.493889111273,
                0.547764034977,
            ],
        },
    );
}

/// The whole temporal path at once: the `c_min` skip, the per-member
/// mismatch variance derived from the member's own match distance, and
/// the scatter into each member's own region of the accumulator ring.
///
/// Recorded with the covering-block search, every block covering a
/// patch contributes a rectangle. `cross_frame_setup` gives every block
/// its own vector, so the search reaches positions the corner block
/// alone never pointed at.
///
/// Re-recorded for the switch from motion-block confidence to a
/// member's own match distance. The digest below comes from this
/// kernel's own output, not a second implementation, because none
/// exists for the new mechanism. [assert_matches_recorded]'s warning
/// about comparing a kernel to itself is about a silently-broken shader
/// producing zeros, and this recording carries real, non-zero coverage.
#[test]
fn fused_reproduces_recorded_output_across_frames() {
    let s = cross_frame_setup(64, 64, 2);
    assert_matches_recorded(
        "cross frame",
        &run_fused(&s),
        &Digest {
            covered: 15800,
            pixel_mean: 0.398917931934,
            pixel_rms: 0.518592139022,
            weight_mean: 1199.919938422,
            probes: [
                0.838003113388,
                0.574141517596,
                0.299827186817,
                0.000000000000,
                0.774458945874,
                0.000000000000,
                0.236727453142,
                0.979726340630,
            ],
        },
    );
}

/// The same cross-frame run with the mismatch variance off.
///
/// `use_member_sigma` is a `#[comptime]` flag, so it compiles a second
/// program, and the arm with it off is the one that checks the threshold
/// still reads a plain `sigma^2` per member.
///
/// Recorded with the covering-block search, every block covering a
/// patch contributes a rectangle.
#[test]
fn fused_reproduces_recorded_output_without_the_mismatch_variance() {
    let mut s = cross_frame_setup(64, 64, 2);
    s.confidence_variance = false;
    assert_matches_recorded(
        "cross frame, flat sigma",
        &run_fused(&s),
        &Digest {
            covered: 15800,
            pixel_mean: 0.398918693763,
            pixel_rms: 0.518592557620,
            weight_mean: 1244.444444987,
            probes: [
                0.838030815125,
                0.574148050944,
                0.299845377604,
                0.000000000000,
                0.774438040597,
                0.000000000000,
                0.236724853516,
                0.979728698730,
            ],
        },
    );
}

/// A group with members in neighbour frames must scatter into those
/// frames' regions of the ring, not collapse onto the centre frame.
///
/// This is the cross-frame aggregation the temporal path exists for, and
/// it is easy to lose, because the frame a member came from is never
/// written down anywhere between the match and the scatter.
#[test]
fn fused_scatters_into_every_member_frame() {
    let (w, h) = (64u32, 64u32);
    let s = three_frame_ring_with_a_planted_match(w, h);
    let got = run_fused(&s);
    for slot in 0..3 {
        assert!(
            got.frame_weight_sum(slot) > 0,
            "ring slot {slot} received nothing"
        );
    }
    assert_matches_recorded(
        "planted cross-frame match",
        &got,
        &Digest {
            covered: 12288,
            pixel_mean: 0.500274434257,
            pixel_rms: 0.577655447440,
            weight_mean: 1246.296296658,
            probes: [
                0.836406707764,
                0.573966026306,
                0.301191602434,
                0.036788940430,
                0.774992261614,
                0.509891510010,
                0.239036560059,
                0.979254982688,
            ],
        },
    );
}
