use super::{
    Digest,
    Setup,
    assert_listed_variants_match_recorded,
    assert_matches_recorded,
    cross_frame_setup,
    five_frame_ring_with_jittered_copies,
    run_fused,
    unique_frame,
};
use crate::collab::kernels::fused::CollabVariant;
use crate::collab::tests::helpers::noisy_flat_field;
use crate::tune::tests::fixtures::variant_chunk_tests;

/// One labelled launch with the digest every variant must reproduce.
type RecordedCase = (&'static str, Setup, Digest);

fn assert_case_matches_recorded(case: RecordedCase, variants: &[CollabVariant]) {
    let (label, setup, want) = case;
    assert_listed_variants_match_recorded(label, &setup, &want, variants);
}

/// Content without ties, so nothing about the result depends on how the insert breaks one.
fn on_unique_content_case() -> RecordedCase {
    let (width, height) = (128u32, 96u32);
    let frame = unique_frame(width, height);
    let setup = Setup::spatial_only(frame, width, height);
    let want = Digest {
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
    };

    ("unique content", setup, want)
}

/// The unique frame ramps along x, so a one-column shift costs far more than a one-row shift and
/// the search rectangle's x extent never decides anything. Transposing it makes the horizontal
/// bounds load-bearing.
fn on_a_transposed_ramp_case() -> RecordedCase {
    let (width, height) = (128u32, 96u32);
    let source = unique_frame(height, width);
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            frame[(y * width + x) as usize] = source[(x * height + y) as usize];
        }
    }

    let setup = Setup::spatial_only(frame, width, height);
    let want = Digest {
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
    };

    ("transposed ramp", setup, want)
}

/// 104 pixels wide gives 25 reference patches at `STEP = 4`, so the last cube of each row runs dead
/// groups at every group count.
///
/// The dead groups must reach every barrier and write nothing. One that scattered would double the
/// last reference's contribution and move a pixel.
fn when_refs_are_not_a_multiple_of_eight_case() -> RecordedCase {
    let (width, height) = (104u32, 96u32);
    let frame = unique_frame(width, height);
    let setup = Setup::spatial_only(frame, width, height);
    let want = Digest {
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
    };

    ("ragged reference row", setup, want)
}

/// At `spatial_radius = 1` a corner reference sees 2x2 candidates and an edge one 2x3, both keeping
/// four, while an interior one sees 3x3 and fills to eight.
///
/// Every wider configuration fills every group to eight, so this is the run where the 2- and
/// 4-member ladders execute.
fn on_a_short_search_space_case() -> RecordedCase {
    let (width, height) = (64u32, 64u32);
    let frame = unique_frame(width, height);
    let mut setup = Setup::spatial_only(frame, width, height);
    setup.spatial_radius = 1;
    let want = Digest {
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
    };

    ("short search space", setup, want)
}

/// Noise over a flat field has no ramp to separate candidates, so the self-match sentinel and the
/// first-wins insert decide the member set.
fn on_noise_case() -> RecordedCase {
    let (width, height) = (64u32, 64u32);
    let frame = noisy_flat_field(width, height, 0.5, 0.05);
    let setup = Setup::spatial_only(frame, width, height);
    let want = Digest {
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
    };

    ("noise", setup, want)
}

/// The kernel applies the correlation profile at the threshold, while the recording applied it to
/// each member's variance before the ladder.
///
/// The ladder only averages and the profile is constant along the stack axis, so both orders agree
/// in exact arithmetic. Every other run uses the all-ones `rho = 0` profile, which cannot tell them
/// apart. `0.86` is the shipped table's high end.
fn under_correlation_shaping_case() -> RecordedCase {
    let (width, height) = (64u32, 64u32);
    let frame = noisy_flat_field(width, height, 0.5, 0.05);
    let mut setup = Setup::spatial_only(frame, width, height);
    setup.rho = 0.86;
    let want = Digest {
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
    };

    ("correlation shaping", setup, want)
}

/// Covers the `c_min` skip, the volume grid with its single-frame fallback, and the scatter into
/// each member's own region of the ring.
///
/// Every block has its own vector, so the search reaches positions the corner block never pointed
/// at, and the confidences straddle `c_min`, so some groups build a grid and others fall back. No
/// second implementation exists, so the digest is this kernel's own output. It carries real,
/// non-zero coverage, which a silently broken shader cannot reproduce.
fn across_frames_case() -> RecordedCase {
    let setup = cross_frame_setup(64, 64, 2);
    let want = Digest {
        covered: 16453,
        pixel_mean: 0.441598762473,
        pixel_rms: 0.546715666545,
        weight_mean: 1045.608471951,
        probes: [
            0.837928771973,
            0.574595237938,
            0.300403234153,
            0.000000000000,
            0.775989927049,
            0.000000000000,
            0.237124125163,
            0.979660034180,
        ],
    };

    ("cross frame", setup, want)
}

variant_chunk_tests! {
    assert_case_matches_recorded(on_unique_content_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_on_unique_content_in_the_first_half => 0,
        fused_reproduces_recorded_output_on_unique_content_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(on_a_transposed_ramp_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_on_a_transposed_ramp_in_the_first_half => 0,
        fused_reproduces_recorded_output_on_a_transposed_ramp_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(when_refs_are_not_a_multiple_of_eight_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_when_refs_are_not_a_multiple_of_eight_in_the_first_half => 0,
        fused_reproduces_recorded_output_when_refs_are_not_a_multiple_of_eight_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(on_a_short_search_space_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_on_a_short_search_space_in_the_first_half => 0,
        fused_reproduces_recorded_output_on_a_short_search_space_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(on_noise_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_on_noise_in_the_first_half => 0,
        fused_reproduces_recorded_output_on_noise_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(under_correlation_shaping_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_under_correlation_shaping_in_the_first_half => 0,
        fused_reproduces_recorded_output_under_correlation_shaping_in_the_second_half => 1,
    ]
}

variant_chunk_tests! {
    assert_case_matches_recorded(across_frames_case()), 2 chunks,
    [
        fused_reproduces_recorded_output_across_frames_in_the_first_half => 0,
        fused_reproduces_recorded_output_across_frames_in_the_second_half => 1,
    ]
}

/// Each neighbour holds the centre plus its own jitter, so every volume keeps a different three of
/// them and every ring slot receives members somewhere.
///
/// The frame a member came from is never written down between the match and the scatter, which
/// makes it easy to lose. No second implementation exists, so the digest is this kernel's own output.
#[test]
fn fused_scatters_into_every_member_frame() {
    let setup = five_frame_ring_with_jittered_copies(64, 64);
    let got = run_fused(&setup);

    for slot in 0..5 {
        let slot_sum = got.frame_weight_sum(slot);
        assert!(slot_sum > 0, "ring slot {slot} received nothing");
    }

    let want = Digest {
        covered: 19992,
        pixel_mean: 0.488864379137,
        pixel_rms: 0.570867709963,
        weight_mean: 1233.333334961,
        probes: [
            0.836363474528,
            0.574619293213,
            0.300458908081,
            0.034561157227,
            0.774574279785,
            0.513134002686,
            0.238787333171,
            0.979087829590,
        ],
    };

    assert_matches_recorded("planted cross-frame match", &got, &want);
}
