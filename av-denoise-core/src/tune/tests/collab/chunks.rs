use cubecl::prelude::*;

use super::luma_spec;
use crate::Nl4dOptions;
use crate::collab::supports_f16_search;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_f32;
use crate::tune::sites::{COLLAB_FUSED, CollabOutputs, CollabVariant};
use crate::tune::tests::engine::{nl4d_outputs, noisy_frames};
use crate::tune::tests::equivalence::assert_listed_candidates_match_baseline;
use crate::tune::tests::fixtures::{CollabSpec, collab_site, collab_site_f16, variant_chunk_tests};
use crate::tune::tests::runtime::{TestRuntime, test_client};
use crate::tune::{KernelPlan, SiteCandidate};

fn assert_groups_retired(client: &ComputeClient<TestRuntime>, outputs: &CollabOutputs, label: &str) {
    let weights = read_f32(client, &outputs.group_weight, outputs.refs).expect("weights read back");
    let retired = weights.iter().any(|&weight| weight > 0.0);

    assert!(retired, "{label}: the baseline retired no group");
}

/// Checks each of `variants` that is legal and fits this device against the baseline, bit for bit.
///
/// A device without f16 support has no f16 case to check.
fn assert_variants_match_baseline(spec: CollabSpec, variants: &[CollabVariant]) {
    let client = test_client();
    let label = format!("{spec:?}");

    if spec.f16_search {
        let supported = supports_f16_search(&client);
        if !supported {
            return;
        }

        let site = collab_site_f16(&client, &spec);
        let baseline = assert_listed_candidates_match_baseline(&site, &label, variants);
        assert_groups_retired(&client, &baseline, &label);
        return;
    }

    let site = collab_site(&client, &spec);
    let baseline = assert_listed_candidates_match_baseline(&site, &label, variants);
    assert_groups_retired(&client, &baseline, &label);
}

fn odd_width_luma_spec() -> CollabSpec {
    CollabSpec::new(70, 45, ChannelMode::Luma, 2)
}

variant_chunk_tests! {
    assert_variants_match_baseline(odd_width_luma_spec()), 4 chunks,
    [
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_first_variant_quarter => 0,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_second_variant_quarter => 1,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_third_variant_quarter => 2,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn pooled_chroma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Chroma, 1);
    spec.pooled = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(pooled_chroma_spec()), 20 chunks,
    [
        pooled_chroma_matches_the_baseline_in_the_first_variant_twentieth => 0,
        pooled_chroma_matches_the_baseline_in_the_second_variant_twentieth => 1,
        pooled_chroma_matches_the_baseline_in_the_third_variant_twentieth => 2,
        pooled_chroma_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        pooled_chroma_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        pooled_chroma_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        pooled_chroma_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        pooled_chroma_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        pooled_chroma_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        pooled_chroma_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        pooled_chroma_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        pooled_chroma_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        pooled_chroma_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        pooled_chroma_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        pooled_chroma_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        pooled_chroma_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        pooled_chroma_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        pooled_chroma_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        pooled_chroma_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        pooled_chroma_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

/// 133 by 33 clamps the last reference onto the right and bottom frame edges, and its 33
/// references per row leave dead groups in the last cube of each row at every group count.
fn frame_edge_yuv_spec() -> CollabSpec {
    CollabSpec::new(133, 33, ChannelMode::Yuv, 2)
}

variant_chunk_tests! {
    assert_variants_match_baseline(frame_edge_yuv_spec()), 40 chunks,
    [
        yuv_at_the_frame_edges_matches_the_baseline_in_the_first_variant_fortieth => 0,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_second_variant_fortieth => 1,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_third_variant_fortieth => 2,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourth_variant_fortieth => 3,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifth_variant_fortieth => 4,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixth_variant_fortieth => 5,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventh_variant_fortieth => 6,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighth_variant_fortieth => 7,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_ninth_variant_fortieth => 8,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_tenth_variant_fortieth => 9,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eleventh_variant_fortieth => 10,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twelfth_variant_fortieth => 11,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirteenth_variant_fortieth => 12,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourteenth_variant_fortieth => 13,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifteenth_variant_fortieth => 14,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixteenth_variant_fortieth => 15,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventeenth_variant_fortieth => 16,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighteenth_variant_fortieth => 17,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_nineteenth_variant_fortieth => 18,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twentieth_variant_fortieth => 19,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_first_variant_fortieth => 20,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_second_variant_fortieth => 21,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_third_variant_fortieth => 22,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_fourth_variant_fortieth => 23,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_fifth_variant_fortieth => 24,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_sixth_variant_fortieth => 25,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_seventh_variant_fortieth => 26,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_eighth_variant_fortieth => 27,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twenty_ninth_variant_fortieth => 28,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirtieth_variant_fortieth => 29,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_first_variant_fortieth => 30,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_second_variant_fortieth => 31,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_third_variant_fortieth => 32,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_fourth_variant_fortieth => 33,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_fifth_variant_fortieth => 34,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_sixth_variant_fortieth => 35,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_seventh_variant_fortieth => 36,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_eighth_variant_fortieth => 37,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirty_ninth_variant_fortieth => 38,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fortieth_variant_fortieth => 39,
    ]
}

/// 104 pixels gives 25 references per row, so the last cube of a row has dead groups at every
/// group count.
fn dead_groups_spec() -> CollabSpec {
    CollabSpec::new(104, 40, ChannelMode::Luma, 0)
}

variant_chunk_tests! {
    assert_variants_match_baseline(dead_groups_spec()), 4 chunks,
    [
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_first_variant_quarter => 0,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_second_variant_quarter => 1,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_third_variant_quarter => 2,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn smaller_than_the_search_window_spec() -> CollabSpec {
    CollabSpec::new(40, 24, ChannelMode::Yuv, 1)
}

variant_chunk_tests! {
    assert_variants_match_baseline(smaller_than_the_search_window_spec()), 40 chunks,
    [
        a_frame_inside_the_search_window_matches_the_baseline_in_the_first_variant_fortieth => 0,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_second_variant_fortieth => 1,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_third_variant_fortieth => 2,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourth_variant_fortieth => 3,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifth_variant_fortieth => 4,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixth_variant_fortieth => 5,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventh_variant_fortieth => 6,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighth_variant_fortieth => 7,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_ninth_variant_fortieth => 8,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_tenth_variant_fortieth => 9,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eleventh_variant_fortieth => 10,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twelfth_variant_fortieth => 11,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirteenth_variant_fortieth => 12,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourteenth_variant_fortieth => 13,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifteenth_variant_fortieth => 14,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixteenth_variant_fortieth => 15,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventeenth_variant_fortieth => 16,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighteenth_variant_fortieth => 17,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_nineteenth_variant_fortieth => 18,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twentieth_variant_fortieth => 19,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_first_variant_fortieth => 20,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_second_variant_fortieth => 21,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_third_variant_fortieth => 22,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_fourth_variant_fortieth => 23,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_fifth_variant_fortieth => 24,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_sixth_variant_fortieth => 25,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_seventh_variant_fortieth => 26,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_eighth_variant_fortieth => 27,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twenty_ninth_variant_fortieth => 28,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirtieth_variant_fortieth => 29,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_first_variant_fortieth => 30,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_second_variant_fortieth => 31,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_third_variant_fortieth => 32,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_fourth_variant_fortieth => 33,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_fifth_variant_fortieth => 34,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_sixth_variant_fortieth => 35,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_seventh_variant_fortieth => 36,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_eighth_variant_fortieth => 37,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirty_ninth_variant_fortieth => 38,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fortieth_variant_fortieth => 39,
    ]
}

fn flat_spec(f16_search: bool) -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Luma, 1);
    spec.flat = true;
    spec.f16_search = f16_search;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(flat_spec(false)), 4 chunks,
    [
        flat_content_matches_the_baseline_in_the_first_variant_quarter => 0,
        flat_content_matches_the_baseline_in_the_second_variant_quarter => 1,
        flat_content_matches_the_baseline_in_the_third_variant_quarter => 2,
        flat_content_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(flat_spec(true)), 4 chunks,
    [
        flat_content_matches_the_baseline_with_the_f16_search_in_the_first_variant_quarter => 0,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_second_variant_quarter => 1,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_third_variant_quarter => 2,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_fourth_variant_quarter => 3,
    ]
}

fn f16_pooled_luma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(70, 45, ChannelMode::Luma, 2);
    spec.f16_search = true;
    spec.pooled = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_pooled_luma_spec()), 4 chunks,
    [
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_first_variant_quarter => 0,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_second_variant_quarter => 1,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_third_variant_quarter => 2,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn f16_yuv_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    spec.f16_search = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_yuv_spec()), 40 chunks,
    [
        the_f16_search_on_yuv_matches_the_baseline_in_the_first_variant_fortieth => 0,
        the_f16_search_on_yuv_matches_the_baseline_in_the_second_variant_fortieth => 1,
        the_f16_search_on_yuv_matches_the_baseline_in_the_third_variant_fortieth => 2,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourth_variant_fortieth => 3,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifth_variant_fortieth => 4,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixth_variant_fortieth => 5,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventh_variant_fortieth => 6,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighth_variant_fortieth => 7,
        the_f16_search_on_yuv_matches_the_baseline_in_the_ninth_variant_fortieth => 8,
        the_f16_search_on_yuv_matches_the_baseline_in_the_tenth_variant_fortieth => 9,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eleventh_variant_fortieth => 10,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twelfth_variant_fortieth => 11,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirteenth_variant_fortieth => 12,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourteenth_variant_fortieth => 13,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifteenth_variant_fortieth => 14,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixteenth_variant_fortieth => 15,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventeenth_variant_fortieth => 16,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighteenth_variant_fortieth => 17,
        the_f16_search_on_yuv_matches_the_baseline_in_the_nineteenth_variant_fortieth => 18,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twentieth_variant_fortieth => 19,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_first_variant_fortieth => 20,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_second_variant_fortieth => 21,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_third_variant_fortieth => 22,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_fourth_variant_fortieth => 23,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_fifth_variant_fortieth => 24,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_sixth_variant_fortieth => 25,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_seventh_variant_fortieth => 26,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_eighth_variant_fortieth => 27,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twenty_ninth_variant_fortieth => 28,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirtieth_variant_fortieth => 29,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_first_variant_fortieth => 30,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_second_variant_fortieth => 31,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_third_variant_fortieth => 32,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_fourth_variant_fortieth => 33,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_fifth_variant_fortieth => 34,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_sixth_variant_fortieth => 35,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_seventh_variant_fortieth => 36,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_eighth_variant_fortieth => 37,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirty_ninth_variant_fortieth => 38,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fortieth_variant_fortieth => 39,
    ]
}

fn f16_frame_edge_chroma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(133, 33, ChannelMode::Chroma, 2);
    spec.f16_search = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_frame_edge_chroma_spec()), 20 chunks,
    [
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_first_variant_twentieth => 0,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_second_variant_twentieth => 1,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_third_variant_twentieth => 2,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        the_f16_search_on_edge_chroma_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

/// [assert_variants_match_baseline] with the f16 search and the pooled threshold each off and on.
fn assert_switches_match_baseline(spec: CollabSpec, variants: &[CollabVariant]) {
    for f16_search in [false, true] {
        for pooled in [false, true] {
            let switched = CollabSpec {
                f16_search,
                pooled,
                ..spec
            };
            assert_variants_match_baseline(switched, variants);
        }
    }
}

fn site_chroma_spec() -> CollabSpec {
    CollabSpec::new(64, 48, ChannelMode::Chroma, 2)
}

/// Yuv takes one switch setting per test, since each yuv variant compiles several times slower.
fn site_yuv_spec(f16_search: bool, pooled: bool) -> CollabSpec {
    let spec = CollabSpec::new(72, 40, ChannelMode::Yuv, 1);

    CollabSpec {
        f16_search,
        pooled,
        ..spec
    }
}

variant_chunk_tests! {
    assert_switches_match_baseline(luma_spec()), 20 chunks,
    [
        every_luma_switch_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_switches_match_baseline(site_chroma_spec()), 40 chunks,
    [
        every_chroma_switch_matches_the_site_baseline_in_the_first_variant_fortieth => 0,
        every_chroma_switch_matches_the_site_baseline_in_the_second_variant_fortieth => 1,
        every_chroma_switch_matches_the_site_baseline_in_the_third_variant_fortieth => 2,
        every_chroma_switch_matches_the_site_baseline_in_the_fourth_variant_fortieth => 3,
        every_chroma_switch_matches_the_site_baseline_in_the_fifth_variant_fortieth => 4,
        every_chroma_switch_matches_the_site_baseline_in_the_sixth_variant_fortieth => 5,
        every_chroma_switch_matches_the_site_baseline_in_the_seventh_variant_fortieth => 6,
        every_chroma_switch_matches_the_site_baseline_in_the_eighth_variant_fortieth => 7,
        every_chroma_switch_matches_the_site_baseline_in_the_ninth_variant_fortieth => 8,
        every_chroma_switch_matches_the_site_baseline_in_the_tenth_variant_fortieth => 9,
        every_chroma_switch_matches_the_site_baseline_in_the_eleventh_variant_fortieth => 10,
        every_chroma_switch_matches_the_site_baseline_in_the_twelfth_variant_fortieth => 11,
        every_chroma_switch_matches_the_site_baseline_in_the_thirteenth_variant_fortieth => 12,
        every_chroma_switch_matches_the_site_baseline_in_the_fourteenth_variant_fortieth => 13,
        every_chroma_switch_matches_the_site_baseline_in_the_fifteenth_variant_fortieth => 14,
        every_chroma_switch_matches_the_site_baseline_in_the_sixteenth_variant_fortieth => 15,
        every_chroma_switch_matches_the_site_baseline_in_the_seventeenth_variant_fortieth => 16,
        every_chroma_switch_matches_the_site_baseline_in_the_eighteenth_variant_fortieth => 17,
        every_chroma_switch_matches_the_site_baseline_in_the_nineteenth_variant_fortieth => 18,
        every_chroma_switch_matches_the_site_baseline_in_the_twentieth_variant_fortieth => 19,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_first_variant_fortieth => 20,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_second_variant_fortieth => 21,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_third_variant_fortieth => 22,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_fourth_variant_fortieth => 23,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_fifth_variant_fortieth => 24,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_sixth_variant_fortieth => 25,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_seventh_variant_fortieth => 26,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_eighth_variant_fortieth => 27,
        every_chroma_switch_matches_the_site_baseline_in_the_twenty_ninth_variant_fortieth => 28,
        every_chroma_switch_matches_the_site_baseline_in_the_thirtieth_variant_fortieth => 29,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_first_variant_fortieth => 30,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_second_variant_fortieth => 31,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_third_variant_fortieth => 32,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_fourth_variant_fortieth => 33,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_fifth_variant_fortieth => 34,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_sixth_variant_fortieth => 35,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_seventh_variant_fortieth => 36,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_eighth_variant_fortieth => 37,
        every_chroma_switch_matches_the_site_baseline_in_the_thirty_ninth_variant_fortieth => 38,
        every_chroma_switch_matches_the_site_baseline_in_the_fortieth_variant_fortieth => 39,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, false)), 40 chunks,
    [
        yuv_matches_the_baseline_through_the_site_in_the_first_variant_fortieth => 0,
        yuv_matches_the_baseline_through_the_site_in_the_second_variant_fortieth => 1,
        yuv_matches_the_baseline_through_the_site_in_the_third_variant_fortieth => 2,
        yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_fortieth => 3,
        yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_fortieth => 4,
        yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_fortieth => 5,
        yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_fortieth => 6,
        yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_fortieth => 7,
        yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_fortieth => 8,
        yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_fortieth => 9,
        yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_fortieth => 10,
        yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_fortieth => 11,
        yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_fortieth => 12,
        yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_fortieth => 13,
        yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_fortieth => 14,
        yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_fortieth => 15,
        yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_fortieth => 16,
        yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_fortieth => 17,
        yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_fortieth => 18,
        yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_fortieth => 19,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_first_variant_fortieth => 20,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_second_variant_fortieth => 21,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_third_variant_fortieth => 22,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_fourth_variant_fortieth => 23,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_fifth_variant_fortieth => 24,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_sixth_variant_fortieth => 25,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_seventh_variant_fortieth => 26,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_eighth_variant_fortieth => 27,
        yuv_matches_the_baseline_through_the_site_in_the_twenty_ninth_variant_fortieth => 28,
        yuv_matches_the_baseline_through_the_site_in_the_thirtieth_variant_fortieth => 29,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_first_variant_fortieth => 30,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_second_variant_fortieth => 31,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_third_variant_fortieth => 32,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_fourth_variant_fortieth => 33,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_fifth_variant_fortieth => 34,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_sixth_variant_fortieth => 35,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_seventh_variant_fortieth => 36,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_eighth_variant_fortieth => 37,
        yuv_matches_the_baseline_through_the_site_in_the_thirty_ninth_variant_fortieth => 38,
        yuv_matches_the_baseline_through_the_site_in_the_fortieth_variant_fortieth => 39,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, true)), 40 chunks,
    [
        pooled_yuv_matches_the_baseline_through_the_site_in_the_first_variant_fortieth => 0,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_second_variant_fortieth => 1,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_third_variant_fortieth => 2,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_fortieth => 3,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_fortieth => 4,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_fortieth => 5,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_fortieth => 6,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_fortieth => 7,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_fortieth => 8,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_fortieth => 9,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_fortieth => 10,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_fortieth => 11,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_fortieth => 12,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_fortieth => 13,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_fortieth => 14,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_fortieth => 15,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_fortieth => 16,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_fortieth => 17,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_fortieth => 18,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_fortieth => 19,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_first_variant_fortieth => 20,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_second_variant_fortieth => 21,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_third_variant_fortieth => 22,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_fourth_variant_fortieth => 23,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_fifth_variant_fortieth => 24,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_sixth_variant_fortieth => 25,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_seventh_variant_fortieth => 26,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_eighth_variant_fortieth => 27,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twenty_ninth_variant_fortieth => 28,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirtieth_variant_fortieth => 29,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_first_variant_fortieth => 30,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_second_variant_fortieth => 31,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_third_variant_fortieth => 32,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_fourth_variant_fortieth => 33,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_fifth_variant_fortieth => 34,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_sixth_variant_fortieth => 35,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_seventh_variant_fortieth => 36,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_eighth_variant_fortieth => 37,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirty_ninth_variant_fortieth => 38,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fortieth_variant_fortieth => 39,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, false)), 40 chunks,
    [
        f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_fortieth => 0,
        f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_fortieth => 1,
        f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_fortieth => 2,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_fortieth => 3,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_fortieth => 4,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_fortieth => 5,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_fortieth => 6,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_fortieth => 7,
        f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_fortieth => 8,
        f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_fortieth => 9,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_fortieth => 10,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_fortieth => 11,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_fortieth => 12,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_fortieth => 13,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_fortieth => 14,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_fortieth => 15,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_fortieth => 16,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_fortieth => 17,
        f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_fortieth => 18,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_fortieth => 19,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_first_variant_fortieth => 20,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_second_variant_fortieth => 21,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_third_variant_fortieth => 22,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_fourth_variant_fortieth => 23,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_fifth_variant_fortieth => 24,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_sixth_variant_fortieth => 25,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_seventh_variant_fortieth => 26,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_eighth_variant_fortieth => 27,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_ninth_variant_fortieth => 28,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirtieth_variant_fortieth => 29,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_first_variant_fortieth => 30,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_second_variant_fortieth => 31,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_third_variant_fortieth => 32,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_fourth_variant_fortieth => 33,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_fifth_variant_fortieth => 34,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_sixth_variant_fortieth => 35,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_seventh_variant_fortieth => 36,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_eighth_variant_fortieth => 37,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_ninth_variant_fortieth => 38,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fortieth_variant_fortieth => 39,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, true)), 40 chunks,
    [
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_fortieth => 0,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_fortieth => 1,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_fortieth => 2,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_fortieth => 3,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_fortieth => 4,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_fortieth => 5,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_fortieth => 6,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_fortieth => 7,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_fortieth => 8,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_fortieth => 9,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_fortieth => 10,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_fortieth => 11,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_fortieth => 12,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_fortieth => 13,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_fortieth => 14,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_fortieth => 15,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_fortieth => 16,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_fortieth => 17,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_fortieth => 18,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_fortieth => 19,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_first_variant_fortieth => 20,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_second_variant_fortieth => 21,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_third_variant_fortieth => 22,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_fourth_variant_fortieth => 23,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_fifth_variant_fortieth => 24,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_sixth_variant_fortieth => 25,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_seventh_variant_fortieth => 26,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_eighth_variant_fortieth => 27,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twenty_ninth_variant_fortieth => 28,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirtieth_variant_fortieth => 29,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_first_variant_fortieth => 30,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_second_variant_fortieth => 31,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_third_variant_fortieth => 32,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_fourth_variant_fortieth => 33,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_fifth_variant_fortieth => 34,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_sixth_variant_fortieth => 35,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_seventh_variant_fortieth => 36,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_eighth_variant_fortieth => 37,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirty_ninth_variant_fortieth => 38,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fortieth_variant_fortieth => 39,
    ]
}

/// Runs nl4d over `channels` with each of `variants` forced and checks it against the baseline.
fn assert_forced_variants_leave_nl4d_output_unchanged(channels: ChannelMode, variants: &[CollabVariant]) {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let planes = channels.count() as usize;
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, planes, 7);
    let baseline_plan = KernelPlan::baseline();
    let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

    for &variant in variants {
        if !variant.distinct_for(channels.count()) {
            continue;
        }

        let name = variant.name();
        let plan = KernelPlan::baseline().with_site(COLLAB_FUSED, name);
        let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
        assert_eq!(forced, baseline, "{channels:?} {name}");
    }
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Luma), 10 chunks,
    [
        forcing_the_first_variant_tenth_leaves_luma_nl4d_output_unchanged => 0,
        forcing_the_second_variant_tenth_leaves_luma_nl4d_output_unchanged => 1,
        forcing_the_third_variant_tenth_leaves_luma_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_tenth_leaves_luma_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_tenth_leaves_luma_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_tenth_leaves_luma_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_tenth_leaves_luma_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_tenth_leaves_luma_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_tenth_leaves_luma_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_tenth_leaves_luma_nl4d_output_unchanged => 9,
    ]
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Chroma), 20 chunks,
    [
        forcing_the_first_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 0,
        forcing_the_second_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 1,
        forcing_the_third_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 9,
        forcing_the_eleventh_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 10,
        forcing_the_twelfth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 11,
        forcing_the_thirteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 12,
        forcing_the_fourteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 13,
        forcing_the_fifteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 14,
        forcing_the_sixteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 15,
        forcing_the_seventeenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 16,
        forcing_the_eighteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 17,
        forcing_the_nineteenth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 18,
        forcing_the_twentieth_variant_twentieth_leaves_chroma_nl4d_output_unchanged => 19,
    ]
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Yuv), 40 chunks,
    [
        forcing_the_first_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 0,
        forcing_the_second_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 1,
        forcing_the_third_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 9,
        forcing_the_eleventh_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 10,
        forcing_the_twelfth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 11,
        forcing_the_thirteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 12,
        forcing_the_fourteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 13,
        forcing_the_fifteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 14,
        forcing_the_sixteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 15,
        forcing_the_seventeenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 16,
        forcing_the_eighteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 17,
        forcing_the_nineteenth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 18,
        forcing_the_twentieth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 19,
        forcing_the_twenty_first_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 20,
        forcing_the_twenty_second_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 21,
        forcing_the_twenty_third_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 22,
        forcing_the_twenty_fourth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 23,
        forcing_the_twenty_fifth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 24,
        forcing_the_twenty_sixth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 25,
        forcing_the_twenty_seventh_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 26,
        forcing_the_twenty_eighth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 27,
        forcing_the_twenty_ninth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 28,
        forcing_the_thirtieth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 29,
        forcing_the_thirty_first_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 30,
        forcing_the_thirty_second_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 31,
        forcing_the_thirty_third_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 32,
        forcing_the_thirty_fourth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 33,
        forcing_the_thirty_fifth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 34,
        forcing_the_thirty_sixth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 35,
        forcing_the_thirty_seventh_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 36,
        forcing_the_thirty_eighth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 37,
        forcing_the_thirty_ninth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 38,
        forcing_the_fortieth_variant_fortieth_leaves_yuv_nl4d_output_unchanged => 39,
    ]
}
