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
use crate::tune::tests::runtime::test_client;
use crate::tune::{KernelPlan, SiteCandidate};

fn assert_groups_retired(client: &Client, outputs: &CollabOutputs, label: &str) {
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
    assert_variants_match_baseline(pooled_chroma_spec()), 10 chunks,
    [
        pooled_chroma_matches_the_baseline_in_the_first_variant_tenth => 0,
        pooled_chroma_matches_the_baseline_in_the_second_variant_tenth => 1,
        pooled_chroma_matches_the_baseline_in_the_third_variant_tenth => 2,
        pooled_chroma_matches_the_baseline_in_the_fourth_variant_tenth => 3,
        pooled_chroma_matches_the_baseline_in_the_fifth_variant_tenth => 4,
        pooled_chroma_matches_the_baseline_in_the_sixth_variant_tenth => 5,
        pooled_chroma_matches_the_baseline_in_the_seventh_variant_tenth => 6,
        pooled_chroma_matches_the_baseline_in_the_eighth_variant_tenth => 7,
        pooled_chroma_matches_the_baseline_in_the_ninth_variant_tenth => 8,
        pooled_chroma_matches_the_baseline_in_the_tenth_variant_tenth => 9,
    ]
}

/// 133 by 33 clamps the last reference onto the right and bottom frame edges, and its 33
/// references per row leave dead groups in the last cube of each row at every group count.
fn frame_edge_yuv_spec() -> CollabSpec {
    CollabSpec::new(133, 33, ChannelMode::Yuv, 2)
}

variant_chunk_tests! {
    assert_variants_match_baseline(frame_edge_yuv_spec()), 20 chunks,
    [
        yuv_at_the_frame_edges_matches_the_baseline_in_the_first_variant_twentieth => 0,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_second_variant_twentieth => 1,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_third_variant_twentieth => 2,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
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
    assert_variants_match_baseline(smaller_than_the_search_window_spec()), 20 chunks,
    [
        a_frame_inside_the_search_window_matches_the_baseline_in_the_first_variant_twentieth => 0,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_second_variant_twentieth => 1,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_third_variant_twentieth => 2,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
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
    assert_variants_match_baseline(f16_yuv_spec()), 20 chunks,
    [
        the_f16_search_on_yuv_matches_the_baseline_in_the_first_variant_twentieth => 0,
        the_f16_search_on_yuv_matches_the_baseline_in_the_second_variant_twentieth => 1,
        the_f16_search_on_yuv_matches_the_baseline_in_the_third_variant_twentieth => 2,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        the_f16_search_on_yuv_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        the_f16_search_on_yuv_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        the_f16_search_on_yuv_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

fn f16_frame_edge_chroma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(133, 33, ChannelMode::Chroma, 2);
    spec.f16_search = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_frame_edge_chroma_spec()), 10 chunks,
    [
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_first_variant_tenth => 0,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_second_variant_tenth => 1,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_third_variant_tenth => 2,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_fourth_variant_tenth => 3,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_fifth_variant_tenth => 4,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_sixth_variant_tenth => 5,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_seventh_variant_tenth => 6,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_eighth_variant_tenth => 7,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_ninth_variant_tenth => 8,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_tenth_variant_tenth => 9,
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
    assert_switches_match_baseline(site_chroma_spec()), 20 chunks,
    [
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, false)), 20 chunks,
    [
        yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, true)), 20 chunks,
    [
        pooled_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, false)), 20 chunks,
    [
        f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, true)), 20 chunks,
    [
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
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
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Chroma), 10 chunks,
    [
        forcing_the_first_variant_tenth_leaves_chroma_nl4d_output_unchanged => 0,
        forcing_the_second_variant_tenth_leaves_chroma_nl4d_output_unchanged => 1,
        forcing_the_third_variant_tenth_leaves_chroma_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_tenth_leaves_chroma_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 9,
    ]
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Yuv), 20 chunks,
    [
        forcing_the_first_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 0,
        forcing_the_second_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 1,
        forcing_the_third_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 9,
        forcing_the_eleventh_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 10,
        forcing_the_twelfth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 11,
        forcing_the_thirteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 12,
        forcing_the_fourteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 13,
        forcing_the_fifteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 14,
        forcing_the_sixteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 15,
        forcing_the_seventeenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 16,
        forcing_the_eighteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 17,
        forcing_the_nineteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 18,
        forcing_the_twentieth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 19,
    ]
}
