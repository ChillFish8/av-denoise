use super::golden::collab_corpus;
use super::{Aggregated, Setup, run_fused_variant};
use crate::collab::kernels::fused::{Centre, ChannelView, CollabVariant, NeighbourWalk, Walk};
use crate::tune::SiteCandidate;
use crate::tune::tests::fixtures::variant_chunk_tests;

/// The baseline variant with the given walk.
pub(super) fn baseline_variant(walk: Walk) -> CollabVariant {
    CollabVariant {
        walk,
        ..CollabVariant::BASELINE
    }
}

pub(super) fn assert_same_output(label: &str, want: &Aggregated, got: &Aggregated) {
    assert_eq!(
        got.group_weight, want.group_weight,
        "{label}: group weights differ"
    );
    assert_eq!(got.accum, want.accum, "{label}: accum differs");
    assert_eq!(got.wsum, want.wsum, "{label}: wsum differs");
}

#[test]
fn sliding_centre_matches_the_baseline_over_the_collab_corpus() {
    for (name, setup) in collab_corpus() {
        for walk in [Walk::Clipped, Walk::Uniform] {
            let baseline = baseline_variant(walk);
            let sliding = CollabVariant {
                centre: Centre::Sliding,
                ..baseline
            };
            let label = format!("{name}/{walk:?}");

            let want = run_fused_variant(&setup, baseline);
            let got = run_fused_variant(&setup, sliding);

            assert_same_output(&label, &want, &got);
        }
    }
}

/// Asserts the sliding neighbour walk matches the row walk with `centre`, over the collab corpus
/// under both the clipped and uniform walks.
fn assert_neighbour_walks_agree(centre: Centre) {
    for (name, setup) in collab_corpus() {
        for walk in [Walk::Clipped, Walk::Uniform] {
            let rows = CollabVariant {
                centre,
                ..baseline_variant(walk)
            };
            let sliding = CollabVariant {
                neighbour: NeighbourWalk::Sliding,
                ..rows
            };
            let label = format!("{name}/{walk:?}/{centre:?}");

            let want = run_fused_variant(&setup, rows);
            let got = run_fused_variant(&setup, sliding);

            assert_same_output(&label, &want, &got);
        }
    }
}

#[test]
fn sliding_neighbour_walk_matches_the_baseline_over_the_collab_corpus() {
    assert_neighbour_walks_agree(Centre::Global);
}

#[test]
fn sliding_neighbour_walk_matches_the_baseline_over_the_collab_corpus_with_the_sliding_centre() {
    assert_neighbour_walks_agree(Centre::Sliding);
}

/// The collab corpus entries with more than one channel.
fn multi_channel_corpus() -> Vec<(&'static str, Setup)> {
    collab_corpus()
        .into_iter()
        .filter(|(_, setup)| setup.channel_mode.count() > 1)
        .collect()
}

/// Asserts each scalar-view variant in `variants` matches its line-view twin on every entry of
/// `corpus`.
fn assert_scalar_views_match_lines(corpus: Vec<(&'static str, Setup)>, variants: &[CollabVariant]) {
    let scalar_variants = variants
        .iter()
        .filter(|variant| variant.channel_view == ChannelView::Scalar);
    for &scalar in scalar_variants {
        let lines = CollabVariant {
            channel_view: ChannelView::Lines,
            ..scalar
        };

        for (name, setup) in &corpus {
            let label = format!("{name}/{}", scalar.name());

            let want = run_fused_variant(setup, lines);
            let got = run_fused_variant(setup, scalar);

            assert_same_output(&label, &want, &got);
        }
    }
}

variant_chunk_tests! {
    assert_scalar_views_match_lines(multi_channel_corpus()), 40 chunks,
    [
        the_scalar_view_matches_its_line_twin_in_the_first_variant_fortieth => 0,
        the_scalar_view_matches_its_line_twin_in_the_second_variant_fortieth => 1,
        the_scalar_view_matches_its_line_twin_in_the_third_variant_fortieth => 2,
        the_scalar_view_matches_its_line_twin_in_the_fourth_variant_fortieth => 3,
        the_scalar_view_matches_its_line_twin_in_the_fifth_variant_fortieth => 4,
        the_scalar_view_matches_its_line_twin_in_the_sixth_variant_fortieth => 5,
        the_scalar_view_matches_its_line_twin_in_the_seventh_variant_fortieth => 6,
        the_scalar_view_matches_its_line_twin_in_the_eighth_variant_fortieth => 7,
        the_scalar_view_matches_its_line_twin_in_the_ninth_variant_fortieth => 8,
        the_scalar_view_matches_its_line_twin_in_the_tenth_variant_fortieth => 9,
        the_scalar_view_matches_its_line_twin_in_the_eleventh_variant_fortieth => 10,
        the_scalar_view_matches_its_line_twin_in_the_twelfth_variant_fortieth => 11,
        the_scalar_view_matches_its_line_twin_in_the_thirteenth_variant_fortieth => 12,
        the_scalar_view_matches_its_line_twin_in_the_fourteenth_variant_fortieth => 13,
        the_scalar_view_matches_its_line_twin_in_the_fifteenth_variant_fortieth => 14,
        the_scalar_view_matches_its_line_twin_in_the_sixteenth_variant_fortieth => 15,
        the_scalar_view_matches_its_line_twin_in_the_seventeenth_variant_fortieth => 16,
        the_scalar_view_matches_its_line_twin_in_the_eighteenth_variant_fortieth => 17,
        the_scalar_view_matches_its_line_twin_in_the_nineteenth_variant_fortieth => 18,
        the_scalar_view_matches_its_line_twin_in_the_twentieth_variant_fortieth => 19,
        the_scalar_view_matches_its_line_twin_in_the_twenty_first_variant_fortieth => 20,
        the_scalar_view_matches_its_line_twin_in_the_twenty_second_variant_fortieth => 21,
        the_scalar_view_matches_its_line_twin_in_the_twenty_third_variant_fortieth => 22,
        the_scalar_view_matches_its_line_twin_in_the_twenty_fourth_variant_fortieth => 23,
        the_scalar_view_matches_its_line_twin_in_the_twenty_fifth_variant_fortieth => 24,
        the_scalar_view_matches_its_line_twin_in_the_twenty_sixth_variant_fortieth => 25,
        the_scalar_view_matches_its_line_twin_in_the_twenty_seventh_variant_fortieth => 26,
        the_scalar_view_matches_its_line_twin_in_the_twenty_eighth_variant_fortieth => 27,
        the_scalar_view_matches_its_line_twin_in_the_twenty_ninth_variant_fortieth => 28,
        the_scalar_view_matches_its_line_twin_in_the_thirtieth_variant_fortieth => 29,
        the_scalar_view_matches_its_line_twin_in_the_thirty_first_variant_fortieth => 30,
        the_scalar_view_matches_its_line_twin_in_the_thirty_second_variant_fortieth => 31,
        the_scalar_view_matches_its_line_twin_in_the_thirty_third_variant_fortieth => 32,
        the_scalar_view_matches_its_line_twin_in_the_thirty_fourth_variant_fortieth => 33,
        the_scalar_view_matches_its_line_twin_in_the_thirty_fifth_variant_fortieth => 34,
        the_scalar_view_matches_its_line_twin_in_the_thirty_sixth_variant_fortieth => 35,
        the_scalar_view_matches_its_line_twin_in_the_thirty_seventh_variant_fortieth => 36,
        the_scalar_view_matches_its_line_twin_in_the_thirty_eighth_variant_fortieth => 37,
        the_scalar_view_matches_its_line_twin_in_the_thirty_ninth_variant_fortieth => 38,
        the_scalar_view_matches_its_line_twin_in_the_fortieth_variant_fortieth => 39,
    ]
}
