use super::golden::collab_corpus;
use super::{Aggregated, run_fused_variant};
use crate::collab::kernels::fused::{Centre, CollabVariant, Walk};

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
