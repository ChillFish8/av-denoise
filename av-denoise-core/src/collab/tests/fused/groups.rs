use super::{Aggregated, cross_frame_setup, flat_noise_setup, run_fused_groups};
use crate::collab::COLLAB_GROUPS;

fn assert_same(label: &str, got: &Aggregated, want: &Aggregated) {
    assert_eq!(got.accum, want.accum, "{label} accum differs");
    assert_eq!(got.wsum, want.wsum, "{label} wsum differs");
    assert_eq!(
        got.group_weight, want.group_weight,
        "{label} group weight differs"
    );
}

#[test]
fn collab_group_counts_match_on_a_cross_frame_ring() {
    let setup = cross_frame_setup(96, 64, 2);
    let want = run_fused_groups(&setup, COLLAB_GROUPS);

    for groups in [4, 16] {
        let got = run_fused_groups(&setup, groups);
        assert_same(&format!("{groups} groups"), &got, &want);
    }
}

#[test]
fn collab_group_counts_match_at_odd_width() {
    let setup = flat_noise_setup(200, 48, 0.05);
    let want = run_fused_groups(&setup, COLLAB_GROUPS);

    for groups in [4, 16] {
        let got = run_fused_groups(&setup, groups);
        assert_same(&format!("{groups} groups"), &got, &want);
    }
}
