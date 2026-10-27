use crate::tune::{KernelPlan, PipelineProfile, SiteKey};

fn window_key(width: u32) -> SiteKey {
    SiteKey::builder("nlm_window", width, 64).param("pr", 2).build()
}

#[test]
fn a_missing_entry_means_the_baseline() {
    let plan = KernelPlan::baseline();
    let key = window_key(64);

    assert_eq!(plan.get(&key), None);
    assert!(plan.is_empty());
}

#[test]
fn an_exact_entry_wins_over_a_site_wide_one() {
    let key = window_key(64);
    let mut plan = KernelPlan::baseline().with_site("nlm_window", "16x16");
    plan.insert(key.clone(), "32x16");

    assert_eq!(plan.get(&key), Some("32x16"));
}

#[test]
fn a_site_wide_entry_covers_keys_with_none_of_their_own() {
    let plan = KernelPlan::baseline().with_site("nlm_window", "16x16");
    let wide_key = window_key(4096);
    let other_site = SiteKey::builder("nlm_accumulate", 64, 64).build();

    assert_eq!(plan.get(&wide_key), Some("16x16"));
    assert_eq!(plan.get(&other_site), None);
}

#[test]
fn filling_keeps_existing_entries() {
    let key = window_key(64);
    let other_key = window_key(4096);
    let mut plan = KernelPlan::baseline();
    plan.insert(key.clone(), "32x16");

    let mut stored = KernelPlan::baseline();
    stored.insert(key.clone(), "16x16");
    stored.insert(other_key.clone(), "16x16");
    plan.fill_from(&stored);

    assert_eq!(plan.get(&key), Some("32x16"));
    assert_eq!(plan.get(&other_key), Some("16x16"));
    assert_eq!(plan.len(), 2);
}

#[test]
fn profile_names_round_trip() {
    for profile in PipelineProfile::ALL {
        let name = profile.name();
        let parsed = PipelineProfile::from_name(name);
        assert_eq!(parsed, Some(profile), "{name}");
    }

    assert_eq!(PipelineProfile::default(), PipelineProfile::Baseline);
}
