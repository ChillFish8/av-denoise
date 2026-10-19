use crate::tune::{
    CandidateRecord,
    CompileProgress,
    FidelityClass,
    PipelineProfile,
    ProfileRanking,
    ProfileRecord,
    SiteKey,
    SiteRanking,
    TuneResult,
    choose,
    choose_profile,
};

fn record(name: &str, fidelity: FidelityClass, baseline: bool, median_ns: u64) -> CandidateRecord {
    CandidateRecord {
        name: name.to_string(),
        fidelity,
        baseline,
        median_ns,
        samples: 20,
    }
}

fn baseline_record(median_ns: u64) -> CandidateRecord {
    record("baseline", FidelityClass::Exact, true, median_ns)
}

#[test]
fn the_lowest_median_wins_when_it_is_clear_of_the_rest() {
    let records = [
        baseline_record(1000),
        record("fast", FidelityClass::Exact, false, 900),
    ];
    let chosen = choose(&records, false).expect("a candidate");

    assert_eq!(chosen.name, "fast");
}

#[test]
fn a_candidate_within_two_percent_of_the_baseline_loses_to_it() {
    let records = [
        baseline_record(1000),
        record("barely_faster", FidelityClass::Exact, false, 985),
    ];
    let chosen = choose(&records, false).expect("a candidate");

    assert!(chosen.baseline);
}

#[test]
fn a_rounding_winner_within_two_percent_of_an_exact_one_loses_to_it() {
    let records = [
        baseline_record(1000),
        record("exact_fast", FidelityClass::Exact, false, 800),
        record("rounding_fast", FidelityClass::Rounding, false, 790),
    ];
    let chosen = choose(&records, false).expect("a candidate");

    assert_eq!(chosen.name, "exact_fast");
}

#[test]
fn a_rounding_winner_clear_of_every_exact_one_wins() {
    let records = [
        baseline_record(1000),
        record("exact_fast", FidelityClass::Exact, false, 900),
        record("rounding_fast", FidelityClass::Rounding, false, 700),
    ];
    let chosen = choose(&records, false).expect("a candidate");

    assert_eq!(chosen.name, "rounding_fast");
}

/// The margin is measured from the fastest median only, so faithfulness never chains past it.
#[test]
fn hysteresis_does_not_chain_past_the_fastest_median() {
    let records = [
        baseline_record(1030),
        record("exact_fast", FidelityClass::Exact, false, 1015),
        record("rounding_fast", FidelityClass::Rounding, false, 1000),
    ];
    let chosen = choose(&records, false).expect("a candidate");

    assert_eq!(chosen.name, "exact_fast");
}

#[test]
fn deterministic_only_considers_exact_candidates() {
    let records = [
        baseline_record(1000),
        record("rounding_fast", FidelityClass::Rounding, false, 500),
        record("exact_fast", FidelityClass::Exact, false, 900),
    ];
    let chosen = choose(&records, true).expect("a candidate");

    assert_eq!(chosen.name, "exact_fast");
}

#[test]
fn no_records_choose_nothing() {
    assert!(choose(&[], false).is_none());
}

fn key(site: &'static str) -> SiteKey {
    SiteKey::builder(site, 64, 64).build()
}

#[test]
fn the_plan_holds_each_sites_choice() {
    let window_key = key("nlm_window");
    let mut result = TuneResult::default();
    let window_ranking = SiteRanking {
        candidates: vec![
            record("32x16", FidelityClass::Exact, false, 800),
            record("32x8", FidelityClass::Exact, true, 1000),
        ],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    result.sites.insert(window_key.clone(), window_ranking);

    let plan = result.kernel_plan(false);

    assert_eq!(plan.get(&window_key), Some("32x16"));
}

#[test]
fn unknown_names_drop_their_key() {
    let window_key = key("nlm_window");
    let accumulate_key = key("nlm_accumulate");
    let mut result = TuneResult::default();
    let known = SiteRanking {
        candidates: vec![record("32x8", FidelityClass::Exact, true, 1000)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    let unknown = SiteRanking {
        candidates: vec![
            record("32x8", FidelityClass::Exact, true, 1000),
            record("64x64", FidelityClass::Exact, false, 900),
        ],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    result.sites.insert(window_key.clone(), known);
    result.sites.insert(accumulate_key.clone(), unknown);

    let dropped = result.retain_known_with(|site, name| site == "nlm_window" || name != "64x64");

    assert_eq!(dropped, 1);
    assert!(result.sites.contains_key(&window_key));
    assert!(!result.sites.contains_key(&accumulate_key));
}

#[test]
fn a_newer_result_replaces_keys_it_holds_and_keeps_the_rest() {
    let window_key = key("nlm_window");
    let collab_key = key("collab_fused");
    let mut stored = TuneResult::default();
    let old_ranking = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    stored.sites.insert(window_key.clone(), old_ranking.clone());
    stored.sites.insert(collab_key.clone(), old_ranking);

    let mut newer = TuneResult::default();
    let new_ranking = SiteRanking {
        candidates: vec![baseline_record(500)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    newer.sites.insert(window_key.clone(), new_ranking.clone());

    stored.merge(newer);

    assert_eq!(stored.sites[&window_key], new_ranking);
    assert!(stored.sites.contains_key(&collab_key));
}

#[test]
fn a_failed_key_is_settled_without_a_ranking() {
    let window_key = key("nlm_window");
    let collab_key = key("collab_fused");
    let mut result = TuneResult::default();
    result.failed.insert(window_key.clone());

    let plan = result.kernel_plan(false);

    assert!(result.is_settled(&window_key));
    assert!(!result.is_settled(&collab_key));
    assert!(result.sites.is_empty());
    assert_eq!(plan.get(&window_key), None);
}

#[test]
fn a_partial_ranking_is_not_settled_but_its_choice_runs() {
    let window_key = key("nlm_window");
    let mut result = TuneResult::default();
    let ranking = SiteRanking {
        candidates: vec![
            record("32x16", FidelityClass::Exact, false, 800),
            record("32x8", FidelityClass::Exact, true, 1000),
        ],
        partial: true,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    result.sites.insert(window_key.clone(), ranking);

    let plan = result.kernel_plan(false);

    assert!(!result.is_settled(&window_key));
    assert!(result.is_partial(&window_key));
    assert_eq!(plan.get(&window_key), Some("32x16"));
}

#[test]
fn a_full_ranking_replaces_a_partial_one() {
    let window_key = key("nlm_window");
    let mut stored = TuneResult::default();
    let partial = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: true,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    stored.sites.insert(window_key.clone(), partial);

    let mut newer = TuneResult::default();
    let full = SiteRanking {
        candidates: vec![baseline_record(900)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    newer.sites.insert(window_key.clone(), full.clone());

    stored.merge(newer);

    assert_eq!(stored.sites[&window_key], full);
    assert!(stored.is_settled(&window_key));
    assert!(!stored.is_partial(&window_key));
}

#[test]
fn merging_keeps_the_newer_outcome_of_each_key() {
    let window_key = key("nlm_window");
    let accumulate_key = key("nlm_accumulate");
    let collab_key = key("collab_fused");
    let ranking = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    let mut stored = TuneResult::default();
    stored.sites.insert(window_key.clone(), ranking.clone());
    stored.failed.insert(collab_key.clone());
    stored.failed.insert(accumulate_key.clone());

    let mut newer = TuneResult::default();
    newer.failed.insert(window_key.clone());
    newer.sites.insert(collab_key.clone(), ranking);

    stored.merge(newer);

    assert!(!stored.sites.contains_key(&window_key));
    assert!(stored.failed.contains(&window_key));
    assert!(stored.sites.contains_key(&collab_key));
    assert!(!stored.failed.contains(&collab_key));
    assert!(stored.failed.contains(&accumulate_key));
}

fn profile_ranking(baseline_ns: u64, f32_ns: u64) -> ProfileRanking {
    ProfileRanking {
        complete: true,
        profiles: vec![
            ProfileRecord {
                profile: PipelineProfile::Baseline,
                median_frame_ns: baseline_ns,
                blocks: 3,
            },
            ProfileRecord {
                profile: PipelineProfile::F32Search,
                median_frame_ns: f32_ns,
                blocks: 3,
            },
        ],
        sites: Vec::new(),
    }
}

#[test]
fn a_profile_must_beat_the_baseline_by_three_percent() {
    let barely = profile_ranking(1000, 975);
    let clearly = profile_ranking(1000, 960);

    assert_eq!(choose_profile(&barely, false), PipelineProfile::Baseline);
    assert_eq!(choose_profile(&clearly, false), PipelineProfile::F32Search);
}

#[test]
fn deterministic_always_picks_the_baseline_profile() {
    let clearly = profile_ranking(1000, 500);

    assert_eq!(choose_profile(&clearly, true), PipelineProfile::Baseline);
}

#[test]
fn a_configuration_is_complete_only_when_every_profile_key_is() {
    let mut result = TuneResult::default();
    let complete = profile_ranking(1000, 1000);
    let mut partial = profile_ranking(1000, 1000);
    partial.complete = false;
    result.profiles.insert("luma".to_string(), complete);
    result.profiles.insert("chroma".to_string(), partial);

    let luma_only = ["luma".to_string()];
    let both = ["luma".to_string(), "chroma".to_string()];
    let missing = ["luma".to_string(), "yuv".to_string()];

    assert!(result.is_complete(&luma_only));
    assert!(!result.is_complete(&both));
    assert!(!result.is_complete(&missing));
}

#[test]
fn a_partial_ranking_never_replaces_a_full_one() {
    let window_key = key("nlm_window");
    let mut stored = TuneResult::default();
    let full = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    stored.sites.insert(window_key.clone(), full.clone());

    let mut newer = TuneResult::default();
    let progress = CompileProgress { left: 3, total: 8 };
    let partial = SiteRanking {
        candidates: vec![baseline_record(900)],
        partial: true,
        progress,
        identity: String::new(),
    };
    newer.sites.insert(window_key.clone(), partial);

    stored.merge(newer);

    assert_eq!(stored.sites[&window_key], full);
    assert!(stored.is_settled(&window_key));
}

#[test]
fn a_partial_ranking_replaces_an_older_partial_one() {
    let window_key = key("nlm_window");
    let mut stored = TuneResult::default();
    let older_progress = CompileProgress { left: 5, total: 8 };
    let older = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: true,
        progress: older_progress,
        identity: String::new(),
    };
    stored.sites.insert(window_key.clone(), older);

    let mut newer = TuneResult::default();
    let newer_progress = CompileProgress { left: 2, total: 8 };
    let partial = SiteRanking {
        candidates: vec![baseline_record(900)],
        partial: true,
        progress: newer_progress,
        identity: String::new(),
    };
    newer.sites.insert(window_key.clone(), partial.clone());

    stored.merge(newer);

    assert_eq!(stored.sites[&window_key], partial);
}

#[test]
fn a_partial_ranking_that_timed_fewer_candidates_keeps_the_older_partial_one() {
    let window_key = key("nlm_window");
    let mut stored = TuneResult::default();
    let baseline = baseline_record(1000);
    let faster = record("32x16", FidelityClass::Exact, false, 800);
    let older = SiteRanking {
        candidates: vec![baseline, faster],
        partial: true,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    stored.sites.insert(window_key.clone(), older.clone());

    let mut newer = TuneResult::default();
    let fewer = SiteRanking {
        candidates: vec![baseline_record(900)],
        partial: true,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    newer.sites.insert(window_key.clone(), fewer);

    stored.merge(newer);

    assert_eq!(stored.sites[&window_key], older);
}

#[test]
fn a_complete_profile_stops_being_complete_once_one_of_its_sites_is_partial() {
    let window_key = key("nlm_window");
    let mut result = TuneResult::default();
    let mut profile = profile_ranking(1000, 1000);
    profile.sites = vec![window_key.clone()];
    result.profiles.insert("luma".to_string(), profile);
    let ranking = SiteRanking {
        candidates: vec![baseline_record(1000)],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    result.sites.insert(window_key.clone(), ranking);
    let profile_keys = ["luma".to_string()];

    let complete_while_full = result.is_complete(&profile_keys);
    let full = result.sites.get_mut(&window_key).expect("the key is ranked");
    full.partial = true;
    let complete_once_partial = result.is_complete(&profile_keys);

    assert!(complete_while_full);
    assert!(!complete_once_partial);
}
