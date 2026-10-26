use std::sync::Arc;
use std::time::{Duration, Instant};

use super::fixtures::{AccumulateSpec, accumulate_site, regularise_site};
use super::probe::{ProbeCandidate, ProbeSite, small_device_limits};
use super::regularise::regularise_specs;
use super::runtime::test_client;
use crate::nlmeans::ChannelMode;
use crate::tests::compilation_cache::install_test_compilation_cache;
use crate::tune::ledger::Observed;
use crate::tune::precompile::CompileAccount;
use crate::tune::sites::{
    AccumulateCandidate,
    CollabVariant,
    RegulariseCandidate,
    WindowCandidate,
    candidate_known,
};
use crate::tune::{
    CandidateRecord,
    CompileProgress,
    EngineKernels,
    FIRST_COMPILE_ESTIMATE,
    FidelityClass,
    KernelPlan,
    KernelSelector,
    SiteCandidate,
    SiteKey,
    SiteRanking,
    TuneLedger,
    TunePhase,
    TuneResult,
    VariantSite,
};

fn selector_for(ledger: &Arc<TuneLedger>) -> KernelSelector {
    let client = test_client();
    let kernels = EngineKernels {
        ledger: Some(ledger.clone()),
        ..EngineKernels::default()
    };
    let limits = small_device_limits();

    KernelSelector::with_limits(&client, kernels, limits)
}

fn assert_all_known<C: SiteCandidate>() {
    for &candidate in C::ALL {
        let name = candidate.name();
        assert!(candidate_known(C::SITE, name), "{} {name} is known", C::SITE);
    }
}

#[test]
fn every_shipped_candidate_name_is_known() {
    assert_all_known::<WindowCandidate>();
    assert_all_known::<AccumulateCandidate>();
    assert_all_known::<CollabVariant>();
    assert_all_known::<RegulariseCandidate>();
}

#[test]
fn unknown_sites_and_names_are_not_known() {
    assert!(!candidate_known("nlm_window", "64x64"));
    assert!(!candidate_known("retired_site", "32x8"));
}

#[test]
fn a_stored_result_drops_keys_with_unknown_names() {
    let mut result = TuneResult::default();
    let key = SiteKey::builder("nlm_window", 64, 64).build();
    let retired = CandidateRecord {
        name: "64x64".to_string(),
        fidelity: FidelityClass::Exact,
        baseline: false,
        median_ns: 10,
        samples: 20,
    };
    let ranking = SiteRanking {
        candidates: vec![retired],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
        search: None,
    };
    result.sites.insert(key, ranking);

    let dropped = result.retain_known();

    assert_eq!(dropped, 1);
    assert!(result.sites.is_empty());
}

#[test]
fn an_idle_ledger_runs_its_seeded_plan() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let ledger = TuneLedger::new(false);
    let mut plan = KernelPlan::baseline();
    plan.insert(site.key(), "medium");
    ledger.seed(&plan);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Medium]);
    assert_eq!(ledger.seen().get(&site.key()), Some(&3));
}

#[test]
fn a_census_times_each_key_and_counts_its_candidates() {
    let client = test_client();
    let spec = AccumulateSpec {
        width: 64,
        height: 48,
        channels: ChannelMode::Luma,
        q_x: 1,
        q_y: 0,
    };
    let site = accumulate_site(&client, &spec);
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Census);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the first run succeeds");
    selector.run(&site).expect("the second run succeeds");
    let census = ledger.take_census();
    let entry = census.get(&site.key()).expect("the key was timed");

    assert_eq!(entry.launches, 2);
    assert_eq!(entry.candidates, 2);
    assert!(ledger.take_census().is_empty(), "taking the census clears it");
}

#[test]
fn a_key_that_is_not_the_target_runs_its_plan() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);
    let other_key = SiteKey::builder("probe", 8192, 8192).build();
    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(other_key, deadline);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Small]);
    assert!(!ledger.target_settled());
}

#[test]
fn the_target_key_is_settled_as_failed_and_runs_its_baseline() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.fail_scratch = true;
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);
    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(site.key(), deadline);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the run succeeds");
    let launches = site.launches();

    assert_eq!(launches, vec![ProbeCandidate::Small]);
    assert!(ledger.target_settled());
    assert!(ledger.failed().contains(&site.key()));
    assert!(ledger.result().failed.contains(&site.key()));
    assert!(ledger.result().sites.is_empty());
}

#[test]
fn a_failed_target_key_runs_the_site_baseline() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.fail_scratch = true;
    site.site_baseline = Some(ProbeCandidate::Wide);
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);
    let key = site.key();
    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(key.clone(), deadline);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the run succeeds");
    let launches = site.launches();

    assert_eq!(launches, vec![ProbeCandidate::Wide]);
    assert!(ledger.target_settled());
    assert!(ledger.failed().contains(&key));
}

#[test]
fn a_failed_key_drops_its_seeded_choice_and_runs_the_baseline() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.fail_scratch = true;
    let ledger = TuneLedger::new(false);
    let mut plan = KernelPlan::baseline();
    plan.insert(site.key(), "medium");
    ledger.seed(&plan);
    ledger.set_phase(TunePhase::Tune);
    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(site.key(), deadline);
    let selector = selector_for(&ledger);

    selector.run(&site).expect("the tuning run succeeds");
    ledger.set_phase(TunePhase::Idle);
    selector.run(&site).expect("the idle run succeeds");
    let launches = site.launches();

    assert_eq!(launches, vec![ProbeCandidate::Small, ProbeCandidate::Small]);
    assert_eq!(ledger.plan().get(&site.key()), None);
}

#[test]
fn only_one_caller_claims_the_target() {
    let key = SiteKey::builder("probe", 64, 64).build();
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);
    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(key.clone(), deadline);

    let first = ledger.observe(&key, 3);
    let second = ledger.observe(&key, 3);

    assert!(matches!(first, Observed::Tune { .. }));
    assert!(matches!(second, Observed::Idle));
    assert!(!ledger.target_settled());
}

#[test]
fn deterministic_ledgers_count_only_exact_representatives() {
    let client = test_client();
    let specs = regularise_specs();
    let site = regularise_site(&client, &specs[0]);
    let deterministic = TuneLedger::new(true);
    let rounding = TuneLedger::new(false);
    let deterministic_selector = selector_for(&deterministic);
    let rounding_selector = selector_for(&rounding);

    deterministic_selector.run(&site).expect("the run succeeds");
    rounding_selector.run(&site).expect("the run succeeds");

    assert_eq!(deterministic.seen().get(&site.key()), Some(&1));
    assert_eq!(rounding.seen().get(&site.key()), Some(&2));
}

#[test]
fn compile_time_adds_up_per_key_and_in_total() {
    let ledger = TuneLedger::new(false);
    let window_key = SiteKey::builder("window_probe", 64, 64).build();
    let collab_key = SiteKey::builder("collab_probe", 64, 64).build();

    ledger.add_compile(
        &window_key,
        Duration::from_millis(300),
        Duration::from_millis(300),
        1,
    );
    ledger.add_compile(
        &collab_key,
        Duration::from_millis(500),
        Duration::from_millis(500),
        1,
    );
    ledger.add_compile(
        &window_key,
        Duration::from_millis(200),
        Duration::from_millis(200),
        1,
    );
    let per_key = ledger.compile_times();

    assert_eq!(ledger.compile_time(), Duration::from_secs(1));
    assert_eq!(per_key.get(&window_key), Some(&Duration::from_millis(500)));
    assert_eq!(per_key.get(&collab_key), Some(&Duration::from_millis(500)));
}

#[test]
fn a_ledger_without_a_compile_ceiling_never_runs_out() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("probe", 64, 64).build();

    ledger.add_compile(&key, Duration::from_secs(3600), Duration::from_secs(3600), 1);

    assert_eq!(ledger.compile_left(), Duration::MAX);
}

#[test]
fn the_compile_ceiling_shrinks_with_compile_time_and_restarts_each_pass() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("probe", 64, 64).build();
    let ceiling = Duration::from_secs(30);

    ledger.begin_compile_pass(ceiling);
    ledger.add_compile(&key, Duration::from_secs(20), Duration::from_secs(20), 1);
    let partly_spent = ledger.compile_left();
    ledger.add_compile(&key, Duration::from_secs(20), Duration::from_secs(20), 1);
    let fully_spent = ledger.compile_left();
    ledger.begin_compile_pass(ceiling);

    assert_eq!(partly_spent, Duration::from_secs(10));
    assert_eq!(fully_spent, Duration::ZERO);
    assert_eq!(ledger.compile_left(), ceiling);
    assert_eq!(ledger.compile_time(), Duration::ZERO);
    assert!(ledger.compile_times().is_empty());
}

#[test]
fn the_compile_estimate_is_the_slowest_batch_mean_of_the_pass() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("probe", 64, 64).build();
    let ceiling = Duration::from_secs(120);

    ledger.begin_compile_pass(ceiling);
    let before_any = ledger.compile_estimate();
    ledger.add_compile(&key, Duration::from_secs(48), Duration::from_secs(48), 4);
    ledger.add_compile(&key, Duration::from_secs(22), Duration::from_secs(22), 2);
    let after_two = ledger.compile_estimate();
    ledger.begin_compile_pass(ceiling);

    assert_eq!(before_any, FIRST_COMPILE_ESTIMATE);
    assert_eq!(after_two, Duration::from_secs(12));
    assert_eq!(ledger.compile_estimate(), FIRST_COMPILE_ESTIMATE);
}

#[test]
fn cheap_batches_never_lower_the_compile_estimate_below_the_first_estimate() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("probe", 64, 64).build();

    ledger.begin_compile_pass(Duration::from_secs(30));
    ledger.add_compile(&key, Duration::from_secs(8), Duration::from_secs(8), 4);

    assert_eq!(ledger.compile_estimate(), FIRST_COMPILE_ESTIMATE);
}

#[test]
fn a_batch_that_compiled_nothing_leaves_the_compile_estimate_alone() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("probe", 64, 64).build();

    ledger.add_compile(&key, Duration::from_millis(5), Duration::from_millis(5), 0);

    assert_eq!(ledger.compile_estimate(), FIRST_COMPILE_ESTIMATE);
}

#[test]
fn a_batch_knows_which_of_its_candidates_a_stored_ranking_names() {
    install_test_compilation_cache();
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder(ProbeCandidate::SITE, 64, 64).build();
    let small = CandidateRecord {
        name: "small".to_string(),
        fidelity: FidelityClass::Exact,
        baseline: true,
        median_ns: 20,
        samples: 20,
    };
    let wide = CandidateRecord {
        name: "wide".to_string(),
        ..small.clone()
    };
    let ranking = SiteRanking {
        candidates: vec![small, wide],
        partial: true,
        progress: CompileProgress::default(),
        identity: "probe 64x64".to_string(),
        search: None,
    };
    let mut stored = TuneResult::default();
    stored.sites.insert(key.clone(), ranking);

    ledger.seed_compiled(&stored);
    let account = CompileAccount {
        ledger: &ledger,
        key: &key,
    };
    let limits = account.limits("probe 64x64", ProbeCandidate::ALL, &[0, 1, 2, 3]);
    let later_batch = account.limits("probe 64x64", ProbeCandidate::ALL, &[1, 2]);
    let other_size = account.limits("probe 62x64", ProbeCandidate::ALL, &[0, 1, 2, 3]);

    assert_eq!(limits.compiled_before, vec![0, 2]);
    assert_eq!(later_batch.compiled_before, vec![2]);
    assert!(other_size.compiled_before.is_empty());
}

#[test]
fn a_ranking_stored_without_a_compile_identity_seeds_nothing() {
    install_test_compilation_cache();
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder(ProbeCandidate::SITE, 64, 64).build();
    let small = CandidateRecord {
        name: "small".to_string(),
        fidelity: FidelityClass::Exact,
        baseline: true,
        median_ns: 20,
        samples: 20,
    };
    let ranking = SiteRanking {
        candidates: vec![small],
        partial: true,
        progress: CompileProgress::default(),
        identity: String::new(),
        search: None,
    };
    let mut stored = TuneResult::default();
    stored.sites.insert(key.clone(), ranking);

    ledger.seed_compiled(&stored);
    let account = CompileAccount {
        ledger: &ledger,
        key: &key,
    };
    let limits = account.limits("", ProbeCandidate::ALL, &[0, 1]);

    assert!(limits.compiled_before.is_empty());
}

#[test]
fn only_cold_compile_time_sets_the_compile_estimate() {
    let ledger = TuneLedger::new(false);
    let key = SiteKey::builder("collab_fused", 64, 64).build();
    let spent = Duration::from_secs(40);
    let cold_spent = Duration::from_secs(24);

    ledger.add_compile(&key, spent, cold_spent, 2);

    assert_eq!(ledger.compile_estimate(), Duration::from_secs(12));
    assert_eq!(ledger.compile_time(), spent);
}
