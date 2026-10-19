use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::fixtures::{
    AccumulateSpec,
    RegulariseContent,
    RegulariseSpec,
    WindowSpec,
    accumulate_site,
    regularise_site,
    window_site,
};
use super::probe::{ProbeCandidate, ProbeSite, small_device_limits};
use super::runtime::{TestRuntime, test_client, wgpu_only};
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_f32;
use crate::tune::precompile::CompileAccount;
use crate::tune::session::{compile_members, timed_count};
use crate::tune::sites::{AccumOutputs, AccumulateCandidate, AccumulateSite, WindowCandidate};
use crate::tune::{
    CompareResult,
    DeviceLimits,
    EngineKernels,
    Fidelity,
    FidelityClass,
    KernelSelector,
    LaunchShape,
    SiteCandidate,
    SiteKey,
    TuneLedger,
    TunePhase,
    VariantSite,
    fit_candidates,
};

wgpu_only! {
    use crate::tune::gpu::profile_launches;
    use crate::tune::median;
}

fn tuning_selector(ledger: &Arc<TuneLedger>, limits: DeviceLimits) -> KernelSelector<TestRuntime> {
    let client = test_client();
    let kernels = EngineKernels {
        ledger: Some(ledger.clone()),
        ..EngineKernels::default()
    };

    KernelSelector::with_limits(&client, kernels, limits)
}

fn tune_once<S: VariantSite<TestRuntime>>(site: &S, deterministic: bool) -> Arc<TuneLedger> {
    let ledger = TuneLedger::new(deterministic);
    tune_on(site, ledger)
}

/// Tunes `site` once under a pass whose compile ceiling is `ceiling`.
fn tune_within<S: VariantSite<TestRuntime>>(site: &S, ceiling: Duration) -> Arc<TuneLedger> {
    let ledger = TuneLedger::new(false);
    ledger.begin_compile_pass(ceiling);

    tune_on(site, ledger)
}

fn tune_on<S: VariantSite<TestRuntime>>(site: &S, ledger: Arc<TuneLedger>) -> Arc<TuneLedger> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    ledger.set_phase(TunePhase::Tune);

    let key = site.key();
    let deadline = Instant::now() + Duration::from_secs(20);
    ledger.set_target(key, deadline);

    let selector = tuning_selector(&ledger, limits);
    selector.run(site).expect("the tuned run succeeds");

    ledger
}

#[test]
fn tuning_a_key_ranks_the_screened_candidates_and_runs_the_choice() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Luma);
    let site = window_site(&client, &spec);
    let key = site.key();

    let ledger = tune_once(&site, false);
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");
    let names: Vec<&str> = ranking
        .candidates
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    let plan = ledger.plan();

    assert!(ledger.target_settled());
    let chosen = plan.get(&key).expect("the key has a choice");
    let unique: HashSet<&str> = names.iter().copied().collect();

    assert!(names.contains(&"32x8"), "{names:?}");
    assert!(names.contains(&chosen), "{chosen} is not in {names:?}");
    assert_eq!(unique.len(), names.len(), "{names:?}");
    for name in &names {
        let parsed = WindowCandidate::from_name(name);
        assert!(parsed.is_some(), "{name} is not a window candidate");
    }

    assert!(ranking.candidates.iter().all(|record| record.samples >= 15));
}

#[test]
fn tuning_a_key_records_its_compile_time_apart_from_timing() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Luma);
    let site = window_site(&client, &spec);
    let key = site.key();

    let started = Instant::now();
    let ledger = tune_once(&site, false);
    let run_elapsed = started.elapsed();
    let compile_time = ledger.compile_time();
    let per_key = ledger.compile_times();

    assert!(compile_time > Duration::ZERO);
    assert!(compile_time < run_elapsed, "{compile_time:?} of {run_elapsed:?}");
    assert_eq!(per_key.get(&key), Some(&compile_time));
}

#[test]
fn a_spent_compile_ceiling_ranks_only_the_baseline() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Luma);
    let site = window_site(&client, &spec);
    let key = site.key();
    let baseline = site.baseline();

    let ledger = tune_within(&site, Duration::ZERO);
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");
    let names: Vec<&str> = ranking
        .candidates
        .iter()
        .map(|record| record.name.as_str())
        .collect();

    assert_eq!(names, vec![baseline.name()]);
}

#[test]
fn a_key_cut_short_by_the_compile_ceiling_is_ranked_as_partial() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Luma);
    let site = window_site(&client, &spec);
    let key = site.key();

    let ledger = tune_within(&site, Duration::ZERO);
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");

    let progress = ranking.progress;

    assert!(ranking.partial);
    assert!(!result.is_settled(&key));
    assert!(ledger.plan().get(&key).is_some(), "the partial choice still runs");
    assert!(progress.left > 0, "{progress:?}");
    assert!(
        progress.left < progress.total,
        "the baseline compiled: {progress:?}"
    );
}

#[test]
fn a_key_tuned_within_the_compile_ceiling_is_not_partial() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Luma);
    let site = window_site(&client, &spec);
    let key = site.key();

    let ledger = tune_within(&site, Duration::from_secs(600));
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");

    assert!(!ranking.partial);
    assert!(result.is_settled(&key));
    assert_eq!(ranking.progress.left, 0);
}

#[test]
fn a_candidate_that_fails_to_compile_does_not_make_the_ranking_partial() {
    let client = test_client();
    let spec = AccumulateSpec {
        width: 96,
        height: 64,
        channels: ChannelMode::Luma,
        q_x: 1,
        q_y: 1,
    };
    let inner = accumulate_site(&client, &spec);
    let site = FaultySite {
        inner,
        broken_panics: true,
    };
    let key = site.key();

    let ledger = tune_within(&site, Duration::from_secs(600));
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");
    let names: Vec<&str> = ranking
        .candidates
        .iter()
        .map(|record| record.name.as_str())
        .collect();

    assert_eq!(names, vec!["baseline"]);
    assert!(!ranking.partial);
    assert!(result.is_settled(&key));
}

#[test]
fn a_baseline_that_does_not_compile_fails_the_site() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.panicking = vec![ProbeCandidate::Small];
    let ledger = TuneLedger::new(false);
    let key = site.key();
    let compile = CompileAccount {
        ledger: &ledger,
        key: &key,
    };

    let result = compile_members(&site, ProbeCandidate::ALL, &[0, 1, 2], &(), compile);
    let error = result.expect_err("the site fails without its baseline");

    assert!(
        error.to_string().contains("the baseline did not compile"),
        "{error:#}"
    );
}

/// Candidates are timed into scratch outputs, so the real accumulators see one launch only.
#[test]
fn candidate_timing_leaves_the_real_outputs_untouched() {
    let client = test_client();
    let spec = WindowSpec::pair(96, 64, ChannelMode::Chroma);
    let tuned_site = window_site(&client, &spec);
    let reference_site = window_site(&client, &spec);
    let baseline = reference_site.baseline();
    reference_site.launch(baseline, &reference_site.outputs);

    // Deterministic tuning keeps the chosen candidate exact, so its one real launch matches the baseline.
    tune_once(&tuned_site, true);

    let tuned_outputs = &tuned_site.outputs;
    let reference_outputs = &reference_site.outputs;
    let tuned = read_f32(&client, &tuned_outputs.accum, tuned_outputs.accum_len).expect("readback");
    let reference =
        read_f32(&client, &reference_outputs.accum, reference_outputs.accum_len).expect("readback");
    let tuned_bits: Vec<u32> = tuned.iter().map(|value| value.to_bits()).collect();
    let reference_bits: Vec<u32> = reference.iter().map(|value| value.to_bits()).collect();

    assert_eq!(tuned_bits, reference_bits);
}

#[test]
fn deterministic_tuning_never_ranks_a_rounding_candidate() {
    let client = test_client();
    let spec = RegulariseSpec {
        width: 128,
        height: 64,
        content: RegulariseContent::Textured,
        field_lambda: 1.0,
        neighbours: 4,
    };
    let site = regularise_site(&client, &spec);
    let key = site.key();
    let limits = DeviceLimits::from_client(&client);
    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
    let rounding_fits = fitted
        .iter()
        .any(|candidate| candidate.fidelity().class() == FidelityClass::Rounding);

    assert!(
        rounding_fits,
        "a rounding candidate is allowed without the deterministic filter"
    );

    let ledger = tune_once(&site, true);
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");

    assert!(
        ranking
            .candidates
            .iter()
            .all(|record| record.fidelity == FidelityClass::Exact)
    );
}

/// A candidate that writes nothing, the way a launch silently dropped by the compiler does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultCandidate {
    Baseline,
    Broken,
}

impl SiteCandidate for FaultCandidate {
    const SITE: &'static str = "fault_probe";
    const ALL: &'static [Self] = &[FaultCandidate::Baseline, FaultCandidate::Broken];

    fn name(self) -> &'static str {
        match self {
            FaultCandidate::Baseline => "baseline",
            FaultCandidate::Broken => "broken",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// The accumulate site with an extra candidate that launches nothing and so always looks fastest.
#[derive(Clone)]
struct FaultySite {
    inner: AccumulateSite<TestRuntime>,
    /// Whether the extra candidate's launch panics, the way a candidate that fails to compile does.
    broken_panics: bool,
}

impl VariantSite<TestRuntime> for FaultySite {
    type Candidate = FaultCandidate;
    type Outputs = AccumOutputs;

    fn client(&self) -> &ComputeClient<TestRuntime> {
        self.inner.client()
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(FaultCandidate::SITE, self.inner.width, self.inner.height).build()
    }

    fn shape(&self, _candidate: FaultCandidate) -> LaunchShape {
        self.inner.shape(AccumulateCandidate::Thin32x16)
    }

    fn shared_bytes(&self, _candidate: FaultCandidate) -> usize {
        0
    }

    fn launch(&self, candidate: FaultCandidate, outputs: &AccumOutputs) {
        if candidate == FaultCandidate::Baseline {
            self.inner.launch(AccumulateCandidate::Thin32x16, outputs);
        } else if self.broken_panics {
            panic!("the broken candidate failed to compile");
        }
    }

    fn real_outputs(&self) -> AccumOutputs {
        self.inner.real_outputs()
    }

    fn scratch_outputs(&self) -> AccumOutputs {
        self.inner.scratch_outputs()
    }

    fn clear_outputs(&self, outputs: &AccumOutputs) {
        self.inner.clear_outputs(outputs);
    }

    fn compare(
        &self,
        baseline: &AccumOutputs,
        candidate: &AccumOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        self.inner.compare(baseline, candidate, fidelity)
    }
}

#[test]
fn a_candidate_with_wrong_output_is_never_ranked_or_chosen() {
    let client = test_client();
    let spec = AccumulateSpec {
        width: 96,
        height: 64,
        channels: ChannelMode::Luma,
        q_x: 1,
        q_y: 1,
    };
    let inner = accumulate_site(&client, &spec);
    let site = FaultySite {
        inner,
        broken_panics: false,
    };
    let key = site.key();

    let ledger = tune_once(&site, false);
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");
    let names: Vec<&str> = ranking
        .candidates
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    let plan = ledger.plan();

    assert_eq!(names, vec!["baseline"]);
    assert_eq!(plan.get(&key), Some("baseline"));
}

#[test]
fn a_failing_tune_runs_the_baseline_and_stores_nothing() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.fail_scratch = true;
    let key = site.key();
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);

    let deadline = Instant::now() + Duration::from_secs(5);
    ledger.set_target(key.clone(), deadline);

    let limits = small_device_limits();
    let selector = tuning_selector(&ledger, limits);
    selector.run(&site).expect("a failed tune never fails the run");

    let failed = ledger.failed();
    let result = ledger.result();
    let launches = site.launches();

    assert!(ledger.target_settled());
    assert!(failed.contains(&key));
    assert!(result.sites.is_empty());
    assert_eq!(launches, vec![ProbeCandidate::Small]);
}

// These tests pin how the wgpu runtime times profiles.
wgpu_only! {
    /// The probe launches no GPU work, so the wgpu device timer reports no time for any sample.
    #[test]
    fn a_site_whose_timer_reports_nothing_fails_and_runs_its_baseline() {
        let client = test_client();
        let site = ProbeSite::new(&client);
        let key = site.key();
        let ledger = TuneLedger::new(false);
        ledger.set_phase(TunePhase::Tune);

        let deadline = Instant::now() + Duration::from_secs(5);
        ledger.set_target(key.clone(), deadline);

        let limits = small_device_limits();
        let selector = tuning_selector(&ledger, limits);
        selector.run(&site).expect("a failed tune never fails the run");

        let failed = ledger.failed();
        let result = ledger.result();
        let launches = site.launches();

        assert!(ledger.target_settled());
        assert!(failed.contains(&key));
        assert!(result.sites.is_empty());
        assert_eq!(launches.last(), Some(&ProbeCandidate::Small));
    }

    fn median_profile(site: &AccumulateSite<TestRuntime>, outputs: &AccumOutputs, launches: u32) -> Duration {
        let baseline = site.baseline();
        let mut timings = Vec::new();

        for _ in 0..5 {
            let timing = profile_launches(site, baseline, outputs, launches).expect("profiling succeeds");
            timings.push(timing);
        }

        median(&timings)
    }

    #[test]
    fn a_batch_longer_than_one_compute_pass_is_timed_in_full() {
        let client = test_client();
        let spec = AccumulateSpec {
            width: 96,
            height: 64,
            channels: ChannelMode::Luma,
            q_x: 1,
            q_y: 1,
        };
        let site = accumulate_site(&client, &spec);
        let outputs = site.scratch_outputs();

        let short = median_profile(&site, &outputs, 32);
        let long = median_profile(&site, &outputs, 256);
        let ratio = long.as_secs_f64() / short.as_secs_f64();

        // Timing only the first 32-launch pass gives a ratio near 1. A full batch lands between 4 and 8,
        // lower on fast cards and when parallel tests share the device.
        assert!(ratio > 2.5, "256 launches took {long:?}, 32 took {short:?}");
    }
}

/// One cell of a 3 by 4 grid, `index = row * 4 + column`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GridCandidate(usize);

impl SiteCandidate for GridCandidate {
    const SITE: &'static str = "grid";
    const ALL: &'static [Self] = &[
        GridCandidate(0),
        GridCandidate(1),
        GridCandidate(2),
        GridCandidate(3),
        GridCandidate(4),
        GridCandidate(5),
        GridCandidate(6),
        GridCandidate(7),
        GridCandidate(8),
        GridCandidate(9),
        GridCandidate(10),
        GridCandidate(11),
    ];

    fn name(self) -> &'static str {
        "cell"
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// A site that varies the column in stage 0 and the row in stage 1.
#[derive(Clone)]
struct GridSite {
    client: ComputeClient<TestRuntime>,
}

impl VariantSite<TestRuntime> for GridSite {
    type Candidate = GridCandidate;
    type Outputs = ();

    const STAGES: usize = 2;

    fn client(&self) -> &ComputeClient<TestRuntime> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(GridCandidate::SITE, 64, 64).build()
    }

    fn shape(&self, _candidate: GridCandidate) -> LaunchShape {
        LaunchShape { x: 8, y: 8 }
    }

    fn shared_bytes(&self, _candidate: GridCandidate) -> usize {
        0
    }

    fn in_stage(&self, stage: usize, seed: GridCandidate, candidate: GridCandidate) -> bool {
        let same_row = seed.0 / 4 == candidate.0 / 4;
        let same_column = seed.0 % 4 == candidate.0 % 4;
        if stage == 0 { same_row } else { same_column }
    }

    fn launch(&self, _candidate: GridCandidate, _outputs: &()) {}

    fn real_outputs(&self) {}

    fn scratch_outputs(&self) {}

    fn clear_outputs(&self, _outputs: &()) {}

    fn compare(
        &self,
        _baseline: &(),
        _candidate: &(),
        _fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        Ok(CompareResult::Pass)
    }
}

#[test]
fn a_staged_site_is_counted_by_what_it_times() {
    let client = test_client();
    let site = GridSite { client };
    let allowed = GridCandidate::ALL;

    let count = timed_count(&site, allowed);

    assert_eq!(count, 10);
}

#[test]
fn an_unstaged_site_is_counted_whole() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let allowed = ProbeCandidate::ALL;

    let count = timed_count(&site, allowed);

    assert_eq!(count, allowed.len());
}
