use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
use crate::tune::gpu::{bind, read_f32};
use crate::tune::precompile::CompileAccount;
use crate::tune::session::{compile_members, timed_count};
use crate::tune::sites::{AccumOutputs, AccumulateCandidate, AccumulateSite, WindowCandidate};
use crate::tune::{
    CandidateRecord,
    CompareResult,
    CompileProgress,
    DeviceLimits,
    EngineKernels,
    ErrorBound,
    Fidelity,
    FidelityClass,
    IntegerBound,
    KernelSelector,
    LaunchShape,
    SiteCandidate,
    SiteKey,
    SiteRanking,
    TuneLedger,
    TunePhase,
    TuneResult,
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

/// The window bound the integer probe's candidates are held to against the baseline.
const INTEGER_PROBE_BOUND: IntegerBound = IntegerBound::Bound(ErrorBound {
    min_psnr: 80.0,
    max_abs: 1.0e-3,
});

/// What the reference and the follower add to the first accumulator. It is inside the bound but never
/// bit identical to the baseline.
const NUDGE_OFFSET: f32 = 1.0e-6;

#[cube(launch_unchecked)]
fn nudge_first_value(values: &mut Array<f32>, offset: f32) {
    if ABSOLUTE_POS == 0 {
        values[0usize] += offset;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerProbe {
    Baseline,
    Reference,
    Follower,
    Stray,
}

impl SiteCandidate for IntegerProbe {
    const SITE: &'static str = "integer_probe";
    const ALL: &'static [Self] = &[
        IntegerProbe::Baseline,
        IntegerProbe::Reference,
        IntegerProbe::Follower,
        IntegerProbe::Stray,
    ];

    fn name(self) -> &'static str {
        match self {
            IntegerProbe::Baseline => "baseline",
            IntegerProbe::Reference => "reference",
            IntegerProbe::Follower => "follower",
            IntegerProbe::Stray => "stray",
        }
    }

    fn fidelity(self) -> Fidelity {
        match self {
            IntegerProbe::Baseline => Fidelity::Exact,
            _ => Fidelity::ExactInteger {
                reference: INTEGER_PROBE_BOUND,
            },
        }
    }
}

/// The accumulate site with an integer reference and two exact integer candidates checked against it.
///
/// The follower launches what the reference launches. In an output check that is the baseline with one
/// accumulator nudged by [NUDGE_OFFSET]. The stray launches the plain baseline, so it matches the
/// baseline but not the reference. `SCREENED` sets whether the candidates are pre-screened.
#[derive(Clone)]
struct IntegerProbeSite<const SCREENED: bool> {
    inner: AccumulateSite<TestRuntime>,
    /// Whether the reference launches nothing, so it fails its check against the baseline.
    reference_breaks: bool,
    /// Whether the reference's launch panics, the way a candidate that fails to compile does.
    reference_compile_fails: bool,
    /// Every candidate launched, in order.
    launches: Arc<Mutex<Vec<IntegerProbe>>>,
    /// Whether the next launch follows a cleared output set, which only an output check does.
    ///
    /// The nudge runs only then, so timed launches stay as cheap as the baseline.
    cleared: Arc<AtomicBool>,
}

impl<const SCREENED: bool> IntegerProbeSite<SCREENED> {
    fn new(reference_breaks: bool, reference_compile_fails: bool) -> Self {
        let client = test_client();
        let spec = AccumulateSpec {
            width: 96,
            height: 64,
            channels: ChannelMode::Luma,
            q_x: 1,
            q_y: 1,
        };
        let inner = accumulate_site(&client, &spec);

        Self {
            inner,
            reference_breaks,
            reference_compile_fails,
            launches: Arc::default(),
            cleared: Arc::default(),
        }
    }

    fn launch_count(&self, candidate: IntegerProbe) -> usize {
        let launches = self.launches.lock().expect("the launch log is not poisoned");
        launches.iter().filter(|&&launched| launched == candidate).count()
    }

    fn launch_reference(&self, outputs: &AccumOutputs, checking: bool) {
        if self.reference_breaks {
            return;
        }

        self.inner.launch(AccumulateCandidate::Thin32x16, outputs);
        if !checking {
            return;
        }

        let client = self.inner.client();
        let values = bind::<TestRuntime>(&outputs.accum, outputs.accum_len);
        let cube_count = CubeCount::new_single();
        let cube_dim = CubeDim::new_1d(1);

        // SAFETY: the accumulators hold at least one value, and only the first is written.
        unsafe {
            nudge_first_value::launch_unchecked::<TestRuntime>(
                client,
                cube_count,
                cube_dim,
                values,
                NUDGE_OFFSET,
            );
        }
    }
}

impl<const SCREENED: bool> VariantSite<TestRuntime> for IntegerProbeSite<SCREENED> {
    type Candidate = IntegerProbe;
    type Outputs = AccumOutputs;

    const PRESCREEN: bool = SCREENED;

    fn client(&self) -> &ComputeClient<TestRuntime> {
        self.inner.client()
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(IntegerProbe::SITE, self.inner.width, self.inner.height).build()
    }

    fn shape(&self, _candidate: IntegerProbe) -> LaunchShape {
        self.inner.shape(AccumulateCandidate::Thin32x16)
    }

    fn shared_bytes(&self, _candidate: IntegerProbe) -> usize {
        0
    }

    fn launch(&self, candidate: IntegerProbe, outputs: &AccumOutputs) {
        {
            let mut launches = self.launches.lock().expect("the launch log is not poisoned");
            launches.push(candidate);
        }

        let checking = self.cleared.swap(false, Ordering::Relaxed);
        let compile_fails = candidate == IntegerProbe::Reference && self.reference_compile_fails;
        if compile_fails {
            panic!("the reference failed to compile");
        }

        match candidate {
            IntegerProbe::Baseline | IntegerProbe::Stray => {
                self.inner.launch(AccumulateCandidate::Thin32x16, outputs);
            },
            IntegerProbe::Reference | IntegerProbe::Follower => {
                self.launch_reference(outputs, checking);
            },
        }
    }

    fn real_outputs(&self) -> AccumOutputs {
        self.inner.real_outputs()
    }

    fn scratch_outputs(&self) -> AccumOutputs {
        self.inner.scratch_outputs()
    }

    fn clear_outputs(&self, outputs: &AccumOutputs) {
        self.cleared.store(true, Ordering::Relaxed);
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

    fn integer_reference(&self) -> Option<IntegerProbe> {
        Some(IntegerProbe::Reference)
    }
}

fn ranked_names(ledger: &TuneLedger, key: &SiteKey) -> (Vec<String>, bool) {
    let result = ledger.result();
    let ranking = result.sites.get(key).expect("the key was ranked");
    let names = ranking
        .candidates
        .iter()
        .map(|record| record.name.clone())
        .collect();

    (names, ranking.partial)
}

fn ranked_progress(ledger: &TuneLedger, key: &SiteKey) -> CompileProgress {
    let result = ledger.result();
    let ranking = result.sites.get(key).expect("the key was ranked");

    ranking.progress
}

/// A stored result that ranks the named candidates of `site`, so a later run treats them as compiled.
fn stored_compiled<S: VariantSite<TestRuntime>>(site: &S, names: &[&str]) -> TuneResult {
    let candidates = names
        .iter()
        .map(|name| CandidateRecord {
            name: name.to_string(),
            fidelity: FidelityClass::ExactInteger,
            baseline: false,
            median_ns: 1,
            samples: 1,
        })
        .collect();
    let ranking = SiteRanking {
        candidates,
        partial: false,
        progress: CompileProgress::default(),
        identity: site.compile_identity(),
    };
    let key = site.key();
    let mut stored = TuneResult::default();
    stored.sites.insert(key, ranking);

    stored
}

/// Launches of a candidate that compiles and is checked but never timed.
const COMPILE_AND_CHECK_LAUNCHES: usize = 2;

/// A candidate that passes its check is timed from the first round on, so the launch log shows which
/// candidates passed even when a loaded device ends the tune before it ranks them.
#[test]
fn an_exact_integer_candidate_is_checked_against_the_integer_reference() {
    let site = IntegerProbeSite::<false>::new(false, false);

    tune_once(&site, false);
    let reference_launches = site.launch_count(IntegerProbe::Reference);
    let follower_launches = site.launch_count(IntegerProbe::Follower);
    let stray_launches = site.launch_count(IntegerProbe::Stray);

    assert!(
        reference_launches > COMPILE_AND_CHECK_LAUNCHES,
        "the reference is timed"
    );
    assert!(
        follower_launches > COMPILE_AND_CHECK_LAUNCHES,
        "the follower is timed"
    );
    assert_eq!(
        stray_launches, COMPILE_AND_CHECK_LAUNCHES,
        "the stray is dropped before timing"
    );
}

#[test]
fn exact_integer_candidates_are_dropped_when_their_reference_fails() {
    let site = IntegerProbeSite::<false>::new(true, false);
    let key = site.key();

    let ledger = tune_once(&site, false);
    let (names, _) = ranked_names(&ledger, &key);

    assert_eq!(names, vec!["baseline"]);
}

#[test]
fn a_spent_compile_ceiling_skips_the_exact_integer_candidates_without_panicking() {
    let site = IntegerProbeSite::<true>::new(false, true);
    let key = site.key();

    let ledger = tune_within(&site, Duration::ZERO);
    let (names, partial) = ranked_names(&ledger, &key);
    let reference_launches = site.launch_count(IntegerProbe::Reference);

    assert_eq!(names, vec!["baseline"]);
    assert!(partial);
    assert_eq!(reference_launches, 0, "the skipped reference never launches");
}

/// The follower and the stray load from the cache under the spent ceiling, so only the reference is
/// skipped. The ranking still counts them as left.
#[test]
fn a_skipped_reference_counts_the_followers_it_removes_as_left() {
    let site = IntegerProbeSite::<false>::new(false, false);
    let key = site.key();
    let ledger = TuneLedger::new(false);
    ledger.begin_compile_pass(Duration::ZERO);
    let compiled = stored_compiled(&site, &["follower", "stray"]);
    ledger.seed_compiled(&compiled);

    let ledger = tune_on(&site, ledger);
    let (names, partial) = ranked_names(&ledger, &key);
    let progress = ranked_progress(&ledger, &key);
    let reference_launches = site.launch_count(IntegerProbe::Reference);
    let follower_launches = site.launch_count(IntegerProbe::Follower);
    let stray_launches = site.launch_count(IntegerProbe::Stray);

    assert_eq!(names, vec!["baseline"]);
    assert!(partial);
    assert_eq!(progress, CompileProgress { left: 3, total: 4 });
    assert_eq!(reference_launches, 0);
    assert_eq!(follower_launches, 1, "the follower only loads");
    assert_eq!(stray_launches, 1, "the stray only loads");
}

/// The followers compile, so only the reference's failure keeps them from the pre-screen.
#[test]
fn a_reference_that_fails_to_compile_drops_followers_that_did_compile() {
    let site = IntegerProbeSite::<true>::new(false, true);
    let key = site.key();

    let ledger = tune_within(&site, Duration::from_secs(600));
    let (names, partial) = ranked_names(&ledger, &key);
    let progress = ranked_progress(&ledger, &key);
    let follower_launches = site.launch_count(IntegerProbe::Follower);
    let stray_launches = site.launch_count(IntegerProbe::Stray);

    assert_eq!(names, vec!["baseline"]);
    assert!(!partial);
    assert_eq!(progress.left, 0, "a failed compile is a drop, not a skip");
    assert_eq!(follower_launches, 1, "the follower launches only to compile");
    assert_eq!(stray_launches, 1, "the stray launches only to compile");
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
