use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::consts::FIRST_COMPILE_ESTIMATE;
use super::helpers::{HelperFactory, HelperSource};
use super::key::SiteKey;
use super::plan::KernelPlan;
use super::result::{SiteRanking, TuneResult, choose};

/// What the selector does at each launch while a ledger is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TunePhase {
    /// Runs the planned candidate.
    #[default]
    Idle,
    /// Runs the planned candidate and times it.
    Census,
    /// Tunes the target key in place when its site is reached.
    Tune,
}

/// One site key's cost over a census.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CensusEntry {
    /// Total time of every launch at this key during the census.
    pub total: Duration,
    pub launches: u32,
    /// Candidates tuning this key times, from those that fit the device and pass the deterministic filter.
    pub candidates: usize,
}

#[derive(Debug, Clone)]
struct Target {
    key: SiteKey,
    deadline: Instant,
    /// Whether one caller is tuning the target, so every other caller runs untuned.
    claimed: bool,
    settled: bool,
}

#[derive(Debug, Default)]
struct LedgerState {
    phase: TunePhase,
    plan: KernelPlan,
    result: TuneResult,
    census: BTreeMap<SiteKey, CensusEntry>,
    seen: BTreeMap<SiteKey, usize>,
    target: Option<Target>,
    compile: CompileState,
}

/// The time a pass spends compiling candidates, and its ceiling.
#[derive(Debug, Default)]
struct CompileState {
    total: Duration,
    by_key: BTreeMap<SiteKey, Duration>,
    /// No ceiling means compiling never runs out.
    ceiling: Option<Duration>,
    /// The highest mean compile time per candidate of any stage this pass.
    slowest_mean: Option<Duration>,
}

/// Tuning state shared by every engine one tuning pass drives.
///
/// Engines built with a ledger consult it at every launch site. Its phase and target say what each
/// launch does, and it collects the census and rankings those launches produce. Every method holds
/// the lock only briefly and never across GPU work.
#[derive(Debug, Default)]
pub struct TuneLedger {
    deterministic: bool,
    state: Mutex<LedgerState>,
    helpers: HelperSource,
}

/// The ledger's choice for one key.
pub(crate) enum LedgerChoice {
    /// The key's tuning failed, so it runs the baseline.
    Baseline,
    Named(String),
    /// The ledger has no choice for the key.
    Unplanned,
}

/// What the selector does at one launch.
pub(crate) enum Observed {
    Idle,
    Census,
    Tune { deadline: Instant },
}

impl TuneLedger {
    /// A ledger in the idle phase.
    ///
    /// With `deterministic` set, only exact candidates are tuned or run.
    pub fn new(deterministic: bool) -> Arc<Self> {
        let ledger = Self {
            deterministic,
            state: Mutex::default(),
            helpers: HelperSource::default(),
        };

        Arc::new(ledger)
    }

    pub fn deterministic(&self) -> bool {
        self.deterministic
    }

    /// Hands the ledger a way to make clients that compile candidates in parallel.
    ///
    /// The factory runs when the first stage compiles. Only the first factory is kept.
    pub fn set_compile_helpers(&self, factory: HelperFactory) {
        self.helpers.set_factory(factory);
    }

    pub fn has_compile_helpers(&self) -> bool {
        self.helpers.is_set()
    }

    /// The compile helpers that run on `R`, made on the first call, and empty when there are none.
    pub(crate) fn compile_helpers<R: Runtime>(&self) -> &[ComputeClient<R>] {
        self.helpers.clients::<R>()
    }

    /// Stops using the compile helpers after every one of them failed in a stage.
    pub(crate) fn fail_compile_helpers(&self) {
        self.helpers.mark_failed();
    }

    fn lock(&self) -> MutexGuard<'_, LedgerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds every choice in `plan` the ledger has none for yet.
    pub fn seed(&self, plan: &KernelPlan) {
        let mut state = self.lock();
        state.plan.fill_from(plan);
    }

    pub fn phase(&self) -> TunePhase {
        self.lock().phase
    }

    pub fn set_phase(&self, phase: TunePhase) {
        self.lock().phase = phase;
    }

    /// Names the key to tune when its site is next reached, finishing by `deadline`.
    pub fn set_target(&self, key: SiteKey, deadline: Instant) {
        let target = Target {
            key,
            deadline,
            claimed: false,
            settled: false,
        };

        self.lock().target = Some(target);
    }

    pub fn clear_target(&self) {
        self.lock().target = None;
    }

    /// Whether the target has been tuned or has failed, or no target is set.
    pub fn target_settled(&self) -> bool {
        let state = self.lock();
        state.target.as_ref().is_none_or(|target| target.settled)
    }

    /// Returns the census and clears it.
    pub fn take_census(&self) -> BTreeMap<SiteKey, CensusEntry> {
        let mut state = self.lock();
        std::mem::take(&mut state.census)
    }

    /// Every key reached since the ledger was built, with how many candidates tuning it times.
    pub fn seen(&self) -> BTreeMap<SiteKey, usize> {
        self.lock().seen.clone()
    }

    /// Keys whose tuning failed.
    pub fn failed(&self) -> BTreeSet<SiteKey> {
        self.lock().result.failed.clone()
    }

    /// The rankings recorded so far.
    pub fn result(&self) -> TuneResult {
        self.lock().result.clone()
    }

    /// The current choice for every key, seeded and newly tuned.
    pub fn plan(&self) -> KernelPlan {
        self.lock().plan.clone()
    }

    /// Records that `key` was reached and says what its launch should do.
    ///
    /// Only the first caller to reach an unsettled target tunes it. The target stays claimed until
    /// it settles, so every other caller at that key runs untuned.
    pub(crate) fn observe(&self, key: &SiteKey, candidates: usize) -> Observed {
        let mut state = self.lock();
        state.seen.insert(key.clone(), candidates);

        match state.phase {
            TunePhase::Idle => Observed::Idle,
            TunePhase::Census => Observed::Census,
            TunePhase::Tune => claim_target(&mut state, key),
        }
    }

    pub(crate) fn choice(&self, key: &SiteKey) -> LedgerChoice {
        let state = self.lock();

        if state.result.failed.contains(key) {
            return LedgerChoice::Baseline;
        }

        match state.plan.get(key) {
            Some(name) => LedgerChoice::Named(name.to_owned()),
            None => LedgerChoice::Unplanned,
        }
    }

    pub(crate) fn add_census(&self, key: &SiteKey, candidates: usize, elapsed: Duration) {
        let mut state = self.lock();
        let entry = state.census.entry(key.clone()).or_default();
        entry.total += elapsed;
        entry.launches += 1;
        entry.candidates = candidates;
    }

    /// Starts a pass's compile time at zero, with `ceiling` as the most it may spend compiling.
    pub fn begin_compile_pass(&self, ceiling: Duration) {
        let compile = CompileState {
            ceiling: Some(ceiling),
            ..CompileState::default()
        };

        self.lock().compile = compile;
    }

    /// Adds `spent`, the time one stage took to compile `compiled` candidates, to the time spent
    /// compiling the candidates of `key`.
    pub(crate) fn add_compile(&self, key: &SiteKey, spent: Duration, compiled: usize) {
        let mut state = self.lock();
        state.compile.total += spent;

        if compiled > 0 {
            let count = u32::try_from(compiled).unwrap_or(u32::MAX);
            let mean = spent / count;
            let slowest = state
                .compile
                .slowest_mean
                .map_or(mean, |slowest| slowest.max(mean));
            state.compile.slowest_mean = Some(slowest);
        }

        let owned = key.clone();
        let entry = state.compile.by_key.entry(owned).or_default();
        *entry += spent;
    }

    /// The time spent compiling candidates, which the budget leaves out.
    pub fn compile_time(&self) -> Duration {
        self.lock().compile.total
    }

    /// The time spent compiling each key's candidates.
    pub fn compile_times(&self) -> BTreeMap<SiteKey, Duration> {
        self.lock().compile.by_key.clone()
    }

    /// The compile time one more candidate is expected to take.
    ///
    /// It is the highest mean per candidate of any stage this pass, and never less than
    /// [FIRST_COMPILE_ESTIMATE](crate::tune::FIRST_COMPILE_ESTIMATE).
    pub fn compile_estimate(&self) -> Duration {
        let state = self.lock();
        let observed = state.compile.slowest_mean.unwrap_or(FIRST_COMPILE_ESTIMATE);
        observed.max(FIRST_COMPILE_ESTIMATE)
    }

    /// The compile time the pass has left, unlimited when no pass set a ceiling.
    pub fn compile_left(&self) -> Duration {
        let state = self.lock();

        match state.compile.ceiling {
            Some(ceiling) => ceiling.saturating_sub(state.compile.total),
            None => Duration::MAX,
        }
    }

    /// Stores `ranking` for `key` and makes its choice the one that runs.
    pub(crate) fn record_ranking(&self, key: SiteKey, ranking: SiteRanking) {
        let mut state = self.lock();
        let chosen = choose(&ranking.candidates, self.deterministic);
        if let Some(record) = chosen {
            state.plan.insert(key.clone(), &record.name);
        }

        state.result.failed.remove(&key);
        state.result.sites.insert(key.clone(), ranking);
        settle(&mut state, &key);
    }

    pub(crate) fn record_failure(&self, key: SiteKey) {
        let mut state = self.lock();
        state.plan.remove(&key);
        state.result.sites.remove(&key);
        state.result.failed.insert(key.clone());
        settle(&mut state, &key);
    }
}

/// Hands the target to the caller at `key` when it is unsettled and nobody holds it yet.
fn claim_target(state: &mut LedgerState, key: &SiteKey) -> Observed {
    let Some(target) = state.target.as_mut() else {
        return Observed::Idle;
    };

    let claimable = !target.settled && !target.claimed && &target.key == key;
    if !claimable {
        return Observed::Idle;
    }

    target.claimed = true;

    Observed::Tune {
        deadline: target.deadline,
    }
}

fn settle(state: &mut LedgerState, key: &SiteKey) {
    if let Some(target) = state.target.as_mut()
        && &target.key == key
    {
        target.claimed = false;
        target.settled = true;
    }
}
