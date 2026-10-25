use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::consts::BASELINE_INDEX;
use super::gpu::sync;
use super::helpers::{compilation_cache_enabled, reload_compilation_cache, reload_marker};
use super::key::SiteKey;
use super::ledger::TuneLedger;
use super::site::{SiteCandidate, VariantSite};

/// How long one batch may compile, and which of its candidates an earlier run compiled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CompileLimits {
    /// The compile time the pass has left.
    pub(crate) allowance: Duration,
    /// The compile time one more cold candidate is expected to take.
    pub(crate) estimate: Duration,
    /// Indices an earlier run compiled into the on-disk cache.
    pub(crate) compiled_before: Vec<usize>,
}

/// What compiling one batch of candidates did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Precompiled {
    /// Indices whose candidate compiled, in the order given.
    pub kept: Vec<usize>,
    /// Indices whose candidate did not compile, each with the reason.
    pub dropped: Vec<(usize, String)>,
    /// Indices left uncompiled because the compile ceiling ran out.
    pub skipped: Vec<usize>,
    /// The wall time the compiles took, one run of each candidate included.
    pub spent: Duration,
    /// Whether each candidate no earlier run compiled was compiled with a sync of its own.
    pub one_by_one: bool,
    /// How many helper clients compiled the batch, zero when the site's own client did.
    pub helpers: usize,
    /// The part of `spent` the site's own client took to load what the helpers compiled.
    pub load: Duration,
    /// Whether the helpers compiled nothing while the site's own client compiled a candidate they
    /// failed, which happens when the helpers' device is lost.
    pub helpers_failed: bool,
    /// How many kept candidates an earlier run had compiled, so they loaded instead.
    pub reused: usize,
    /// The time spent loading candidates from the on-disk cache, summed over every client.
    pub reused_load: Duration,
}

impl Precompiled {
    /// Keeps every index without compiling anything.
    pub fn keeping(indices: &[usize]) -> Self {
        let kept = indices.to_vec();

        Self {
            kept,
            ..Self::default()
        }
    }

    /// Why the candidate at `index` was dropped, if it was.
    pub fn reason_for(&self, index: usize) -> Option<&str> {
        self.dropped
            .iter()
            .find(|(dropped, _)| *dropped == index)
            .map(|(_, reason)| reason.as_str())
    }
}

/// Where one key's compile time goes, and how much compiling its pass has left.
#[derive(Clone, Copy)]
pub(crate) struct CompileAccount<'a> {
    pub(crate) ledger: &'a TuneLedger,
    pub(crate) key: &'a SiteKey,
}

impl CompileAccount<'_> {
    /// The pass's compile allowance and estimate, with the indices in `indices` an earlier run
    /// compiled for exactly `identity`.
    pub(crate) fn limits<C: SiteCandidate>(
        &self,
        identity: &str,
        allowed: &[C],
        indices: &[usize],
    ) -> CompileLimits {
        let allowance = self.ledger.compile_left();
        let estimate = self.ledger.compile_estimate();
        let names = self.ledger.compiled_before(self.key, identity);
        let compiled_before = indices
            .iter()
            .copied()
            .filter(|&index| names.contains(allowed[index].name()))
            .collect();

        CompileLimits {
            allowance,
            estimate,
            compiled_before,
        }
    }

    /// The clients that compile candidates in parallel, empty when there are none.
    pub(crate) fn helpers<R: Runtime>(&self) -> &[ComputeClient<R>] {
        self.ledger.compile_helpers::<R>()
    }

    /// Stops using the helpers after they failed candidates the tuned client then compiled.
    pub(crate) fn fail_helpers(&self) {
        self.ledger.fail_compile_helpers();
    }

    /// Adds one batch's compile time to the key.
    pub(crate) fn record(&self, precompiled: &Precompiled) {
        let compiled = precompiled.kept.len() + precompiled.dropped.len();
        let cold = compiled.saturating_sub(precompiled.reused);
        let cold_spent = precompiled.spent.saturating_sub(precompiled.reused_load);
        self.ledger
            .add_compile(self.key, precompiled.spent, cold_spent, cold);
    }
}

/// Compiles the candidates at `indices` of `allowed` by launching each once into `outputs`.
///
/// A kernel compiles on its first launch and stays compiled for the client's device, so a candidate
/// launched here compiles nothing when it is timed. With `helpers`, a site that can copy itself onto
/// them, and the on-disk compilation cache on, the helpers compile the candidates in parallel and the
/// site's client then loads them from the cache one at a time. Otherwise the candidates an earlier
/// run compiled load first, one at a time. Then, when the allowance left covers the estimate for
/// every other candidate, their launches are queued and the device is synced once. Otherwise, or
/// when a launch panics or the sync fails, each candidate is launched and synced on its own, so a
/// failure names its candidate. A failed candidate is dropped with the reason, and nothing here
/// panics.
///
/// The baseline is at [BASELINE_INDEX](crate::tune::BASELINE_INDEX) and always compiles. Once the
/// allowance has passed, every other candidate not yet launched is skipped, unless an earlier run
/// compiled it. Those still load, since a load from the cache takes about a tenth of a second, until
/// one of them takes [SLOW_LOAD] or longer and so was not in the cache. Queued launches compile on
/// the device thread, so a batch checks the allowance mostly before it starts. One candidate at a
/// time, each helper, and each load after the helpers check it before each candidate.
pub(crate) fn precompile<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    limits: &CompileLimits,
    helpers: &[ComputeClient<R>],
) -> Precompiled {
    let started = Instant::now();
    let ceiling = Ceiling {
        started,
        allowance: limits.allowance,
        compiled_before: &limits.compiled_before,
        missed: AtomicBool::new(false),
        reused_nanos: AtomicU64::new(0),
    };
    let replicas = replicas(site, helpers);

    let mut precompiled = match replicas {
        Some(replicas) => parallel(site, replicas, allowed, indices, outputs, &ceiling),
        None => serial(site, allowed, indices, outputs, &ceiling, limits.estimate),
    };

    precompiled.spent = started.elapsed();
    precompiled.reused = precompiled
        .kept
        .iter()
        .filter(|index| limits.compiled_before.contains(index))
        .count();
    precompiled.reused_load = ceiling.reused_load();
    precompiled
}

/// Compiles on the site's own client.
///
/// The candidates an earlier run compiled load first, each with a sync of its own, so a slow one
/// shows the cache missed it. The rest compile in one batch when the allowance left covers the
/// estimate for each of them.
fn serial<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
    estimate: Duration,
) -> Precompiled {
    let (reused, cold): (Vec<usize>, Vec<usize>) =
        indices.iter().copied().partition(|&index| ceiling.reused(index));
    let mut precompiled = one_by_one(site, allowed, &reused, outputs, ceiling);

    let left = ceiling.left();
    let batched = fits_in_one_batch(left, estimate, cold.len());
    let rest = if batched {
        batch(site, allowed, &cold, outputs, ceiling)
    } else {
        one_by_one(site, allowed, &cold, outputs, ceiling)
    };

    precompiled.kept.extend(rest.kept);
    precompiled.dropped.extend(rest.dropped);
    precompiled.skipped.extend(rest.skipped);
    precompiled.one_by_one = rest.one_by_one;
    in_given_order(&mut precompiled, indices);
    precompiled
}

/// Sorts what `precompiled` kept, dropped and skipped into the order of `indices`.
fn in_given_order(precompiled: &mut Precompiled, indices: &[usize]) {
    let position = |index: &usize| indices.iter().position(|member| member == index);

    precompiled.kept.sort_by_key(position);
    precompiled.skipped.sort_by_key(position);
    precompiled.dropped.sort_by_key(|(index, _)| position(index));
}

/// A copy of `site` on each helper, or none when there are no helpers, the site cannot copy itself
/// or the on-disk compilation cache is off.
fn replicas<R: Runtime, S: VariantSite<R>>(site: &S, helpers: &[ComputeClient<R>]) -> Option<Vec<S>> {
    if helpers.is_empty() || !compilation_cache_enabled() {
        return None;
    }

    let copies: Option<Vec<S>> = helpers.iter().map(|helper| site.on_client(helper)).collect();
    assert_eq!(
        copies.is_some(),
        S::COPIES,
        "{} sets COPIES to {} but its on_client disagrees",
        S::Candidate::SITE,
        S::COPIES
    );

    copies
}

/// A helper compile at least this long built a kernel the on-disk cache did not hold yet.
const NEW_COMPILE: Duration = Duration::from_millis(500);

/// A load on the site's client at least this long compiled the kernel instead of loading it.
pub(crate) const SLOW_LOAD: Duration = Duration::from_secs(1);

/// How many compiles in a row may fail on one helper before it stops, as a lost device would.
const HELPER_FAILURE_LIMIT: usize = 2;

/// What a helper did with the candidate at one position.
enum Outcome {
    /// It compiled, in this long.
    Compiled(Duration),
    Failed(String),
    Skipped,
}

/// The candidates the helpers share, and what they have done with them so far.
struct HelperQueue<'a> {
    indices: &'a [usize],
    next: AtomicUsize,
    done: Mutex<Vec<(usize, Outcome)>>,
}

impl<'a> HelperQueue<'a> {
    fn new(indices: &'a [usize]) -> Self {
        Self {
            indices,
            next: AtomicUsize::new(0),
            done: Mutex::new(Vec::new()),
        }
    }

    /// The next unclaimed position and its index.
    fn claim(&self) -> Option<(usize, usize)> {
        let position = self.next.fetch_add(1, Ordering::Relaxed);
        let index = self.indices.get(position)?;

        Some((position, *index))
    }

    fn record(&self, position: usize, outcome: Outcome) {
        let mut done = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        done.push((position, outcome));
    }

    fn into_done(self) -> Vec<(usize, Outcome)> {
        self.done.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What the helpers did with a batch, by index.
#[derive(Default)]
struct HelperResults {
    compiled: Vec<usize>,
    skipped: Vec<usize>,
    /// Whether any helper compile built a kernel the cache did not hold.
    any_new: bool,
    /// Indices the helpers failed or never finished.
    failed: Vec<usize>,
}

/// Compiles the candidates on the helpers in parallel, then loads them on the site's own client.
///
/// The baseline stays off the helpers, since the census already compiled it on the site's client.
/// Each helper takes the next candidate, checks the ceiling, then compiles it with
/// [VariantSite::compile_only] on its own copy of the site and syncs. A helper stops after
/// [HELPER_FAILURE_LIMIT] failures in a row. Every outcome is recorded as it happens, so a helper
/// that panics loses only the candidate it was on. The ceiling can run over by up to one compile on
/// each helper.
///
/// When a helper built anything new, the site's client rereads the on-disk cache through a marker
/// named after the site's exact compile identity and the compiled set. Then it loads the baseline,
/// the compiled candidates and every candidate a helper failed or never finished, one at a time (see
/// [load_one_by_one] for how the ceiling applies). A load that has to compile is slow but still
/// correct, and is logged.
fn parallel<R: Runtime, S: VariantSite<R>>(
    site: &S,
    replicas: Vec<S>,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
) -> Precompiled {
    let helpers = replicas.len();
    let helper_indices: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&index| index != BASELINE_INDEX)
        .collect();
    let queue = HelperQueue::new(&helper_indices);

    run_helpers(replicas, allowed, &queue, ceiling);

    let results = sort_outcomes(queue, allowed, &helper_indices);

    if results.any_new {
        reload_for(site, allowed, &results.compiled);
    }

    let on_site: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|index| !results.skipped.contains(index))
        .collect();
    let load_started = Instant::now();
    let mut loaded = load_one_by_one(site, allowed, &on_site, &results.compiled, outputs, ceiling);
    loaded.load = load_started.elapsed();
    loaded.helpers = helpers;
    loaded.helpers_failed = helpers_failed(&results, &loaded);

    loaded.skipped.extend(results.skipped);
    loaded
        .skipped
        .sort_by_key(|skipped| indices.iter().position(|index| index == skipped));
    loaded
}

/// Whether the helpers compiled nothing while the site's client compiled a candidate they failed.
///
/// A candidate that fails on the site's client too is a genuine compile failure, which says nothing
/// about the helpers.
fn helpers_failed(results: &HelperResults, loaded: &Precompiled) -> bool {
    let site_compiled_one = results.failed.iter().any(|index| loaded.kept.contains(index));

    results.compiled.is_empty() && site_compiled_one
}

/// Runs one compile loop per helper and waits for every one of them.
fn run_helpers<R: Runtime, S: VariantSite<R>>(
    replicas: Vec<S>,
    allowed: &[S::Candidate],
    queue: &HelperQueue<'_>,
    ceiling: &Ceiling<'_>,
) {
    let clients: Vec<ComputeClient<R>> = replicas.iter().map(|replica| replica.client().clone()).collect();

    std::thread::scope(|scope| {
        let workers: Vec<_> = replicas
            .into_iter()
            .map(|replica| scope.spawn(move || compile_on_helper(&replica, allowed, queue, ceiling)))
            .collect();

        for worker in workers {
            let joined = worker.join();
            if joined.is_err() {
                tracing::warn!("a compile helper panicked, so the tuned client compiles what it held");
            }
        }
    });

    // Each copy's buffers are gone by now, so the cleanup can release its pool pages.
    for client in clients {
        client.memory_cleanup();
    }
}

/// Compiles candidates from `queue` on one helper until the queue is empty or
/// [HELPER_FAILURE_LIMIT] compiles in a row fail.
fn compile_on_helper<R: Runtime, S: VariantSite<R>>(
    replica: &S,
    allowed: &[S::Candidate],
    queue: &HelperQueue<'_>,
    ceiling: &Ceiling<'_>,
) {
    let outputs = replica.real_outputs();
    let mut failures_in_a_row = 0;

    while let Some((position, index)) = queue.claim() {
        let reused = ceiling.reused(index);
        if ceiling.skips(index, reused) {
            queue.record(position, Outcome::Skipped);
            continue;
        }

        let candidate = allowed[index];
        let started = Instant::now();
        let compiled = compile_only_one(replica, candidate, &outputs);
        let took = started.elapsed();
        ceiling.observe(took, reused);

        match compiled {
            Ok(()) => {
                failures_in_a_row = 0;
                queue.record(position, Outcome::Compiled(took));
            },
            Err(reason) => {
                failures_in_a_row += 1;
                queue.record(position, Outcome::Failed(reason));
            },
        }

        if failures_in_a_row >= HELPER_FAILURE_LIMIT {
            break;
        }
    }
}

/// Sorts the helpers' outcomes by index, counting every position no helper finished as a retry.
fn sort_outcomes<C: SiteCandidate>(
    queue: HelperQueue<'_>,
    allowed: &[C],
    helper_indices: &[usize],
) -> HelperResults {
    let mut done = queue.into_done();
    let reported: Vec<usize> = done.iter().map(|(position, _)| *position).collect();
    let unfinished = (0..helper_indices.len()).filter(|position| !reported.contains(position));
    for position in unfinished {
        let reason = "no helper finished it".to_string();
        done.push((position, Outcome::Failed(reason)));
    }

    done.sort_by_key(|(position, _)| *position);

    let mut results = HelperResults::default();

    for (position, outcome) in done {
        let index = helper_indices[position];

        match outcome {
            Outcome::Compiled(took) => {
                results.compiled.push(index);
                results.any_new |= took >= NEW_COMPILE;
            },
            Outcome::Failed(reason) => {
                let candidate = allowed[index];
                tracing::debug!(
                    site = C::SITE,
                    candidate = candidate.name(),
                    %reason,
                    "a helper did not compile a candidate, so the tuned client tries it"
                );
                results.failed.push(index);
            },
            Outcome::Skipped => results.skipped.push(index),
        }
    }

    results
}

/// Makes the site's client reread the on-disk cache, with a marker named after the site's exact
/// compile identity and the candidates the helpers compiled.
fn reload_for<R: Runtime, S: VariantSite<R>>(site: &S, allowed: &[S::Candidate], compiled: &[usize]) {
    let names: Vec<&str> = compiled.iter().map(|&index| allowed[index].name()).collect();
    let identity = site.compile_identity();
    let marker = reload_marker(&identity, &names);
    let reloaded = reload_compilation_cache(site.client(), marker);

    if let Err(error) = reloaded {
        tracing::warn!(%error, "cannot reread the compilation cache, so the candidates compile again");
    }
}

/// Loads each candidate in `indices` on the site's own client with a sync of its own.
///
/// A candidate from `from_helpers` is already compiled, so it loads whatever the ceiling, which costs
/// about a tenth of a second. A slow such load means the cache missed, and a miss repeats, so after
/// one the helper candidates past the ceiling are skipped too. Every other candidate checks the
/// ceiling with [Ceiling::skips] before it loads.
fn load_one_by_one<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    from_helpers: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
) -> Precompiled {
    let mut loaded = Precompiled {
        one_by_one: true,
        ..Precompiled::default()
    };

    for &index in indices {
        let from_helper = from_helpers.contains(&index);
        let reused = from_helper || ceiling.reused(index);
        if ceiling.skips(index, reused) {
            loaded.skipped.push(index);
            continue;
        }

        let candidate = allowed[index];
        let started = Instant::now();
        let compiled = compile_one(site, candidate, outputs);
        let took = started.elapsed();
        ceiling.observe(took, reused);

        if from_helper && took >= SLOW_LOAD {
            tracing::warn!(
                site = S::Candidate::SITE,
                candidate = candidate.name(),
                ?took,
                "a candidate a helper compiled compiled again on the tuned client, so the cache missed it"
            );
        }

        match compiled {
            Ok(()) => loaded.kept.push(index),
            Err(reason) => drop_candidate(&mut loaded, allowed, index, reason),
        }
    }

    loaded
}

/// Whether `allowance` is longer than `candidates` compiles of `estimate` each.
pub(crate) fn fits_in_one_batch(allowance: Duration, estimate: Duration, candidates: usize) -> bool {
    let count = u32::try_from(candidates).unwrap_or(u32::MAX);
    let needed = estimate.saturating_mul(count);

    allowance > needed
}

/// Queues every launch and syncs once, falling back to one candidate at a time when that fails.
fn batch<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
) -> Precompiled {
    let queued = launch_all(site, allowed, indices, outputs, ceiling);
    let synced = sync(site.client());

    match queued {
        Ok(batch) if synced.is_ok() => batch,
        _ => one_by_one(site, allowed, indices, outputs, ceiling),
    }
}

/// When compiling began, how long it may run, and which candidates load from the cache instead.
struct Ceiling<'a> {
    started: Instant,
    allowance: Duration,
    compiled_before: &'a [usize],
    /// Whether a candidate that should have loaded from the cache took [SLOW_LOAD] or longer.
    missed: AtomicBool,
    /// The time candidates an earlier run compiled took, in nanoseconds.
    reused_nanos: AtomicU64,
}

impl Ceiling<'_> {
    /// Whether an earlier run compiled the candidate at `index`.
    fn reused(&self, index: usize) -> bool {
        self.compiled_before.contains(&index)
    }

    /// The allowance not yet spent.
    fn left(&self) -> Duration {
        let elapsed = self.started.elapsed();
        self.allowance.saturating_sub(elapsed)
    }

    fn reused_load(&self) -> Duration {
        let nanos = self.reused_nanos.load(Ordering::Relaxed);
        Duration::from_nanos(nanos)
    }

    /// Whether the candidate at `index` is skipped.
    ///
    /// Only a non-baseline candidate past the allowance is. A `reused` one, which should load from
    /// the cache, is skipped only once some such load turned out slow.
    fn skips(&self, index: usize, reused: bool) -> bool {
        if index == BASELINE_INDEX || self.started.elapsed() < self.allowance {
            return false;
        }

        !reused || self.missed.load(Ordering::Relaxed)
    }

    /// Notes that a candidate took `took`, which for a `reused` one of [SLOW_LOAD] or longer means
    /// the cache missed it.
    fn observe(&self, took: Duration, reused: bool) {
        if !reused {
            return;
        }

        let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
        self.reused_nanos.fetch_add(nanos, Ordering::Relaxed);

        if took >= SLOW_LOAD {
            self.missed.store(true, Ordering::Relaxed);
        }
    }
}

/// Launches every candidate once without waiting, stopping at the first launch that panics.
fn launch_all<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
) -> Result<Precompiled, String> {
    let mut queued = Precompiled::default();

    for &index in indices {
        let reused = ceiling.reused(index);
        if ceiling.skips(index, reused) {
            queued.skipped.push(index);
            continue;
        }

        let candidate = allowed[index];
        launch_caught(site, candidate, outputs)?;
        queued.kept.push(index);
    }

    Ok(queued)
}

/// Launches `candidate` once, turning a panic into a reason.
fn launch_caught<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
) -> Result<(), String> {
    let name = candidate.name();
    let attempt = AssertUnwindSafe(|| site.launch(candidate, outputs));
    let launched = catch_unwind(attempt);

    launched.map_err(|_| format!("launching {name} panicked"))
}

/// Compiles each candidate with a sync of its own and drops those that fail.
fn one_by_one<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling<'_>,
) -> Precompiled {
    let mut precompiled = Precompiled {
        one_by_one: true,
        ..Precompiled::default()
    };

    for &index in indices {
        let reused = ceiling.reused(index);
        if ceiling.skips(index, reused) {
            precompiled.skipped.push(index);
            continue;
        }

        let candidate = allowed[index];
        let started = Instant::now();
        let compiled = compile_one(site, candidate, outputs);
        let took = started.elapsed();
        ceiling.observe(took, reused);

        match compiled {
            Ok(()) => precompiled.kept.push(index),
            Err(reason) => drop_candidate(&mut precompiled, allowed, index, reason),
        }
    }

    precompiled
}

/// Logs and records a candidate that did not compile.
fn drop_candidate<C: SiteCandidate>(
    precompiled: &mut Precompiled,
    allowed: &[C],
    index: usize,
    reason: String,
) {
    let candidate = allowed[index];
    tracing::warn!(
        site = C::SITE,
        candidate = candidate.name(),
        %reason,
        "dropping a candidate that did not compile"
    );

    precompiled.dropped.push((index, reason));
}

/// Compiles `candidate` with [VariantSite::compile_only] and syncs, returning the reason when either
/// fails.
fn compile_only_one<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
) -> Result<(), String> {
    let name = candidate.name();
    let attempt = AssertUnwindSafe(|| site.compile_only(candidate, outputs));
    let caught = catch_unwind(attempt);
    if caught.is_err() {
        return Err(format!("compiling {name} panicked"));
    }

    let synced = sync(site.client());
    synced.map_err(|error| format!("{error:#}"))
}

/// Launches `candidate` once and syncs, returning the reason when either fails.
fn compile_one<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
) -> Result<(), String> {
    launch_caught(site, candidate, outputs)?;
    let synced = sync(site.client());

    synced.map_err(|error| format!("{error:#}"))
}
