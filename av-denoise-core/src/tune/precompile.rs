use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::consts::BASELINE_INDEX;
use super::gpu::sync;
use super::key::SiteKey;
use super::ledger::TuneLedger;
use super::site::{SiteCandidate, VariantSite};

/// What compiling one stage's candidates did.
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
    /// The compile time the pass has left.
    pub(crate) fn left(&self) -> Duration {
        self.ledger.compile_left()
    }

    /// Adds one stage's compile time to the key.
    pub(crate) fn record(&self, spent: Duration) {
        self.ledger.add_compile(self.key, spent);
    }
}

/// Compiles the candidates at `indices` of `allowed` by launching each once into `outputs`.
///
/// A kernel compiles on its first launch and stays compiled for the client's device, so a candidate
/// launched here compiles nothing when it is timed. Every launch is queued, then the device is synced
/// once. When a launch panics or the sync fails, each candidate is launched and synced again on its own,
/// so the failure names its candidate. A failed candidate is dropped with the reason, and nothing here
/// panics.
///
/// The baseline is at [BASELINE_INDEX](crate::tune::BASELINE_INDEX) and always compiles. Once
/// `allowance` has passed, every other candidate not yet launched is skipped. Queued launches compile
/// on the device thread, so in the batch the allowance is mostly checked before the stage starts.
pub(crate) fn precompile<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    allowance: Duration,
) -> Precompiled {
    let started = Instant::now();
    let ceiling = Ceiling { started, allowance };
    let queued = launch_all(site, allowed, indices, outputs, &ceiling);
    let synced = sync(site.client());

    let mut precompiled = match queued {
        Ok(batch) if synced.is_ok() => batch,
        _ => one_by_one(site, allowed, indices, outputs, &ceiling),
    };

    if !precompiled.skipped.is_empty() {
        tracing::info!(
            site = S::Candidate::SITE,
            skipped = precompiled.skipped.len(),
            "the compile ceiling is spent, so the remaining candidates are skipped for this run"
        );
    }

    precompiled.spent = started.elapsed();
    precompiled
}

/// When compiling began and how long it may run.
struct Ceiling {
    started: Instant,
    allowance: Duration,
}

impl Ceiling {
    /// Whether the candidate at `index` is skipped, which only happens to a non-baseline candidate
    /// once the allowance has passed.
    fn skips(&self, index: usize) -> bool {
        index != BASELINE_INDEX && self.started.elapsed() >= self.allowance
    }
}

/// Launches every candidate once without waiting, stopping at the first launch that panics.
fn launch_all<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    ceiling: &Ceiling,
) -> Result<Precompiled, String> {
    let mut queued = Precompiled::default();

    for &index in indices {
        if ceiling.skips(index) {
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
    ceiling: &Ceiling,
) -> Precompiled {
    let mut precompiled = Precompiled::default();

    for &index in indices {
        if ceiling.skips(index) {
            precompiled.skipped.push(index);
            continue;
        }

        let candidate = allowed[index];
        let compiled = compile_one(site, candidate, outputs);

        match compiled {
            Ok(()) => precompiled.kept.push(index),
            Err(reason) => {
                tracing::warn!(
                    site = S::Candidate::SITE,
                    candidate = candidate.name(),
                    %reason,
                    "dropping a candidate that did not compile"
                );
                precompiled.dropped.push((index, reason));
            },
        }
    }

    precompiled
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
