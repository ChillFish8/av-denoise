mod keep;
pub(super) mod pass;
pub(super) mod profiles;

use std::time::{Duration, Instant};

use cubecl::prelude::*;

use self::keep::{keep_result, summarise};
use self::pass::{PassInputs, run_pass};
use super::config::{CachePhase, TuningConfig, TuningMode};
use super::engine_set::EngineSet;
use super::frame::TuningFrame;
use super::identity::describe;
use super::ledger::{TuneLedger, TunePhase};
use super::lock::{TuneLock, TuneLockOutcome};
use super::profile_key::profile_key;
use super::search::SearchMode;
use super::setup::{KernelSetup, device_priority};
use super::store::{reload, resolve};
use crate::engine::EngineSpec;

/// What a tuning pass asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuneRequest {
    /// Tunes every key again, including those the store already holds.
    pub force: bool,
    /// How widely each key is searched.
    ///
    /// [SearchMode::Exhaustive] times every candidate and every profile with no compile ceiling,
    /// under the [TUNE_EXHAUSTIVE](crate::tune::TUNE_EXHAUSTIVE) budget. Its report then sums up
    /// how the families search compares.
    pub mode: SearchMode,
}

/// How a tuning pass ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuneOutcome {
    /// Tuning ran and its result was kept.
    Tuned,
    /// The store already held a complete result for every engine.
    AlreadyTuned,
    /// Nothing was tuned, for the reason given. Stored results are used.
    Skipped(String),
}

/// What a tuning pass did.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneReport {
    pub outcome: TuneOutcome,
    /// Site keys fully ranked in this pass, by the main pass and the alternative profile passes together.
    pub tuned_keys: usize,
    /// Site keys this pass ranked partially, which a later run tunes again.
    pub partial_keys: usize,
    /// The time spent compiling candidates, which the budget leaves out.
    pub compile: Duration,
    /// The whole call's wall time, any wait for the tune lock included.
    pub elapsed: Duration,
    /// Whether the result reached the store on disk.
    pub saved: bool,
    /// How the families search and the device table compare with the timings, after an exhaustive
    /// pass.
    pub summary: Option<String>,
}

impl TuneReport {
    fn finished(outcome: TuneOutcome, started: Instant) -> Self {
        Self {
            outcome,
            tuned_keys: 0,
            partial_keys: 0,
            compile: Duration::ZERO,
            elapsed: started.elapsed(),
            saved: false,
            summary: None,
        }
    }

    fn skipped(reason: impl Into<String>, started: Instant) -> Self {
        let reason = reason.into();
        tracing::info!(
            %reason,
            "kernel tuning skipped, launch sites without stored results run their baseline"
        );

        Self::finished(TuneOutcome::Skipped(reason), started)
    }
}

/// Tunes the engines `specs` describe on `frames`, and stores the result.
///
/// The pass builds its own engines and drops them afterwards. [TuningMode::Off] tunes nothing, and
/// any other mode tunes.
pub fn tune_frames(
    device: &cubecl::Device,
    client: &Client,
    specs: &[EngineSpec],
    frames: &[TuningFrame],
    config: &TuningConfig,
    request: &TuneRequest,
) -> TuneReport {
    let started = Instant::now();

    if config.mode == TuningMode::Off {
        return TuneReport::skipped("tuning is off", started);
    }

    let ledger = TuneLedger::new(config.deterministic);
    let setup = KernelSetup {
        ledger: Some(ledger),
        profile_override: None,
    };
    let built = EngineSet::build(device, client, specs, config, &setup);
    let mut set = match built {
        Ok(set) => set,
        Err(error) => {
            let reason = format!("{error:#}");
            return TuneReport::skipped(reason, started);
        },
    };

    tune_set(&mut set, frames, config, request)
}

/// Whether the store already holds a complete result for every engine `specs` describe.
///
/// Answers `true` when there is nothing to tune, because tuning is not [TuningMode::Auto].
pub fn is_tuned(
    device: &cubecl::Device,
    client: &Client,
    specs: &[EngineSpec],
    config: &TuningConfig,
) -> bool {
    if config.mode != TuningMode::Auto {
        return true;
    }

    let identity = describe(device, client);
    let profile_keys: Vec<String> = specs.iter().map(profile_key).collect();
    let stored = resolve(config.store.as_deref(), &identity);

    stored.is_complete(&profile_keys)
}

/// Tunes `set` in place on `frames`, then saves and remembers the result.
///
/// The streams are reset and the grain measured on `frames` is dropped afterwards, so the set is
/// ready for a fresh stream.
pub(crate) fn tune_set(
    set: &mut EngineSet,
    frames: &[TuningFrame],
    config: &TuningConfig,
    request: &TuneRequest,
) -> TuneReport {
    let started = Instant::now();

    let Some(ledger) = set.ledger().cloned() else {
        return TuneReport::skipped("tuning is not on for this engine set", started);
    };

    if frames.is_empty() {
        return TuneReport::skipped("there are no frames to tune on", started);
    }

    let store = config.store.as_deref();
    let hook = config.cache_hook.as_ref();
    let lock = TuneLock::acquire(store, hook);
    let budget_started = Instant::now();
    let identity = set.device().identity.clone();
    let table = device_priority(set.device());
    let profile_keys = set.profile_keys();
    let stored = reload(store, &identity);
    let deterministic = ledger.deterministic();
    let stored_plan = stored.kernel_plan(deterministic);
    ledger.seed(&stored_plan);
    ledger.seed_compiled(&stored);

    if matches!(lock, TuneLockOutcome::TimedOut) {
        return TuneReport::skipped("another process is still tuning", started);
    }

    if !request.force && stored.is_complete(&profile_keys) {
        return TuneReport::finished(TuneOutcome::AlreadyTuned, started);
    }

    tracing::debug!("warming up to see whether tuning earns back its time");

    ledger.set_search_mode(request.mode);

    let inputs = PassInputs {
        identity: &identity,
        ledger: &ledger,
        frames,
        stored: &stored,
        config,
        request,
        started: budget_started,
    };
    let pass = run_pass(set, &inputs);
    set.reset();
    ledger.clear_target();
    ledger.set_phase(TunePhase::Idle);
    ledger.set_search_mode(SearchMode::Families);
    set.discard_grain();

    let report = match pass {
        Ok(summary) => {
            let replayed = match request.mode {
                SearchMode::Exhaustive => {
                    let text = summarise(&identity, &ledger, &summary, table);
                    Some(text)
                },
                SearchMode::Families => None,
            };

            let mut report = keep_result(&inputs, &profile_keys, summary, started);
            report.summary = replayed;
            report
        },
        Err(error) => {
            let reason = format!("{error:#}");
            TuneReport::skipped(reason, started)
        },
    };

    // Tuning compiles kernels nothing else has, so they reach the shared cache before the next
    // process tunes.
    let compiled = !ledger.compile_time().is_zero();
    if let Some(hook) = hook
        && compiled
    {
        hook(CachePhase::KernelsCompiled);
    }

    drop(lock);
    report
}
