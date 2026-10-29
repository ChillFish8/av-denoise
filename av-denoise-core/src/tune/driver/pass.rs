use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::TuneRequest;
use super::profiles::{ProfilePhase, compare_profiles};
use crate::tune::budget::{
    Budget,
    BudgetRequest,
    KeyValue,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    key_values,
    pending_keys,
    site_min_cost,
    timing_spent,
    tuning_budget,
    tuning_order,
};
use crate::tune::config::TuningConfig;
use crate::tune::consts::{HARD_SITE_CAP, SITE_CAP, TUNE_EXHAUSTIVE, WARM_WINDOW};
use crate::tune::device::DeviceIdentity;
use crate::tune::engine_set::EngineSet;
use crate::tune::frame::TuningFrame;
use crate::tune::key::SiteKey;
use crate::tune::ledger::{CensusEntry, TuneLedger, TunePhase};
use crate::tune::result::TuneResult;
use crate::tune::search::SearchMode;

/// How many passes over the tuning frames a target key gets to be reached and tuned.
pub(super) const PASSES_PER_KEY: usize = 2;

/// Candidates a site is assumed to have when costing a site not yet seen.
pub(super) const TYPICAL_CANDIDATES: usize = 3;

/// Time a target may take beyond its site caps, for its re-warm, pre-screen and output checks.
const SETTLE_MARGIN: Duration = Duration::from_secs(5);

/// Hands out a fixed set of frames in order, starting over at the end.
pub(crate) struct FrameCycle<'a> {
    frames: &'a [TuningFrame],
    pub(crate) next: usize,
}

impl<'a> FrameCycle<'a> {
    pub(crate) fn new(frames: &'a [TuningFrame]) -> Self {
        Self { frames, next: 0 }
    }

    fn next_frame(&mut self) -> &'a TuningFrame {
        let frame = &self.frames[self.next];
        self.next = (self.next + 1) % self.frames.len();
        frame
    }

    fn len(&self) -> usize {
        self.frames.len()
    }
}

/// What every step of one pass reads.
pub(super) struct PassInputs<'a> {
    pub(super) identity: &'a DeviceIdentity,
    pub(super) ledger: &'a Arc<TuneLedger>,
    pub(super) frames: &'a [TuningFrame],
    pub(super) stored: &'a TuneResult,
    pub(super) config: &'a TuningConfig,
    pub(super) request: &'a TuneRequest,
    /// When the budget clock started, which is once the tune lock is held.
    pub(super) started: Instant,
}

/// What the warm-up measured.
pub(super) struct WarmUpReading {
    /// The steady per-frame wall time.
    pub(super) frame_time: Duration,
    /// How long the first frames took, which is mostly kernel compilation.
    pub(super) priming: Duration,
}

/// What tuning a list of keys did.
pub(crate) struct TunedKeys {
    /// The frame error that ended tuning early, if one did.
    pub(crate) frame_error: Option<anyhow::Error>,
}

/// What the timed part of a pass produced.
pub(super) struct PassSummary {
    pub(super) frame_time: Duration,
    /// Keys worth at least 2% of the census device time, which completeness needs settled.
    pub(super) significant: Vec<SiteKey>,
    /// Whether the census timed any launch. Completeness needs it.
    pub(super) census_measured: bool,
    pub(super) profiles: ProfilePhase,
}

pub(super) fn run_pass(set: &mut EngineSet, inputs: &PassInputs<'_>) -> Result<PassSummary, anyhow::Error> {
    let mut cycle = FrameCycle::new(inputs.frames);
    let warmed = warm_up(set, &mut cycle, inputs)?;
    let Some(reading) = warmed else {
        anyhow::bail!("the clip is too short or the budget too small to earn back tuning");
    };

    let frame_time = reading.frame_time;

    let budget = pass_budget(inputs.config.budget, inputs.request.mode, frame_time);
    let Some(budget) = budget else {
        anyhow::bail!("the clip is too short to earn back tuning");
    };

    let spent = inputs.started.elapsed();
    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    if !can_afford(&budget, spent, one_site) {
        anyhow::bail!("the tuning budget does not cover the warm-up and one site");
    }

    tracing::info!(
        adapter = %inputs.identity.adapter,
        backend = %inputs.identity.backend,
        "tuning kernels for this device, this takes a while once per device and configuration"
    );

    let compile_ceiling = match inputs.request.mode {
        SearchMode::Families => budget.total,
        SearchMode::Exhaustive => Duration::MAX,
    };
    inputs.ledger.begin_compile_pass(compile_ceiling);

    let census = take_census(set, inputs.ledger, &mut cycle)?;
    let census_measured = check_census(&census);
    let census_time = census_frame_time(&census);
    let significant = significant_keys(&census, census_time);
    let known = known_keys(inputs);
    let order = pending_keys(&census, &known, census_time, budget.full);
    let tuned = tune_keys(set, inputs.ledger, &mut cycle, &order, &budget, inputs.started);

    let profiles = match tuned.frame_error {
        Some(_) => ProfilePhase::Unfinished,
        None => {
            let mode = inputs.ledger.search_mode();
            compare_profiles(set, inputs, &budget, &reading, mode)
        },
    };

    Ok(PassSummary {
        frame_time,
        significant,
        census_measured,
        profiles,
    })
}

/// The budget `request` gets in `mode` once the warm-up reads `frame_time`.
///
/// An exhaustive pass always gets [TUNE_EXHAUSTIVE](crate::tune::TUNE_EXHAUSTIVE).
fn pass_budget(request: BudgetRequest, mode: SearchMode, frame_time: Duration) -> Option<Budget> {
    match mode {
        SearchMode::Families => tuning_budget(request, frame_time),
        SearchMode::Exhaustive => Some(Budget {
            total: TUNE_EXHAUSTIVE,
            full: true,
        }),
    }
}

/// Runs the next frame through `set` and returns the time it took.
///
/// Returns `None` while the window is still filling and no frame comes out.
pub(super) fn timed_frame(
    set: &mut EngineSet,
    cycle: &mut FrameCycle<'_>,
) -> Result<Option<Duration>, anyhow::Error> {
    let frame = cycle.next_frame();
    let timed = set.timed_frame(frame)?;

    Ok(timed)
}

/// Pushes frames until the first denoised frame comes out.
///
/// The first frames compile the kernels, so the warm-up starts timing only after them. Returns how
/// long priming took.
fn prime(set: &mut EngineSet, cycle: &mut FrameCycle<'_>) -> Result<Duration, anyhow::Error> {
    let started = Instant::now();
    let span = set.window_span();
    let frame_limit = span.frame_count() * 2;

    for _ in 0..frame_limit {
        let timed = timed_frame(set, cycle)?;
        if timed.is_some() {
            let priming = started.elapsed();
            return Ok(priming);
        }
    }

    anyhow::bail!("the engines produced no frame to time")
}

/// Runs frames until per-frame time holds steady, and returns its median with the priming time.
///
/// Returns `None` once a reading shows the clip is too short to earn back tuning. Frame time only
/// falls as clocks rise, so a clip that is too short at the current time stays too short. Also
/// returns `None` once the budget no longer covers the time spent plus one site.
fn warm_up(
    set: &mut EngineSet,
    cycle: &mut FrameCycle<'_>,
    inputs: &PassInputs<'_>,
) -> Result<Option<WarmUpReading>, anyhow::Error> {
    let budget = inputs.config.budget;
    let priming = prime(set, cycle)?;

    let started = Instant::now();
    let mut tracker = WarmUpTracker::new();
    let mut timed_frames = 0usize;

    loop {
        let timed = timed_frame(set, cycle)?;
        let Some(frame_time) = timed else {
            anyhow::bail!("the engines stopped producing frames");
        };

        timed_frames += 1;
        let elapsed = started.elapsed();
        let state = tracker.push(frame_time, elapsed);

        match state {
            WarmUpState::Stable(median) | WarmUpState::Capped(median) => {
                let reading = WarmUpReading {
                    frame_time: median,
                    priming,
                };
                return Ok(Some(reading));
            },
            WarmUpState::Warming => {},
        }

        let reading_ready = timed_frames > WARM_WINDOW;
        let affordable = tuning_budget(budget, frame_time).is_some();
        if reading_ready && !affordable {
            return Ok(None);
        }

        let recent_median = tracker.recent_median();
        let spent = inputs.started.elapsed();
        if warm_up_overruns(budget, recent_median, spent) {
            tracing::info!("the tuning budget runs out during the warm-up");
            return Ok(None);
        }
    }
}

/// Whether the warm-up has spent so much of the budget that one site no longer fits after it.
///
/// An automatic budget is worked out from `frame_time`. A request that earns no budget at all is
/// not an overrun here.
pub(crate) fn warm_up_overruns(request: BudgetRequest, frame_time: Duration, spent: Duration) -> bool {
    let budget = tuning_budget(request, frame_time);
    let Some(budget) = budget else {
        return false;
    };

    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    !can_afford(&budget, spent, one_site)
}

/// Times every launch of one frame and returns each key's cost.
pub(super) fn take_census(
    set: &mut EngineSet,
    ledger: &TuneLedger,
    cycle: &mut FrameCycle<'_>,
) -> Result<BTreeMap<SiteKey, CensusEntry>, anyhow::Error> {
    ledger.set_phase(TunePhase::Census);
    let timed = timed_frame(set, cycle);
    ledger.set_phase(TunePhase::Idle);
    timed?;

    let census = ledger.take_census();
    Ok(census)
}

/// Whether the census timed any launch for a nonzero time.
pub(crate) fn census_measured(census: &BTreeMap<SiteKey, CensusEntry>) -> bool {
    census.values().any(|entry| !entry.total.is_zero())
}

/// [census_measured], with a warning when it is not.
pub(super) fn check_census(census: &BTreeMap<SiteKey, CensusEntry>) -> bool {
    let measured = census_measured(census);
    if !measured {
        tracing::warn!("the census timed no launch, so this configuration is tuned again later");
    }

    measured
}

/// The device time of every launch in the census, at least one nanosecond.
pub(super) fn census_frame_time(census: &BTreeMap<SiteKey, CensusEntry>) -> Duration {
    let total: Duration = census.values().map(|entry| entry.total).sum();
    total.max(Duration::from_nanos(1))
}

/// The keys worth at least 2% of frame time.
pub(super) fn significant_keys(
    census: &BTreeMap<SiteKey, CensusEntry>,
    frame_time: Duration,
) -> Vec<SiteKey> {
    let values = key_values(census);
    let order = tuning_order(values, frame_time, false);

    order.into_iter().map(|value| value.key).collect()
}

/// The keys this pass leaves alone, which are those it ranked or failed so far.
///
/// Without `force`, the stored keys are left alone too.
pub(super) fn known_keys(inputs: &PassInputs<'_>) -> TuneResult {
    let ranked_so_far = inputs.ledger.result();

    if inputs.request.force {
        return ranked_so_far;
    }

    let mut known = inputs.stored.clone();
    known.merge(ranked_so_far);
    known
}

/// Tunes each key in `order` in place while the budget and the compile ceiling last.
///
/// A frame that fails ends tuning, keeping the keys ranked before it.
pub(crate) fn tune_keys(
    set: &mut EngineSet,
    ledger: &TuneLedger,
    cycle: &mut FrameCycle<'_>,
    order: &[KeyValue],
    budget: &Budget,
    started: Instant,
) -> TunedKeys {
    let frame_limit = cycle.len() * PASSES_PER_KEY + 1;
    let mut frame_error = None;
    ledger.set_phase(TunePhase::Tune);

    for (position, value) in order.iter().enumerate() {
        let spent = budget_spent(ledger, started);
        let needed = site_min_cost(value.candidates);
        if !can_afford(budget, spent, needed) {
            tracing::info!("the tuning budget is spent, the remaining keys run their baseline");
            break;
        }

        if ledger.compile_left().is_zero() {
            let skipped = key_names(&order[position..]);
            tracing::info!(
                ?skipped,
                "the compile ceiling is spent, the remaining keys run their baseline"
            );
            break;
        }

        let remaining = budget.total.saturating_sub(spent);
        let deadline = Instant::now() + remaining;
        let site_cap = SITE_CAP.min(remaining);
        let wait_limit = site_cap + HARD_SITE_CAP + SETTLE_MARGIN;
        ledger.set_target(value.key.clone(), deadline);
        tracing::info!(key = %value.key, candidates = value.candidates, "tuning a launch site");

        let reached = drive_until_settled(set, ledger, cycle, frame_limit, wait_limit);
        match reached {
            Ok(true) => {},
            Ok(false) => {
                tracing::warn!(key = %value.key, "the launch site was not reached, it keeps its baseline")
            },
            Err(error) => {
                tracing::warn!(%error, "a frame failed while tuning, stopping");
                frame_error = Some(error);
                break;
            },
        }
    }

    ledger.clear_target();
    ledger.set_phase(TunePhase::Idle);

    TunedKeys { frame_error }
}

fn key_names(values: &[KeyValue]) -> Vec<String> {
    values.iter().map(|value| value.key.to_string()).collect()
}

/// The budget spent since `started`, leaving out the time spent compiling candidates.
pub(super) fn budget_spent(ledger: &TuneLedger, started: Instant) -> Duration {
    let elapsed = started.elapsed();
    let compile = ledger.compile_time();

    timing_spent(elapsed, compile)
}

/// Pushes frames until the ledger's target settles. Returns whether it did.
///
/// It gives up after `frame_limit` frames, or once `wait_limit` has passed between frames. A
/// target settles inside the frame that reaches it, so the limits only end a wait for a site the
/// frames never reach.
fn drive_until_settled(
    set: &mut EngineSet,
    ledger: &TuneLedger,
    cycle: &mut FrameCycle<'_>,
    frame_limit: usize,
    wait_limit: Duration,
) -> Result<bool, anyhow::Error> {
    let started = Instant::now();

    for _ in 0..frame_limit {
        timed_frame(set, cycle)?;

        if ledger.target_settled() {
            return Ok(true);
        }

        if started.elapsed() >= wait_limit {
            return Ok(false);
        }
    }

    Ok(false)
}
