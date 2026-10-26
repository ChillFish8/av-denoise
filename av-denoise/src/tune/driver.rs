use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use av_denoise_core::tune::{
    Budget,
    BudgetRequest,
    CensusEntry,
    DeviceIdentity,
    DeviceProfile,
    HARD_SITE_CAP,
    KeyValue,
    Legality,
    PROFILE_BLOCK_FRAMES,
    PROFILE_MIN_BLOCKS,
    PipelineProfile,
    PriorityTable,
    ProfileRanking,
    ProfileRecord,
    SITE_CAP,
    SearchMode,
    SiteKey,
    TuneLedger,
    TunePhase,
    TuneResult,
    WARM_WINDOW,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    key_values,
    median,
    pending_keys,
    profile_wins,
    site_min_cost,
    timing_spent,
    tuning_budget,
    tuning_order,
};

use super::mode::TuningMode;
use super::setup::{KernelSetup, device_priority};
use super::store::TuneStore;
use super::{reload, remember, resolve};
use crate::backend::selected_device;
use crate::host::Algorithm;
use crate::planar::{
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    plane_profile_keys,
    push_needs_retry,
};
use crate::warmup::{TuneLock, TuneLockOutcome};

/// How many passes over the tuning frames a target key gets to be reached and tuned.
const PASSES_PER_KEY: usize = 2;

/// Candidates a site is assumed to have when costing a site not yet seen.
const TYPICAL_CANDIDATES: usize = 3;

/// Time a target may take beyond its site caps, for its re-warm, pre-screen and output checks.
const SETTLE_MARGIN: Duration = Duration::from_secs(5);

/// What a tuning pass asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneRequest {
    pub budget: BudgetRequest,
    /// Tunes every key again, including those the store already holds.
    pub force: bool,
}

/// How a tuning pass ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuneOutcome {
    /// Tuning ran and its result was kept.
    Tuned,
    /// The store already held a complete result for every plane.
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
        }
    }

    fn skipped(reason: impl Into<String>, started: Instant) -> Self {
        let reason = reason.into();
        tracing::info!(%reason, "kernel tuning skipped, stored results are used");

        Self::finished(TuneOutcome::Skipped(reason), started)
    }
}

/// Hands out a fixed set of frames in order, starting over at the end.
pub(super) struct FrameCycle<'a> {
    frames: &'a [Planes],
    pub(super) next: usize,
}

impl<'a> FrameCycle<'a> {
    pub(super) fn new(frames: &'a [Planes]) -> Self {
        Self { frames, next: 0 }
    }

    fn next_frame(&mut self) -> &'a Planes {
        let frame = &self.frames[self.next];
        self.next = (self.next + 1) % self.frames.len();
        frame
    }

    fn len(&self) -> usize {
        self.frames.len()
    }
}

/// What every step of one pass reads.
struct PassInputs<'a> {
    ledger: &'a Arc<TuneLedger>,
    frames: &'a [Planes],
    stored: &'a TuneResult,
    request: &'a TuneRequest,
    /// When the budget clock started, which is once the tune lock is held.
    started: Instant,
}

/// Where the profile comparison ended.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ProfilePhase {
    /// Only the baseline profile exists for this configuration on this device.
    NotApplicable,
    /// Compared, with the keys worth at least 2% of each alternative's census device time.
    Compared {
        records: Vec<ProfileRecord>,
        significant: Vec<SiteKey>,
    },
    /// The comparison could not run or finish, so a later run tries again.
    Unfinished,
}

/// Block times of one alternative against the baseline, and whether every significant key was ranked.
pub(super) struct ProfileComparison {
    pub(super) baseline_times: Vec<Duration>,
    pub(super) alternative_times: Vec<Duration>,
    /// Keys worth at least 2% of the alternative's census device time.
    pub(super) significant: Vec<SiteKey>,
    /// Whether the alternative's census timed a launch and every key worth at least 2% of its
    /// device time is settled.
    pub(super) finished: bool,
}

/// How comparing one alternative profile with the baseline went.
pub(super) enum ProfileTrial<P> {
    /// Both were timed, and `built` runs the alternative.
    Timed { comparison: ProfileComparison, built: P },
    /// The alternative cannot be built.
    Unbuilt,
    /// The comparison failed part way.
    Failed,
    /// No tuning budget is left to compare it.
    OutOfBudget,
}

/// The fastest alternative profile compared so far.
pub(super) struct BestAlternative<P> {
    pub(super) profile: PipelineProfile,
    pub(super) median: Duration,
    pub(super) built: P,
}

/// What timing the alternative profiles produced.
pub(super) struct ProfileRun<P> {
    pub(super) phase: ProfilePhase,
    /// The fastest alternative, when it beats the baseline by the profile margin.
    pub(super) winner: Option<BestAlternative<P>>,
}

/// What the warm-up measured.
struct WarmUpReading {
    /// The steady per-frame wall time.
    frame_time: Duration,
    /// How long the first frames took, which is mostly kernel compilation.
    priming: Duration,
}

/// What tuning a list of keys did.
pub(super) struct TunedKeys {
    /// The frame error that ended tuning early, if one did.
    pub(super) frame_error: Option<anyhow::Error>,
}

/// What the timed part of a pass produced.
struct PassSummary {
    frame_time: Duration,
    /// Keys worth at least 2% of the census device time, which completeness needs settled.
    significant: Vec<SiteKey>,
    /// Whether the census timed any launch. Completeness needs it.
    census_measured: bool,
    profiles: ProfilePhase,
}

/// Tunes `planar` in place on `frames`, then saves and remembers the result.
///
/// The streams are reset and the grain measured on `frames` is dropped afterwards, so the denoiser
/// is ready for a fresh stream.
pub(crate) fn tune_planar(
    planar: &mut PlanarDenoiser,
    frames: &[Planes],
    request: &TuneRequest,
) -> TuneReport {
    let started = Instant::now();

    let Some(ledger) = planar.ledger().cloned() else {
        return TuneReport::skipped("tuning is not on for this denoiser", started);
    };

    if frames.is_empty() {
        return TuneReport::skipped("there are no frames to tune on", started);
    }

    let lock = TuneLock::acquire();
    let budget_started = Instant::now();
    let hosts = planar.hosts();
    let identity = hosts[0].device_profile().identity.clone();
    let profile_keys: Vec<String> = hosts.iter().map(|host| host.profile_key().to_string()).collect();
    let stored = reload(&identity);
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

    tracing::info!("tuning kernels for this device, this takes a while once per device and configuration");

    let inputs = PassInputs {
        ledger: &ledger,
        frames,
        stored: &stored,
        request,
        started: budget_started,
    };
    let pass = run_pass(planar, &inputs);
    planar.reset_streams();
    ledger.clear_target();
    ledger.set_phase(TunePhase::Idle);
    discard_grain_chunks(planar);

    let report = match pass {
        Ok(summary) => keep_result(&identity, &profile_keys, &stored, &ledger, summary, started),
        Err(error) => {
            let reason = format!("{error:#}");
            TuneReport::skipped(reason, started)
        },
    };

    drop(lock);
    report
}

/// Drops the grain chunks measured on the tuning frames, so a later drain holds only real frames.
fn discard_grain_chunks(planar: &mut PlanarDenoiser) {
    let drained = planar.drain_grain_chunks();
    if let Err(error) = drained {
        tracing::warn!(%error, "cannot drop the grain measured while tuning");
    }
}

fn run_pass(planar: &mut PlanarDenoiser, inputs: &PassInputs<'_>) -> Result<PassSummary, anyhow::Error> {
    let mut cycle = FrameCycle::new(inputs.frames);
    let warmed = warm_up(planar, &mut cycle, inputs)?;
    let Some(reading) = warmed else {
        anyhow::bail!("the clip is too short or the budget too small to earn back tuning");
    };

    let frame_time = reading.frame_time;

    let budget = tuning_budget(inputs.request.budget, frame_time);
    let Some(budget) = budget else {
        anyhow::bail!("the clip is too short to earn back tuning");
    };

    let spent = inputs.started.elapsed();
    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    if !can_afford(&budget, spent, one_site) {
        anyhow::bail!("the tuning budget does not cover the warm-up and one site");
    }

    inputs.ledger.begin_compile_pass(budget.total);

    let census = take_census(planar, inputs.ledger, &mut cycle)?;
    let census_measured = check_census(&census);
    let census_time = census_frame_time(&census);
    let significant = significant_keys(&census, census_time);
    let known = known_keys(inputs);
    let order = pending_keys(&census, &known, census_time, budget.full);
    let tuned = tune_keys(planar, inputs.ledger, &mut cycle, &order, &budget, inputs.started);

    let profiles = match tuned.frame_error {
        Some(_) => ProfilePhase::Unfinished,
        None => {
            let mode = inputs.ledger.search_mode();
            compare_profiles(planar, inputs, &budget, &reading, mode)
        },
    };

    Ok(PassSummary {
        frame_time,
        significant,
        census_measured,
        profiles,
    })
}

/// Pushes one frame and waits for the frame it readies, returning the time both took.
///
/// Returns `None` while the window is still filling and no frame comes out.
fn timed_frame(
    planar: &mut PlanarDenoiser,
    cycle: &mut FrameCycle<'_>,
) -> Result<Option<Duration>, anyhow::Error> {
    let planes = cycle.next_frame();
    let started = Instant::now();
    let pushed = planar.push(planes);

    if push_needs_retry(pushed)? {
        planar.recv()?;
        planar.push(planes)?;
    }

    let received = planar.recv()?;
    let elapsed = started.elapsed();

    Ok(received.map(|_| elapsed))
}

/// Pushes frames until the first denoised frame comes out.
///
/// The first frames compile the kernels, so the warm-up starts timing only after them. Returns how
/// long priming took.
fn prime(planar: &mut PlanarDenoiser, cycle: &mut FrameCycle<'_>) -> Result<Duration, anyhow::Error> {
    let started = Instant::now();
    let span = planar.window_span();
    let frame_limit = span.frame_count() * 2;

    for _ in 0..frame_limit {
        let timed = timed_frame(planar, cycle)?;
        if timed.is_some() {
            let priming = started.elapsed();
            return Ok(priming);
        }
    }

    anyhow::bail!("the denoiser produced no frame to time")
}

/// Runs frames until per-frame time holds steady, and returns its median with the priming time.
///
/// Returns `None` once a reading shows the clip is too short to earn back tuning. Frame time only
/// falls as clocks rise, so a clip that is too short at the current time stays too short. Also
/// returns `None` once the budget no longer covers the time spent plus one site.
fn warm_up(
    planar: &mut PlanarDenoiser,
    cycle: &mut FrameCycle<'_>,
    inputs: &PassInputs<'_>,
) -> Result<Option<WarmUpReading>, anyhow::Error> {
    let budget = inputs.request.budget;
    let priming = prime(planar, cycle)?;

    let started = Instant::now();
    let mut tracker = WarmUpTracker::new();
    let mut timed_frames = 0usize;

    loop {
        let timed = timed_frame(planar, cycle)?;
        let Some(frame_time) = timed else {
            anyhow::bail!("the denoiser stopped producing frames");
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
pub(super) fn warm_up_overruns(request: BudgetRequest, frame_time: Duration, spent: Duration) -> bool {
    let budget = tuning_budget(request, frame_time);
    let Some(budget) = budget else {
        return false;
    };

    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    !can_afford(&budget, spent, one_site)
}

/// Times every launch of one frame and returns each key's cost.
fn take_census(
    planar: &mut PlanarDenoiser,
    ledger: &TuneLedger,
    cycle: &mut FrameCycle<'_>,
) -> Result<BTreeMap<SiteKey, CensusEntry>, anyhow::Error> {
    ledger.set_phase(TunePhase::Census);
    let timed = timed_frame(planar, cycle);
    ledger.set_phase(TunePhase::Idle);
    timed?;

    let census = ledger.take_census();
    Ok(census)
}

/// Whether the census timed any launch for a nonzero time.
pub(super) fn census_measured(census: &BTreeMap<SiteKey, CensusEntry>) -> bool {
    census.values().any(|entry| !entry.total.is_zero())
}

/// [census_measured], with a warning when it is not.
fn check_census(census: &BTreeMap<SiteKey, CensusEntry>) -> bool {
    let measured = census_measured(census);
    if !measured {
        tracing::warn!("the census timed no launch, so this configuration is tuned again later");
    }

    measured
}

/// The device time of every launch in the census, at least one nanosecond.
fn census_frame_time(census: &BTreeMap<SiteKey, CensusEntry>) -> Duration {
    let total: Duration = census.values().map(|entry| entry.total).sum();
    total.max(Duration::from_nanos(1))
}

/// The keys worth at least 2% of frame time.
fn significant_keys(census: &BTreeMap<SiteKey, CensusEntry>, frame_time: Duration) -> Vec<SiteKey> {
    let values = key_values(census);
    let order = tuning_order(values, frame_time, false);

    order.into_iter().map(|value| value.key).collect()
}

/// The keys this pass leaves alone, which are those it ranked or failed so far.
///
/// Without `force`, the stored keys are left alone too.
fn known_keys(inputs: &PassInputs<'_>) -> TuneResult {
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
pub(super) fn tune_keys(
    planar: &mut PlanarDenoiser,
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

        let reached = drive_until_settled(planar, ledger, cycle, frame_limit, wait_limit);
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
fn budget_spent(ledger: &TuneLedger, started: Instant) -> Duration {
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
    planar: &mut PlanarDenoiser,
    ledger: &TuneLedger,
    cycle: &mut FrameCycle<'_>,
    frame_limit: usize,
    wait_limit: Duration,
) -> Result<bool, anyhow::Error> {
    let started = Instant::now();

    for _ in 0..frame_limit {
        timed_frame(planar, cycle)?;

        if ledger.target_settled() {
            return Ok(true);
        }

        if started.elapsed() >= wait_limit {
            return Ok(false);
        }
    }

    Ok(false)
}

/// The profiles worth comparing with the baseline for this denoiser on `device`, in the order
/// `table` gives them.
///
/// Every profile here differs from the baseline on this device. NLMeans has none.
pub(crate) fn alternative_profiles_for(
    is_nl4d: bool,
    device: &DeviceProfile,
    table: &PriorityTable,
) -> Vec<PipelineProfile> {
    if !is_nl4d {
        return Vec::new();
    }

    table
        .profiles
        .iter()
        .copied()
        .filter(|profile| differs_from_baseline(*profile, device))
        .collect()
}

/// Whether nl4d running `profile` on `device` builds a different pipeline from the baseline.
fn differs_from_baseline(profile: PipelineProfile, device: &DeviceProfile) -> bool {
    let legality = Legality::for_device(device);

    match profile {
        PipelineProfile::Baseline => false,
        PipelineProfile::F32Search => f32_profile_applies(true, device),
        PipelineProfile::SplitSearch => true,
        PipelineProfile::PackedAtomics => legality.packed_atomics,
    }
}

fn alternative_profiles(planar: &PlanarDenoiser) -> Vec<PipelineProfile> {
    let is_nl4d = matches!(planar.options().algorithm, Algorithm::Nl4d(_));
    let hosts = planar.hosts();
    let device = hosts[0].device_profile();
    let table = device_priority(device);

    alternative_profiles_for(is_nl4d, device, table)
}

/// Whether `F32Search` differs from the baseline, which needs nl4d and a legal f16 search.
pub(crate) fn f32_profile_applies(is_nl4d: bool, device: &DeviceProfile) -> bool {
    let legality = Legality::for_device(device);
    let runs_f16 = device.limits.supports_f16_search() && legality.f16_search;

    is_nl4d && runs_f16
}

/// Compares the applicable profiles with the baseline on whole frames, in the device table's order.
///
/// Only the fastest alternative so far stays alive. The winner, if it beats the baseline by the
/// profile margin, replaces `planar`. Each cost estimate counts the alternative's kernel compilation
/// as one more priming. A configuration where no alternative can be built has no profile to compare.
fn compare_profiles(
    planar: &mut PlanarDenoiser,
    inputs: &PassInputs<'_>,
    budget: &Budget,
    reading: &WarmUpReading,
    mode: SearchMode,
) -> ProfilePhase {
    let alternatives = alternative_profiles(planar);
    if alternatives.is_empty() || inputs.ledger.deterministic() {
        return ProfilePhase::NotApplicable;
    }

    let block_frames = (2 * PROFILE_MIN_BLOCKS * PROFILE_BLOCK_FRAMES) as u32;
    let warm_frames = (inputs.frames.len() + WARM_WINDOW) as u32;
    let frames_cost = reading.frame_time * (block_frames + warm_frames);
    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    let needed = frames_cost + one_site + reading.priming;

    let run = time_profiles(&alternatives, mode, |profile| {
        try_profile(planar, inputs, budget, needed, profile)
    });

    if let Some(mut winner) = run.winner {
        tracing::info!(
            profile = winner.profile.name(),
            "a pipeline profile is faster on this device"
        );
        std::mem::swap(planar, &mut winner.built);
    }

    run.phase
}

/// Builds `profile` and compares it with `planar`, when the budget still covers `needed`.
fn try_profile(
    planar: &mut PlanarDenoiser,
    inputs: &PassInputs<'_>,
    budget: &Budget,
    needed: Duration,
    profile: PipelineProfile,
) -> ProfileTrial<PlanarDenoiser> {
    let spent = budget_spent(inputs.ledger, inputs.started);
    if !can_afford(budget, spent, needed) {
        tracing::info!(
            profile = profile.name(),
            "no tuning budget is left to compare this profile"
        );
        return ProfileTrial::OutOfBudget;
    }

    let created = build_alternative(planar, inputs.ledger, profile);
    let mut alternative = match created {
        Ok(alternative) => alternative,
        Err(error) => {
            tracing::warn!(%error, profile = profile.name(), "cannot build a profile to compare");
            return ProfileTrial::Unbuilt;
        },
    };

    let compared = compare_with(planar, &mut alternative, inputs, budget);
    match compared {
        Ok(comparison) => ProfileTrial::Timed {
            comparison,
            built: alternative,
        },
        Err(error) => {
            tracing::warn!(%error, profile = profile.name(), "the profile comparison failed");
            ProfileTrial::Failed
        },
    }
}

/// Times `alternatives` in order through `trial`, and picks the winner.
///
/// [SearchMode::Families] stops at the first profile that does not beat the best so far, the
/// baseline included, by the profile margin. The profiles after it stay untimed and are left out
/// of the records. [SearchMode::Exhaustive] times every profile.
pub(super) fn time_profiles<P>(
    alternatives: &[PipelineProfile],
    mode: SearchMode,
    mut trial: impl FnMut(PipelineProfile) -> ProfileTrial<P>,
) -> ProfileRun<P> {
    let mut baseline_times = Vec::new();
    let mut records = Vec::new();
    let mut significant = Vec::new();
    let mut finished = true;
    let mut best: Option<BestAlternative<P>> = None;

    for (position, &profile) in alternatives.iter().enumerate() {
        let tried = trial(profile);
        let (comparison, built) = match tried {
            ProfileTrial::Timed { comparison, built } => (comparison, built),
            ProfileTrial::Unbuilt => continue,
            ProfileTrial::Failed => {
                finished = false;
                continue;
            },
            ProfileTrial::OutOfBudget => {
                finished = false;
                break;
            },
        };

        finished = finished && comparison.finished;
        significant.extend(comparison.significant);
        baseline_times.extend(comparison.baseline_times);

        let alternative_median = median(&comparison.alternative_times);
        records.push(ProfileRecord {
            profile,
            median_frame_ns: alternative_median.as_nanos() as u64,
            blocks: PROFILE_MIN_BLOCKS as u32,
        });

        let baseline_median = median(&baseline_times);
        let best_median = best
            .as_ref()
            .map_or(baseline_median, |best| best.median.min(baseline_median));
        let wins = profile_wins(best_median, alternative_median);

        let beats_best = best.as_ref().is_none_or(|best| alternative_median < best.median);
        if beats_best {
            best = Some(BestAlternative {
                profile,
                median: alternative_median,
                built,
            });
        }

        if mode == SearchMode::Families && !wins {
            let untimed: Vec<&str> = alternatives[position + 1..]
                .iter()
                .map(|profile| profile.name())
                .collect();
            if !untimed.is_empty() {
                tracing::info!(
                    profile = profile.name(),
                    ?untimed,
                    "this profile does not beat the best so far, so the rest stay untimed"
                );
            }

            break;
        }
    }

    if records.is_empty() {
        let phase = if finished {
            ProfilePhase::NotApplicable
        } else {
            ProfilePhase::Unfinished
        };
        return ProfileRun { phase, winner: None };
    }

    let baseline_median = median(&baseline_times);
    let baseline_blocks = (baseline_times.len() / PROFILE_BLOCK_FRAMES) as u32;
    let baseline_record = ProfileRecord {
        profile: PipelineProfile::Baseline,
        median_frame_ns: baseline_median.as_nanos() as u64,
        blocks: baseline_blocks,
    };
    records.insert(0, baseline_record);

    let winner = best.filter(|best| profile_wins(baseline_median, best.median));

    significant.sort();
    significant.dedup();

    let phase = if finished {
        ProfilePhase::Compared { records, significant }
    } else {
        ProfilePhase::Unfinished
    };

    ProfileRun { phase, winner }
}

/// Builds a denoiser like `planar`, on the same device, that runs `profile` and tunes through
/// `ledger`.
pub(super) fn build_alternative(
    planar: &PlanarDenoiser,
    ledger: &Arc<TuneLedger>,
    profile: PipelineProfile,
) -> Result<PlanarDenoiser, anyhow::Error> {
    let setup = KernelSetup {
        ledger: Some(ledger.clone()),
        profile_override: Some(profile),
    };
    let options = planar.options().clone();
    let layout = planar.layout();
    let placement = planar.placement();

    PlanarDenoiser::create_with(&options, layout, placement, setup)
}

/// Tunes the keys only `alternative` reaches, then times both denoisers in interleaved blocks.
fn compare_with(
    baseline: &mut PlanarDenoiser,
    alternative: &mut PlanarDenoiser,
    inputs: &PassInputs<'_>,
    budget: &Budget,
) -> Result<ProfileComparison, anyhow::Error> {
    let mut baseline_cycle = FrameCycle::new(inputs.frames);
    let mut alternative_cycle = FrameCycle::new(inputs.frames);
    let fill_limit = inputs.frames.len() * PASSES_PER_KEY + WARM_WINDOW;
    let mut produced = 0;

    for _ in 0..fill_limit {
        let timed = timed_frame(alternative, &mut alternative_cycle)?;
        if timed.is_some() {
            produced += 1;
        }

        if produced >= WARM_WINDOW {
            break;
        }
    }

    let census = take_census(alternative, inputs.ledger, &mut alternative_cycle)?;
    let census_measured = check_census(&census);
    let census_time = census_frame_time(&census);
    let significant = significant_keys(&census, census_time);
    let known = known_keys(inputs);
    let order = pending_keys(&census, &known, census_time, budget.full);
    let tuned = tune_keys(
        alternative,
        inputs.ledger,
        &mut alternative_cycle,
        &order,
        budget,
        inputs.started,
    );

    if let Some(error) = tuned.frame_error {
        return Err(error);
    }

    let mut baseline_times = Vec::new();
    let mut alternative_times = Vec::new();

    for _ in 0..PROFILE_MIN_BLOCKS {
        time_block(baseline, &mut baseline_cycle, &mut baseline_times)?;
        time_block(alternative, &mut alternative_cycle, &mut alternative_times)?;
    }

    let mut settled = inputs.stored.clone();
    let ranked = inputs.ledger.result();
    settled.merge(ranked);
    let keys_settled = significant.iter().all(|key| settled.is_settled(key));
    let finished = census_measured && keys_settled;

    Ok(ProfileComparison {
        baseline_times,
        alternative_times,
        significant,
        finished,
    })
}

/// Times one block of whole frames on the wall clock.
///
/// Each frame is read back before the next is pushed, so every time covers the frame's GPU work.
fn time_block(
    planar: &mut PlanarDenoiser,
    cycle: &mut FrameCycle<'_>,
    times: &mut Vec<Duration>,
) -> Result<(), anyhow::Error> {
    let mut timed_frames = 0;

    while timed_frames < PROFILE_BLOCK_FRAMES {
        let timed = timed_frame(planar, cycle)?;
        if let Some(frame_time) = timed {
            times.push(frame_time);
            timed_frames += 1;
        }
    }

    Ok(())
}

/// Merges this pass's rankings into the stored result, then remembers and saves it.
///
/// Each profile key is complete when every key worth at least 2% of frame time is settled and the
/// profiles were compared or do not apply. A key with a partial ranking is not settled. Each profile
/// key keeps those keys, so it stops being complete if any of them is later left partial.
fn keep_result(
    identity: &DeviceIdentity,
    profile_keys: &[String],
    stored: &TuneResult,
    ledger: &TuneLedger,
    summary: PassSummary,
    started: Instant,
) -> TuneReport {
    let mut merged = stored.clone();
    let ranked = ledger.result();
    let compile = ledger.compile_time();
    let tuned_keys = ranked.sites.values().filter(|ranking| !ranking.partial).count();
    let ranked_keys: Vec<SiteKey> = ranked.sites.keys().cloned().collect();
    merged.merge(ranked);

    let partial_keys = ranked_keys.iter().filter(|key| merged.is_partial(key)).count();

    let keys_done = summary.significant.iter().all(|key| merged.is_settled(key));
    let profiles_done = summary.profiles != ProfilePhase::Unfinished;
    let complete = summary.census_measured && keys_done && profiles_done;

    let baseline_only = vec![ProfileRecord {
        profile: PipelineProfile::Baseline,
        median_frame_ns: summary.frame_time.as_nanos() as u64,
        blocks: 0,
    }];
    let mut sites = summary.significant;
    let compared = match summary.profiles {
        ProfilePhase::Compared { records, significant } => {
            sites.extend(significant);
            Some(records)
        },
        _ => None,
    };

    sites.sort();
    sites.dedup();

    for profile_key in profile_keys {
        let previous = stored.profiles.get(profile_key);
        let kept_records = match (&compared, previous) {
            (Some(records), _) => records.clone(),
            (None, Some(previous)) if previous.profiles.len() > 1 => previous.profiles.clone(),
            (None, _) => baseline_only.clone(),
        };
        let ranking = ProfileRanking {
            complete,
            profiles: kept_records,
            sites: sites.clone(),
        };
        merged.profiles.insert(profile_key.clone(), ranking);
    }

    remember(identity, &merged);
    let saved = save_result(identity, &merged);

    TuneReport {
        outcome: TuneOutcome::Tuned,
        tuned_keys,
        partial_keys,
        compile,
        elapsed: started.elapsed(),
        saved,
    }
}

fn save_result(identity: &DeviceIdentity, result: &TuneResult) -> bool {
    let Some(store) = TuneStore::open_default() else {
        tracing::info!("kernel caching is off, so the tuning result lasts for this run only");
        return false;
    };

    let written = store.save(identity, result);
    match written {
        Ok(path) => {
            tracing::info!(?path, "saved the tuning result");
            true
        },
        Err(error) => {
            tracing::warn!(%error, "cannot save the tuning result, this run still uses it");
            false
        },
    }
}

/// Tunes the planes `options` denoises at `layout` on `frames`, and saves the result.
///
/// The pass builds its own denoiser and drops it afterwards. [TuningMode::Off] tunes nothing, and
/// any other mode tunes.
pub fn tune_planes(
    options: &PlaneOptions,
    layout: FrameLayout,
    frames: &[Planes],
    request: &TuneRequest,
) -> TuneReport {
    let started = Instant::now();

    if options.tuning == TuningMode::Off {
        return TuneReport::skipped("tuning is off", started);
    }

    let tuning_options = PlaneOptions {
        tuning: TuningMode::Auto,
        ..options.clone()
    };
    let created = PlanarDenoiser::create(&tuning_options, layout);
    let mut planar = match created {
        Ok(planar) => planar,
        Err(error) => {
            let reason = format!("{error:#}");
            return TuneReport::skipped(reason, started);
        },
    };

    tune_planar(&mut planar, frames, request)
}

/// Whether the store already holds a complete result for every plane `options` denoises.
///
/// Answers `true` when there is nothing to tune, because tuning is not [TuningMode::Auto] or no
/// device can be opened.
pub fn is_tuned(options: &PlaneOptions, layout: FrameLayout) -> bool {
    if options.tuning != TuningMode::Auto {
        return true;
    }

    let selected = selected_device(&options.accelerators, &options.device);
    let Ok(device) = selected else {
        return true;
    };

    let profile_keys = plane_profile_keys(options, layout);
    let stored = resolve(&device.identity);
    stored.is_complete(&profile_keys)
}
