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
    ProfileRanking,
    ProfileRecord,
    SITE_CAP,
    SiteKey,
    TuneLedger,
    TunePhase,
    TuneResult,
    WARM_WINDOW,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    median,
    profile_wins,
    site_min_cost,
    tuning_budget,
    tuning_order,
};

use super::mode::TuningMode;
use super::setup::KernelSetup;
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
    /// Site keys ranked in this pass, by the main pass and the alternative profile pass together.
    pub tuned_keys: usize,
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
struct FrameCycle<'a> {
    frames: &'a [Planes],
    next: usize,
}

impl<'a> FrameCycle<'a> {
    fn new(frames: &'a [Planes]) -> Self {
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
enum ProfilePhase {
    /// Only the baseline profile exists for this configuration on this device.
    NotApplicable,
    Compared(Vec<ProfileRecord>),
    /// The comparison could not run or finish, so a later run tries again.
    Unfinished,
}

/// What timing both profiles on whole frames found.
struct ProfileComparison {
    records: Vec<ProfileRecord>,
    /// Whether the `F32Search` census timed a launch and every key worth at least 2% of its device
    /// time is settled.
    finished: bool,
    alternative_wins: bool,
}

/// What the warm-up measured.
struct WarmUpReading {
    /// The steady per-frame wall time.
    frame_time: Duration,
    /// How long the first frames took, which is mostly kernel compilation.
    priming: Duration,
}

/// What tuning a list of keys did.
struct TunedKeys {
    /// The frame error that ended tuning early, if one did.
    frame_error: Option<anyhow::Error>,
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

    let census = take_census(planar, inputs.ledger, &mut cycle)?;
    let census_measured = check_census(&census);
    let census_time = census_frame_time(&census);
    let significant = significant_keys(&census, census_time);
    let known = known_keys(inputs);
    let order = pending_keys(census, &known, census_time, budget.full);
    let tuned = tune_keys(planar, inputs.ledger, &mut cycle, &order, &budget, inputs.started);

    let profiles = match tuned.frame_error {
        Some(_) => ProfilePhase::Unfinished,
        None => compare_profiles(planar, inputs, &budget, &reading),
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

fn key_values(census: &BTreeMap<SiteKey, CensusEntry>) -> Vec<KeyValue> {
    census
        .iter()
        .map(|(key, entry)| KeyValue {
            key: key.clone(),
            per_frame: entry.total,
            candidates: entry.candidates,
        })
        .collect()
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

/// The census keys not yet settled in `known`, most valuable first.
fn pending_keys(
    census: BTreeMap<SiteKey, CensusEntry>,
    known: &TuneResult,
    frame_time: Duration,
    full: bool,
) -> Vec<KeyValue> {
    let mut values = key_values(&census);
    values.retain(|value| !known.is_settled(&value.key));

    tuning_order(values, frame_time, full)
}

/// Tunes each key in `order` in place while the budget lasts.
///
/// A frame that fails ends tuning, keeping the keys ranked before it.
fn tune_keys(
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

    for value in order {
        let spent = started.elapsed();
        let needed = site_min_cost(value.candidates);
        if !can_afford(budget, spent, needed) {
            tracing::info!("the tuning budget is spent, the remaining keys run their baseline");
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

fn supports_f32_profile(planar: &PlanarDenoiser) -> bool {
    let is_nl4d = matches!(planar.options().algorithm, Algorithm::Nl4d(_));
    let hosts = planar.hosts();
    let device = hosts[0].device_profile();

    f32_profile_applies(is_nl4d, device)
}

/// Whether `F32Search` differs from the baseline, which needs nl4d and a legal f16 search.
pub(crate) fn f32_profile_applies(is_nl4d: bool, device: &DeviceProfile) -> bool {
    let legality = Legality::for_device(device);
    let runs_f16 = device.limits.supports_f16_search() && legality.f16_search;

    is_nl4d && runs_f16
}

/// Compares the baseline profile with `F32Search` on whole frames, when that profile applies.
///
/// A winning `F32Search` denoiser replaces `planar`. The cost estimate counts the alternative's
/// kernel compilation as one more priming. A configuration whose alternative cannot be built has
/// no profile to compare.
fn compare_profiles(
    planar: &mut PlanarDenoiser,
    inputs: &PassInputs<'_>,
    budget: &Budget,
    reading: &WarmUpReading,
) -> ProfilePhase {
    if !supports_f32_profile(planar) || inputs.ledger.deterministic() {
        return ProfilePhase::NotApplicable;
    }

    let block_frames = (2 * PROFILE_MIN_BLOCKS * PROFILE_BLOCK_FRAMES) as u32;
    let warm_frames = (inputs.frames.len() + WARM_WINDOW) as u32;
    let frames_cost = reading.frame_time * (block_frames + warm_frames);
    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    let needed = frames_cost + one_site + reading.priming;
    let spent = inputs.started.elapsed();
    if !can_afford(budget, spent, needed) {
        tracing::info!("no tuning budget is left to compare pipeline profiles");
        return ProfilePhase::Unfinished;
    }

    let setup = KernelSetup {
        ledger: Some(inputs.ledger.clone()),
        profile_override: Some(PipelineProfile::F32Search),
    };
    let options = planar.options().clone();
    let layout = planar.layout();
    let created = PlanarDenoiser::create_with(&options, layout, setup);
    let mut alternative = match created {
        Ok(alternative) => alternative,
        Err(error) => {
            tracing::warn!(%error, "cannot build the f32 search pipeline, only the baseline profile runs");
            return ProfilePhase::NotApplicable;
        },
    };

    let compared = compare_with(planar, &mut alternative, inputs, budget);
    let comparison = match compared {
        Ok(comparison) => comparison,
        Err(error) => {
            tracing::warn!(%error, "the profile comparison failed");
            return ProfilePhase::Unfinished;
        },
    };

    if comparison.alternative_wins {
        tracing::info!("the f32 search profile is faster on this device");
        std::mem::swap(planar, &mut alternative);
    }

    if comparison.finished {
        ProfilePhase::Compared(comparison.records)
    } else {
        ProfilePhase::Unfinished
    }
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
    let order = pending_keys(census, &known, census_time, budget.full);
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

    let baseline_median = median(&baseline_times);
    let alternative_median = median(&alternative_times);
    let blocks = PROFILE_MIN_BLOCKS as u32;
    let records = vec![
        ProfileRecord {
            profile: PipelineProfile::Baseline,
            median_frame_ns: baseline_median.as_nanos() as u64,
            blocks,
        },
        ProfileRecord {
            profile: PipelineProfile::F32Search,
            median_frame_ns: alternative_median.as_nanos() as u64,
            blocks,
        },
    ];

    let mut settled = inputs.stored.clone();
    let ranked = inputs.ledger.result();
    settled.merge(ranked);
    let keys_settled = significant.iter().all(|key| settled.is_settled(key));
    let finished = census_measured && keys_settled;
    let alternative_wins = profile_wins(baseline_median, alternative_median);

    Ok(ProfileComparison {
        records,
        finished,
        alternative_wins,
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
/// profiles were compared or do not apply.
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
    let tuned_keys = ranked.sites.len();
    merged.merge(ranked);

    let keys_done = summary.significant.iter().all(|key| merged.is_settled(key));
    let profiles_done = summary.profiles != ProfilePhase::Unfinished;
    let complete = summary.census_measured && keys_done && profiles_done;

    let baseline_only = vec![ProfileRecord {
        profile: PipelineProfile::Baseline,
        median_frame_ns: summary.frame_time.as_nanos() as u64,
        blocks: 0,
    }];
    let compared = match summary.profiles {
        ProfilePhase::Compared(records) => Some(records),
        _ => None,
    };

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
        };
        merged.profiles.insert(profile_key.clone(), ranking);
    }

    remember(identity, &merged);
    let saved = save_result(identity, &merged);

    TuneReport {
        outcome: TuneOutcome::Tuned,
        tuned_keys,
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
