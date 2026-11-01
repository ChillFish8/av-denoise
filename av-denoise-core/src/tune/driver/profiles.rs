use std::time::Duration;

use super::pass::{
    FrameCycle,
    PASSES_PER_KEY,
    PassInputs,
    TYPICAL_CANDIDATES,
    WarmUpReading,
    budget_spent,
    census_frame_time,
    check_census,
    known_keys,
    significant_keys,
    take_census,
    timed_frame,
    tune_keys,
};
use crate::engine::EngineSpec;
use crate::tune::budget::{Budget, can_afford, pending_keys, profile_wins, site_min_cost};
use crate::tune::consts::{PROFILE_BLOCK_FRAMES, PROFILE_MIN_BLOCKS, WARM_WINDOW};
use crate::tune::decide::median;
use crate::tune::device::DeviceProfile;
use crate::tune::engine_set::EngineSet;
use crate::tune::key::SiteKey;
use crate::tune::legality::Legality;
use crate::tune::plan::PipelineProfile;
use crate::tune::priority::PriorityTable;
use crate::tune::result::ProfileRecord;
use crate::tune::search::SearchMode;
use crate::tune::setup::device_priority;

/// Where the profile comparison ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProfilePhase {
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
pub(crate) struct ProfileComparison {
    pub(crate) baseline_times: Vec<Duration>,
    pub(crate) alternative_times: Vec<Duration>,
    /// Keys worth at least 2% of the alternative's census device time.
    pub(crate) significant: Vec<SiteKey>,
    /// Whether the alternative's census timed a launch and every key worth at least 2% of its
    /// device time is settled.
    pub(crate) finished: bool,
}

/// How comparing one alternative profile with the baseline went.
pub(crate) enum ProfileTrial<P> {
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
pub(crate) struct BestAlternative<P> {
    pub(crate) profile: PipelineProfile,
    pub(crate) median: Duration,
    pub(crate) built: P,
}

/// What timing the alternative profiles produced.
pub(crate) struct ProfileRun<P> {
    pub(crate) phase: ProfilePhase,
    /// The fastest alternative, when it beats the baseline by the profile margin.
    pub(crate) winner: Option<BestAlternative<P>>,
}

/// The profiles worth comparing with the baseline for this group on `device`, in the order
/// `table` gives them.
///
/// Every profile here differs from the baseline on this device. A group without nl4d has none.
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
    match profile {
        PipelineProfile::Baseline => false,
        PipelineProfile::F32Search => f32_profile_applies(true, device),
        PipelineProfile::SplitSearch => true,
    }
}

/// The alternative profiles for `set`, which has some when any of its engines is nl4d.
fn alternative_profiles(set: &EngineSet) -> Vec<PipelineProfile> {
    let is_nl4d = set.specs().iter().any(EngineSpec::is_nl4d);
    let device = set.device();
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
/// profile margin, replaces `set`. Each cost estimate counts the alternative's kernel compilation
/// as one more priming. A configuration where no alternative can be built has no profile to compare.
pub(super) fn compare_profiles(
    set: &mut EngineSet,
    inputs: &PassInputs<'_>,
    budget: &Budget,
    reading: &WarmUpReading,
    mode: SearchMode,
) -> ProfilePhase {
    let alternatives = alternative_profiles(set);
    if alternatives.is_empty() || inputs.ledger.deterministic() {
        return ProfilePhase::NotApplicable;
    }

    let block_frames = (2 * PROFILE_MIN_BLOCKS * PROFILE_BLOCK_FRAMES) as u32;
    let warm_frames = (inputs.frames.len() + WARM_WINDOW) as u32;
    let frames_cost = reading.frame_time * (block_frames + warm_frames);
    let one_site = site_min_cost(TYPICAL_CANDIDATES);
    let needed = frames_cost + one_site + reading.priming;

    let run = time_profiles(&alternatives, mode, |profile| {
        try_profile(set, inputs, budget, needed, profile)
    });

    if let Some(mut winner) = run.winner {
        tracing::info!(
            profile = winner.profile.name(),
            "a pipeline profile is faster on this device"
        );
        std::mem::swap(set, &mut winner.built);
    }

    run.phase
}

/// Builds `profile` and compares it with `set`, when the budget still covers `needed`.
fn try_profile(
    set: &mut EngineSet,
    inputs: &PassInputs<'_>,
    budget: &Budget,
    needed: Duration,
    profile: PipelineProfile,
) -> ProfileTrial<EngineSet> {
    let spent = budget_spent(inputs.ledger, inputs.started);
    if !can_afford(budget, spent, needed) {
        tracing::info!(
            profile = profile.name(),
            "no tuning budget is left to compare this profile"
        );
        return ProfileTrial::OutOfBudget;
    }

    let created = set.alternative(inputs.config, profile);
    let mut alternative = match created {
        Ok(alternative) => alternative,
        Err(error) => {
            tracing::warn!(%error, profile = profile.name(), "cannot build a profile to compare");
            return ProfileTrial::Unbuilt;
        },
    };

    let compared = compare_with(set, &mut alternative, inputs, budget);
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
pub(crate) fn time_profiles<P>(
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

/// Tunes the keys only `alternative` reaches, then times both sets in interleaved blocks.
fn compare_with(
    baseline: &mut EngineSet,
    alternative: &mut EngineSet,
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
/// Each frame waits for the device before the next is pushed, so every time covers the frame's GPU
/// work.
fn time_block(
    set: &mut EngineSet,
    cycle: &mut FrameCycle<'_>,
    times: &mut Vec<Duration>,
) -> Result<(), anyhow::Error> {
    let mut timed_frames = 0;

    while timed_frames < PROFILE_BLOCK_FRAMES {
        let timed = timed_frame(set, cycle)?;
        if let Some(frame_time) = timed {
            times.push(frame_time);
            timed_frames += 1;
        }
    }

    Ok(())
}
