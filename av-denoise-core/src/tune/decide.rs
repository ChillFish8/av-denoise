use std::time::Duration;

use anyhow::Context;

use super::consts::{
    EXHAUSTIVE_SITE_CAP,
    HARD_SITE_CAP,
    HYSTERESIS,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    PRESCREEN_MARGIN,
    PRESCREEN_SAMPLES,
    RACE_MARGIN,
    RACE_MIN_SAMPLES,
    SAMPLES_PER_WAIT,
    SITE_CAP,
};
use super::fidelity::FidelityClass;

/// One timing sample, `launches` back-to-back launches that took `total` together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTiming {
    pub total: Duration,
    pub launches: u32,
}

impl SampleTiming {
    /// The time of one launch.
    pub fn per_launch(&self) -> Duration {
        let launches = self.launches.max(1);
        self.total / launches
    }
}

/// Times candidates by their index.
pub trait Sampler {
    /// Takes one sample of the candidate at `index`.
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error>;

    /// Takes `count` samples of the candidate at `index`.
    ///
    /// It returns all `count` timings or an error.
    fn sample_batch(&mut self, index: usize, count: usize) -> Result<Vec<SampleTiming>, anyhow::Error> {
        let mut timings = Vec::with_capacity(count);
        for _ in 0..count {
            let timing = self.sample(index)?;
            timings.push(timing);
        }

        Ok(timings)
    }

    /// Time spent on the current site so far.
    ///
    /// It increases with every sample.
    fn elapsed(&self) -> Duration;

    /// The name of the candidate at `index`, for logs.
    fn candidate_name(&self, index: usize) -> String {
        index.to_string()
    }

    /// The fidelity class of the candidate at `index`.
    fn fidelity(&self, _index: usize) -> FidelityClass {
        FidelityClass::Exact
    }

    /// Leaves `spent` out of [Sampler::elapsed] from now on.
    fn exclude(&mut self, _spent: Duration) {}
}

/// The samples one candidate has collected.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Samples {
    per_launch: Vec<Duration>,
    spent: Duration,
}

impl Samples {
    /// Adds one sample.
    pub fn push(&mut self, timing: SampleTiming) {
        let per_launch = timing.per_launch();
        self.per_launch.push(per_launch);
        self.spent += timing.total;
    }

    /// How many samples were taken.
    pub fn count(&self) -> usize {
        self.per_launch.len()
    }

    /// The total time these samples took.
    pub fn spent(&self) -> Duration {
        self.spent
    }

    /// The per-launch time at fraction `rank` of the sorted samples, by nearest rank.
    pub fn quantile(&self, rank: f64) -> Duration {
        let rank = rank.clamp(0.0, 1.0);
        nearest_rank(&self.per_launch, rank)
    }

    /// The median per-launch time.
    pub fn median(&self) -> Duration {
        median(&self.per_launch)
    }

    fn has_minimums(&self, limits: &DecideLimits) -> bool {
        self.count() >= limits.min_samples && self.spent >= limits.min_time
    }
}

/// The median of `durations`, by nearest rank.
///
/// An empty slice gives zero.
pub fn median(durations: &[Duration]) -> Duration {
    nearest_rank(durations, 0.5)
}

fn nearest_rank(durations: &[Duration], rank: f64) -> Duration {
    let mut sorted = durations.to_vec();
    sorted.sort_unstable();

    let Some(last) = sorted.len().checked_sub(1) else {
        return Duration::ZERO;
    };

    let position = (last as f64 * rank).round() as usize;
    sorted[position]
}

/// The thresholds the decision loop applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecideLimits {
    /// The fewest samples every candidate takes before the stop rule applies.
    pub min_samples: usize,
    /// The least sampled time every candidate collects before the stop rule applies.
    pub min_time: Duration,
    /// The time after which sampling stops once every minimum is met.
    ///
    /// It never shortens a minimum.
    pub site_cap: Duration,
    /// A site still short of its minimums here fails.
    pub hard_cap: Duration,
    /// Short samples each candidate takes in the pre-screen.
    pub prescreen_samples: usize,
    /// How much slower than the fastest a candidate may be and survive the pre-screen.
    pub prescreen_margin: f64,
    /// Whether candidates that can no longer win are raced out before their minimums.
    pub race: bool,
}

impl Default for DecideLimits {
    fn default() -> Self {
        Self {
            min_samples: MIN_SAMPLES,
            min_time: MIN_CANDIDATE_TIME,
            site_cap: SITE_CAP,
            hard_cap: HARD_SITE_CAP,
            prescreen_samples: PRESCREEN_SAMPLES,
            prescreen_margin: PRESCREEN_MARGIN,
            race: true,
        }
    }
}

impl DecideLimits {
    /// The limits of an exhaustive search.
    ///
    /// It keeps the usual minimums, with a wider site cap, no hard cap and no racing.
    pub fn exhaustive() -> Self {
        Self {
            site_cap: EXHAUSTIVE_SITE_CAP,
            hard_cap: Duration::MAX,
            race: false,
            ..Self::default()
        }
    }
}

/// One candidate's samples at the end of a site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateStats {
    pub index: usize,
    pub samples: Samples,
}

/// How one site's sampling ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteTiming {
    /// Every candidate timed, with the samples it took.
    ///
    /// The candidates timed to the end come first, then the raced-out ones, each in the order given.
    pub timed: Vec<CandidateStats>,
    /// Candidates dropped after a sample failed.
    pub failed: Vec<usize>,
    /// Candidates raced out before their minimums, in the order given.
    pub raced: Vec<usize>,
    /// Whether the site cap ended sampling before the leader separated.
    pub capped: bool,
}

/// Drops every candidate more than the pre-screen margin slower than the fastest.
///
/// Each candidate takes a few short samples and is judged on their median. `keep`, the baseline,
/// always survives so it reaches full timing. A candidate whose sample fails is dropped, and a
/// failing `keep` fails the pre-screen.
pub fn prescreen<S: Sampler>(
    sampler: &mut S,
    candidates: &[usize],
    keep: usize,
    limits: &DecideLimits,
) -> Result<Vec<usize>, anyhow::Error> {
    let mut medians = Vec::with_capacity(candidates.len());

    for &index in candidates {
        let mut samples = Samples::default();

        for _ in 0..limits.prescreen_samples {
            let sampled = sampler.sample(index);

            match sampled {
                Ok(timing) => samples.push(timing),
                Err(error) if index == keep => {
                    let error = error.context("the baseline failed during the pre-screen");
                    return Err(error);
                },
                Err(error) => {
                    tracing::warn!(index, %error, "dropping a candidate that failed the pre-screen");
                    break;
                },
            }
        }

        if samples.count() == limits.prescreen_samples {
            let sample_median = samples.median();
            medians.push((index, sample_median));
        }
    }

    let fastest = medians.iter().map(|&(_, sample_median)| sample_median).min();
    let Some(fastest) = fastest else {
        anyhow::bail!("no candidate survived the pre-screen");
    };

    let threshold = fastest.as_secs_f64() * (1.0 + limits.prescreen_margin);
    let mut survivors = Vec::with_capacity(medians.len());

    for (index, sample_median) in medians {
        let survives = index == keep || sample_median.as_secs_f64() <= threshold;
        if survives {
            survivors.push(index);
            continue;
        }

        let name = sampler.candidate_name(index);
        tracing::debug!(candidate = %name, median = ?sample_median, "the pre-screen dropped a candidate");
    }

    Ok(survivors)
}

/// Samples `candidates` round-robin until the leader separates or the site cap passes.
///
/// Each round takes [SAMPLES_PER_WAIT](crate::tune::SAMPLES_PER_WAIT) samples of every candidate.
/// Sampling never stops before every candidate has its minimum samples and time. After that it
/// stops once the leader's median sits below the runner-up's lower quartile, or once the sampler's
/// elapsed time reaches the site cap. A site still short of its minimums at the hard cap fails. A
/// failing candidate is dropped. A failing `baseline` fails the site. A round that adds no elapsed
/// time fails the site.
///
/// When the limits race, each round ends by racing out the candidates that can no longer win or tie
/// the leader. A raced-out candidate takes no more samples, and the minimum, separation and cap
/// checks look only at the candidates still being sampled. The baseline, the leader, the fastest
/// Exact candidate and any Exact candidate that could still tie it are never raced out.
pub fn sample_until_settled<S: Sampler>(
    sampler: &mut S,
    candidates: &[usize],
    baseline: usize,
    limits: &DecideLimits,
) -> Result<SiteTiming, anyhow::Error> {
    let mut active: Vec<CandidateStats> = candidates
        .iter()
        .map(|&index| CandidateStats {
            index,
            samples: Samples::default(),
        })
        .collect();
    let mut raced = Vec::new();
    let mut failed = Vec::new();

    loop {
        let elapsed_before = sampler.elapsed();
        let mut position = 0;

        while position < active.len() {
            let index = active[position].index;
            let sampled = sampler.sample_batch(index, SAMPLES_PER_WAIT);

            match sampled {
                Ok(timings) => {
                    for timing in timings {
                        active[position].samples.push(timing);
                    }

                    position += 1;
                },
                Err(error) if index == baseline => {
                    return Err(error).context("the baseline failed while being timed");
                },
                Err(error) => {
                    tracing::warn!(index, %error, "dropping a candidate that failed while being timed");
                    failed.push(index);
                    active.remove(position);
                },
            }
        }

        if limits.race {
            race_out(sampler, &mut active, &mut raced, baseline);
        }

        let minimums_met = active.iter().all(|stats| stats.samples.has_minimums(limits));
        let elapsed = sampler.elapsed();

        let stalled = elapsed <= elapsed_before;

        if !minimums_met {
            if elapsed >= limits.hard_cap {
                anyhow::bail!("the candidates did not reach their minimum samples within the hard cap");
            }

            if stalled {
                anyhow::bail!("candidates stopped accruing time before their minimums");
            }

            continue;
        }

        if leader_separated(&active) {
            let timing = site_timing(candidates, active, raced, failed, false);
            return Ok(timing);
        }

        if stalled {
            anyhow::bail!("the candidates stopped accruing time before the leader separated");
        }

        if elapsed >= limits.site_cap {
            let timing = site_timing(candidates, active, raced, failed, true);
            return Ok(timing);
        }
    }
}

/// Moves every candidate in `active` that can no longer win or tie the leader into `raced`.
///
/// The leader is the candidate with the lowest median. A candidate is raced out once it has
/// [RACE_MIN_SAMPLES](crate::tune::RACE_MIN_SAMPLES) samples and its lower quartile sits above the
/// leader's median by more than the hysteresis plus [RACE_MARGIN](crate::tune::RACE_MARGIN). An Exact
/// candidate must also sit that far above the fastest Exact candidate's median. The baseline, the
/// leader and the Exact candidate with the lowest median are never raced out.
fn race_out<S: Sampler>(
    sampler: &S,
    active: &mut Vec<CandidateStats>,
    raced: &mut Vec<CandidateStats>,
    baseline: usize,
) {
    let Some(leader) = fastest_of(active.iter()) else {
        return;
    };

    let exact = active
        .iter()
        .filter(|stats| sampler.fidelity(stats.index) == FidelityClass::Exact);
    let fastest_exact = fastest_of(exact);
    let fastest_exact_index = fastest_exact.map(|stats| stats.index);
    let fastest_exact_median = fastest_exact.map(|stats| stats.samples.median());

    let leader_index = leader.index;
    let leader_median = leader.samples.median();
    let threshold = race_threshold(leader_median);
    let exact_threshold = fastest_exact_median.map(race_threshold);

    let mut position = 0;
    while position < active.len() {
        let stats = &active[position];
        let protected = stats.index == baseline
            || stats.index == leader_index
            || Some(stats.index) == fastest_exact_index;
        let sampled_enough = stats.samples.count() >= RACE_MIN_SAMPLES;
        let lower_quartile = stats.samples.quantile(0.25).as_secs_f64();
        let is_exact = sampler.fidelity(stats.index) == FidelityClass::Exact;
        let beaten_by_exact = exact_threshold.is_none_or(|exact_limit| lower_quartile > exact_limit);
        let beaten = lower_quartile > threshold && (!is_exact || beaten_by_exact);

        if protected || !sampled_enough || !beaten {
            position += 1;
            continue;
        }

        let stats = active.remove(position);
        let name = sampler.candidate_name(stats.index);
        let samples = stats.samples.count();
        let median = stats.samples.median();
        tracing::debug!(candidate = %name, samples, median = ?median, "raced a candidate out");

        raced.push(stats);
    }
}

/// The lower quartile, in seconds, above which a candidate can no longer win or tie `median`.
fn race_threshold(median: Duration) -> f64 {
    median.as_secs_f64() * (1.0 + HYSTERESIS + RACE_MARGIN)
}

/// The candidate with the lowest median, the first of any tie.
fn fastest_of<'a>(stats: impl Iterator<Item = &'a CandidateStats>) -> Option<&'a CandidateStats> {
    stats.min_by_key(|stats| stats.samples.median())
}

/// The end of a site, with the raced-out candidates after the active ones in the order given.
fn site_timing(
    candidates: &[usize],
    active: Vec<CandidateStats>,
    mut raced: Vec<CandidateStats>,
    failed: Vec<usize>,
    capped: bool,
) -> SiteTiming {
    raced.sort_by_key(|stats| candidates.iter().position(|&index| index == stats.index));
    let raced_indices = raced.iter().map(|stats| stats.index).collect();

    let mut timed = active;
    timed.extend(raced);

    SiteTiming {
        timed,
        failed,
        raced: raced_indices,
        capped,
    }
}

/// Whether the fastest median sits below the runner-up's lower quartile.
fn leader_separated(active: &[CandidateStats]) -> bool {
    if active.len() < 2 {
        return true;
    }

    let mut by_median: Vec<&CandidateStats> = active.iter().collect();
    by_median.sort_by_key(|stats| stats.samples.median());

    let leader_median = by_median[0].samples.median();
    let runner_up_lower_quartile = by_median[1].samples.quantile(0.25);

    leader_median < runner_up_lower_quartile
}
