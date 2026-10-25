use std::time::Duration;

use anyhow::Context;

use super::consts::{
    HARD_SITE_CAP,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    PRESCREEN_MARGIN,
    PRESCREEN_SAMPLES,
    SITE_CAP,
};

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

    /// Time spent on the current site so far.
    ///
    /// It increases with every sample.
    fn elapsed(&self) -> Duration;

    /// The name of the candidate at `index`, for logs.
    fn candidate_name(&self, index: usize) -> String {
        index.to_string()
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
        }
    }
}

impl DecideLimits {
    /// The limits of an exhaustive search, with the usual minimums and no site or hard cap.
    pub fn exhaustive() -> Self {
        Self {
            site_cap: Duration::MAX,
            hard_cap: Duration::MAX,
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
    /// Candidates timed to the end, in the order they were given.
    pub timed: Vec<CandidateStats>,
    /// Candidates dropped after a sample failed.
    pub failed: Vec<usize>,
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
/// Sampling never stops before every candidate has its minimum samples and time. After that it
/// stops once the leader's median sits below the runner-up's lower quartile, or once the sampler's
/// elapsed time reaches the site cap. A site still short of its minimums at the hard cap fails. A
/// failing candidate is dropped. A failing `baseline` fails the site. A round that adds no elapsed
/// time fails the site.
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
    let mut failed = Vec::new();

    loop {
        let elapsed_before = sampler.elapsed();
        let mut position = 0;

        while position < active.len() {
            let index = active[position].index;
            let sampled = sampler.sample(index);

            match sampled {
                Ok(timing) => {
                    active[position].samples.push(timing);
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
            return Ok(SiteTiming {
                timed: active,
                failed,
                capped: false,
            });
        }

        if stalled {
            anyhow::bail!("the candidates stopped accruing time before the leader separated");
        }

        if elapsed >= limits.site_cap {
            return Ok(SiteTiming {
                timed: active,
                failed,
                capped: true,
            });
        }
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
