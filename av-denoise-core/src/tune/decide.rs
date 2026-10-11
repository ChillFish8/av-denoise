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
                    return Err(error.context("the baseline failed during the pre-screen"));
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
    let survivors = medians
        .into_iter()
        .filter(|&(index, sample_median)| index == keep || sample_median.as_secs_f64() <= threshold)
        .map(|(index, _)| index)
        .collect();

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

/// How a site's candidates are timed in stages.
pub struct Staging<'a> {
    /// Stages to run, 1 or more.
    pub stages: usize,
    /// The fastest candidates each later stage starts from.
    pub keep: usize,
    /// Whether each stage pre-screens its own candidates.
    pub prescreen: bool,
    /// The candidates stage `stage` times around candidate `seed`, as indices.
    pub members: &'a dyn Fn(usize, usize) -> Vec<usize>,
}

/// Times candidates stage by stage and returns every candidate timed.
///
/// The first stage times the members around `baseline`. Each later stage times the members around
/// each of the `keep` fastest candidates so far, leaving out any an earlier stage already took. A
/// stage pre-screens its members when asked, `check` drops those whose output differs, and the rest
/// are sampled until settled. The baseline is always a member of the first stage. A later stage
/// where no member passes the pre-screen times nothing and the site keeps what it has. The site cap
/// counts the whole site. The hard cap restarts with each stage, so a later stage always reaches its
/// minimums.
pub fn sample_in_stages<S: Sampler>(
    sampler: &mut S,
    staging: &Staging<'_>,
    baseline: usize,
    limits: &DecideLimits,
    mut check: impl FnMut(&[usize]) -> Vec<usize>,
) -> Result<SiteTiming, anyhow::Error> {
    let mut timed: Vec<CandidateStats> = Vec::new();
    let mut failed = Vec::new();
    let mut capped = false;
    let mut tried: Vec<usize> = Vec::new();
    let mut seeds = vec![baseline];

    for stage in 0..staging.stages {
        let mut members = fresh_members(staging, stage, &seeds, &tried);

        if stage == 0 && !members.contains(&baseline) {
            members.insert(0, baseline);
        }

        tried.extend(&members);

        if members.is_empty() {
            continue;
        }

        let screened = if !staging.prescreen {
            members
        } else if stage == 0 {
            prescreen(sampler, &members, baseline, limits)?
        } else {
            let survivors = prescreen(sampler, &members, baseline, limits);

            match survivors {
                Ok(survivors) => survivors,
                Err(error) => {
                    tracing::warn!(stage, %error, "skipping a stage with no pre-screen survivor");
                    continue;
                },
            }
        };

        let checked = check(&screened);
        let stage_limits = DecideLimits {
            hard_cap: sampler.elapsed() + limits.hard_cap,
            ..*limits
        };
        let stage_timing = sample_until_settled(sampler, &checked, baseline, &stage_limits)?;

        timed.extend(stage_timing.timed);
        failed.extend(stage_timing.failed);
        capped |= stage_timing.capped;
        seeds = fastest(&timed, staging.keep);
    }

    Ok(SiteTiming {
        timed,
        failed,
        capped,
    })
}

/// Stage `stage`'s members around every seed, in seed order, with no repeats and nothing in `tried`.
fn fresh_members(staging: &Staging<'_>, stage: usize, seeds: &[usize], tried: &[usize]) -> Vec<usize> {
    let mut members = Vec::new();

    for &seed in seeds {
        let around = (staging.members)(stage, seed);

        for index in around {
            let fresh = !tried.contains(&index) && !members.contains(&index);
            if fresh {
                members.push(index);
            }
        }
    }

    members
}

/// The indices of the `keep` timed candidates with the lowest medians.
fn fastest(timed: &[CandidateStats], keep: usize) -> Vec<usize> {
    let mut by_median: Vec<&CandidateStats> = timed.iter().collect();
    by_median.sort_by_key(|stats| stats.samples.median());

    by_median.iter().take(keep).map(|stats| stats.index).collect()
}
