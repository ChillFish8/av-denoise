use std::collections::BTreeMap;
use std::time::Duration;

use crate::tune::{
    DecideLimits,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    SampleTiming,
    Sampler,
    Samples,
    prescreen,
    sample_until_settled,
};

/// Returns each candidate's scripted durations in turn, repeating the last one.
///
/// Elapsed time is the sum of every sample handed out, so tests never wait on a clock.
pub(super) struct ScriptedSampler {
    scripts: BTreeMap<usize, Vec<Duration>>,
    served: BTreeMap<usize, usize>,
    pub(super) failing: Vec<usize>,
    elapsed: Duration,
}

impl ScriptedSampler {
    pub(super) fn new(scripts: &[(usize, &[u64])]) -> Self {
        let mut scripted = BTreeMap::new();
        for &(index, micros) in scripts {
            let durations = micros.iter().map(|&value| Duration::from_micros(value)).collect();
            scripted.insert(index, durations);
        }

        Self {
            scripts: scripted,
            served: BTreeMap::new(),
            failing: Vec::new(),
            elapsed: Duration::ZERO,
        }
    }

    pub(super) fn served(&self, index: usize) -> usize {
        self.served.get(&index).copied().unwrap_or(0)
    }
}

impl Sampler for ScriptedSampler {
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error> {
        if self.failing.contains(&index) {
            anyhow::bail!("candidate {index} failed to launch");
        }

        let script = &self.scripts[&index];
        let position = self.served(index).min(script.len() - 1);
        let total = script[position];
        *self.served.entry(index).or_insert(0) += 1;
        self.elapsed += total;

        Ok(SampleTiming { total, launches: 1 })
    }

    fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

#[test]
fn quantiles_use_the_nearest_rank() {
    let mut samples = Samples::default();
    for micros in [5u64, 1, 4, 2, 3] {
        let total = Duration::from_micros(micros);
        samples.push(SampleTiming { total, launches: 1 });
    }

    assert_eq!(samples.median(), Duration::from_micros(3));
    assert_eq!(samples.quantile(0.25), Duration::from_micros(2));
    assert_eq!(samples.quantile(0.75), Duration::from_micros(4));
    assert_eq!(samples.count(), 5);
    assert_eq!(samples.spent(), Duration::from_micros(15));
}

#[test]
fn a_sample_is_divided_by_its_launch_count() {
    let timing = SampleTiming {
        total: Duration::from_millis(8),
        launches: 4,
    };

    assert_eq!(timing.per_launch(), Duration::from_millis(2));
}

#[test]
fn a_separated_leader_stops_as_soon_as_every_minimum_is_met() {
    let mut sampler = ScriptedSampler::new(&[(0, &[2_000]), (1, &[1_000])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1], 0, &limits).expect("sampling succeeds");

    // Candidate 1 reaches 250 ms after 250 rounds of 1 ms, the later of the two minimums.
    assert!(!timing.capped);
    assert_eq!(sampler.served(1), 250);
    assert_eq!(sampler.served(0), 250);
    for stats in &timing.timed {
        assert!(stats.samples.count() >= MIN_SAMPLES);
        assert!(stats.samples.spent() >= MIN_CANDIDATE_TIME);
    }
}

#[test]
fn overlapping_candidates_run_until_the_site_cap() {
    let alternating: Vec<u64> = (0..4000)
        .map(|round| if round % 2 == 0 { 900 } else { 1100 })
        .collect();
    let mut sampler = ScriptedSampler::new(&[(0, &alternating), (1, &alternating)]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1], 0, &limits).expect("sampling succeeds");

    assert!(timing.capped);
    assert!(sampler.elapsed() >= limits.site_cap);
}

#[test]
fn the_cap_never_shortens_the_minimums() {
    let mut sampler = ScriptedSampler::new(&[(0, &[1_000]), (1, &[1_000])]);
    let limits = DecideLimits {
        site_cap: Duration::from_millis(10),
        ..DecideLimits::default()
    };

    let timing = sample_until_settled(&mut sampler, &[0, 1], 0, &limits).expect("sampling succeeds");

    for stats in &timing.timed {
        assert!(stats.samples.spent() >= MIN_CANDIDATE_TIME);
    }
}

/// Reports no device time while wall time passes, as a site with no GPU work would.
struct StalledSampler {
    elapsed: Duration,
}

impl Sampler for StalledSampler {
    fn sample(&mut self, _index: usize) -> Result<SampleTiming, anyhow::Error> {
        self.elapsed += Duration::from_millis(10);

        Ok(SampleTiming {
            total: Duration::ZERO,
            launches: 1,
        })
    }

    fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

#[test]
fn a_site_that_never_reaches_its_minimums_fails_at_the_hard_cap() {
    let mut sampler = StalledSampler {
        elapsed: Duration::ZERO,
    };
    let limits = DecideLimits::default();

    let result = sample_until_settled(&mut sampler, &[0, 1], 0, &limits);

    assert!(result.is_err());
    assert!(sampler.elapsed >= limits.hard_cap);
}

#[test]
fn a_failing_candidate_is_dropped_and_the_rest_carry_on() {
    let mut sampler = ScriptedSampler::new(&[(0, &[2_000]), (1, &[1_000]), (2, &[500])]);
    sampler.failing.push(2);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");
    let timed: Vec<usize> = timing.timed.iter().map(|stats| stats.index).collect();

    assert_eq!(timing.failed, vec![2]);
    assert_eq!(timed, vec![0, 1]);
}

#[test]
fn a_failing_baseline_fails_the_site() {
    let mut sampler = ScriptedSampler::new(&[(0, &[2_000]), (1, &[1_000])]);
    sampler.failing.push(0);
    let limits = DecideLimits::default();

    let result = sample_until_settled(&mut sampler, &[0, 1], 0, &limits);

    assert!(result.is_err());
}

#[test]
fn the_prescreen_drops_candidates_more_than_a_quarter_slower() {
    let mut sampler = ScriptedSampler::new(&[(0, &[1_300]), (1, &[1_000]), (2, &[1_240]), (3, &[1_260])]);
    let limits = DecideLimits::default();

    let survivors = prescreen(&mut sampler, &[0, 1, 2, 3], 0, &limits).expect("prescreen succeeds");

    // Candidate 0 is 30% slower than the fastest but is the baseline, so it always survives.
    assert_eq!(survivors, vec![0, 1, 2]);
    assert_eq!(sampler.served(1), 3);
}

#[test]
fn the_prescreen_drops_a_failing_candidate() {
    let mut sampler = ScriptedSampler::new(&[(0, &[1_000]), (1, &[1_000])]);
    sampler.failing.push(1);
    let limits = DecideLimits::default();

    let survivors = prescreen(&mut sampler, &[0, 1], 0, &limits).expect("prescreen succeeds");

    assert_eq!(survivors, vec![0]);
}

#[test]
fn zero_length_samples_fail_the_site() {
    let mut sampler = ScriptedSampler::new(&[(0, &[0]), (1, &[0])]);
    let limits = DecideLimits::default();

    let result = sample_until_settled(&mut sampler, &[0, 1], 0, &limits);

    assert!(result.is_err());
}

#[test]
fn samples_that_stall_after_the_minimums_fail_the_site() {
    let mut script = vec![1_000u64; 300];
    script.push(0);
    let mut sampler = ScriptedSampler::new(&[(0, &script), (1, &script)]);
    let limits = DecideLimits::default();

    let result = sample_until_settled(&mut sampler, &[0, 1], 0, &limits);

    assert!(result.is_err());
}

#[test]
fn a_quantile_rank_past_one_gives_the_slowest_sample() {
    let mut samples = Samples::default();
    for micros in [1u64, 2, 3] {
        let total = Duration::from_micros(micros);
        samples.push(SampleTiming { total, launches: 1 });
    }

    assert_eq!(samples.quantile(2.0), Duration::from_micros(3));
}
