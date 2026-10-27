use std::collections::BTreeMap;
use std::time::Duration;

use crate::tune::{
    DecideLimits,
    FidelityClass,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    RACE_MIN_SAMPLES,
    SAMPLES_PER_WAIT,
    SampleTiming,
    Sampler,
    Samples,
    SiteTiming,
    prescreen,
    sample_until_settled,
};

/// Returns each candidate's scripted durations in turn, repeating the last one.
///
/// Elapsed time is the sum of every sample handed out, so tests never wait on a clock. A candidate
/// in `failing_from` fails every sample once it has been served that many. A candidate missing from
/// `fidelities` is Exact.
pub(super) struct ScriptedSampler {
    scripts: BTreeMap<usize, Vec<Duration>>,
    served: BTreeMap<usize, usize>,
    pub(super) failing: Vec<usize>,
    pub(super) failing_from: BTreeMap<usize, usize>,
    pub(super) fidelities: BTreeMap<usize, FidelityClass>,
    pub(super) batches: Vec<usize>,
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
            failing_from: BTreeMap::new(),
            fidelities: BTreeMap::new(),
            batches: Vec::new(),
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

        let fails_from = self.failing_from.get(&index).copied();
        if fails_from.is_some_and(|limit| self.served(index) >= limit) {
            anyhow::bail!("candidate {index} failed mid-batch");
        }

        let script = &self.scripts[&index];
        let position = self.served(index).min(script.len() - 1);
        let total = script[position];
        *self.served.entry(index).or_insert(0) += 1;
        self.elapsed += total;

        Ok(SampleTiming { total, launches: 1 })
    }

    fn sample_batch(&mut self, index: usize, count: usize) -> Result<Vec<SampleTiming>, anyhow::Error> {
        self.batches.push(count);

        let mut timings = Vec::with_capacity(count);
        for _ in 0..count {
            let timing = self.sample(index)?;
            timings.push(timing);
        }

        Ok(timings)
    }

    fn elapsed(&self) -> Duration {
        self.elapsed
    }

    fn fidelity(&self, index: usize) -> FidelityClass {
        let fidelity = self.fidelities.get(&index).copied();
        fidelity.unwrap_or(FidelityClass::Exact)
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

    // Candidate 1 reaches 250 ms after 250 samples of 1 ms, rounded up to whole rounds of samples.
    assert!(!timing.capped);
    assert_eq!(sampler.served(1), 252);
    assert_eq!(sampler.served(0), 252);
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
fn rounds_take_several_samples_per_wait() {
    let mut sampler = ScriptedSampler::new(&[(0, &[2_000]), (1, &[1_000])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1], 0, &limits).expect("sampling succeeds");

    assert!(!sampler.batches.is_empty());
    assert!(sampler.batches.iter().all(|&count| count == SAMPLES_PER_WAIT));
    for stats in &timing.timed {
        assert_eq!(stats.samples.count() % SAMPLES_PER_WAIT, 0);
    }
}

#[test]
fn a_failure_in_a_batched_sample_drops_only_that_candidate() {
    let mut sampler = ScriptedSampler::new(&[(0, &[2_000]), (1, &[1_000]), (2, &[500])]);
    sampler.failing_from.insert(2, SAMPLES_PER_WAIT + 1);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");
    let timed: Vec<usize> = timing.timed.iter().map(|stats| stats.index).collect();

    assert_eq!(timing.failed, vec![2]);
    assert_eq!(timed, vec![0, 1]);
    for stats in &timing.timed {
        assert!(stats.samples.count() >= MIN_SAMPLES);
        assert!(stats.samples.spent() >= MIN_CANDIDATE_TIME);
        assert_eq!(stats.samples.count(), sampler.served(stats.index));
    }
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

fn timed_indices(timing: &SiteTiming) -> Vec<usize> {
    timing.timed.iter().map(|stats| stats.index).collect()
}

fn sample_count(timing: &SiteTiming, index: usize) -> usize {
    let stats = timing.timed.iter().find(|stats| stats.index == index);
    let stats = stats.expect("the candidate is timed");
    stats.samples.count()
}

fn assert_minimums(timing: &SiteTiming, index: usize) {
    let stats = timing.timed.iter().find(|stats| stats.index == index);
    let stats = stats.expect("the candidate is timed");

    assert!(stats.samples.count() >= MIN_SAMPLES, "{index}");
    assert!(stats.samples.spent() >= MIN_CANDIDATE_TIME, "{index}");
}

#[test]
fn a_clear_loser_is_raced_out_with_fewer_samples() {
    let mut sampler = ScriptedSampler::new(&[(0, &[100]), (1, &[102]), (2, &[150])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");

    assert_eq!(timing.raced, vec![2]);
    assert_eq!(timed_indices(&timing), vec![0, 1, 2]);
    assert!(sample_count(&timing, 2) < sample_count(&timing, 0));
    assert!(sample_count(&timing, 2) >= RACE_MIN_SAMPLES);
    assert_minimums(&timing, 0);
    assert_minimums(&timing, 1);
}

#[test]
fn the_baseline_is_never_raced_out() {
    let mut sampler = ScriptedSampler::new(&[(0, &[150]), (1, &[100]), (2, &[100])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");

    assert!(timing.raced.is_empty());
    assert_minimums(&timing, 0);
}

#[test]
fn the_fastest_exact_candidate_is_never_raced_out() {
    let mut sampler = ScriptedSampler::new(&[(1, &[100]), (2, &[140]), (3, &[150])]);
    sampler.fidelities.insert(1, FidelityClass::Approximate);
    let limits = DecideLimits::default();

    // The baseline sits in an earlier batch, as it does in a search's later batches.
    let timing = sample_until_settled(&mut sampler, &[1, 2, 3], 0, &limits).expect("sampling succeeds");

    assert_eq!(timing.raced, vec![3]);
    assert_minimums(&timing, 1);
    assert_minimums(&timing, 2);
}

#[test]
fn ties_within_the_hysteresis_are_never_raced_out() {
    let mut sampler = ScriptedSampler::new(&[(0, &[100]), (1, &[101]), (2, &[101]), (3, &[102])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2, 3], 0, &limits).expect("sampling succeeds");

    assert!(timing.raced.is_empty());
    for index in 0..4 {
        assert_minimums(&timing, index);
    }
}

#[test]
fn no_candidate_is_raced_out_before_its_first_samples() {
    // Only the first round is slow, so the candidate looks a clear loser only before it has its first samples.
    let mut slow_start = vec![300u64; SAMPLES_PER_WAIT];
    slow_start.push(104);
    let mut sampler = ScriptedSampler::new(&[(0, &[100]), (1, &[104]), (2, &slow_start)]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");

    assert!(timing.raced.is_empty());
    assert_minimums(&timing, 2);
}

fn sample_pair(script: &[(usize, &[u64])], race: bool) -> SiteTiming {
    let mut sampler = ScriptedSampler::new(script);
    let limits = DecideLimits {
        race,
        ..DecideLimits::default()
    };

    sample_until_settled(&mut sampler, &[0, 1], 0, &limits).expect("sampling succeeds")
}

fn fastest_index(timing: &SiteTiming) -> usize {
    let fastest = timing.timed.iter().min_by_key(|stats| stats.samples.median());
    let fastest = fastest.expect("a candidate is timed");
    fastest.index
}

#[test]
fn two_candidate_sites_end_as_before() {
    let scripts: [&[(usize, &[u64])]; 2] = [&[(0, &[150]), (1, &[100])], &[(0, &[105]), (1, &[100])]];

    for script in scripts {
        let raced = sample_pair(script, true);
        let unraced = sample_pair(script, false);

        assert!(raced.raced.is_empty());
        assert_eq!(raced, unraced);
    }

    // A baseline leading by more than the race threshold races the other out but keeps its pick.
    let leading_baseline: &[(usize, &[u64])] = &[(0, &[100]), (1, &[150])];
    let raced = sample_pair(leading_baseline, true);
    let unraced = sample_pair(leading_baseline, false);

    assert_eq!(raced.raced, vec![1]);
    assert_eq!(fastest_index(&raced), fastest_index(&unraced));
    assert_eq!(raced.timed[0], unraced.timed[0]);
}

#[test]
fn candidates_inside_the_race_margin_are_never_raced_out() {
    let mut sampler = ScriptedSampler::new(&[(0, &[100]), (1, &[105]), (2, &[150])]);
    let limits = DecideLimits::default();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");

    assert_eq!(timing.raced, vec![2]);
    assert_minimums(&timing, 1);
}

#[test]
fn close_exact_runners_up_are_never_raced_out() {
    let mut sampler = ScriptedSampler::new(&[(1, &[100]), (2, &[140]), (3, &[141]), (4, &[155])]);
    sampler.fidelities.insert(1, FidelityClass::Approximate);
    let limits = DecideLimits::default();

    // The 141 candidate could still tie the fastest Exact, and the 155 one sits beyond the margin.
    let timing = sample_until_settled(&mut sampler, &[1, 2, 3, 4], 0, &limits).expect("sampling succeeds");

    assert_eq!(timing.raced, vec![4]);
    assert_minimums(&timing, 2);
    assert_minimums(&timing, 3);
}

#[test]
fn exhaustive_limits_never_race() {
    let mut sampler = ScriptedSampler::new(&[(0, &[100]), (1, &[102]), (2, &[150])]);
    let limits = DecideLimits::exhaustive();

    let timing = sample_until_settled(&mut sampler, &[0, 1, 2], 0, &limits).expect("sampling succeeds");

    assert!(timing.raced.is_empty());
    for index in 0..3 {
        assert_minimums(&timing, index);
    }
}
