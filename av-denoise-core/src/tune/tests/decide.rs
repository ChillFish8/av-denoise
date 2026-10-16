use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

use crate::tune::{
    DecideLimits,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    PRESCREEN_SAMPLES,
    Precompiled,
    SampleTiming,
    Sampler,
    Samples,
    SiteTiming,
    Staging,
    prescreen,
    sample_in_stages,
    sample_in_stages_with,
    sample_until_settled,
};

/// Returns each candidate's scripted durations in turn, repeating the last one.
///
/// Elapsed time is the sum of every sample handed out, so tests never wait on a clock.
struct ScriptedSampler {
    scripts: BTreeMap<usize, Vec<Duration>>,
    served: BTreeMap<usize, usize>,
    failing: Vec<usize>,
    elapsed: Duration,
}

impl ScriptedSampler {
    fn new(scripts: &[(usize, &[u64])]) -> Self {
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

    fn served(&self, index: usize) -> usize {
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

/// A 3 by 4 grid of candidates, `index = row * 4 + column`, timed at these microseconds.
///
/// Candidate 0 is the baseline. Stage 0 varies the column in the seed's row and stage 1 varies the
/// row in the seed's column, the way a site varies its search dimensions and then its group count.
const GRID_MICROS: [u64; 12] = [1_000, 900, 800, 850, 1_200, 700, 760, 820, 1_300, 980, 990, 640];

const GRID_COLUMNS: usize = 4;

fn grid_sampler() -> ScriptedSampler {
    let scripts: Vec<(usize, &[u64])> = GRID_MICROS
        .iter()
        .enumerate()
        .map(|(index, micros)| (index, std::slice::from_ref(micros)))
        .collect();

    ScriptedSampler::new(&scripts)
}

fn grid_members(stage: usize, seed: usize) -> Vec<usize> {
    let seed_row = seed / GRID_COLUMNS;
    let seed_column = seed % GRID_COLUMNS;

    (0..GRID_MICROS.len())
        .filter(|&index| {
            let same_row = index / GRID_COLUMNS == seed_row;
            let same_column = index % GRID_COLUMNS == seed_column;
            if stage == 0 { same_row } else { same_column }
        })
        .collect()
}

fn grid_staging(stages: usize, prescreen: bool) -> Staging<'static> {
    Staging {
        stages,
        keep: 3,
        prescreen,
        members: &grid_members,
    }
}

fn timed_indices(timing: &SiteTiming) -> Vec<usize> {
    let mut indices: Vec<usize> = timing.timed.iter().map(|stats| stats.index).collect();
    indices.sort_unstable();
    indices
}

#[test]
fn one_stage_around_the_baseline_times_only_its_members() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(1, false);

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, |indices| indices.to_vec())
        .expect("sampling succeeds");

    assert_eq!(timed_indices(&timing), vec![0, 1, 2, 3]);
    assert_eq!(sampler.served(4), 0);
}

/// Columns 2, 3 and 1 are the fastest of the first row, so the second stage times rows 1 and 2 of
/// those columns and never column 0.
#[test]
fn a_later_stage_varies_its_dimension_around_the_fastest_so_far() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(2, false);

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, |indices| indices.to_vec())
        .expect("sampling succeeds");

    assert_eq!(timed_indices(&timing), vec![0, 1, 2, 3, 5, 6, 7, 9, 10, 11]);
    assert_eq!(sampler.served(4), 0);
    assert_eq!(sampler.served(8), 0);
}

#[test]
fn a_later_stage_never_retimes_a_candidate() {
    let mut first_only = grid_sampler();
    let mut both = grid_sampler();
    let limits = DecideLimits::default();
    let keep_all = |indices: &[usize]| indices.to_vec();

    sample_in_stages(&mut first_only, &grid_staging(1, false), 0, &limits, keep_all)
        .expect("sampling succeeds");
    sample_in_stages(&mut both, &grid_staging(2, false), 0, &limits, keep_all).expect("sampling succeeds");

    for index in 0..GRID_COLUMNS {
        assert_eq!(both.served(index), first_only.served(index), "candidate {index}");
    }
}

/// Candidate 1 is 41% slower than the fastest of the whole grid but within 25% of its own stage's
/// fastest, so it survives. Candidates 7, 9 and 10 are more than 25% slower than candidate 11, the
/// fastest of their stage, and are dropped after their pre-screen samples.
#[test]
fn the_prescreen_runs_within_each_stage() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(2, true);

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, |indices| indices.to_vec())
        .expect("sampling succeeds");
    let timed = timed_indices(&timing);

    assert!(timed.contains(&1));

    for dropped in [7, 9, 10] {
        assert!(!timed.contains(&dropped), "candidate {dropped} was timed");
        assert_eq!(sampler.served(dropped), PRESCREEN_SAMPLES, "candidate {dropped}");
    }
}

#[test]
fn a_candidate_whose_output_check_fails_is_never_timed() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(2, false);
    let drop_six = |indices: &[usize]| indices.iter().copied().filter(|&index| index != 6).collect();

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, drop_six).expect("sampling succeeds");

    assert!(!timed_indices(&timing).contains(&6));
    assert_eq!(sampler.served(6), 0);
}

#[test]
fn a_later_stage_where_every_member_fails_the_prescreen_keeps_the_earlier_ranking() {
    let mut sampler = grid_sampler();
    sampler.failing = vec![5, 6, 7, 9, 10, 11];
    let limits = DecideLimits::default();
    let staging = grid_staging(2, true);

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, |indices| indices.to_vec())
        .expect("the earlier stage still ranks");

    assert_eq!(timed_indices(&timing), vec![0, 1, 2, 3]);
}

#[test]
fn the_baseline_is_timed_even_when_the_first_stage_leaves_it_out() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let without_baseline = |_stage: usize, _seed: usize| vec![1, 2, 3];
    let staging = Staging {
        stages: 1,
        keep: 3,
        prescreen: false,
        members: &without_baseline,
    };

    let timing = sample_in_stages(&mut sampler, &staging, 0, &limits, |indices| indices.to_vec())
        .expect("sampling succeeds");

    assert_eq!(timed_indices(&timing), vec![0, 1, 2, 3]);
}

/// A scripted sampler whose elapsed time also counts a shared compile clock, less what it leaves out.
struct CompilingSampler {
    inner: ScriptedSampler,
    compile_clock: Rc<Cell<Duration>>,
    excluded: Duration,
}

impl CompilingSampler {
    fn new(inner: ScriptedSampler, compile_clock: &Rc<Cell<Duration>>) -> Self {
        let compile_clock = compile_clock.clone();

        Self {
            inner,
            compile_clock,
            excluded: Duration::ZERO,
        }
    }
}

impl Sampler for CompilingSampler {
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error> {
        self.inner.sample(index)
    }

    fn elapsed(&self) -> Duration {
        let sampled = self.inner.elapsed();
        let compiled = self.compile_clock.get();
        let total = sampled + compiled;

        total.saturating_sub(self.excluded)
    }

    fn exclude(&mut self, spent: Duration) {
        self.excluded += spent;
    }
}

/// Two candidates whose samples overlap, so only the site cap ends their timing.
fn overlapping_pair() -> (CompilingSampler, Rc<Cell<Duration>>) {
    let alternating: Vec<u64> = (0..4000)
        .map(|round| if round % 2 == 0 { 900 } else { 1100 })
        .collect();
    let inner = ScriptedSampler::new(&[(0, &alternating), (1, &alternating)]);
    let compile_clock = Rc::new(Cell::new(Duration::ZERO));
    let sampler = CompilingSampler::new(inner, &compile_clock);

    (sampler, compile_clock)
}

fn pair_members(_stage: usize, _seed: usize) -> Vec<usize> {
    vec![0, 1]
}

fn pair_staging() -> Staging<'static> {
    Staging {
        stages: 1,
        keep: 3,
        prescreen: false,
        members: &pair_members,
    }
}

#[test]
fn compile_time_does_not_shorten_the_timing_before_the_site_cap() {
    let (mut sampler, compile_clock) = overlapping_pair();
    let limits = DecideLimits::default();
    let staging = pair_staging();
    let compile_spent = Duration::from_secs(30);
    let slow_compile = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let advanced = compile_clock.get() + compile_spent;
        compile_clock.set(advanced);

        let mut precompiled = Precompiled::keeping(indices);
        precompiled.spent = compile_spent;
        Ok(precompiled)
    };
    let keep_all = |indices: &[usize]| indices.to_vec();

    let timing = sample_in_stages_with(&mut sampler, &staging, 0, &limits, slow_compile, keep_all)
        .expect("sampling succeeds");

    assert!(timing.capped);
    assert!(sampler.inner.elapsed() >= limits.site_cap);
    assert_eq!(sampler.excluded, compile_spent);
}

/// The same compile, reported as taking no time, eats the site cap before any timing.
#[test]
fn unreported_compile_time_ends_the_timing_at_its_minimums() {
    let (mut sampler, compile_clock) = overlapping_pair();
    let limits = DecideLimits::default();
    let staging = pair_staging();
    let silent_compile = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let advanced = compile_clock.get() + Duration::from_secs(30);
        compile_clock.set(advanced);

        Ok(Precompiled::keeping(indices))
    };
    let keep_all = |indices: &[usize]| indices.to_vec();

    sample_in_stages_with(&mut sampler, &staging, 0, &limits, silent_compile, keep_all)
        .expect("sampling succeeds");

    assert!(sampler.inner.elapsed() < limits.site_cap);
}

#[test]
fn a_member_that_does_not_compile_is_never_checked_or_timed() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(2, false);
    let drop_six = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let kept = indices.iter().copied().filter(|&index| index != 6).collect();
        let dropped = indices
            .iter()
            .filter(|&&index| index == 6)
            .map(|&index| (index, "too much shared memory".to_string()))
            .collect();

        Ok(Precompiled {
            kept,
            dropped,
            ..Precompiled::default()
        })
    };
    let checked = RefCell::new(Vec::new());
    let record_checked = |indices: &[usize]| {
        checked.borrow_mut().extend_from_slice(indices);
        indices.to_vec()
    };

    let timing = sample_in_stages_with(&mut sampler, &staging, 0, &limits, drop_six, record_checked)
        .expect("sampling succeeds");

    assert!(!timed_indices(&timing).contains(&6));
    assert!(!checked.borrow().contains(&6));
    assert_eq!(sampler.served(6), 0);
}

#[test]
fn a_failing_precompile_fails_the_site() {
    let mut sampler = grid_sampler();
    let limits = DecideLimits::default();
    let staging = grid_staging(2, false);
    let failing = |_indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        anyhow::bail!("the baseline did not compile")
    };
    let keep_all = |indices: &[usize]| indices.to_vec();

    let result = sample_in_stages_with(&mut sampler, &staging, 0, &limits, failing, keep_all);

    assert!(result.is_err());
    assert_eq!(sampler.served(0), 0);
}
