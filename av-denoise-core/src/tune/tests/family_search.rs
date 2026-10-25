use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::decide::ScriptedSampler;
use super::probe::{ProbeCandidate, ProbeSite};
use super::runtime::{TestRuntime, test_client};
use crate::tune::sites::{Centre, CollabVariant, NeighbourWalk, Walk};
use crate::tune::{
    Basis,
    DecideLimits,
    DeviceLimits,
    Dimension,
    DimensionFamilies,
    DimensionValue,
    EngineKernels,
    Families,
    FamilyId,
    Fidelity,
    FlatFamily,
    KernelSelector,
    MIN_SAMPLES,
    PRESCREEN_SAMPLES,
    PipelineProfile,
    Precompiled,
    PriorityTable,
    SampleTiming,
    Sampler,
    SearchMode,
    SearchOutcome,
    SearchSpace,
    SearchStep,
    SiteCandidate,
    SitePriority,
    TuneLedger,
    TunePhase,
    VariantSite,
    first_timed,
    search_candidates,
};

/// A flat site of three families, `a` holding the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlatFake {
    A0,
    A1,
    A2,
    B0,
    B1,
    C0,
    C1,
}

const FLAT_FAMILIES: &[FlatFamily<FlatFake>] = &[
    FlatFamily {
        id: FamilyId("a"),
        representative: FlatFake::A0,
        members: &[FlatFake::A0, FlatFake::A1, FlatFake::A2],
    },
    FlatFamily {
        id: FamilyId("b"),
        representative: FlatFake::B0,
        members: &[FlatFake::B0, FlatFake::B1],
    },
    FlatFamily {
        id: FamilyId("c"),
        representative: FlatFake::C0,
        members: &[FlatFake::C0, FlatFake::C1],
    },
];

impl SiteCandidate for FlatFake {
    const SITE: &'static str = "flat_fake";
    const ALL: &'static [Self] = &[
        FlatFake::A0,
        FlatFake::A1,
        FlatFake::A2,
        FlatFake::B0,
        FlatFake::B1,
        FlatFake::C0,
        FlatFake::C1,
    ];

    fn name(self) -> &'static str {
        match self {
            FlatFake::A0 => "a0",
            FlatFake::A1 => "a1",
            FlatFake::A2 => "a2",
            FlatFake::B0 => "b0",
            FlatFake::B1 => "b1",
            FlatFake::C0 => "c0",
            FlatFake::C1 => "c1",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }

    fn families() -> Families<Self> {
        Families::Flat(FLAT_FAMILIES)
    }
}

/// A 3 by 3 grid of rows and columns, where the site does not build row 2 with column 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GridFake {
    row: usize,
    column: usize,
}

const fn cell(row: usize, column: usize) -> GridFake {
    GridFake { row, column }
}

const GRID_DIMENSIONS: &[Dimension] = &[
    Dimension {
        name: "row",
        values: &[
            DimensionValue { id: FamilyId("row0") },
            DimensionValue { id: FamilyId("row1") },
            DimensionValue { id: FamilyId("row2") },
        ],
    },
    Dimension {
        name: "column",
        values: &[
            DimensionValue {
                id: FamilyId("column0"),
            },
            DimensionValue {
                id: FamilyId("column1"),
            },
            DimensionValue {
                id: FamilyId("column2"),
            },
        ],
    },
];

fn grid_value_of(candidate: GridFake, dimension: usize) -> usize {
    if dimension == 0 {
        candidate.row
    } else {
        candidate.column
    }
}

fn grid_compose(values: &[usize]) -> Option<GridFake> {
    let candidate = cell(values[0], values[1]);
    GridFake::ALL.contains(&candidate).then_some(candidate)
}

impl SiteCandidate for GridFake {
    const SITE: &'static str = "grid_fake";
    const ALL: &'static [Self] = &[
        cell(0, 0),
        cell(0, 1),
        cell(0, 2),
        cell(1, 0),
        cell(1, 1),
        cell(1, 2),
        cell(2, 0),
        cell(2, 1),
    ];

    fn name(self) -> &'static str {
        const NAMES: [[&str; 3]; 3] = [
            ["r0c0", "r0c1", "r0c2"],
            ["r1c0", "r1c1", "r1c2"],
            ["r2c0", "r2c1", "r2c2"],
        ];
        NAMES[self.row][self.column]
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }

    fn families() -> Families<Self> {
        Families::Dimensions(DimensionFamilies {
            dimensions: GRID_DIMENSIONS,
            value_of: grid_value_of,
            compose: grid_compose,
        })
    }
}

/// The flat fake's entry, timing `order` first and leaving out `skip`.
const fn flat_priority(order: &'static [FamilyId], skip: &'static [FamilyId]) -> SitePriority {
    SitePriority {
        site: FlatFake::SITE,
        basis: Basis::Expected,
        order,
        skip,
    }
}

const FORWARD: PriorityTable = PriorityTable {
    sites: &[flat_priority(&[FamilyId("a"), FamilyId("b"), FamilyId("c")], &[])],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

const REVERSED: PriorityTable = PriorityTable {
    sites: &[flat_priority(&[FamilyId("c"), FamilyId("b"), FamilyId("a")], &[])],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

const SKIPS_B: PriorityTable = PriorityTable {
    sites: &[flat_priority(&[], &[FamilyId("b")])],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

const SKIPS_A: PriorityTable = PriorityTable {
    sites: &[flat_priority(&[], &[FamilyId("a")])],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

/// What one search did, with every batch it compiled.
struct Searched {
    outcome: SearchOutcome,
    compiled: Vec<Vec<usize>>,
    sampler: ScriptedSampler,
}

impl Searched {
    fn compiled_all(&self) -> Vec<usize> {
        self.compiled.iter().flatten().copied().collect()
    }

    fn timed(&self) -> Vec<usize> {
        let mut timed: Vec<usize> = self
            .outcome
            .timing
            .timed
            .iter()
            .map(|stats| stats.index)
            .collect();
        timed.sort_unstable();
        timed
    }
}

/// A sampler that times each allowed candidate at `micros` of it, every sample alike.
fn scripted<C: SiteCandidate>(allowed: &[C], micros: impl Fn(C) -> u64) -> ScriptedSampler {
    let times: Vec<u64> = allowed.iter().map(|&candidate| micros(candidate)).collect();
    let scripts: Vec<(usize, &[u64])> = times
        .iter()
        .enumerate()
        .map(|(index, time)| (index, std::slice::from_ref(time)))
        .collect();

    ScriptedSampler::new(&scripts)
}

fn limits_for(mode: SearchMode) -> DecideLimits {
    match mode {
        SearchMode::Families => DecideLimits::default(),
        SearchMode::Exhaustive => DecideLimits::exhaustive(),
    }
}

/// Searches `allowed` with every candidate compiling and passing its output check.
fn search<C: SiteCandidate>(
    allowed: &[C],
    sampler: ScriptedSampler,
    priority: &PriorityTable,
    mode: SearchMode,
    prescreen: bool,
) -> Searched {
    let mut sampler = sampler;
    let compiled = RefCell::new(Vec::new());
    let log_compile = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        compiled.borrow_mut().push(indices.to_vec());
        let kept = Precompiled::keeping(indices);
        Ok(kept)
    };
    let keep_all = |indices: &[usize]| indices.to_vec();
    let space = SearchSpace {
        allowed,
        priority,
        mode,
        prescreen,
    };
    let limits = limits_for(mode);

    let outcome =
        search_candidates(&mut sampler, space, &limits, log_compile, keep_all).expect("the search succeeds");
    let compiled = compiled.into_inner();

    Searched {
        outcome,
        compiled,
        sampler,
    }
}

fn index_of<C: SiteCandidate>(allowed: &[C], candidate: C) -> usize {
    allowed
        .iter()
        .position(|&listed| listed == candidate)
        .expect("the candidate is allowed")
}

fn indices_of<C: SiteCandidate>(allowed: &[C], candidates: &[C]) -> Vec<usize> {
    candidates
        .iter()
        .map(|&candidate| index_of(allowed, candidate))
        .collect()
}

fn opened_names(searched: &Searched) -> Vec<&'static str> {
    let mut names: Vec<&str> = searched
        .outcome
        .search
        .opened
        .iter()
        .map(|family| family.0)
        .collect();
    names.sort_unstable();
    names
}

fn sorted(mut indices: Vec<usize>) -> Vec<usize> {
    indices.sort_unstable();
    indices
}

/// `a` is the fastest, `b` within the margin of it and `c` beyond it.
fn flat_micros(candidate: FlatFake) -> u64 {
    match candidate {
        FlatFake::A0 => 1_000,
        FlatFake::A1 => 980,
        FlatFake::A2 => 1_020,
        FlatFake::B0 => 1_030,
        FlatFake::B1 => 960,
        FlatFake::C0 => 1_300,
        FlatFake::C1 => 500,
    }
}

fn flat_search(priority: &PriorityTable) -> Searched {
    let allowed = FlatFake::ALL;
    let sampler = scripted(allowed, flat_micros);

    search(allowed, sampler, priority, SearchMode::Families, false)
}

#[test]
fn only_representatives_compile_before_any_family_opens() {
    let allowed = FlatFake::ALL;
    let searched = flat_search(&FORWARD);
    let representatives = indices_of(allowed, &[FlatFake::A0, FlatFake::B0, FlatFake::C0]);

    assert_eq!(searched.compiled[0], representatives);
}

#[test]
fn a_family_within_the_margin_opens_and_compiles_its_members() {
    let allowed = FlatFake::ALL;
    let searched = flat_search(&FORWARD);
    let opened = indices_of(allowed, &[FlatFake::A1, FlatFake::A2, FlatFake::B1]);
    let search = &searched.outcome.search;
    let b1 = index_of(allowed, FlatFake::B1);
    let second_batch = sorted(searched.compiled[1].clone());

    assert_eq!(searched.compiled.len(), 2);
    assert_eq!(second_batch, opened);
    assert_eq!(search.opened, vec![FamilyId("a"), FamilyId("b")]);
    assert_eq!(search.candidates[b1].step, Some(SearchStep::Opened));
}

#[test]
fn a_family_beyond_the_margin_is_never_compiled() {
    let allowed = FlatFake::ALL;
    let searched = flat_search(&FORWARD);
    let c1 = index_of(allowed, FlatFake::C1);

    assert!(!searched.compiled_all().contains(&c1));
    assert_eq!(searched.sampler.served(c1), 0);
    assert_eq!(searched.outcome.search.candidates[c1].step, None);
}

#[test]
fn a_skipped_family_is_never_compiled() {
    let allowed = FlatFake::ALL;
    let searched = flat_search(&SKIPS_B);
    let family_b = indices_of(allowed, &[FlatFake::B0, FlatFake::B1]);
    let compiled = searched.compiled_all();

    for index in family_b {
        assert!(!compiled.contains(&index), "candidate {index} compiled");
    }

    assert_eq!(searched.outcome.search.skipped, vec![FamilyId("b")]);
}

/// The baseline's family is slow and the table asks to skip it, yet the baseline is still timed.
#[test]
fn the_baseline_is_always_timed_even_when_its_family_is_slow() {
    let allowed = FlatFake::ALL;
    let slow_baseline = |candidate: FlatFake| match candidate {
        FlatFake::A0 | FlatFake::A1 | FlatFake::A2 => 2_000,
        other => flat_micros(other),
    };
    let sampler = scripted(allowed, slow_baseline);

    let searched = search(allowed, sampler, &SKIPS_A, SearchMode::Families, true);
    let a1 = index_of(allowed, FlatFake::A1);

    assert!(searched.timed().contains(&0));
    assert!(!searched.compiled_all().contains(&a1));
    assert!(searched.outcome.search.skipped.is_empty());
}

#[test]
fn a_family_with_an_illegal_representative_is_skipped() {
    let allowed = [
        FlatFake::A0,
        FlatFake::A1,
        FlatFake::A2,
        FlatFake::B1,
        FlatFake::C0,
        FlatFake::C1,
    ];
    let sampler = scripted(&allowed, flat_micros);

    let searched = search(&allowed, sampler, &FORWARD, SearchMode::Families, false);
    let b1 = index_of(&allowed, FlatFake::B1);

    assert!(!searched.compiled_all().contains(&b1));
    assert_eq!(searched.outcome.search.skipped, vec![FamilyId("b")]);
}

#[test]
fn a_dimension_value_with_an_illegal_representative_is_never_built_up() {
    let allowed = [cell(0, 0), cell(0, 1), cell(0, 2), cell(1, 1), cell(2, 0)];
    let fast_column = |candidate: GridFake| match (candidate.row, candidate.column) {
        (0, 1) => 900,
        (1, 1) => 500,
        _ => 1_000,
    };
    let sampler = scripted(&allowed, fast_column);

    let neutral = PriorityTable::neutral();
    let searched = search(&allowed, sampler, neutral, SearchMode::Families, false);
    let r1c1 = index_of(&allowed, cell(1, 1));

    assert!(!searched.compiled_all().contains(&r1c1));
    assert_eq!(searched.outcome.search.skipped, vec![FamilyId("row1")]);
}

/// From `r2c0`, the only fastest representative, a round could only reach `r2c1` and `r2c2`. The
/// first is not allowed and the second is not built, so the round is empty.
#[test]
fn an_illegal_combination_is_left_out_of_the_opened_cross_product() {
    let allowed = [
        cell(0, 0),
        cell(0, 1),
        cell(0, 2),
        cell(1, 0),
        cell(1, 1),
        cell(2, 0),
    ];
    let fast_row = |candidate: GridFake| match (candidate.row, candidate.column) {
        (2, 0) => 800,
        _ => 1_000,
    };
    let sampler = scripted(&allowed, fast_row);

    let neutral = PriorityTable::neutral();
    let searched = search(&allowed, sampler, neutral, SearchMode::Families, false);
    let representatives = indices_of(
        &allowed,
        &[cell(0, 0), cell(1, 0), cell(2, 0), cell(0, 1), cell(0, 2)],
    );

    assert_eq!(searched.compiled, vec![representatives]);
    assert_eq!(
        searched.outcome.search.opened,
        vec![FamilyId("row2"), FamilyId("column0")]
    );
}

#[test]
fn exhaustive_mode_compiles_and_times_every_allowed_candidate() {
    let allowed = FlatFake::ALL;
    let sampler = scripted(allowed, flat_micros);

    let searched = search(allowed, sampler, &SKIPS_B, SearchMode::Exhaustive, true);
    let every: Vec<usize> = (0..allowed.len()).collect();
    let search = &searched.outcome.search;

    assert_eq!(searched.compiled, vec![every.clone()]);
    assert_eq!(searched.timed(), every);
    assert!(search.skipped.is_empty());
    assert_eq!(search.opened, vec![FamilyId("a"), FamilyId("b"), FamilyId("c")]);

    // C0 is more than a quarter slower than C1, so only a pre-screen would drop it.
    let c0 = index_of(allowed, FlatFake::C0);
    assert!(searched.sampler.served(c0) >= MIN_SAMPLES);
    assert!(
        search
            .candidates
            .iter()
            .all(|candidate| candidate.step == Some(SearchStep::Exhaustive))
    );
}

#[test]
fn exhaustive_limits_lift_both_caps_and_keep_the_minimums() {
    let limits = DecideLimits::exhaustive();
    let usual = DecideLimits::default();

    assert_eq!(limits.site_cap, Duration::MAX);
    assert_eq!(limits.hard_cap, Duration::MAX);
    assert_eq!(limits.min_samples, usual.min_samples);
    assert_eq!(limits.min_time, usual.min_time);
}

#[test]
fn the_table_order_sets_the_timing_order_not_what_opens() {
    let allowed = FlatFake::ALL;
    let forward = flat_search(&FORWARD);
    let reversed = flat_search(&REVERSED);
    let forward_first = indices_of(allowed, &[FlatFake::A0, FlatFake::B0, FlatFake::C0]);
    let reversed_first = indices_of(allowed, &[FlatFake::C0, FlatFake::B0, FlatFake::A0]);
    let forward_opened = opened_names(&forward);
    let reversed_opened = opened_names(&reversed);
    let forward_compiled = sorted(forward.compiled_all());
    let reversed_compiled = sorted(reversed.compiled_all());

    assert_eq!(forward.compiled[0], forward_first);
    assert_eq!(reversed.compiled[0], reversed_first);
    assert_eq!(forward_opened, reversed_opened);
    assert_eq!(forward_compiled, reversed_compiled);
}

/// `r0c1` is slower than the baseline alone, but with `r1c0`, the fastest representative, it wins.
#[test]
fn the_build_up_reaches_a_combination_no_single_representative_shows() {
    let allowed = GridFake::ALL;
    let pair_wins = |candidate: GridFake| match (candidate.row, candidate.column) {
        (0, 0) => 1_000,
        (1, 0) => 950,
        (0, 1) => 1_100,
        (1, 1) => 700,
        _ => 1_200,
    };
    let sampler = scripted(allowed, pair_wins);

    let neutral = PriorityTable::neutral();
    let searched = search(allowed, sampler, neutral, SearchMode::Families, false);
    let search = &searched.outcome.search;
    let r1c1 = index_of(allowed, cell(1, 1));

    assert_eq!(search.opened, vec![FamilyId("row1"), FamilyId("column1")]);
    assert_eq!(search.candidates[r1c1].step, Some(SearchStep::BuildUp(1)));
    assert!(searched.timed().contains(&r1c1));
}

/// `r1c1` is under 1% faster than `r1c0`, so the search stays on `r1c0` and never reaches `r2c1`.
#[test]
fn the_build_up_stops_when_a_round_gains_less_than_the_threshold() {
    let allowed = GridFake::ALL;
    let small_gain = |candidate: GridFake| match (candidate.row, candidate.column) {
        (1, 0) => 900,
        (1, 1) => 895,
        (2, 1) => 100,
        _ => 1_000,
    };
    let sampler = scripted(allowed, small_gain);

    let neutral = PriorityTable::neutral();
    let searched = search(allowed, sampler, neutral, SearchMode::Families, false);
    let r2c1 = index_of(allowed, cell(2, 1));

    assert_eq!(searched.compiled.len(), 2);
    assert!(!searched.compiled_all().contains(&r2c1));
    assert_eq!(
        searched.outcome.search.opened,
        vec![FamilyId("row1"), FamilyId("column0")]
    );
}

#[test]
fn the_build_up_never_compiles_a_candidate_it_does_not_reach() {
    let allowed = GridFake::ALL;
    let row_one_only = |candidate: GridFake| match (candidate.row, candidate.column) {
        (1, 0) => 900,
        _ => 1_000,
    };
    let sampler = scripted(allowed, row_one_only);

    let neutral = PriorityTable::neutral();
    let searched = search(allowed, sampler, neutral, SearchMode::Families, false);
    let representatives = indices_of(
        allowed,
        &[cell(0, 0), cell(1, 0), cell(2, 0), cell(0, 1), cell(0, 2)],
    );
    let first_round = indices_of(allowed, &[cell(1, 1), cell(1, 2)]);
    let r2c1 = index_of(allowed, cell(2, 1));

    assert_eq!(searched.compiled, vec![representatives, first_round]);
    assert_eq!(searched.sampler.served(r2c1), 0);
}

/// Batching costs 5% alone and gains 6% with the tiled centre, and the sliding neighbour walk gains
/// 3%. So the build-up moves twice, then times 4 groups on top and stops, since 4 groups gains only
/// 0.5%.
fn collab_micros(variant: CollabVariant) -> u64 {
    let centre: f64 = match variant.centre {
        Centre::Global => 1.0,
        Centre::Tiled => 0.88,
        Centre::Sliding => 0.97,
    };
    let batched = match (variant.batched, variant.centre) {
        (false, _) => 1.0,
        (true, Centre::Tiled) => 0.94,
        (true, _) => 1.05,
    };
    let groups = match variant.groups {
        4 => 0.995,
        16 => 1.02,
        _ => 1.0,
    };
    let walk = match variant.walk {
        Walk::Clipped => 1.0,
        Walk::Uniform => 1.1,
    };
    let neighbour = match variant.neighbour {
        NeighbourWalk::Rows => 1.0,
        NeighbourWalk::Sliding => 0.97,
    };
    let scale = centre * batched * groups * walk * neighbour;

    (10_000.0 * scale).round() as u64
}

#[test]
fn the_build_up_finds_a_three_dimension_collab_winner() {
    let allowed = CollabVariant::ALL;
    let sampler = scripted(allowed, collab_micros);

    let neutral = PriorityTable::neutral();
    let searched = search(allowed, sampler, neutral, SearchMode::Families, true);
    let pick = CollabVariant {
        centre: Centre::Tiled,
        batched: true,
        neighbour: NeighbourWalk::Sliding,
        ..CollabVariant::BASELINE
    };
    let regrouped = CollabVariant { groups: 4, ..pick };
    let pick_index = index_of(allowed, pick);
    let regrouped_index = index_of(allowed, regrouped);
    let candidates = &searched.outcome.search.candidates;
    let fastest = searched
        .outcome
        .timing
        .timed
        .iter()
        .min_by_key(|stats| stats.samples.median())
        .expect("candidates were timed");

    assert_eq!(allowed[0], CollabVariant::BASELINE);
    assert_eq!(fastest.index, regrouped_index);
    assert_eq!(candidates[pick_index].step, Some(SearchStep::BuildUp(2)));
    assert_eq!(candidates[regrouped_index].step, Some(SearchStep::BuildUp(3)));
    assert_eq!(searched.compiled_all().len(), 21);
}

#[test]
fn the_first_timed_count_is_the_allowed_unskipped_representatives() {
    let allowed = [
        FlatFake::A0,
        FlatFake::A1,
        FlatFake::B1,
        FlatFake::C0,
        FlatFake::C1,
    ];

    let families = first_timed(&allowed, &FORWARD, SearchMode::Families);
    let skipping = first_timed(FlatFake::ALL, &SKIPS_B, SearchMode::Families);
    let exhaustive = first_timed(&allowed, &FORWARD, SearchMode::Exhaustive);

    assert_eq!(families, 2);
    assert_eq!(skipping, 2);
    assert_eq!(exhaustive, allowed.len());
}

/// A scripted sampler whose elapsed time also counts a shared compile clock, less what it leaves out.
struct CompilingSampler {
    inner: ScriptedSampler,
    compile_clock: Rc<Cell<Duration>>,
    excluded: Duration,
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

const PAIR: [FlatFake; 2] = [FlatFake::A0, FlatFake::B0];

/// Two representatives whose samples overlap, so only the site cap ends their timing.
fn overlapping_pair() -> (CompilingSampler, Rc<Cell<Duration>>) {
    let alternating: Vec<u64> = (0..4000)
        .map(|round| if round % 2 == 0 { 900 } else { 1100 })
        .collect();
    let inner = ScriptedSampler::new(&[(0, &alternating), (1, &alternating)]);
    let clock = Cell::new(Duration::ZERO);
    let compile_clock = Rc::new(clock);
    let sampler = CompilingSampler {
        inner,
        compile_clock: compile_clock.clone(),
        excluded: Duration::ZERO,
    };

    (sampler, compile_clock)
}

fn pair_space() -> SearchSpace<'static, FlatFake> {
    SearchSpace {
        allowed: &PAIR,
        priority: &FORWARD,
        mode: SearchMode::Families,
        prescreen: false,
    }
}

#[test]
fn compile_time_does_not_shorten_the_timing_before_the_site_cap() {
    let (mut sampler, compile_clock) = overlapping_pair();
    let limits = DecideLimits::default();
    let compile_spent = Duration::from_secs(30);
    let slow_compile = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let advanced = compile_clock.get() + compile_spent;
        compile_clock.set(advanced);

        let mut precompiled = Precompiled::keeping(indices);
        precompiled.spent = compile_spent;
        Ok(precompiled)
    };
    let keep_all = |indices: &[usize]| indices.to_vec();
    let space = pair_space();

    let outcome =
        search_candidates(&mut sampler, space, &limits, slow_compile, keep_all).expect("the search succeeds");

    assert!(outcome.timing.capped);
    assert!(sampler.inner.elapsed() >= limits.site_cap);
    assert_eq!(sampler.excluded, compile_spent);
}

/// The same compile, reported as taking no time, eats the site cap before any timing.
#[test]
fn unreported_compile_time_ends_the_timing_at_its_minimums() {
    let (mut sampler, compile_clock) = overlapping_pair();
    let limits = DecideLimits::default();
    let silent_compile = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let advanced = compile_clock.get() + Duration::from_secs(30);
        compile_clock.set(advanced);

        let kept = Precompiled::keeping(indices);
        Ok(kept)
    };
    let keep_all = |indices: &[usize]| indices.to_vec();
    let space = pair_space();

    search_candidates(&mut sampler, space, &limits, silent_compile, keep_all).expect("the search succeeds");

    assert!(sampler.inner.elapsed() < limits.site_cap);
}

#[test]
fn a_member_that_does_not_compile_is_never_checked_or_timed() {
    let allowed = FlatFake::ALL;
    let mut sampler = scripted(allowed, flat_micros);
    let limits = DecideLimits::default();
    let a1 = index_of(allowed, FlatFake::A1);
    let drop_a1 = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let kept = indices.iter().copied().filter(|&index| index != a1).collect();
        let dropped = indices
            .iter()
            .filter(|&&index| index == a1)
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
    let space = SearchSpace {
        allowed,
        priority: &FORWARD,
        mode: SearchMode::Families,
        prescreen: false,
    };

    let outcome = search_candidates(&mut sampler, space, &limits, drop_a1, record_checked)
        .expect("the search succeeds");
    let timed: Vec<usize> = outcome.timing.timed.iter().map(|stats| stats.index).collect();

    assert!(!timed.contains(&a1));
    assert!(!checked.borrow().contains(&a1));
    assert_eq!(sampler.served(a1), 0);
}

#[test]
fn a_candidate_whose_output_check_fails_is_never_timed() {
    let allowed = FlatFake::ALL;
    let mut sampler = scripted(allowed, flat_micros);
    let limits = DecideLimits::default();
    let b1 = index_of(allowed, FlatFake::B1);
    let keep_all = |indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        let kept = Precompiled::keeping(indices);
        Ok(kept)
    };
    let drop_b1 = |indices: &[usize]| indices.iter().copied().filter(|&index| index != b1).collect();
    let space = SearchSpace {
        allowed,
        priority: &FORWARD,
        mode: SearchMode::Families,
        prescreen: false,
    };

    let outcome =
        search_candidates(&mut sampler, space, &limits, keep_all, drop_b1).expect("the search succeeds");
    let timed: Vec<usize> = outcome.timing.timed.iter().map(|stats| stats.index).collect();

    assert!(!timed.contains(&b1));
    assert_eq!(sampler.served(b1), 0);
}

#[test]
fn a_failing_precompile_fails_the_site() {
    let allowed = FlatFake::ALL;
    let mut sampler = scripted(allowed, flat_micros);
    let limits = DecideLimits::default();
    let failing = |_indices: &[usize]| -> Result<Precompiled, anyhow::Error> {
        anyhow::bail!("the baseline did not compile")
    };
    let keep_all = |indices: &[usize]| indices.to_vec();
    let space = SearchSpace {
        allowed,
        priority: &FORWARD,
        mode: SearchMode::Families,
        prescreen: false,
    };

    let result = search_candidates(&mut sampler, space, &limits, failing, keep_all);

    assert!(result.is_err());
    assert_eq!(sampler.served(0), 0);
}

/// `A2` is within a quarter of the representatives' fastest but more than a quarter slower than
/// `B1`, the fastest of the opened batch, so the batch's own pre-screen drops it.
#[test]
fn the_prescreen_runs_within_each_batch() {
    let allowed = FlatFake::ALL;
    let opened_spread = |candidate: FlatFake| match candidate {
        FlatFake::A2 => 1_200,
        FlatFake::B1 => 700,
        other => flat_micros(other),
    };
    let sampler = scripted(allowed, opened_spread);

    let searched = search(allowed, sampler, &FORWARD, SearchMode::Families, true);
    let a2 = index_of(allowed, FlatFake::A2);

    assert!(!searched.timed().contains(&a2));
    assert_eq!(searched.sampler.served(a2), PRESCREEN_SAMPLES);
}

#[test]
fn a_later_batch_where_every_member_fails_the_prescreen_keeps_the_earlier_ranking() {
    let allowed = FlatFake::ALL;
    let mut sampler = scripted(allowed, flat_micros);
    sampler.failing = indices_of(allowed, &[FlatFake::A1, FlatFake::A2, FlatFake::B1]);

    let searched = search(allowed, sampler, &FORWARD, SearchMode::Families, true);
    let timed_first = indices_of(allowed, &[FlatFake::A0, FlatFake::B0]);
    let representatives = sorted(timed_first);

    assert_eq!(searched.timed(), representatives);
}

fn tuning_selector(ledger: &Arc<TuneLedger>) -> KernelSelector<TestRuntime> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let kernels = EngineKernels {
        ledger: Some(ledger.clone()),
        ..EngineKernels::default()
    };

    KernelSelector::with_limits(&client, kernels, limits)
}

#[test]
fn a_key_with_only_its_baseline_stays_untimed() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.illegal = vec![ProbeCandidate::Medium, ProbeCandidate::Wide, ProbeCandidate::Huge];
    let key = site.key();
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);
    let deadline = Instant::now() + Duration::from_secs(20);
    ledger.set_target(key.clone(), deadline);

    let selector = tuning_selector(&ledger);
    selector.run(&site).expect("the tuned run succeeds");
    let result = ledger.result();
    let ranking = result.sites.get(&key).expect("the key was ranked");

    assert_eq!(ranking.candidates.len(), 1);
    assert_eq!(ranking.candidates[0].samples, 0);
    assert_eq!(site.launches(), vec![ProbeCandidate::Small]);
}
