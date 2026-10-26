use std::time::Duration;

use av_denoise_core::bench_api::tune_sites::{
    Centre,
    CollabVariant,
    NeighbourWalk,
    RegulariseCandidate,
    Walk,
};
use av_denoise_core::tune::{
    BASELINE_FAMILY,
    Basis,
    CandidateRecord,
    DecideLimits,
    FamilyId,
    FidelityClass,
    PipelineProfile,
    Precompiled,
    PriorityTable,
    ProfileRecord,
    SampleTiming,
    Sampler,
    SearchMode,
    SearchOutcome,
    SearchSpace,
    SearchStep,
    SearchedCandidate,
    SiteCandidate,
    SiteKey,
    SitePriority,
    SiteRanking,
    SiteSearch,
    SiteTiming,
    TuneResult,
    choose,
    search_candidates,
};

use crate::tune::report::{format_report, replay, replay_profiles};

const FLAT_SITE: &str = "flat_report";

const FORWARD: PriorityTable = PriorityTable {
    sites: &[SitePriority {
        site: FLAT_SITE,
        basis: Basis::Expected,
        order: &[FamilyId("a"), FamilyId("b"), FamilyId("c")],
        skip: &[],
    }],
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
    ],
};

const MEASURED: PriorityTable = PriorityTable {
    sites: &[SitePriority {
        site: FLAT_SITE,
        basis: Basis::Expected,
        order: &[FamilyId("b"), FamilyId("a"), FamilyId("c")],
        skip: &[],
    }],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

const SKIPS_B: PriorityTable = PriorityTable {
    sites: &[SitePriority {
        site: FLAT_SITE,
        basis: Basis::Expected,
        order: &[],
        skip: &[FamilyId("b")],
    }],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

const SKIPS_NEIGHBOUR_SLIDING: PriorityTable = PriorityTable {
    sites: &[SitePriority {
        site: CollabVariant::SITE,
        basis: Basis::Expected,
        order: &[],
        skip: &[FamilyId("neighbour_sliding")],
    }],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

/// A flat site of three families, `a` holding the baseline, each led by its `0` member.
const FLAT: [(&str, &str); 6] = [
    ("a0", "a"),
    ("a1", "a"),
    ("b0", "b"),
    ("b1", "b"),
    ("c0", "c"),
    ("c1", "c"),
];

fn flat_search() -> SiteSearch {
    let candidates = FLAT
        .iter()
        .map(|&(name, family)| {
            let id = FamilyId(family);
            let represents = name.ends_with('0').then_some(id);

            SearchedCandidate {
                name,
                families: vec![id],
                represents,
                step: Some(SearchStep::Exhaustive),
            }
        })
        .collect();

    SiteSearch {
        mode: SearchMode::Exhaustive,
        dimensions: Vec::new(),
        candidates,
        opened: vec![FamilyId("a"), FamilyId("b"), FamilyId("c")],
        skipped: Vec::new(),
    }
}

fn record(name: &str, micros: u64) -> CandidateRecord {
    CandidateRecord {
        name: name.to_string(),
        fidelity: FidelityClass::Exact,
        baseline: name == "a0",
        median_ns: micros * 1_000,
        samples: 15,
    }
}

fn records(timings: &[(&str, u64)]) -> Vec<CandidateRecord> {
    timings
        .iter()
        .map(|&(name, micros)| record(name, micros))
        .collect()
}

/// Family `b` opens within the margin and holds the fastest member. Family `c` stays shut.
fn opening_timings() -> Vec<CandidateRecord> {
    records(&[
        ("a0", 100),
        ("a1", 98),
        ("b0", 103),
        ("b1", 90),
        ("c0", 120),
        ("c1", 119),
    ])
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

#[test]
fn replay_finds_the_pruned_winner_from_exhaustive_timings() {
    let search = flat_search();
    let timings = opening_timings();

    let replayed = replay(FLAT_SITE, &search, &timings, &FORWARD);
    let pruned = &replayed.pruned.as_ref().expect("the pruned search picks").pick;

    assert!(replayed.matches(), "{replayed:?}");
    assert_eq!(pruned.name, "b1");
    assert_eq!(replayed.slowdown(), 0.0);
    assert_eq!(replayed.pruned_tried, 5);
    assert_eq!(replayed.allowed, 6);
}

#[test]
fn replay_reports_the_slowdown_when_the_pruned_winner_differs() {
    let search = flat_search();
    let timings = records(&[
        ("a0", 100),
        ("a1", 98),
        ("b0", 103),
        ("b1", 90),
        ("c0", 120),
        ("c1", 80),
    ]);

    let replayed = replay(FLAT_SITE, &search, &timings, &FORWARD);
    let pruned = &replayed.pruned.as_ref().expect("the pruned search picks").pick;
    let full = &replayed.full.as_ref().expect("the full search picks").pick;

    assert!(!replayed.matches());
    assert_eq!(pruned.name, "b1");
    assert_eq!(full.name, "c1");
    assert!((replayed.slowdown() - 0.125).abs() < 1e-9, "{replayed:?}");
}

#[test]
fn replay_reports_each_familys_representative_gap() {
    let search = flat_search();
    let timings = opening_timings();

    let replayed = replay(FLAT_SITE, &search, &timings, &FORWARD);
    let gaps: Vec<(&str, &str, u64, &str, u64)> = replayed
        .families
        .iter()
        .map(|gap| {
            let representative = &gap.representative;
            let best = &gap.best;
            let representative_micros = micros(representative.median);
            let best_micros = micros(best.median);

            (
                gap.family.0,
                representative.name.as_str(),
                representative_micros,
                best.name.as_str(),
                best_micros,
            )
        })
        .collect();

    assert_eq!(
        gaps,
        vec![
            ("a", "a0", 100, "a1", 98),
            ("b", "b0", 103, "b1", 90),
            ("c", "c0", 120, "c1", 119),
        ]
    );

    let b_gap = replayed.families[1].gap();
    assert!((b_gap - (103.0 / 90.0 - 1.0)).abs() < 1e-9, "{b_gap}");
}

#[test]
fn replay_reports_where_the_table_order_disagrees() {
    let search = flat_search();
    let timings = records(&[
        ("a0", 100),
        ("a1", 80),
        ("b0", 95),
        ("b1", 90),
        ("c0", 120),
        ("c1", 70),
    ]);

    let forward = replay(FLAT_SITE, &search, &timings, &FORWARD);
    let measured = replay(FLAT_SITE, &search, &timings, &MEASURED);
    let expected = vec![FamilyId("b"), FamilyId("a"), FamilyId("c")];

    assert_eq!(forward.measured_order, expected);
    assert_ne!(forward.table_order, forward.measured_order);
    assert_eq!(measured.table_order, expected);
    assert_eq!(measured.measured_order, expected);
}

#[test]
fn replay_never_opens_a_family_whose_representative_was_not_timed() {
    let search = flat_search();
    let timings = records(&[("a0", 100), ("a1", 98), ("b1", 90), ("c0", 120), ("c1", 119)]);

    let replayed = replay(FLAT_SITE, &search, &timings, &FORWARD);
    let pruned = &replayed.pruned.as_ref().expect("the pruned search picks").pick;

    assert_eq!(pruned.name, "a1");
    assert_eq!(replayed.pruned_tried, 4);
    assert!(!replayed.matches());
}

#[test]
fn replay_leaves_out_a_family_the_table_skips() {
    let search = flat_search();
    let timings = opening_timings();

    let replayed = replay(FLAT_SITE, &search, &timings, &SKIPS_B);
    let pruned = &replayed.pruned.as_ref().expect("the pruned search picks").pick;

    assert_eq!(pruned.name, "a1");
    assert_eq!(replayed.pruned_tried, 3);
}

fn profile(profile: PipelineProfile, micros: u64) -> ProfileRecord {
    ProfileRecord {
        profile,
        median_frame_ns: micros * 1_000,
        blocks: 3,
    }
}

/// Split wins, packed does not beat it by the margin, and f32 is the fastest of all.
fn profile_timings() -> Vec<ProfileRecord> {
    vec![
        profile(PipelineProfile::Baseline, 100),
        profile(PipelineProfile::SplitSearch, 90),
        profile(PipelineProfile::PackedAtomics, 89),
        profile(PipelineProfile::F32Search, 50),
    ]
}

#[test]
fn replay_profiles_follows_the_early_stop() {
    let timings = profile_timings();

    let replayed = replay_profiles(&timings, &FORWARD).expect("there are profiles to replay");

    assert_eq!(
        replayed.timed,
        vec![PipelineProfile::SplitSearch, PipelineProfile::PackedAtomics]
    );
    assert_eq!(replayed.pruned.name, PipelineProfile::PackedAtomics.name());
    assert_eq!(replayed.full.name, PipelineProfile::F32Search.name());
    assert!(!replayed.matches());
    assert_eq!(
        replayed.table_order,
        vec![
            PipelineProfile::SplitSearch,
            PipelineProfile::PackedAtomics,
            PipelineProfile::F32Search,
        ]
    );
    assert_eq!(
        replayed.measured_order,
        vec![
            PipelineProfile::F32Search,
            PipelineProfile::PackedAtomics,
            PipelineProfile::SplitSearch,
        ]
    );
}

#[test]
fn replay_profiles_needs_a_baseline_and_an_alternative() {
    let baseline_only = vec![profile(PipelineProfile::Baseline, 100)];

    let replayed = replay_profiles(&baseline_only, &FORWARD);
    let nothing = replay_profiles(&[], &FORWARD);

    assert!(replayed.is_none());
    assert!(nothing.is_none());
}

#[test]
fn the_report_has_a_block_per_key_then_profiles_then_totals() {
    let search = flat_search();
    let timings = opening_timings();
    let key = SiteKey::builder(FLAT_SITE, 64, 48).build();
    let ranking = SiteRanking {
        candidates: timings,
        partial: false,
        progress: Default::default(),
        identity: String::new(),
        search: Some(search),
    };
    let mut ranked = TuneResult::default();
    ranked.sites.insert(key.clone(), ranking);
    let profiles = profile_timings();

    let text = format_report(&ranked, &profiles, &FORWARD);
    let key_at = text.find(key.as_str()).expect("the key has a block");
    let profiles_at = text.find("pipeline profiles").expect("the profiles have a block");
    let totals_at = text
        .find("matched 1 of 1 keys")
        .expect("the totals close the report");

    assert!(key_at < profiles_at, "{text}");
    assert!(profiles_at < totals_at, "{text}");
    assert!(text.contains("b1"), "{text}");
    assert!(text.contains("compiles 5 of 6"), "{text}");
    assert!(text.contains("Every pruned pick is the full pick."), "{text}");
    assert!(text.contains("The profile early stop did not match."), "{text}");
}

/// Times each candidate at its cost in nanoseconds, plus one nanosecond per index so no two tie.
struct CostSampler<C> {
    allowed: Vec<C>,
    cost: fn(C) -> u64,
    elapsed: Duration,
}

impl<C: SiteCandidate> Sampler for CostSampler<C> {
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error> {
        let candidate = self.allowed[index];
        let nanos = (self.cost)(candidate) + index as u64;
        let total = Duration::from_nanos(nanos);
        self.elapsed += total;

        Ok(SampleTiming { total, launches: 1 })
    }

    fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

fn candidate_records<C: SiteCandidate>(allowed: &[C], timing: &SiteTiming) -> Vec<CandidateRecord> {
    timing
        .timed
        .iter()
        .map(|stats| {
            let candidate = allowed[stats.index];
            let median = stats.samples.median();

            CandidateRecord {
                name: candidate.name().to_string(),
                fidelity: candidate.fidelity().class(),
                baseline: stats.index == 0,
                median_ns: median.as_nanos() as u64,
                samples: stats.samples.count() as u32,
            }
        })
        .collect()
}

/// Runs the families search and the exhaustive search on the same costs, and checks that replaying
/// the exhaustive timings picks and compiles what the families search did.
fn assert_replay_matches_the_search<C: SiteCandidate>(
    allowed: &[C],
    cost: fn(C) -> u64,
    table: &PriorityTable,
) {
    let families = run_search(allowed, cost, table, SearchMode::Families);
    let exhaustive = run_search(allowed, cost, table, SearchMode::Exhaustive);

    let family_records = candidate_records(allowed, &families.timing);
    let family_pick = choose(&family_records, false).expect("the families search picks");
    let family_tried = families
        .search
        .candidates
        .iter()
        .filter(|candidate| candidate.step.is_some())
        .count();

    let exhaustive_records = candidate_records(allowed, &exhaustive.timing);
    let replayed = replay(C::SITE, &exhaustive.search, &exhaustive_records, table);
    let pruned = &replayed.pruned.as_ref().expect("the replay picks").pick;

    assert_eq!(pruned.name, family_pick.name, "{replayed:?}");
    assert_eq!(replayed.pruned_tried, family_tried, "{replayed:?}");
    assert!(replayed.pruned_tried < replayed.allowed, "{replayed:?}");
}

fn run_search<C: SiteCandidate>(
    allowed: &[C],
    cost: fn(C) -> u64,
    table: &PriorityTable,
    mode: SearchMode,
) -> SearchOutcome {
    let mut sampler = CostSampler {
        allowed: allowed.to_vec(),
        cost,
        elapsed: Duration::ZERO,
    };
    let space = SearchSpace {
        allowed,
        priority: table,
        mode,
        prescreen: false,
    };
    let limits = match mode {
        SearchMode::Families => DecideLimits::default(),
        SearchMode::Exhaustive => DecideLimits::exhaustive(),
    };
    let precompile = |indices: &[usize]| Ok(Precompiled::keeping(indices));
    let check = |indices: &[usize]| indices.to_vec();

    search_candidates(&mut sampler, space, &limits, precompile, check).expect("the search runs")
}

/// Batching loses alone and wins with the tiled centre, the sliding neighbour walk gains 3%, and 4
/// groups gains under the build-up threshold.
fn collab_nanos(variant: CollabVariant) -> u64 {
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

    (10_000_000.0 * scale).round() as u64
}

/// The serial family leads, and the cooperative family stays shut although it holds the fastest
/// member.
fn regularise_nanos(candidate: RegulariseCandidate) -> u64 {
    match candidate {
        RegulariseCandidate::Serial8x8 => 10_000_000,
        RegulariseCandidate::Serial8x4 => 9_800_000,
        RegulariseCandidate::Serial4x4 => 10_400_000,
        RegulariseCandidate::Coop32 => 10_600_000,
        RegulariseCandidate::Coop64 => 9_000_000,
    }
}

#[test]
fn replay_matches_the_families_search_on_a_dimension_site() {
    let allowed = CollabVariant::ALL;
    let neutral = PriorityTable::neutral();

    assert_replay_matches_the_search(allowed, collab_nanos, neutral);
    assert_replay_matches_the_search(allowed, collab_nanos, &SKIPS_NEIGHBOUR_SLIDING);
}

#[test]
fn replay_matches_the_families_search_without_an_illegal_representative() {
    let uniform = CollabVariant {
        walk: Walk::Uniform,
        ..CollabVariant::BASELINE
    };
    let allowed: Vec<CollabVariant> = CollabVariant::ALL
        .iter()
        .copied()
        .filter(|&variant| variant != uniform)
        .collect();
    let neutral = PriorityTable::neutral();

    assert_replay_matches_the_search(&allowed, collab_nanos, neutral);
}

#[test]
fn replay_matches_the_families_search_on_a_flat_site() {
    let allowed = RegulariseCandidate::ALL;
    let neutral = PriorityTable::neutral();

    assert_replay_matches_the_search(allowed, regularise_nanos, neutral);
}

#[test]
fn a_pick_kept_over_a_faster_record_names_it() {
    let search = flat_search();
    let timings = records(&[
        ("a0", 100),
        ("a1", 99),
        ("b0", 120),
        ("b1", 130),
        ("c0", 140),
        ("c1", 150),
    ]);
    let key = SiteKey::builder(FLAT_SITE, 64, 48).build();
    let ranking = SiteRanking {
        candidates: timings,
        partial: false,
        progress: Default::default(),
        identity: String::new(),
        search: Some(search),
    };
    let mut ranked = TuneResult::default();
    ranked.sites.insert(key, ranking);

    let text = format_report(&ranked, &[], &FORWARD);

    assert!(
        text.contains("full    a0  0.100 ms  kept over a1 0.099 ms as the baseline within 2%"),
        "{text}"
    );
}

const GRID_SITE: &str = "grid_report";

/// A 3x3 grid of rows and columns, `r0c0` the baseline.
fn grid_search() -> SiteSearch {
    let mut candidates = Vec::new();

    for (name, row, column) in GRID {
        let families = vec![FamilyId(row), FamilyId(column)];
        let represents = match name {
            "r0c0" => Some(BASELINE_FAMILY),
            "r1c0" | "r2c0" => Some(FamilyId(row)),
            "r0c1" | "r0c2" => Some(FamilyId(column)),
            _ => None,
        };

        candidates.push(SearchedCandidate {
            name,
            families,
            represents,
            step: Some(SearchStep::Exhaustive),
        });
    }

    SiteSearch {
        mode: SearchMode::Exhaustive,
        dimensions: vec!["row", "column"],
        candidates,
        opened: Vec::new(),
        skipped: Vec::new(),
    }
}

const GRID: [(&str, &str, &str); 9] = [
    ("r0c0", "r0", "c0"),
    ("r0c1", "r0", "c1"),
    ("r0c2", "r0", "c2"),
    ("r1c0", "r1", "c0"),
    ("r1c1", "r1", "c1"),
    ("r1c2", "r1", "c2"),
    ("r2c0", "r2", "c0"),
    ("r2c1", "r2", "c1"),
    ("r2c2", "r2", "c2"),
];

fn grid_records(timings: &[(&str, u64)]) -> Vec<CandidateRecord> {
    timings
        .iter()
        .map(|&(name, micros)| CandidateRecord {
            name: name.to_string(),
            fidelity: FidelityClass::Exact,
            baseline: name == "r0c0",
            median_ns: micros * 1_000,
            samples: 15,
        })
        .collect()
}

/// `r1c1` failed and has no record, so the round moves to `r1c2` and never reaches `r2c1`.
#[test]
fn a_build_up_round_skips_a_candidate_with_no_record() {
    let search = grid_search();
    let timings = grid_records(&[
        ("r0c0", 100),
        ("r0c1", 110),
        ("r0c2", 115),
        ("r1c0", 90),
        ("r1c2", 85),
        ("r2c0", 120),
        ("r2c1", 50),
        ("r2c2", 200),
    ]);
    let neutral = PriorityTable::neutral();

    let replayed = replay(GRID_SITE, &search, &timings, neutral);
    let pruned = &replayed.pruned.as_ref().expect("the pruned search picks").pick;
    let full = &replayed.full.as_ref().expect("the full search picks").pick;

    assert_eq!(pruned.name, "r1c2");
    assert_eq!(full.name, "r2c1");
    assert_eq!(replayed.pruned_tried, 8);
}
