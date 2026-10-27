use std::collections::BTreeSet;
use std::time::Duration;

use super::consts::{BASELINE_INDEX, BUILD_UP_GAIN, FAMILY_MARGIN};
use super::decide::{CandidateStats, DecideLimits, Sampler, SiteTiming, prescreen, sample_until_settled};
use super::family::{DimensionFamilies, Families, FamilyId, opened_members, representatives};
use super::precompile::Precompiled;
use super::priority::{PriorityTable, family_order, skipped};
use super::site::SiteCandidate;

/// How widely a site key's candidates are searched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Times each family's representative, then only what looks worth it.
    #[default]
    Families,
    /// Times every allowed candidate, with no pre-screen, a wider site cap and no hard cap.
    Exhaustive,
}

/// Why the search tried a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStep {
    /// It stands in for its family, so it was tried first.
    Representative,
    /// Its flat family opened within [FAMILY_MARGIN](crate::tune::FAMILY_MARGIN).
    Opened,
    /// It differs from the best so far in one dimension, in this build-up round counted from 1.
    BuildUp(usize),
    /// The search tried every candidate.
    Exhaustive,
}

/// One allowed candidate as the search saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchedCandidate {
    pub name: &'static str,
    /// Its family on a flat site, or its value in each dimension in order.
    pub families: Vec<FamilyId>,
    /// The family it stands in for, when it is a representative.
    pub represents: Option<FamilyId>,
    /// Why the search tried it, `None` when it never did.
    pub step: Option<SearchStep>,
}

/// How one site key's candidates were searched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteSearch {
    pub mode: SearchMode,
    /// The dimension names, empty on a flat site.
    pub dimensions: Vec<&'static str>,
    /// Every allowed candidate, the baseline first.
    pub candidates: Vec<SearchedCandidate>,
    /// The families the search opened.
    ///
    /// A flat site opens the families within the margin. A dimension site counts the values of the
    /// build-up's pick. An exhaustive search opens every family.
    pub opened: Vec<FamilyId>,
    /// The families a family-level search leaves out, by the table or because their representative is
    /// not allowed.
    ///
    /// An exhaustive search still times them.
    pub skipped: Vec<FamilyId>,
}

/// The timings a search took and how it took them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOutcome {
    pub timing: SiteTiming,
    pub search: SiteSearch,
}

/// Whether a family whose representative has `median` opens next to the best representative's `best`.
pub fn opens_family(median: Duration, best: Duration) -> bool {
    let threshold = best.as_secs_f64() * (1.0 + FAMILY_MARGIN);
    median.as_secs_f64() <= threshold
}

/// Whether a build-up round's best `median` is enough faster than the best so far, `current`.
pub fn builds_up(median: Duration, current: Duration) -> bool {
    let threshold = current.as_secs_f64() * (1.0 - BUILD_UP_GAIN);
    median.as_secs_f64() <= threshold
}

/// How many of `allowed` the search times before it opens anything.
///
/// It is every candidate in an exhaustive search.
pub fn first_timed<C: SiteCandidate>(allowed: &[C], priority: &PriorityTable, mode: SearchMode) -> usize {
    if mode == SearchMode::Exhaustive {
        return allowed.len();
    }

    let first = first_batch(allowed, priority);
    first.representatives.len()
}

/// Times the candidates at one site key in batches and returns every candidate timed.
///
/// `allowed[0]` is the baseline, which is always timed. The families search times every allowed,
/// unskipped family's representative first, in the table's order. A flat site then opens every
/// family within [FAMILY_MARGIN](crate::tune::FAMILY_MARGIN) of the best representative and times its
/// other members. A dimension site builds up instead. From the fastest so far, each round times every
/// candidate one value away from it, and moves to the round's best when that is at least
/// [BUILD_UP_GAIN](crate::tune::BUILD_UP_GAIN) faster. It stops at a round without such a gain, or
/// after one round per dimension. An exhaustive search times every allowed candidate in one batch.
///
/// Each batch goes through `precompile` first, whose time is left out of the sampler's elapsed time.
/// A families search then pre-screens it when the space asks, `check` drops the candidates whose
/// output differs, and the rest are sampled until settled. A later batch where nothing passes the
/// pre-screen times nothing. The site cap counts the whole site, and the hard cap restarts with each
/// batch.
pub fn search_candidates<C: SiteCandidate, S: Sampler>(
    sampler: &mut S,
    space: SearchSpace<'_, C>,
    limits: &DecideLimits,
    precompile: impl FnMut(&[usize]) -> Result<Precompiled, anyhow::Error>,
    check: impl FnMut(&[usize]) -> Vec<usize>,
) -> Result<SearchOutcome, anyhow::Error> {
    let allowed = space.allowed;
    let prescreen = space.prescreen && space.mode == SearchMode::Families;
    let mut search = Search {
        sampler,
        allowed,
        limits,
        prescreen,
        precompile,
        check,
        tried: BTreeSet::new(),
        steps: vec![None; allowed.len()],
        timed: Vec::new(),
        failed: Vec::new(),
        raced: Vec::new(),
        capped: false,
    };

    let families = C::families();
    let first = first_batch(allowed, space.priority);
    let opened = match space.mode {
        SearchMode::Exhaustive => {
            let every: Vec<usize> = (0..allowed.len()).collect();
            search.run_batch(&every, SearchStep::Exhaustive, true)?;
            every_family(&families)
        },
        SearchMode::Families => {
            let indices: Vec<usize> = first.representatives.iter().map(|&(_, index)| index).collect();
            search.run_batch(&indices, SearchStep::Representative, true)?;

            match &families {
                Families::Flat(_) => search.open_families(&families, &first)?,
                Families::Dimensions(dimensions) => search.build_up(dimensions, &first.skipped)?,
            }
        },
    };

    let searched = searched_candidates(allowed, &families, &first, &search.steps);
    let dimensions = dimension_names(&families);

    let site_search = SiteSearch {
        mode: space.mode,
        dimensions,
        candidates: searched,
        opened,
        skipped: first.skipped,
    };
    let timing = SiteTiming {
        timed: search.timed,
        failed: search.failed,
        raced: search.raced,
        capped: search.capped,
    };

    Ok(SearchOutcome {
        timing,
        search: site_search,
    })
}

/// The candidates one site key may run and how to search them.
#[derive(Debug, Clone, Copy)]
pub struct SearchSpace<'a, C> {
    /// The candidates the key may run, the baseline first.
    pub allowed: &'a [C],
    pub priority: &'a PriorityTable,
    pub mode: SearchMode,
    /// Whether each batch is pre-screened in a families search.
    pub prescreen: bool,
}

/// The representatives a families search times first, and the families it leaves out.
struct FirstBatch {
    /// Each unskipped family with its representative's index in `allowed`, in the table's order.
    representatives: Vec<(FamilyId, usize)>,
    /// Every family whose representative is allowed, skipped or not, in declaration order.
    allowed_representatives: Vec<(FamilyId, usize)>,
    skipped: Vec<FamilyId>,
}

/// The allowed representatives in the table's order, leaving out the families the table skips.
///
/// A family whose representative is not allowed is skipped too. The baseline's family is never
/// skipped.
fn first_batch<C: SiteCandidate>(allowed: &[C], priority: &PriorityTable) -> FirstBatch {
    let families = C::families();
    let baseline = allowed[BASELINE_INDEX];
    let table_skips = skipped(priority, C::SITE);
    let mut kept = Vec::new();
    let mut allowed_representatives = Vec::new();
    let mut skipped_families = Vec::new();

    for (family, representative) in representatives(&families, baseline) {
        let index = allowed.iter().position(|&candidate| candidate == representative);
        let Some(index) = index else {
            skipped_families.push(family);
            continue;
        };

        allowed_representatives.push((family, index));

        let skip = index != BASELINE_INDEX && table_skips.contains(&family);
        if skip {
            skipped_families.push(family);
            continue;
        }

        kept.push((family, index));
    }

    let ids: Vec<FamilyId> = kept.iter().map(|&(family, _)| family).collect();
    let order = family_order(priority, C::SITE, &ids);
    let ordered = order
        .iter()
        .filter_map(|family| kept.iter().find(|(kept_family, _)| kept_family == family))
        .copied()
        .collect();

    FirstBatch {
        representatives: ordered,
        allowed_representatives,
        skipped: skipped_families,
    }
}

/// The state of one search as its batches run.
struct Search<'a, C, S, P, K> {
    sampler: &'a mut S,
    allowed: &'a [C],
    limits: &'a DecideLimits,
    prescreen: bool,
    precompile: P,
    check: K,
    /// Every index handed to a batch, so none is tried twice.
    tried: BTreeSet<usize>,
    steps: Vec<Option<SearchStep>>,
    timed: Vec<CandidateStats>,
    failed: Vec<usize>,
    raced: Vec<usize>,
    capped: bool,
}

impl<C, S, P, K> Search<'_, C, S, P, K>
where
    C: SiteCandidate,
    S: Sampler,
    P: FnMut(&[usize]) -> Result<Precompiled, anyhow::Error>,
    K: FnMut(&[usize]) -> Vec<usize>,
{
    /// Compiles, screens, checks and times the indices not yet tried, returning those it timed.
    ///
    /// The first batch holds the baseline, so a failing pre-screen fails the site. A later batch with
    /// no pre-screen survivor times nothing.
    fn run_batch(
        &mut self,
        indices: &[usize],
        step: SearchStep,
        first: bool,
    ) -> Result<Vec<CandidateStats>, anyhow::Error> {
        let fresh: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|index| !self.tried.contains(index))
            .collect();

        if fresh.is_empty() {
            return Ok(Vec::new());
        }

        for &index in &fresh {
            self.tried.insert(index);
            self.steps[index] = Some(step);
        }

        let precompiled = (self.precompile)(&fresh)?;
        self.sampler.exclude(precompiled.spent);
        let kept = precompiled.kept;

        if kept.is_empty() {
            return Ok(Vec::new());
        }

        let screened = if !self.prescreen {
            kept
        } else if first {
            prescreen(self.sampler, &kept, BASELINE_INDEX, self.limits)?
        } else {
            let survivors = prescreen(self.sampler, &kept, BASELINE_INDEX, self.limits);

            match survivors {
                Ok(survivors) => survivors,
                Err(error) => {
                    tracing::warn!(?step, %error, "skipping a batch with no pre-screen survivor");
                    return Ok(Vec::new());
                },
            }
        };

        let checked = (self.check)(&screened);
        let elapsed = self.sampler.elapsed();
        let batch_limits = DecideLimits {
            hard_cap: elapsed.saturating_add(self.limits.hard_cap),
            ..*self.limits
        };
        let timing = sample_until_settled(self.sampler, &checked, BASELINE_INDEX, &batch_limits)?;

        self.failed.extend(timing.failed);
        self.raced.extend(timing.raced);
        self.capped |= timing.capped;
        self.timed.extend(timing.timed.iter().cloned());

        Ok(timing.timed)
    }

    /// Opens every flat family whose representative is within the margin of the best one, and times
    /// the opened families' other members.
    fn open_families(
        &mut self,
        families: &Families<C>,
        first: &FirstBatch,
    ) -> Result<Vec<FamilyId>, anyhow::Error> {
        let timed_representatives: Vec<(FamilyId, Duration)> = first
            .representatives
            .iter()
            .filter_map(|&(family, index)| {
                let median = median_of(&self.timed, index)?;
                Some((family, median))
            })
            .collect();

        let best = timed_representatives.iter().map(|&(_, median)| median).min();
        let Some(best) = best else {
            return Ok(Vec::new());
        };

        let opened: Vec<FamilyId> = timed_representatives
            .iter()
            .filter(|&&(_, median)| opens_family(median, best))
            .map(|&(family, _)| family)
            .collect();

        let baseline = self.allowed[BASELINE_INDEX];
        let members = opened_members(families, baseline, &opened);
        let indices = allowed_indices(self.allowed, &members);
        self.run_batch(&indices, SearchStep::Opened, false)?;

        Ok(opened)
    }

    /// Builds up from the fastest candidate so far one dimension value at a time, and returns the
    /// families of the pick.
    fn build_up(
        &mut self,
        dimensions: &DimensionFamilies<C>,
        skipped_families: &[FamilyId],
    ) -> Result<Vec<FamilyId>, anyhow::Error> {
        let baseline = self.allowed[BASELINE_INDEX];
        let base = values_of(dimensions, baseline);

        let Some(mut current) = fastest(&self.timed) else {
            return Ok(Vec::new());
        };

        for round in 1..=dimensions.dimensions.len() {
            let candidate = self.allowed[current.0];
            let values = values_of(dimensions, candidate);
            let around = one_value_away(dimensions, &values, &base, skipped_families);
            let indices = allowed_indices(self.allowed, &around);

            let step = SearchStep::BuildUp(round);
            let round_timed = self.run_batch(&indices, step, false)?;
            let Some(round_best) = fastest(&round_timed) else {
                break;
            };

            if !builds_up(round_best.1, current.1) {
                break;
            }

            current = round_best;
        }

        let pick = self.allowed[current.0];
        let pick_values = values_of(dimensions, pick);
        let opened = value_ids(dimensions, &pick_values);

        Ok(opened)
    }
}

/// Every candidate that differs from `values` in exactly one dimension and that the site builds.
///
/// A dimension's value in `skipped_families` is left out, unless it is the baseline's value `base`.
fn one_value_away<C>(
    dimensions: &DimensionFamilies<C>,
    values: &[usize],
    base: &[usize],
    skipped_families: &[FamilyId],
) -> Vec<C> {
    let mut around = Vec::new();

    for (dimension_index, dimension) in dimensions.dimensions.iter().enumerate() {
        for (value_index, value) in dimension.values.iter().enumerate() {
            if value_index == values[dimension_index] {
                continue;
            }

            let is_base = value_index == base[dimension_index];
            if !is_base && skipped_families.contains(&value.id) {
                continue;
            }

            let mut moved = values.to_vec();
            moved[dimension_index] = value_index;

            if let Some(candidate) = (dimensions.compose)(&moved) {
                around.push(candidate);
            }
        }
    }

    around
}

/// The indices in `allowed` of `candidates`, in their order, leaving out those not allowed.
fn allowed_indices<C: SiteCandidate>(allowed: &[C], candidates: &[C]) -> Vec<usize> {
    candidates
        .iter()
        .filter_map(|&candidate| allowed.iter().position(|&listed| listed == candidate))
        .collect()
}

fn median_of(timed: &[CandidateStats], index: usize) -> Option<Duration> {
    timed
        .iter()
        .find(|stats| stats.index == index)
        .map(|stats| stats.samples.median())
}

/// The index and median of the timed candidate with the lowest median.
fn fastest(timed: &[CandidateStats]) -> Option<(usize, Duration)> {
    timed
        .iter()
        .map(|stats| (stats.index, stats.samples.median()))
        .min_by_key(|&(_, median)| median)
}

fn values_of<C: Copy>(dimensions: &DimensionFamilies<C>, candidate: C) -> Vec<usize> {
    (0..dimensions.dimensions.len())
        .map(|dimension| (dimensions.value_of)(candidate, dimension))
        .collect()
}

fn value_ids<C>(dimensions: &DimensionFamilies<C>, values: &[usize]) -> Vec<FamilyId> {
    dimensions
        .dimensions
        .iter()
        .zip(values)
        .map(|(dimension, &value)| dimension.values[value].id)
        .collect()
}

fn every_family<C>(families: &Families<C>) -> Vec<FamilyId> {
    match families {
        Families::Flat(flat) => flat.iter().map(|family| family.id).collect(),
        Families::Dimensions(dimensions) => dimensions
            .dimensions
            .iter()
            .flat_map(|dimension| dimension.values.iter().map(|value| value.id))
            .collect(),
    }
}

fn dimension_names<C>(families: &Families<C>) -> Vec<&'static str> {
    match families {
        Families::Flat(_) => Vec::new(),
        Families::Dimensions(dimensions) => dimensions
            .dimensions
            .iter()
            .map(|dimension| dimension.name)
            .collect(),
    }
}

/// The families of `candidate`, one on a flat site and one per dimension otherwise.
fn families_of<C: SiteCandidate>(families: &Families<C>, candidate: C) -> Vec<FamilyId> {
    match families {
        Families::Flat(flat) => flat
            .iter()
            .filter(|family| family.members.contains(&candidate))
            .map(|family| family.id)
            .collect(),
        Families::Dimensions(dimensions) => {
            let values = values_of(dimensions, candidate);
            value_ids(dimensions, &values)
        },
    }
}

fn searched_candidates<C: SiteCandidate>(
    allowed: &[C],
    families: &Families<C>,
    first: &FirstBatch,
    steps: &[Option<SearchStep>],
) -> Vec<SearchedCandidate> {
    allowed
        .iter()
        .enumerate()
        .map(|(index, &candidate)| {
            let represents = first
                .allowed_representatives
                .iter()
                .find(|&&(_, representative)| representative == index)
                .map(|&(family, _)| family);

            SearchedCandidate {
                name: candidate.name(),
                families: families_of(families, candidate),
                represents,
                step: steps[index],
            }
        })
        .collect()
}
