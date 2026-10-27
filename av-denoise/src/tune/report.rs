use std::collections::BTreeSet;
use std::fmt::Write;
use std::time::Duration;

use av_denoise_core::tune::{
    BASELINE_FAMILY,
    BASELINE_INDEX,
    CandidateRecord,
    DeviceIdentity,
    FamilyId,
    HYSTERESIS,
    PipelineProfile,
    PriorityTable,
    ProfileRanking,
    ProfileRecord,
    SearchMode,
    SiteSearch,
    TuneResult,
    builds_up,
    choose,
    choose_profile,
    family_order,
    opens_family,
    profile_wins,
    skipped,
};

/// One candidate or profile with its median.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Pick {
    pub(crate) name: String,
    pub(crate) median: Duration,
}

/// The candidate [choose] keeps from a set of records, with the fastest of them.
///
/// They differ when a more faithful candidate is within [HYSTERESIS] of the fastest.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Choice {
    pub(crate) pick: Pick,
    pub(crate) fastest: Pick,
    /// Whether the pick is the site's baseline.
    pub(crate) baseline: bool,
}

/// A family's representative against its fastest member.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FamilyGap {
    pub(crate) family: FamilyId,
    pub(crate) representative: Pick,
    pub(crate) best: Pick,
}

impl FamilyGap {
    /// How much slower the representative is than the family's best member, as a fraction.
    pub(crate) fn gap(&self) -> f64 {
        slower_by(self.representative.median, self.best.median)
    }
}

/// What the families search would have picked at one site key, replayed from exhaustive timings.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SiteReplay {
    /// The choice from every timed candidate.
    pub(crate) full: Option<Choice>,
    /// The choice from the candidates the families search would have timed.
    pub(crate) pruned: Option<Choice>,
    /// How many candidates the families search would have compiled.
    pub(crate) pruned_tried: usize,
    pub(crate) allowed: usize,
    /// Each family with a timed representative.
    pub(crate) families: Vec<FamilyGap>,
    /// The families in the device table's order.
    pub(crate) table_order: Vec<FamilyId>,
    /// The same families, ordered by their representative's median.
    pub(crate) measured_order: Vec<FamilyId>,
}

impl SiteReplay {
    pub(crate) fn matches(&self) -> bool {
        let pruned = self.pruned.as_ref().map(|choice| &choice.pick.name);
        let full = self.full.as_ref().map(|choice| &choice.pick.name);

        pruned == full
    }

    /// How much slower the pruned pick is than the full pick, as a fraction.
    pub(crate) fn slowdown(&self) -> f64 {
        match (&self.pruned, &self.full) {
            (Some(pruned), Some(full)) => slower_by(pruned.pick.median, full.pick.median),
            _ => 0.0,
        }
    }
}

/// What the profile early stop would have picked, replayed from timing every profile.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProfileReplay {
    /// The pick from every timed profile.
    pub(crate) full: Pick,
    /// The pick from the profiles the early stop would have timed.
    pub(crate) pruned: Pick,
    /// The timed alternatives in the device table's order.
    pub(crate) table_order: Vec<PipelineProfile>,
    /// The same alternatives, fastest first.
    pub(crate) measured_order: Vec<PipelineProfile>,
    /// The alternatives the early stop would have timed.
    pub(crate) timed: Vec<PipelineProfile>,
}

impl ProfileReplay {
    pub(crate) fn matches(&self) -> bool {
        self.pruned.name == self.full.name
    }
}

/// Replays the families search at one key of `site` over the timings of an exhaustive search.
///
/// It applies the rules the tuner applies. The representatives are timed first, leaving out the
/// families `table` or the search skipped. A flat site then opens every family within
/// [opens_family] of the best representative. A dimension site builds up from the fastest
/// representative with [builds_up], one round per dimension at most. A candidate counts as timed
/// only when `timings` holds it, so one that failed to build or to compare never opens a family or
/// leads a round.
pub(crate) fn replay(
    site: &str,
    search: &SiteSearch,
    timings: &[CandidateRecord],
    table: &PriorityTable,
) -> SiteReplay {
    let medians: Vec<Option<Duration>> = search
        .candidates
        .iter()
        .map(|candidate| median_named(timings, candidate.name))
        .collect();
    let mut left_out: Vec<FamilyId> = skipped(table, site).to_vec();
    left_out.extend(&search.skipped);

    let first = first_batch(site, search, table, &left_out);
    let tried = if search.dimensions.is_empty() {
        replay_flat(search, &medians, &first)
    } else {
        replay_build_up(search, &medians, &first, &left_out)
    };

    let tried_names: BTreeSet<&str> = tried.iter().map(|&index| search.candidates[index].name).collect();
    let pruned_records: Vec<CandidateRecord> = timings
        .iter()
        .filter(|record| tried_names.contains(record.name.as_str()))
        .cloned()
        .collect();

    let pruned = choice_of(&pruned_records);
    let full = choice_of(timings);

    let families = family_gaps(search, &medians);
    let ids: Vec<FamilyId> = families.iter().map(|gap| gap.family).collect();
    let table_order = family_order(table, site, &ids);
    let mut by_representative = families.clone();
    by_representative.sort_by_key(|gap| gap.representative.median);
    let measured_order = by_representative.iter().map(|gap| gap.family).collect();

    SiteReplay {
        full,
        pruned,
        pruned_tried: tried.len(),
        allowed: search.candidates.len(),
        families,
        table_order,
        measured_order,
    }
}

/// Replays the profile early stop over `records`, which time every profile.
///
/// The alternatives are taken in `table`'s order. The replay stops after the first one that does
/// not beat the best so far, the baseline included, by the profile margin. Returns `None` without
/// a baseline record or any alternative.
pub(crate) fn replay_profiles(records: &[ProfileRecord], table: &PriorityTable) -> Option<ProfileReplay> {
    let baseline = records
        .iter()
        .find(|record| record.profile == PipelineProfile::Baseline)?;
    let baseline_median = Duration::from_nanos(baseline.median_frame_ns);

    let alternatives: Vec<&ProfileRecord> = table
        .profiles
        .iter()
        .filter(|&&profile| profile != PipelineProfile::Baseline)
        .filter_map(|&profile| profile_record(records, profile))
        .collect();
    if alternatives.is_empty() {
        return None;
    }

    let table_order: Vec<PipelineProfile> = alternatives.iter().map(|record| record.profile).collect();
    let mut timed = Vec::new();
    let mut best: Option<Duration> = None;

    for record in &alternatives {
        let profile = record.profile;
        let median = Duration::from_nanos(record.median_frame_ns);
        let best_so_far = best.map_or(baseline_median, |best| best.min(baseline_median));
        let wins = profile_wins(best_so_far, median);
        let lowest = best.map_or(median, |best| best.min(median));
        best = Some(lowest);
        timed.push(profile);

        if !wins {
            break;
        }
    }

    let mut pruned_records = vec![*baseline];
    let timed_records = timed
        .iter()
        .filter_map(|&profile| profile_record(records, profile));
    pruned_records.extend(timed_records);

    let full = profile_pick(records, records);
    let pruned = profile_pick(&pruned_records, records);

    let mut by_median = alternatives.clone();
    by_median.sort_by_key(|record| record.median_frame_ns);
    let measured_order = by_median.iter().map(|record| record.profile).collect();

    Some(ProfileReplay {
        full,
        pruned,
        table_order,
        measured_order,
        timed,
    })
}

/// The printed summary of an exhaustive tune on the device `identity` names.
///
/// It has a header naming the device, one block per site key `ranked` searched exhaustively, then
/// the profiles in `profiles`, then a line of totals.
pub(crate) fn format_report(
    identity: &DeviceIdentity,
    ranked: &TuneResult,
    profiles: &[ProfileRecord],
    table: &PriorityTable,
) -> String {
    let mut text = String::new();
    let mut replays = Vec::new();

    let _ = writeln!(
        text,
        "Exhaustive tune against the families search, {} on {} ({})",
        identity.adapter, identity.backend, identity.driver
    );

    for (key, ranking) in &ranked.sites {
        let Some(search) = &ranking.search else {
            continue;
        };

        if search.mode != SearchMode::Exhaustive {
            continue;
        }

        let replayed = replay(key.site(), search, &ranking.candidates, table);
        let block = format_site(key.as_str(), &replayed);
        text.push_str(&block);
        replays.push((key.as_str(), replayed));
    }

    let profile_replay = replay_profiles(profiles, table);
    let profile_block = match &profile_replay {
        Some(replayed) => format_profiles(replayed),
        None => "\nno pipeline profiles to compare\n".to_string(),
    };
    text.push_str(&profile_block);

    let totals = format_totals(&replays, profile_replay.as_ref());
    text.push_str(&totals);

    text
}

fn format_site(key: &str, replayed: &SiteReplay) -> String {
    let mut block = String::new();
    let _ = writeln!(block, "\n{key}");

    let full = choice_text(replayed.full.as_ref());
    let _ = writeln!(block, "  full    {full}");

    let pruned = choice_text(replayed.pruned.as_ref());
    let verdict = verdict_text(replayed.matches(), replayed.slowdown());
    let tried = replayed.pruned_tried;
    let allowed = replayed.allowed;
    let _ = writeln!(
        block,
        "  pruned  {pruned}  {verdict}, compiles {tried} of {allowed}"
    );

    for gap in &replayed.families {
        let representative = &gap.representative;
        let best = &gap.best;
        let representative_millis = millis(representative.median);
        let best_millis = millis(best.median);
        let percent = gap.gap() * 100.0;
        let family = gap.family.0;
        let _ = write!(
            block,
            "  family {family}  rep {} {representative_millis:.3} ms",
            representative.name
        );
        let _ = writeln!(
            block,
            "  best {} {best_millis:.3} ms  gap {percent:.1}%",
            best.name
        );
    }

    let table_order = family_names(&replayed.table_order);
    let measured_order = family_names(&replayed.measured_order);
    let order_verdict = order_text(replayed.table_order == replayed.measured_order);
    let _ = writeln!(block, "  table order     {table_order}");
    let _ = writeln!(block, "  measured order  {measured_order}  {order_verdict}");

    block
}

fn format_profiles(replayed: &ProfileReplay) -> String {
    let mut block = String::new();
    let _ = writeln!(block, "\npipeline profiles");

    let full = &replayed.full;
    let full_millis = millis(full.median);
    let _ = writeln!(block, "  full    {}  {full_millis:.2} ms", full.name);

    let pruned = &replayed.pruned;
    let pruned_millis = millis(pruned.median);
    let slowdown = slower_by(pruned.median, full.median);
    let verdict = verdict_text(replayed.matches(), slowdown);
    let timed = replayed.timed.len();
    let compared = replayed.table_order.len();
    let _ = writeln!(
        block,
        "  pruned  {}  {pruned_millis:.2} ms  {verdict}, times {timed} of {compared}",
        pruned.name
    );

    let table_order = profile_names(&replayed.table_order);
    let measured_order = profile_names(&replayed.measured_order);
    let order_verdict = order_text(replayed.table_order == replayed.measured_order);
    let _ = writeln!(block, "  table order     {table_order}");
    let _ = writeln!(block, "  measured order  {measured_order}  {order_verdict}");

    block
}

/// How many keys matched, the worst slowdown and the largest representative gap.
fn format_totals(replays: &[(&str, SiteReplay)], profiles: Option<&ProfileReplay>) -> String {
    let matched = replays.iter().filter(|(_, replayed)| replayed.matches()).count();
    let worst = replays
        .iter()
        .filter(|(_, replayed)| !replayed.matches())
        .map(|(key, replayed)| (*key, replayed.slowdown()))
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let widest = replays
        .iter()
        .flat_map(|(key, replayed)| replayed.families.iter().map(move |gap| (*key, gap)))
        .max_by(|left, right| left.1.gap().total_cmp(&right.1.gap()));

    let keys = replays.len();
    let mut totals = String::new();
    let _ = writeln!(totals, "\nThe pruned search matched {matched} of {keys} keys.");

    match worst {
        Some((key, slowdown)) => {
            let percent = slowdown * 100.0;
            let _ = writeln!(totals, "The worst slowdown is {percent:+.1}% at {key}.");
        },
        None => {
            let _ = writeln!(totals, "Every pruned pick is the full pick.");
        },
    }

    if let Some((key, gap)) = widest {
        let percent = gap.gap() * 100.0;
        let _ = writeln!(
            totals,
            "The largest representative gap is {percent:.1}% for {} at {key}.",
            gap.family.0
        );
    }

    if let Some(replayed) = profiles {
        let profile_verdict = if replayed.matches() {
            "matched"
        } else {
            "did not match"
        };
        let _ = writeln!(totals, "The profile early stop {profile_verdict}.");
    }

    totals
}

/// The representatives the families search times first, in the table's order, with their indices.
///
/// The baseline is always among them. Every other family in `left_out` is not.
fn first_batch(
    site: &str,
    search: &SiteSearch,
    table: &PriorityTable,
    left_out: &[FamilyId],
) -> Vec<(FamilyId, usize)> {
    let kept: Vec<(FamilyId, usize)> = search
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let family = candidate.represents?;
            let skip = index != BASELINE_INDEX && left_out.contains(&family);
            let entry = (family, index);
            (!skip).then_some(entry)
        })
        .collect();

    let ids: Vec<FamilyId> = kept.iter().map(|&(family, _)| family).collect();
    let order = family_order(table, site, &ids);

    order
        .iter()
        .filter_map(|family| kept.iter().find(|(kept_family, _)| kept_family == family))
        .copied()
        .collect()
}

/// The indices a flat site's families search compiles.
fn replay_flat(
    search: &SiteSearch,
    medians: &[Option<Duration>],
    first: &[(FamilyId, usize)],
) -> BTreeSet<usize> {
    let mut tried: BTreeSet<usize> = first.iter().map(|&(_, index)| index).collect();
    let timed_representatives: Vec<(FamilyId, Duration)> = first
        .iter()
        .filter_map(|&(family, index)| {
            let median = medians[index]?;
            Some((family, median))
        })
        .collect();

    let best = timed_representatives.iter().map(|&(_, median)| median).min();
    let Some(best) = best else {
        return tried;
    };

    let opened: Vec<FamilyId> = timed_representatives
        .iter()
        .filter(|&&(_, median)| opens_family(median, best))
        .map(|&(family, _)| family)
        .collect();

    for (index, candidate) in search.candidates.iter().enumerate() {
        let family = candidate.families.first();
        let in_opened = family.is_some_and(|family| opened.contains(family));
        if in_opened {
            tried.insert(index);
        }
    }

    tried
}

/// The indices a dimension site's build-up compiles.
fn replay_build_up(
    search: &SiteSearch,
    medians: &[Option<Duration>],
    first: &[(FamilyId, usize)],
    left_out: &[FamilyId],
) -> BTreeSet<usize> {
    let mut tried: BTreeSet<usize> = first.iter().map(|&(_, index)| index).collect();
    let first_indices: Vec<usize> = first.iter().map(|&(_, index)| index).collect();

    let Some(mut current) = fastest(&first_indices, medians) else {
        return tried;
    };

    let base = &search.candidates[BASELINE_INDEX].families;

    for _ in 0..search.dimensions.len() {
        let values = &search.candidates[current.0].families;
        let around: Vec<usize> = one_value_away(search, values, base, left_out)
            .into_iter()
            .filter(|index| !tried.contains(index))
            .collect();
        tried.extend(&around);

        let Some(round_best) = fastest(&around, medians) else {
            break;
        };

        if !builds_up(round_best.1, current.1) {
            break;
        }

        current = round_best;
    }

    tried
}

/// The indices of the candidates that differ from `values` in exactly one dimension.
///
/// A value in `left_out` is not moved to, unless it is the baseline's value `base`.
fn one_value_away(
    search: &SiteSearch,
    values: &[FamilyId],
    base: &[FamilyId],
    left_out: &[FamilyId],
) -> Vec<usize> {
    let mut around = Vec::new();

    for (index, candidate) in search.candidates.iter().enumerate() {
        let moved: Vec<usize> = (0..values.len())
            .filter(|&dimension| candidate.families[dimension] != values[dimension])
            .collect();
        let &[dimension] = moved.as_slice() else {
            continue;
        };

        let value = candidate.families[dimension];
        let is_base = value == base[dimension];
        if !is_base && left_out.contains(&value) {
            continue;
        }

        around.push(index);
    }

    around
}

/// The index and median of the fastest timed candidate among `indices`.
fn fastest(indices: &[usize], medians: &[Option<Duration>]) -> Option<(usize, Duration)> {
    indices
        .iter()
        .filter_map(|&index| {
            let median = medians[index]?;
            Some((index, median))
        })
        .min_by_key(|&(_, median)| median)
}

/// Each represented family's representative against its fastest member.
///
/// A dimension site's family is one dimension value, and its members are the candidates with that
/// value. The baseline's own entry is left out there, since every candidate shares a dimension
/// with it.
fn family_gaps(search: &SiteSearch, medians: &[Option<Duration>]) -> Vec<FamilyGap> {
    let flat = search.dimensions.is_empty();
    let base = &search.candidates[BASELINE_INDEX].families;
    let mut gaps = Vec::new();

    for (index, candidate) in search.candidates.iter().enumerate() {
        let Some(family) = candidate.represents else {
            continue;
        };

        if !flat && family == BASELINE_FAMILY {
            continue;
        }

        let Some(representative_median) = medians[index] else {
            continue;
        };

        let dimension = if flat {
            Some(0)
        } else {
            varied_dimension(&candidate.families, base)
        };
        let Some(dimension) = dimension else {
            continue;
        };

        let members: Vec<usize> = search
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, member)| member.families.get(dimension) == Some(&family))
            .map(|(member_index, _)| member_index)
            .collect();
        let Some((best_index, best_median)) = fastest(&members, medians) else {
            continue;
        };

        let representative = Pick {
            name: candidate.name.to_string(),
            median: representative_median,
        };
        let best = Pick {
            name: search.candidates[best_index].name.to_string(),
            median: best_median,
        };
        gaps.push(FamilyGap {
            family,
            representative,
            best,
        });
    }

    gaps
}

/// The first dimension where `values` differs from the baseline's `base`.
fn varied_dimension(values: &[FamilyId], base: &[FamilyId]) -> Option<usize> {
    (0..base.len()).find(|&dimension| values[dimension] != base[dimension])
}

fn median_named(timings: &[CandidateRecord], name: &str) -> Option<Duration> {
    let record = timings.iter().find(|record| record.name == name)?;
    let median = Duration::from_nanos(record.median_ns);
    Some(median)
}

/// What [choose] keeps from `records`, with the fastest record.
fn choice_of(records: &[CandidateRecord]) -> Option<Choice> {
    let chosen = choose(records, false)?;
    let fastest = records.iter().min_by_key(|record| record.median_ns)?;

    Some(Choice {
        pick: candidate_pick(chosen),
        fastest: candidate_pick(fastest),
        baseline: chosen.baseline,
    })
}

fn candidate_pick(record: &CandidateRecord) -> Pick {
    Pick {
        name: record.name.clone(),
        median: Duration::from_nanos(record.median_ns),
    }
}

fn profile_record(records: &[ProfileRecord], profile: PipelineProfile) -> Option<&ProfileRecord> {
    records.iter().find(|record| record.profile == profile)
}

/// The profile [choose_profile] picks from `candidates`, with its median from `records`.
fn profile_pick(candidates: &[ProfileRecord], records: &[ProfileRecord]) -> Pick {
    let ranking = ProfileRanking {
        complete: true,
        profiles: candidates.to_vec(),
        sites: Vec::new(),
    };
    let profile = choose_profile(&ranking, false);
    let median_ns = profile_record(records, profile).map_or(0, |record| record.median_frame_ns);

    Pick {
        name: profile.name().to_string(),
        median: Duration::from_nanos(median_ns),
    }
}

/// How much slower `slower` is than `faster`, as a fraction, or zero when `faster` is zero.
fn slower_by(slower: Duration, faster: Duration) -> f64 {
    let faster_seconds = faster.as_secs_f64();
    if faster_seconds == 0.0 {
        return 0.0;
    }

    slower.as_secs_f64() / faster_seconds - 1.0
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// The pick and its median, with the faster record it was kept over when there is one.
fn choice_text(choice: Option<&Choice>) -> String {
    let Some(choice) = choice else {
        return "nothing timed".to_string();
    };

    let pick = &choice.pick;
    let pick_millis = millis(pick.median);
    let mut text = format!("{}  {pick_millis:.3} ms", pick.name);

    let fastest = &choice.fastest;
    if fastest.name == pick.name {
        return text;
    }

    let fastest_millis = millis(fastest.median);
    let percent = HYSTERESIS * 100.0;
    let reason = if choice.baseline {
        "as the baseline"
    } else {
        "as more faithful"
    };
    let _ = write!(
        text,
        "  kept over {} {fastest_millis:.3} ms {reason} within {percent:.0}%",
        fastest.name
    );

    text
}

fn verdict_text(matches: bool, slowdown: f64) -> String {
    if matches {
        return "same".to_string();
    }

    let percent = slowdown * 100.0;
    format!("differs, {percent:+.1}%")
}

fn order_text(matches: bool) -> &'static str {
    if matches { "matches" } else { "differs" }
}

fn family_names(families: &[FamilyId]) -> String {
    let names: Vec<&str> = families.iter().map(|family| family.0).collect();
    names.join(", ")
}

fn profile_names(profiles: &[PipelineProfile]) -> String {
    let names: Vec<&str> = profiles.iter().map(|profile| profile.name()).collect();
    names.join(", ")
}
