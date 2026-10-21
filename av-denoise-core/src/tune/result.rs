use std::collections::{BTreeMap, BTreeSet};

use super::consts::{HYSTERESIS, PROFILE_MARGIN};
use super::fidelity::FidelityClass;
use super::key::SiteKey;
use super::plan::{KernelPlan, PipelineProfile};
use super::sites::candidate_known;

/// Every ranking a tuning pass produced, per site key and per profile key.
///
/// Storing the whole ranking rather than the winner lets a later run pick under different rules,
/// such as `--deterministic`, without tuning again.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TuneResult {
    pub sites: BTreeMap<SiteKey, SiteRanking>,
    /// Keyed by the host's profile key, which names an algorithm, plane geometry and parameters.
    pub profiles: BTreeMap<String, ProfileRanking>,
    /// Site keys whose tuning failed. They hold no ranking and run their baseline.
    pub failed: BTreeSet<SiteKey>,
}

/// The candidates that passed their output check at one site key.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteRanking {
    pub candidates: Vec<CandidateRecord>,
    /// Whether the compile ceiling skipped some of the key's candidates, so a later run tunes it again.
    pub partial: bool,
    /// How far the compile ceiling let tuning get through the key's candidates.
    pub progress: CompileProgress,
    /// The exact compile identity of the site the candidates were compiled for.
    ///
    /// It is empty for a ranking stored before identities were kept.
    pub identity: String,
}

/// How many of a key's candidates the compile ceiling skipped, out of how many tuning tried.
///
/// Each candidate counts once. Both are zero for a key stored before these counts were kept.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompileProgress {
    pub left: usize,
    pub total: usize,
}

/// One candidate's timing at one site key.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRecord {
    pub name: String,
    pub fidelity: FidelityClass,
    pub baseline: bool,
    pub median_ns: u64,
    pub samples: u32,
}

impl CandidateRecord {
    /// Lower is more faithful. The baseline comes first, then exact, exact integer, rounding and
    /// approximate.
    pub fn faithfulness(&self) -> u8 {
        match (self.baseline, self.fidelity) {
            (true, _) => 0,
            (false, FidelityClass::Exact) => 1,
            (false, FidelityClass::ExactInteger) => 2,
            (false, FidelityClass::Rounding) => 3,
            (false, FidelityClass::Approximate) => 4,
        }
    }
}

/// Whole-frame timings for each profile at one profile key.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileRanking {
    /// Whether every site this configuration reached has a ranking and the profiles were compared.
    pub complete: bool,
    pub profiles: Vec<ProfileRecord>,
    /// The site keys worth tuning that this configuration reached, which stay settled while it is
    /// complete.
    pub sites: Vec<SiteKey>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileRecord {
    pub profile: PipelineProfile,
    pub median_frame_ns: u64,
    pub blocks: u32,
}

/// Whether `older` stays when merging `newer`.
///
/// It stays when `newer` is partial and either `older` is full or `older` timed more candidates.
fn keeps_over(older: &SiteRanking, newer: &SiteRanking) -> bool {
    if !newer.partial {
        return false;
    }

    let timed_fewer = newer.candidates.len() < older.candidates.len();
    !older.partial || timed_fewer
}

/// Picks the candidate to run from a site's records.
///
/// The lowest median wins, unless a more faithful candidate is within
/// [HYSTERESIS](crate::tune::HYSTERESIS) of that median. Then the most faithful such candidate
/// wins, and the faster one between equally faithful candidates. With `deterministic` set only
/// exact and exact integer candidates are considered.
pub fn choose(records: &[CandidateRecord], deterministic: bool) -> Option<&CandidateRecord> {
    let eligible: Vec<&CandidateRecord> = records
        .iter()
        .filter(|record| !deterministic || record.fidelity.deterministic())
        .collect();

    let fastest = eligible.iter().map(|record| record.median_ns).min()?;
    let threshold = fastest as f64 * (1.0 + HYSTERESIS);

    eligible
        .into_iter()
        .filter(|record| record.median_ns as f64 <= threshold)
        .min_by_key(|record| (record.faithfulness(), record.median_ns))
}

/// Whether a profile's frame time beats the baseline profile's by more than the profile margin.
pub fn beats_baseline_profile(baseline_frame_ns: u64, profile_frame_ns: u64) -> bool {
    let threshold = baseline_frame_ns as f64 * (1.0 - PROFILE_MARGIN);
    (profile_frame_ns as f64) < threshold
}

/// Picks the profile to run from a profile key's records.
///
/// A profile replaces the baseline profile only when it is more than
/// [PROFILE_MARGIN](crate::tune::PROFILE_MARGIN) faster. `deterministic` always picks the baseline
/// profile.
pub fn choose_profile(ranking: &ProfileRanking, deterministic: bool) -> PipelineProfile {
    if deterministic {
        return PipelineProfile::Baseline;
    }

    let baseline = ranking
        .profiles
        .iter()
        .find(|record| record.profile == PipelineProfile::Baseline);
    let Some(baseline) = baseline else {
        return PipelineProfile::Baseline;
    };

    let winner = ranking
        .profiles
        .iter()
        .filter(|record| record.profile != PipelineProfile::Baseline)
        .filter(|record| beats_baseline_profile(baseline.median_frame_ns, record.median_frame_ns))
        .min_by_key(|record| record.median_frame_ns);

    winner.map_or(PipelineProfile::Baseline, |record| record.profile)
}

impl TuneResult {
    /// Takes every key `newer` holds, keeping the rest of this result.
    ///
    /// A key `newer` ranks is no longer failed, and a key `newer` failed loses its old ranking. A
    /// partial ranking in `newer` never replaces a full one, and replaces an older partial one only
    /// when it timed at least as many candidates.
    pub fn merge(&mut self, newer: TuneResult) {
        for key in newer.sites.keys() {
            self.failed.remove(key);
        }

        for key in &newer.failed {
            self.sites.remove(key);
        }

        for (key, ranking) in newer.sites {
            let older = self.sites.get(&key);
            let keeps_older = older.is_some_and(|older| keeps_over(older, &ranking));
            if keeps_older {
                continue;
            }

            self.sites.insert(key, ranking);
        }

        self.failed.extend(newer.failed);
        self.profiles.extend(newer.profiles);
    }

    /// Whether `key` was fully ranked or its tuning failed, so it needs no more tuning.
    pub fn is_settled(&self, key: &SiteKey) -> bool {
        let fully_ranked = self.sites.get(key).is_some_and(|ranking| !ranking.partial);
        fully_ranked || self.failed.contains(key)
    }

    /// Whether `key` holds a partial ranking, which runs its choice until it is tuned again.
    pub fn is_partial(&self, key: &SiteKey) -> bool {
        self.sites.get(key).is_some_and(|ranking| ranking.partial)
    }

    /// Drops every site key whose site or any candidate name `known` rejects.
    ///
    /// `known` takes a site id and a candidate name. Returns how many keys were dropped, so they
    /// are tuned again.
    pub fn retain_known_with(&mut self, known: impl Fn(&str, &str) -> bool) -> usize {
        let before = self.sites.len();

        self.sites.retain(|key, ranking| {
            let site = key.site();
            ranking.candidates.iter().all(|record| known(site, &record.name))
        });

        before - self.sites.len()
    }

    /// Drops every site key whose site or candidate this build does not know, so it is tuned again.
    pub fn retain_known(&mut self) -> usize {
        self.retain_known_with(candidate_known)
    }

    /// The plan this result implies, one choice per site key.
    pub fn kernel_plan(&self, deterministic: bool) -> KernelPlan {
        let mut plan = KernelPlan::baseline();

        for (key, ranking) in &self.sites {
            let chosen = choose(&ranking.candidates, deterministic);
            if let Some(record) = chosen {
                plan.insert(key.clone(), &record.name);
            }
        }

        plan
    }

    /// The profile to build for `profile_key`, the baseline profile when it was never compared.
    pub fn profile_for(&self, profile_key: &str, deterministic: bool) -> PipelineProfile {
        match self.profiles.get(profile_key) {
            Some(ranking) => choose_profile(ranking, deterministic),
            None => PipelineProfile::Baseline,
        }
    }

    /// Whether every profile key in `profile_keys` was tuned to completion.
    ///
    /// A profile key whose sites include one that is no longer settled, such as a key another
    /// configuration left partial, is not complete.
    pub fn is_complete(&self, profile_keys: &[String]) -> bool {
        profile_keys.iter().all(|profile_key| {
            let ranking = self.profiles.get(profile_key);
            ranking.is_some_and(|ranking| ranking.complete && self.all_settled(&ranking.sites))
        })
    }

    fn all_settled(&self, keys: &[SiteKey]) -> bool {
        keys.iter().all(|key| self.is_settled(key))
    }
}
