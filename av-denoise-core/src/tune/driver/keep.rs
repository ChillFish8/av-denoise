use std::path::Path;
use std::time::Instant;

use super::pass::{PassInputs, PassSummary};
use super::profiles::ProfilePhase;
use super::{TuneOutcome, TuneReport};
use crate::tune::device::DeviceIdentity;
use crate::tune::key::SiteKey;
use crate::tune::ledger::TuneLedger;
use crate::tune::plan::PipelineProfile;
use crate::tune::priority::PriorityTable;
use crate::tune::report::format_report;
use crate::tune::result::{ProfileRanking, ProfileRecord, TuneResult};
use crate::tune::store::{TuneStore, remember};

/// The printed summary of an exhaustive pass, from the rankings it recorded and its profile timings.
pub(super) fn summarise(
    identity: &DeviceIdentity,
    ledger: &TuneLedger,
    summary: &PassSummary,
    table: &PriorityTable,
) -> String {
    let ranked = ledger.result();
    let profiles = match &summary.profiles {
        ProfilePhase::Compared { records, .. } => records.as_slice(),
        _ => &[],
    };

    format_report(identity, &ranked, profiles, table)
}

/// Merges this pass's rankings into the stored result, then remembers and saves it.
///
/// Each profile key is complete when every key worth at least 2% of frame time is settled and the
/// profiles were compared or do not apply. A key with a partial ranking is not settled. Each profile
/// key keeps those keys, so it stops being complete if any of them is later left partial.
pub(super) fn keep_result(
    inputs: &PassInputs<'_>,
    profile_keys: &[String],
    summary: PassSummary,
    started: Instant,
) -> TuneReport {
    let store = inputs.config.store.as_deref();
    let identity = inputs.identity;
    let stored = inputs.stored;
    let ledger = inputs.ledger;

    let mut merged = stored.clone();
    let ranked = ledger.result();
    let compile = ledger.compile_time();
    let tuned_keys = ranked.sites.values().filter(|ranking| !ranking.partial).count();
    let ranked_keys: Vec<SiteKey> = ranked.sites.keys().cloned().collect();
    merged.merge(ranked);

    let partial_keys = ranked_keys.iter().filter(|key| merged.is_partial(key)).count();

    let keys_done = summary.significant.iter().all(|key| merged.is_settled(key));
    let profiles_done = summary.profiles != ProfilePhase::Unfinished;
    let complete = summary.census_measured && keys_done && profiles_done;

    let baseline_only = vec![ProfileRecord {
        profile: PipelineProfile::Baseline,
        median_frame_ns: summary.frame_time.as_nanos() as u64,
        blocks: 0,
    }];
    let mut sites = summary.significant;
    let compared = match summary.profiles {
        ProfilePhase::Compared { records, significant } => {
            sites.extend(significant);
            Some(records)
        },
        _ => None,
    };

    sites.sort();
    sites.dedup();

    for profile_key in profile_keys {
        let previous = stored.profiles.get(profile_key);
        let kept_records = match (&compared, previous) {
            (Some(records), _) => records.clone(),
            (None, Some(previous)) if previous.profiles.len() > 1 => previous.profiles.clone(),
            (None, _) => baseline_only.clone(),
        };
        let ranking = ProfileRanking {
            complete,
            profiles: kept_records,
            sites: sites.clone(),
        };
        merged.profiles.insert(profile_key.clone(), ranking);
    }

    remember(store, identity, &merged);
    let saved = save_result(store, identity, &merged);

    TuneReport {
        outcome: TuneOutcome::Tuned,
        tuned_keys,
        partial_keys,
        compile,
        elapsed: started.elapsed(),
        saved,
        summary: None,
    }
}

/// Writes `result` to the `store` directory, returning whether it landed.
fn save_result(store: Option<&Path>, identity: &DeviceIdentity, result: &TuneResult) -> bool {
    let Some(dir) = store else {
        tracing::info!("there is no tuning store, so the tuning result lasts for this run only");
        return false;
    };

    let store = TuneStore::at(dir);
    let written = store.save(identity, result);
    match written {
        Ok(path) => {
            tracing::info!(?path, "saved the tuning result");
            true
        },
        Err(error) => {
            tracing::warn!(%error, "cannot save the tuning result, this run still uses it");
            false
        },
    }
}
