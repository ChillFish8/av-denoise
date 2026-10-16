use std::collections::BTreeMap;
use std::time::Duration;

use av_denoise_core::tune::{CandidateRecord, CensusEntry, FidelityClass, SiteKey, SiteRanking, TuneResult};

use crate::tune::driver::{census_measured, pending_keys};

fn key(site: &'static str) -> SiteKey {
    SiteKey::builder(site, 64, 48).build()
}

fn entry(total: Duration) -> CensusEntry {
    CensusEntry {
        total,
        launches: 1,
        candidates: 3,
    }
}

#[test]
fn an_empty_census_is_not_measured() {
    let census = BTreeMap::new();

    assert!(!census_measured(&census));
}

#[test]
fn a_census_of_zero_times_is_not_measured() {
    let first_key = key("first");
    let second_key = key("second");
    let first_entry = entry(Duration::ZERO);
    let second_entry = entry(Duration::ZERO);
    let census = BTreeMap::from([(first_key, first_entry), (second_key, second_entry)]);

    assert!(!census_measured(&census));
}

#[test]
fn one_timed_launch_makes_a_census_measured() {
    let first_key = key("first");
    let second_key = key("second");
    let first_entry = entry(Duration::ZERO);
    let second_entry = entry(Duration::from_micros(40));
    let census = BTreeMap::from([(first_key, first_entry), (second_key, second_entry)]);

    assert!(census_measured(&census));
}

fn baseline_ranking(partial: bool) -> SiteRanking {
    let record = CandidateRecord {
        name: "baseline".to_string(),
        fidelity: FidelityClass::Exact,
        baseline: true,
        median_ns: 1000,
        samples: 20,
    };

    SiteRanking {
        candidates: vec![record],
        partial,
    }
}

#[test]
fn partial_keys_are_tuned_again_after_every_unranked_key() {
    let unranked_key = key("unranked");
    let partial_key = key("partial");
    let ranked_key = key("ranked");
    let unranked_entry = entry(Duration::from_micros(10));
    let partial_entry = entry(Duration::from_micros(500));
    let ranked_entry = entry(Duration::from_micros(900));
    let census = BTreeMap::from([
        (unranked_key.clone(), unranked_entry),
        (partial_key.clone(), partial_entry),
        (ranked_key.clone(), ranked_entry),
    ]);

    let partial_ranking = baseline_ranking(true);
    let full_ranking = baseline_ranking(false);
    let mut known = TuneResult::default();
    known.sites.insert(partial_key.clone(), partial_ranking);
    known.sites.insert(ranked_key, full_ranking);
    let frame_time = Duration::from_micros(1410);

    let order = pending_keys(census, &known, frame_time, true);
    let keys: Vec<SiteKey> = order.into_iter().map(|value| value.key).collect();

    assert_eq!(keys, vec![unranked_key, partial_key]);
}
