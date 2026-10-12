use std::collections::BTreeMap;
use std::time::Duration;

use av_denoise_core::tune::{CensusEntry, SiteKey};

use crate::tune::driver::census_measured;

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
