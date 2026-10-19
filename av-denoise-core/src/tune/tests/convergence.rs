use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::probe::{ProbeCandidate, ProbeSite};
use super::runtime::test_client;
use crate::tune::precompile::CompileAccount;
use crate::tune::session::compile_members;
use crate::tune::{
    CandidateRecord,
    CensusEntry,
    CompileProgress,
    FidelityClass,
    SiteCandidate,
    SiteKey,
    SiteRanking,
    TuneLedger,
    TuneResult,
    VariantSite,
    pending_keys,
};

/// Each pass may compile for this long, which covers two cold compiles and starts a third.
const CEILING: Duration = Duration::from_millis(700);

/// How long a candidate the cache lacks takes to compile.
const COLD_COMPILE: Duration = Duration::from_millis(400);

/// More passes than three keys of three cold candidates each could need at two a pass.
const MAX_PASSES: usize = 10;

/// One key's site, its share of frame time and its compile cache, which outlives every pass.
struct Key {
    width: u32,
    per_frame: Duration,
    cold_delay: Duration,
    /// How long loading a candidate the cache holds takes.
    load_delay: Duration,
    cache: Arc<Mutex<Vec<ProbeCandidate>>>,
}

impl Key {
    /// A key whose baseline the census already compiled, with instant cache loads.
    fn new(width: u32, per_frame_micros: u64) -> Self {
        let baseline = vec![ProbeCandidate::Small];

        Self {
            width,
            per_frame: Duration::from_micros(per_frame_micros),
            cold_delay: COLD_COMPILE,
            load_delay: Duration::ZERO,
            cache: Arc::new(Mutex::new(baseline)),
        }
    }

    fn site(&self) -> ProbeSite {
        let client = test_client();
        let mut site = ProbeSite::new(&client);
        site.width = self.width;
        site.cache = Some(self.cache.clone());
        site.cold_delay = self.cold_delay;
        site.launch_delay = self.load_delay;
        site
    }
}

/// The ranking a pass that timed the candidates in `kept` would store.
fn ranking_of(kept: &[usize], progress: CompileProgress, identity: String) -> SiteRanking {
    let candidates = kept
        .iter()
        .map(|&index| {
            let candidate = ProbeCandidate::ALL[index];

            CandidateRecord {
                name: candidate.name().to_string(),
                fidelity: FidelityClass::Exact,
                baseline: index == 0,
                median_ns: 1000 + index as u64,
                samples: 20,
            }
        })
        .collect();

    SiteRanking {
        candidates,
        partial: progress.left > 0,
        progress,
        identity,
    }
}

/// Compiles one key's candidates under `ledger` and records the ranking they would give.
fn tune_key(ledger: &TuneLedger, key: &Key) {
    let site = key.site();
    let site_key = site.key();
    let allowed = ProbeCandidate::ALL;
    let indices: Vec<usize> = (0..allowed.len()).collect();
    let account = CompileAccount {
        ledger,
        key: &site_key,
    };

    let precompiled = compile_members(&site, allowed, &indices, &(), account).expect("the stage compiles");
    let progress = CompileProgress {
        left: precompiled.skipped.len(),
        total: indices.len(),
    };
    let identity = site.compile_identity();
    let ranking = ranking_of(&precompiled.kept, progress, identity);
    ledger.record_ranking(site_key, ranking);
}

/// What one pass did.
struct Pass {
    tuned: Vec<SiteKey>,
    /// Keys whose new ranking timed fewer candidates than the one stored before the pass.
    shrunk: Vec<SiteKey>,
}

/// Runs one short pass over the unsettled keys in order. It stops once the compile ceiling is spent.
fn run_pass(stored: &mut TuneResult, keys: &[Key]) -> Pass {
    let ledger = TuneLedger::new(false);
    ledger.begin_compile_pass(CEILING);
    ledger.seed_compiled(stored);

    let census: BTreeMap<SiteKey, CensusEntry> = keys
        .iter()
        .map(|key| {
            let entry = CensusEntry {
                total: key.per_frame,
                launches: 1,
                candidates: ProbeCandidate::ALL.len(),
            };
            (key.site().key(), entry)
        })
        .collect();
    let frame_time: Duration = keys.iter().map(|key| key.per_frame).sum();
    let order = pending_keys(&census, stored, frame_time, true);
    let mut tuned = Vec::new();

    for value in order {
        if ledger.compile_left().is_zero() {
            break;
        }

        let key = keys
            .iter()
            .find(|key| key.site().key() == value.key)
            .expect("every census key has a site");
        tune_key(&ledger, key);
        tuned.push(value.key);
    }

    let ranked = ledger.result();
    let shrunk = ranked
        .sites
        .iter()
        .filter(|(key, ranking)| {
            let before = stored.sites.get(*key).map_or(0, |older| older.candidates.len());
            ranking.candidates.len() < before
        })
        .map(|(key, _)| key.clone())
        .collect();
    stored.merge(ranked);

    Pass { tuned, shrunk }
}

/// Runs passes until one tunes nothing, and returns how many ran and how often each key was tuned.
fn run_until_idle(stored: &mut TuneResult, keys: &[Key]) -> (usize, BTreeMap<SiteKey, usize>) {
    let mut tunes: BTreeMap<SiteKey, usize> = BTreeMap::new();
    let mut passes = 0;

    while passes < MAX_PASSES {
        let pass = run_pass(stored, keys);
        passes += 1;

        assert!(
            pass.shrunk.is_empty(),
            "pass {passes} lost candidates at {:?}",
            pass.shrunk
        );

        if pass.tuned.is_empty() {
            break;
        }

        for key in pass.tuned {
            *tunes.entry(key).or_default() += 1;
        }
    }

    (passes, tunes)
}

fn unsettled(stored: &TuneResult, keys: &[Key]) -> Vec<SiteKey> {
    keys.iter()
        .map(|key| key.site().key())
        .filter(|key| !stored.is_settled(key))
        .collect()
}

#[test]
fn short_passes_settle_every_key_and_retry_none_forever() {
    let keys = [Key::new(64, 900), Key::new(128, 500), Key::new(256, 100)];
    let mut stored = TuneResult::default();

    let (passes, tunes) = run_until_idle(&mut stored, &keys);
    let left = unsettled(&stored, &keys);

    assert!(left.is_empty(), "{left:?} after {passes} passes");
    assert!(passes < MAX_PASSES, "the last pass still tuned keys");
    assert!(
        tunes.values().all(|&count| count <= 2),
        "a key was tuned more than once after its first partial ranking: {tunes:?}"
    );
}

/// The first retry finishes with about 100 ms of the ceiling left, so the second retry passes the
/// ceiling while it loads candidates the earlier run compiled. Without seeding it would skip them
/// and lose them from its ranking.
#[test]
fn a_retry_past_the_ceiling_keeps_every_candidate_an_earlier_run_compiled() {
    let compiled = vec![
        ProbeCandidate::Small,
        ProbeCandidate::Medium,
        ProbeCandidate::Wide,
    ];
    let load_delay = Duration::from_millis(150);
    let mut first = Key::new(64, 900);
    first.cold_delay = Duration::from_millis(150);
    first.load_delay = load_delay;
    first.cache = Arc::new(Mutex::new(compiled.clone()));
    let mut second = Key::new(128, 500);
    second.load_delay = load_delay;
    second.cache = Arc::new(Mutex::new(compiled));
    let keys = [first, second];

    let mut stored = TuneResult::default();
    for key in &keys {
        let site = key.site();
        let progress = CompileProgress { left: 1, total: 4 };
        let identity = site.compile_identity();
        let ranking = ranking_of(&[0, 1, 2], progress, identity);
        stored.sites.insert(site.key(), ranking);
    }

    let (passes, _) = run_until_idle(&mut stored, &keys);
    let left = unsettled(&stored, &keys);

    assert!(left.is_empty(), "{left:?} after {passes} passes");
}
