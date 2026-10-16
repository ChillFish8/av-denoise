use av_denoise_core::tune::{
    CandidateRecord,
    FidelityClass,
    PipelineProfile,
    ProfileRanking,
    ProfileRecord,
    SiteKey,
    SiteRanking,
};

use crate::tune::{DeviceIdentity, TuneResult, TuneStore, remember, remembered, resolve};

fn identity() -> DeviceIdentity {
    DeviceIdentity {
        backend: "vulkan".to_string(),
        adapter: "Test Adapter".to_string(),
        vendor_id: 0x1002,
        device_id: 0x7551,
        driver: "test driver 1.0".to_string(),
        os: "linux".to_string(),
        compute_capability: None,
    }
}

fn sample_result() -> TuneResult {
    let mut result = TuneResult::default();
    let key = SiteKey::builder("nlm_window", 1920, 1080).param("pr", 2).build();
    let ranking = SiteRanking {
        candidates: vec![
            CandidateRecord {
                name: "32x16".to_string(),
                fidelity: FidelityClass::Exact,
                baseline: false,
                median_ns: 800,
                samples: 40,
            },
            CandidateRecord {
                name: "32x8".to_string(),
                fidelity: FidelityClass::Exact,
                baseline: true,
                median_ns: 1000,
                samples: 40,
            },
        ],
        partial: false,
    };
    result.sites.insert(key, ranking);

    let profiles = ProfileRanking {
        complete: true,
        profiles: vec![ProfileRecord {
            profile: PipelineProfile::Baseline,
            median_frame_ns: 30_000_000,
            blocks: 3,
        }],
    };
    result.profiles.insert("luma".to_string(), profiles);
    result
}

#[test]
fn a_saved_result_loads_back_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();

    let path = store.save(&device, &result).expect("the save succeeds");
    let loaded = store.load(&device);

    assert_eq!(path, store.path_for(&device));
    assert_eq!(loaded, result);
}

#[test]
fn failed_keys_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let mut result = sample_result();
    let failed_key = SiteKey::builder("collab", 1280, 720).param("pr", 1).build();
    result.failed.insert(failed_key);

    store.save(&device, &result).expect("the save succeeds");
    let loaded = store.load(&device);

    assert_eq!(loaded.failed, result.failed);
    assert_eq!(loaded, result);
}

#[test]
fn a_file_without_failed_keys_loads_with_none() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    let path = store.save(&device, &result).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    document.as_object_mut().unwrap().remove("failed");
    let stripped = serde_json::to_string_pretty(&document).unwrap();
    std::fs::write(&path, stripped).unwrap();

    let loaded = store.load(&device);

    assert!(loaded.failed.is_empty());
    assert_eq!(loaded, result);
}

#[test]
fn partial_rankings_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let mut result = sample_result();
    for ranking in result.sites.values_mut() {
        ranking.partial = true;
    }

    store.save(&device, &result).expect("the save succeeds");
    let loaded = store.load(&device);

    assert!(loaded.sites.values().all(|ranking| ranking.partial));
    assert_eq!(loaded, result);
}

#[test]
fn a_file_without_partial_keys_loads_every_ranking_as_full() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    let path = store.save(&device, &result).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    document.as_object_mut().unwrap().remove("partial");
    let stripped = serde_json::to_string_pretty(&document).unwrap();
    std::fs::write(&path, stripped).unwrap();

    let loaded = store.load(&device);

    assert!(loaded.sites.values().all(|ranking| !ranking.partial));
    assert_eq!(loaded, result);
}

#[test]
fn the_file_is_named_by_the_identity_hash() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let hash = device.identity_hash();

    let path = store.path_for(&device);

    assert_eq!(path, dir.path().join(format!("{hash}.json")));
}

#[test]
fn a_save_leaves_no_temporary_file_behind() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();

    store.save(&device, &result).expect("the first save succeeds");
    store
        .save(&device, &result)
        .expect("a save over an existing file succeeds");
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();

    assert_eq!(names.len(), 1, "{names:?}");
}

#[test]
fn a_missing_file_loads_an_empty_result() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());

    let loaded = store.load(&identity());

    assert_eq!(loaded, TuneResult::default());
}

#[test]
fn a_corrupt_file_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let path = store.path_for(&device);
    std::fs::write(&path, b"{ this is not json").unwrap();

    let loaded = store.load(&device);

    assert_eq!(loaded, TuneResult::default());
}

#[test]
fn another_format_version_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    let path = store.save(&device, &result).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let bumped = text.replace("\"format_version\": 1", "\"format_version\": 999");
    std::fs::write(&path, bumped).unwrap();

    let loaded = store.load(&device);

    assert_eq!(loaded, TuneResult::default());
}

#[test]
fn unknown_candidate_names_drop_their_key() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    let path = store.save(&device, &result).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let renamed = text.replace("\"32x16\"", "\"64x64\"");
    std::fs::write(&path, renamed).unwrap();

    let loaded = store.load(&device);

    let profile_keys = ["luma".to_string()];

    assert!(
        loaded.sites.is_empty(),
        "the key naming an unknown candidate is tuned again"
    );
    assert_eq!(loaded.profiles.len(), 1);
    assert!(!loaded.is_complete(&profile_keys), "the next run tunes again");
}

#[test]
fn unknown_fidelity_names_drop_their_key() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    let path = store.save(&device, &result).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let renamed = text.replace("\"fidelity\": \"exact\"", "\"fidelity\": \"mystery\"");
    std::fs::write(&path, renamed).unwrap();

    let loaded = store.load(&device);

    let profile_keys = ["luma".to_string()];

    assert!(
        loaded.sites.is_empty(),
        "the key naming an unknown fidelity is tuned again"
    );
    assert_eq!(loaded.profiles.len(), 1);
    assert!(!loaded.is_complete(&profile_keys), "the next run tunes again");
}

#[test]
fn a_store_with_only_known_keys_stays_complete() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let result = sample_result();
    store.save(&device, &result).unwrap();

    let loaded = store.load(&device);

    let profile_keys = ["luma".to_string()];
    assert!(loaded.is_complete(&profile_keys));
}

#[test]
fn clearing_removes_only_this_devices_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let mut other = identity();
    other.adapter = "Other Adapter".to_string();
    let result = sample_result();
    store.save(&device, &result).unwrap();
    store.save(&other, &result).unwrap();

    let removed = store.clear(&device).unwrap();
    let removed_again = store.clear(&device).unwrap();

    assert!(removed);
    assert!(!removed_again);
    assert!(store.path_for(&other).exists());
}

/// With caching off there is no store, and a result still has to reach every engine in the run.
#[test]
fn results_are_remembered_without_a_store() {
    let device = identity();
    let result = sample_result();

    assert_eq!(remembered(&device), None);
    remember(&device, &result);

    assert_eq!(remembered(&device), Some(result.clone()));
    assert_eq!(resolve(&device), result);
}

#[test]
fn a_file_for_another_identity_loads_empty() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let mut other = identity();
    other.adapter = "Other Adapter".to_string();
    let other_path = store.save(&other, &sample_result()).unwrap();
    std::fs::rename(&other_path, store.path_for(&device)).unwrap();

    let loaded = store.load(&device);

    assert_eq!(loaded, TuneResult::default());
}

#[test]
fn an_unknown_profile_name_drops_its_profile_key() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    let path = store.save(&device, &sample_result()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let renamed = text.replace("\"profile\": \"baseline\"", "\"profile\": \"mystery\"");
    std::fs::write(&path, renamed).unwrap();

    let loaded = store.load(&device);

    assert!(loaded.profiles.is_empty(), "the profile key is compared again");
    assert_eq!(
        loaded.sites,
        sample_result().sites,
        "the rest of the file still decodes"
    );
}

#[test]
fn a_failed_rename_removes_the_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = TuneStore::at(dir.path());
    let device = identity();
    std::fs::create_dir(store.path_for(&device)).unwrap();

    let saved = store.save(&device, &sample_result());

    assert!(saved.is_err());
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
