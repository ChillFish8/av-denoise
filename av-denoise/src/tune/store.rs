use std::collections::{BTreeMap, HashMap};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, PoisonError};

use av_denoise_core::tune::{
    CandidateRecord,
    DeviceIdentity,
    FidelityClass,
    PipelineProfile,
    ProfileRanking,
    ProfileRecord,
    SiteKey,
    SiteRanking,
    TuneResult,
};
use serde::{Deserialize, Serialize};

use crate::cache::compilation_cache_dir;

/// The store's directory inside this build's kernel cache directory.
const TUNE_DIR_NAME: &str = "tune";

/// The version of the file layout below. Any other version is ignored and tuned again.
const FORMAT_VERSION: u32 = 1;

/// Makes each temporary file name unique within this process.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Results this process tuned or loaded, by identity hash.
static REMEMBERED: LazyLock<Mutex<HashMap<String, TuneResult>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Serialize, Deserialize)]
struct StoredFile {
    format_version: u32,
    identity: StoredIdentity,
    sites: BTreeMap<String, Vec<StoredCandidate>>,
    profiles: BTreeMap<String, StoredProfiles>,
    #[serde(default)]
    failed: Vec<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct StoredIdentity {
    backend: String,
    adapter: String,
    vendor_id: u32,
    device_id: u32,
    driver: String,
    os: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredCandidate {
    name: String,
    fidelity: String,
    baseline: bool,
    median_ns: u64,
    samples: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredProfiles {
    complete: bool,
    ranking: Vec<StoredProfile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredProfile {
    profile: String,
    median_frame_ns: u64,
    blocks: u32,
}

/// Something went wrong writing the tuning store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("cannot write the tuning store at {path}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot encode the tuning result")]
    Encode(#[from] serde_json::Error),
}

/// Tuning results on disk, one JSON file per device identity.
///
/// The default store lives in this build's kernel cache directory, so a change to any kernel
/// source starts a fresh store. Writes go to a temporary file in the same directory, which is then
/// renamed into place.
#[derive(Debug, Clone)]
pub struct TuneStore {
    dir: PathBuf,
}

impl TuneStore {
    /// The store in this build's kernel cache directory, `None` when kernel caching is off.
    pub fn open_default() -> Option<Self> {
        let cache = compilation_cache_dir()?;
        let dir = cache.join(TUNE_DIR_NAME);
        Some(Self { dir })
    }

    /// A store that keeps its files in `dir`.
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self { dir }
    }

    /// The file holding the result for `identity`.
    pub fn path_for(&self, identity: &DeviceIdentity) -> PathBuf {
        let hash = identity.identity_hash();
        self.dir.join(format!("{hash}.json"))
    }

    /// Loads the result stored for `identity`.
    ///
    /// A missing file loads empty. An unreadable or corrupt file, or one of another format version
    /// or device, is logged and loads empty. Keys naming a candidate this build does not have, and
    /// profile keys naming a profile it does not have, are dropped so they are tuned again.
    pub fn load(&self, identity: &DeviceIdentity) -> TuneResult {
        let path = self.path_for(identity);
        let read = std::fs::read_to_string(&path);

        let text = match read {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return TuneResult::default(),
            Err(error) => {
                tracing::warn!(?path, %error, "cannot read the tuning store, tuning again");
                return TuneResult::default();
            },
        };

        let decoded = decode(&text, identity);
        let mut result = match decoded {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(?path, %error, "ignoring an unusable tuning store, tuning again");
                return TuneResult::default();
            },
        };

        let dropped = result.retain_known();
        if dropped > 0 {
            tracing::info!(
                dropped,
                "the tuning store names candidates this build lacks, tuning them again"
            );
        }

        result
    }

    /// Writes `result` for `identity`, replacing any earlier file.
    pub fn save(&self, identity: &DeviceIdentity, result: &TuneResult) -> Result<PathBuf, StoreError> {
        std::fs::create_dir_all(&self.dir).map_err(|source| StoreError::Io {
            path: self.dir.clone(),
            source,
        })?;

        let path = self.path_for(identity);
        let stored = encode(identity, result);
        let text = serde_json::to_string_pretty(&stored)?;

        let hash = identity.identity_hash();
        let process = std::process::id();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_path = self.dir.join(format!(".{hash}.{process}.{counter}.tmp"));
        write_synced(&temp_path, &text).map_err(|source| StoreError::Io {
            path: temp_path.clone(),
            source,
        })?;

        let renamed = std::fs::rename(&temp_path, &path);
        if let Err(source) = renamed {
            let _ = std::fs::remove_file(&temp_path);
            return Err(StoreError::Io { path, source });
        }

        Ok(path)
    }

    /// Deletes the file for `identity`. Returns whether there was one.
    pub fn clear(&self, identity: &DeviceIdentity) -> Result<bool, StoreError> {
        let path = self.path_for(identity);
        let removed = std::fs::remove_file(&path);

        match removed {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
            Err(source) => Err(StoreError::Io { path, source }),
        }
    }
}

/// Writes `text` to `path` and flushes it to disk before returning.
fn write_synced(path: &Path, text: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

fn stored_identity(identity: &DeviceIdentity) -> StoredIdentity {
    StoredIdentity {
        backend: identity.backend.clone(),
        adapter: identity.adapter.clone(),
        vendor_id: identity.vendor_id,
        device_id: identity.device_id,
        driver: identity.driver.clone(),
        os: identity.os.clone(),
    }
}

fn encode_candidate(record: &CandidateRecord) -> StoredCandidate {
    StoredCandidate {
        name: record.name.clone(),
        fidelity: record.fidelity.name().to_string(),
        baseline: record.baseline,
        median_ns: record.median_ns,
        samples: record.samples,
    }
}

fn encode_profile(record: &ProfileRecord) -> StoredProfile {
    StoredProfile {
        profile: record.profile.name().to_string(),
        median_frame_ns: record.median_frame_ns,
        blocks: record.blocks,
    }
}

fn encode(identity: &DeviceIdentity, result: &TuneResult) -> StoredFile {
    let sites = result
        .sites
        .iter()
        .map(|(key, ranking)| {
            let candidates = ranking.candidates.iter().map(encode_candidate).collect();
            (key.as_str().to_string(), candidates)
        })
        .collect();

    let profiles = result
        .profiles
        .iter()
        .map(|(profile_key, ranking)| {
            let stored = StoredProfiles {
                complete: ranking.complete,
                ranking: ranking.profiles.iter().map(encode_profile).collect(),
            };
            (profile_key.clone(), stored)
        })
        .collect();

    let failed = result.failed.iter().map(|key| key.as_str().to_string()).collect();

    StoredFile {
        format_version: FORMAT_VERSION,
        identity: stored_identity(identity),
        sites,
        profiles,
        failed,
    }
}

fn decode_candidate(stored: StoredCandidate) -> Option<CandidateRecord> {
    let fidelity = FidelityClass::from_name(&stored.fidelity)?;

    Some(CandidateRecord {
        name: stored.name,
        fidelity,
        baseline: stored.baseline,
        median_ns: stored.median_ns,
        samples: stored.samples,
    })
}

fn decode_profile(stored: StoredProfile) -> Option<ProfileRecord> {
    let profile = PipelineProfile::from_name(&stored.profile)?;

    Some(ProfileRecord {
        profile,
        median_frame_ns: stored.median_frame_ns,
        blocks: stored.blocks,
    })
}

fn decode(text: &str, identity: &DeviceIdentity) -> Result<TuneResult, anyhow::Error> {
    let stored: StoredFile = serde_json::from_str(text)?;

    if stored.format_version != FORMAT_VERSION {
        anyhow::bail!(
            "the store has format version {}, this build reads {FORMAT_VERSION}",
            stored.format_version
        );
    }

    let expected_identity = stored_identity(identity);
    if stored.identity != expected_identity {
        anyhow::bail!("the store belongs to another device");
    }

    let mut result = TuneResult::default();

    for (key_text, candidates) in stored.sites {
        let Some(key) = SiteKey::from_stored(&key_text) else {
            continue;
        };

        let records: Option<Vec<CandidateRecord>> = candidates.into_iter().map(decode_candidate).collect();
        if let Some(records) = records {
            let ranking = SiteRanking { candidates: records };
            result.sites.insert(key, ranking);
        }
    }

    for (profile_key, stored_profiles) in stored.profiles {
        let records: Option<Vec<ProfileRecord>> =
            stored_profiles.ranking.into_iter().map(decode_profile).collect();
        let Some(profiles) = records else {
            continue;
        };

        let ranking = ProfileRanking {
            complete: stored_profiles.complete,
            profiles,
        };
        result.profiles.insert(profile_key, ranking);
    }

    let failed_keys = stored
        .failed
        .iter()
        .filter_map(|key_text| SiteKey::from_stored(key_text));
    result.failed.extend(failed_keys);

    Ok(result)
}

/// Keeps `result` for `identity` for the rest of this process.
pub(crate) fn remember(identity: &DeviceIdentity, result: &TuneResult) {
    let hash = identity.identity_hash();
    let mut remembered = REMEMBERED.lock().unwrap_or_else(PoisonError::into_inner);
    remembered.insert(hash, result.clone());
}

/// The result this process holds for `identity`, if any.
pub(crate) fn remembered(identity: &DeviceIdentity) -> Option<TuneResult> {
    let hash = identity.identity_hash();
    let remembered = REMEMBERED.lock().unwrap_or_else(PoisonError::into_inner);
    remembered.get(&hash).cloned()
}

/// The result engines build from, what this process already holds or else the default store's file.
pub(crate) fn resolve(identity: &DeviceIdentity) -> TuneResult {
    if let Some(result) = remembered(identity) {
        return result;
    }

    reload(identity)
}

/// Reads the default store's file for `identity` afresh and remembers it.
pub(crate) fn reload(identity: &DeviceIdentity) -> TuneResult {
    let store = TuneStore::open_default();
    let loaded = store.map(|store| store.load(identity)).unwrap_or_default();

    remember(identity, &loaded);
    loaded
}
