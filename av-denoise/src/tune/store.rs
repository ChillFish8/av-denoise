use std::path::PathBuf;

use av_denoise_core::tune::{DeviceIdentity, TuneResult};

use crate::cache::compilation_cache_dir;

/// The store's directory inside this build's kernel cache directory.
const TUNE_DIR_NAME: &str = "tune";

/// The directory tuning results are stored in, `None` when kernel caching is off.
///
/// It sits in this build's kernel cache directory, so a change to any kernel source starts a fresh
/// store.
pub fn default_store_dir() -> Option<PathBuf> {
    let cache = compilation_cache_dir()?;
    Some(cache.join(TUNE_DIR_NAME))
}

/// The result engines build from, what this process already holds or else the default store's file.
pub(crate) fn resolve(identity: &DeviceIdentity) -> TuneResult {
    let store = default_store_dir();
    av_denoise_core::tune::resolve(store.as_deref(), identity)
}

/// Reads the default store's file for `identity` afresh and remembers it.
pub(crate) fn reload(identity: &DeviceIdentity) -> TuneResult {
    let store = default_store_dir();
    av_denoise_core::tune::reload(store.as_deref(), identity)
}

/// Keeps `result` for `identity` for the rest of this process.
pub(crate) fn remember(identity: &DeviceIdentity, result: &TuneResult) {
    let store = default_store_dir();
    av_denoise_core::tune::remember(store.as_deref(), identity, result);
}
