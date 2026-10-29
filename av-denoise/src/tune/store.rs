use std::path::PathBuf;

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
