use std::path::PathBuf;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

/// Name of the environment variable that, when set, redirects CubeCL's
/// compilation and autotune caches to the given directory.
pub const COMPILATION_CACHE_ENV: &str = "AV_DENOISE_COMPILATION_CACHE";

/// The CubeCL global config was already initialized before this
/// helper ran, so the override can no longer be installed.
#[derive(Debug, thiserror::Error)]
#[error(
    "CubeCL global config already initialized. Call apply_compilation_cache_env() before any Denoiser::create"
)]
pub struct CacheAlreadyInitialisedError;

/// Apply the `AV_DENOISE_COMPILATION_CACHE` override if set.
///
/// Returns `Ok(Some(path))` if the override was installed, `Ok(None)`
/// if the env var was unset, or `Err` if the global config was already
/// read by something else (e.g. a previously-created CubeCL client).
///
/// Must be called before any CubeCL client is created.
pub fn apply_compilation_cache_env() -> Result<Option<PathBuf>, CacheAlreadyInitialisedError> {
    let Some(raw) = std::env::var_os(COMPILATION_CACHE_ENV) else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);

    let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
    cfg.compilation.cache = Some(CacheConfig::File(path.clone()));
    cfg.autotune.cache = CacheConfig::File(path.clone());

    // `RuntimeConfig::set` panics if the singleton has already been
    // initialized. Catch that so callers get a typed error instead of
    // an abort.
    //
    // Unfortunately, CubeCL doesn't currently expose a graceful variant
    // of this function, so we're left with catching the panic.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CubeClRuntimeConfig::set(cfg);
    }))
    .map_err(|_| CacheAlreadyInitialisedError)?;

    Ok(Some(path))
}
