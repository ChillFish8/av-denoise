//! Where CubeCL keeps its compiled kernels.
//!
//! Compiling GPU kernels takes seconds, so CubeCL caches the results on
//! disk and reuses them on the next run. By default that cache lives
//! wherever CubeCL puts it.
//!
//! Setting the `AV_DENOISE_COMPILATION_CACHE` environment variable moves
//! both the compilation cache and the autotune cache to a directory of
//! your choosing, which is useful for CI runs and containers that want
//! the cache on a mounted volume.
//!
//! [`apply_compilation_cache_env`] reads that variable and installs the
//! override. It has to run before the first [`Denoiser`](crate::Denoiser)
//! is created, because building a CubeCL client locks the global config.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Call this at the top of `main`, before any denoiser exists.
//! if let Some(path) = av_denoise::apply_compilation_cache_env()? {
//!     println!("caching compiled kernels in {}", path.display());
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

/// The environment variable that redirects CubeCL's compilation and
/// autotune caches to a directory of your choosing.
pub const COMPILATION_CACHE_ENV: &str = "AV_DENOISE_COMPILATION_CACHE";

/// The CubeCL global config was already set up before this helper ran,
/// so the override can no longer be installed.
#[derive(Debug, thiserror::Error)]
#[error(
    "CubeCL global config already initialized. Call apply_compilation_cache_env() before any Denoiser::create"
)]
pub struct CacheAlreadyInitialisedError;

/// Applies the `AV_DENOISE_COMPILATION_CACHE` override when it is set.
///
/// Returns `Ok(Some(path))` once the override is installed, or
/// `Ok(None)` when the variable was not set.
///
/// Returns `Err` if something else has already read the global config,
/// which usually means a CubeCL client was created first.
///
/// This must be called before any CubeCL client is created.
pub fn apply_compilation_cache_env() -> Result<Option<PathBuf>, CacheAlreadyInitialisedError> {
    let Some(raw) = std::env::var_os(COMPILATION_CACHE_ENV) else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);

    let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
    cfg.compilation.cache = Some(CacheConfig::File(path.clone()));
    cfg.autotune.cache = CacheConfig::File(path.clone());

    // `RuntimeConfig::set` panics if the singleton is already set up.
    // Catching that turns an abort into a typed error for the caller.
    //
    // CubeCL does not expose a fallible version of this call, so the
    // panic is the only signal available.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CubeClRuntimeConfig::set(cfg);
    }))
    .map_err(|_| CacheAlreadyInitialisedError)?;

    Ok(Some(path))
}
