//! Where CubeCL keeps its compiled kernels.
//!
//! Compiling this crate's kernels takes about ten seconds, and CubeCL
//! caches nothing on its own. Its cache setting defaults to `None`, so
//! every run recompiles from scratch unless something points it at a
//! directory.
//!
//! [`install_compilation_cache`] points it at one. By default, that is
//! [`default_cache_dir`], `av-denoise` inside the platform's cache
//! directory, which turns the ten seconds into a cost paid once per
//! machine rather than once per run.
//!
//! The `AV_DENOISE_COMPILATION_CACHE` environment variable overrides the
//! location, which is what CI runs and containers use to put the cache
//! on a mounted volume. Setting it to `off` disables caching entirely,
//! which is what benchmarking wants, because a warm cache hides the
//! compilation cost that a first run pays.
//!
//! [`install_compilation_cache`] has to run before the first
//! [`Denoiser`](crate::Denoiser) is created, because building a CubeCL
//! client locks the global config.
//!
//! If using `av-denoise` as a library you may want to specify the cache directory
//! path yourself via [`install_compilation_cache_at`] instead.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Call this at the top of `main`, before any denoiser exists.
//! match av_denoise_core::install_compilation_cache()? {
//!     Some(path) => println!("caching compiled kernels in {}", path.display()),
//!     None => println!("kernel caching is off, every run recompiles"),
//! }
//! # Ok(())
//! # }
//! ```

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use etcetera::base_strategy::{BaseStrategy, choose_base_strategy};

/// The environment variable that overrides where compiled kernels are
/// cached, or turns caching off.
pub const COMPILATION_CACHE_ENV: &str = "AV_DENOISE_COMPILATION_CACHE";

/// Where compiled kernels are cached, once an install has settled it.
static CACHE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// The directory name this crate uses inside the user's cache directory.
const CACHE_DIR_NAME: &str = "av-denoise";

/// The values of [`COMPILATION_CACHE_ENV`] that turn caching off.
///
/// Compared without regard to case. `off` is the documented spelling and
/// the others are here so that a reasonable guess does not silently
/// create a directory named `0`.
const DISABLE_WORDS: [&str; 4] = ["off", "0", "false", "none"];

/// Something went wrong installing the kernel cache.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The CubeCL global config was already set up before this helper
    /// ran, so the cache directory can no longer be installed.
    #[error("CubeCL global config already initialized. Install the cache before any Denoiser::create")]
    AlreadyInitialised,
    /// The cache directory does not exist and could not be created.
    #[error("cannot create the kernel cache directory {path}", path = path.display())]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Where compiled kernels go.
///
/// [`Disabled`](CacheLocation::Disabled) is reachable only when
/// [`COMPILATION_CACHE_ENV`] names one of [`DISABLE_WORDS`]. Every
/// platform has a default directory, so nothing else produces it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheLocation {
    /// Nothing is cached and every run recompiles.
    Disabled,
    /// Compiled kernels are written under this directory.
    Dir(PathBuf),
}

/// The directory compiled kernels are cached in when nothing overrides it.
pub fn default_cache_dir() -> PathBuf {
    let platform_cache = choose_base_strategy().ok().map(|s| s.cache_dir());
    if platform_cache.is_none() {
        tracing::warn!("no platform cache directory available, falling back to the temporary directory");
    }
    resolve_default_dir(platform_cache, std::env::temp_dir())
}

fn resolve_default_dir(platform_cache: Option<PathBuf>, temp: PathBuf) -> PathBuf {
    platform_cache.unwrap_or(temp).join(CACHE_DIR_NAME)
}

/// Decides where compiled kernels go, from a default and the environment
/// override alone.
pub(crate) fn resolve_cache_location(
    env: Option<&OsStr>,
    default: impl FnOnce() -> PathBuf,
) -> CacheLocation {
    if let Some(raw) = env {
        let text = raw.to_string_lossy();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if DISABLE_WORDS.iter().any(|w| trimmed.eq_ignore_ascii_case(w)) {
                return CacheLocation::Disabled;
            }
            // `raw.to_str()` fails only when the value is not UTF-8, in
            // which case it cannot be trimmed portably, so it is used
            // unchanged rather than dropped.
            let dir = match raw.to_str() {
                Some(s) => PathBuf::from(s.trim()),
                None => PathBuf::from(raw),
            };
            return CacheLocation::Dir(dir);
        }
    }

    CacheLocation::Dir(default())
}

/// Points CubeCL's compilation and autotune caches at `dir`, creating it
/// if it does not exist.
///
/// This is the entry point for a caller using this crate directly.
pub fn install_compilation_cache_at(dir: &Path) -> Result<(), CacheError> {
    if let Err(source) = std::fs::create_dir_all(dir) {
        return Err(CacheError::Create {
            path: dir.to_path_buf(),
            source,
        });
    }

    set_runtime_config(dir)?;
    let _ = CACHE_DIR.set(Some(dir.to_path_buf()));
    Ok(())
}

/// Installs the cache at [`default_cache_dir`], or the directory
/// [`COMPILATION_CACHE_ENV`] names, unless that variable turns caching off.
///
/// Returns `Ok(None)` only when the variable disables caching.
///
/// A directory that cannot be created is reported through `tracing` logs and then
/// ignored, because denoising works without a cache. A caller that wants a
/// creation failure reported should use [`install_compilation_cache_at`].
pub fn install_compilation_cache() -> Result<Option<PathBuf>, CacheError> {
    let location = resolve_cache_location(
        std::env::var_os(COMPILATION_CACHE_ENV).as_deref(),
        default_cache_dir,
    );

    let CacheLocation::Dir(path) = location else {
        return Ok(None);
    };

    if let Err(err) = std::fs::create_dir_all(&path) {
        tracing::warn!(
            ?path,
            %err,
            "cannot create the kernel cache directory, continuing without a cache"
        );
        return Ok(None);
    }

    set_runtime_config(&path)?;

    // Only an install that reached the config has a directory worth
    // recording. A directory that could not be created has already
    // answered `Ok(None)` above, and latching that would leave
    // `compilation_cache_dir` saying `None` for the rest of the process
    // even if a later call succeeded.
    let _ = CACHE_DIR.set(Some(path.clone()));

    Ok(Some(path))
}

/// Points CubeCL at a cache the first time it runs, and reports where.
///
/// Written for callers that are not a `main`, such as the VapourSynth
/// plugin, where filter creation is the earliest hook there is and runs
/// once per filter rather than once per process.
///
/// A failure to install is reported through `tracing` and then ignored,
/// because a plugin that refuses to denoise is worse than one that
/// recompiles. Failing here means something else configured CubeCL
/// first, which may well have pointed it at a cache of its own. What is
/// lost is knowing where that cache is, which is why this answers `None`
/// rather than guessing.
pub fn install_compilation_cache_once() -> Option<&'static Path> {
    static ONCE: Once = Once::new();

    ONCE.call_once(|| match install_compilation_cache() {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!("kernel caching is off, every run recompiles"),
        Err(err) => tracing::warn!(
            %err,
            "something else configured CubeCL first, leaving its kernel cache alone"
        ),
    });

    compilation_cache_dir()
}

/// The directory compiled kernels are cached in.
///
/// `None` until an install succeeds, and `None` for good when caching is
/// off. Callers that want to sit alongside the cache, such as
/// [`WarmUp`](crate::WarmUp), have nowhere to put their own files until
/// this answers.
pub fn compilation_cache_dir() -> Option<&'static Path> {
    CACHE_DIR.get()?.as_deref()
}

/// Points the CubeCL global config at `path`.
///
/// Split out because both entry points build the same config once they
/// have settled on a directory that exists.
fn set_runtime_config(path: &Path) -> Result<(), CacheError> {
    let mut cfg = CubeClRuntimeConfig::from_current_dir().override_from_env();
    cfg.compilation.cache = Some(CacheConfig::File(path.to_path_buf()));
    cfg.autotune.cache = CacheConfig::File(path.to_path_buf());

    // `RuntimeConfig::set` panics if the singleton is already set up.
    // Catching that turns an abort into a typed error for the caller.
    //
    // CubeCL does not expose a fallible version of this call, so the
    // panic is the only signal available.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CubeClRuntimeConfig::set(cfg);
    }))
    .map_err(|_| CacheError::AlreadyInitialised)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn os(text: &str) -> OsString {
        OsString::from(text)
    }

    fn resolve(env: Option<&str>, default: &str) -> CacheLocation {
        let env = env.map(os);
        let default = PathBuf::from(default);
        resolve_cache_location(env.as_deref(), || default)
    }

    #[test]
    fn with_nothing_set_the_default_wins() {
        assert_eq!(
            resolve(None, "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    #[test]
    fn an_explicit_path_overrides_the_default() {
        assert_eq!(
            resolve(Some("/mnt/cache"), "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/mnt/cache")),
        );
    }

    #[test]
    fn the_disable_words_turn_caching_off() {
        for word in ["off", "OFF", "Off", "0", "false", "FALSE", "none", " off "] {
            assert_eq!(
                resolve(Some(word), "/home/u/.cache/av-denoise"),
                CacheLocation::Disabled,
                "{word} should disable caching",
            );
        }
    }

    #[test]
    fn a_path_containing_a_disable_word_is_still_a_path() {
        assert_eq!(
            resolve(Some("/tmp/offsite"), "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/tmp/offsite")),
        );
    }

    #[test]
    fn an_empty_variable_takes_the_default() {
        assert_eq!(
            resolve(Some(""), "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
        assert_eq!(
            resolve(Some("   "), "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    #[test]
    fn a_padded_explicit_path_is_trimmed() {
        assert_eq!(
            resolve(Some(" /mnt/cache "), "/home/u/.cache/av-denoise"),
            CacheLocation::Dir(PathBuf::from("/mnt/cache")),
        );
    }

    /// A non-UTF-8 value cannot be trimmed portably, so it is used unchanged.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_override_is_carried_through_untrimmed() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        // `0xFF` is not valid UTF-8 in any position, so `bytes` never
        // decodes to a `&str`.
        let bytes = vec![b'/', b'm', b'n', b't', b'/', 0xFF, b'x'];
        let env = OsString::from_vec(bytes.clone());
        assert_eq!(
            resolve_cache_location(Some(&env), || PathBuf::from("/home/u/.cache/av-denoise")),
            CacheLocation::Dir(PathBuf::from(OsString::from_vec(bytes))),
        );
    }

    #[test]
    fn resolve_default_dir_joins_the_platform_cache_directory() {
        assert_eq!(
            resolve_default_dir(Some(PathBuf::from("/home/u/.cache")), PathBuf::from("/tmp"),),
            PathBuf::from("/home/u/.cache/av-denoise"),
        );
    }

    #[test]
    fn resolve_default_dir_falls_back_to_the_temporary_directory() {
        assert_eq!(
            resolve_default_dir(None, PathBuf::from("/tmp")),
            PathBuf::from("/tmp/av-denoise"),
        );
    }
}
