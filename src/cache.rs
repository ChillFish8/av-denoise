//! Where CubeCL keeps its compiled kernels.
//!
//! Compiling this crate's kernels takes about ten seconds, and CubeCL
//! caches nothing on its own. Its cache setting defaults to `None`, so
//! every run recompiles from scratch unless something points it at a
//! directory.
//!
//! [`install_compilation_cache`] points it at one. By default that is
//! `av-denoise` inside the user's cache directory, which turns the ten
//! seconds into a cost paid once per machine rather than once per run.
//! A warm cache takes the 53-frame reference clip from 11.8 s to 1.3 s.
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
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Call this at the top of `main`, before any denoiser exists.
//! match av_denoise::install_compilation_cache()? {
//!     Some(path) => println!("caching compiled kernels in {}", path.display()),
//!     None => println!("kernel caching is off, every run recompiles"),
//! }
//! # Ok(())
//! # }
//! ```

use std::ffi::OsStr;
use std::path::PathBuf;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

/// The environment variable that overrides where compiled kernels are
/// cached, or turns caching off.
pub const COMPILATION_CACHE_ENV: &str = "AV_DENOISE_COMPILATION_CACHE";

/// The directory name this crate uses inside the user's cache directory.
const CACHE_DIR_NAME: &str = "av-denoise";

/// The values of [`COMPILATION_CACHE_ENV`] that turn caching off.
///
/// Compared without regard to case. `off` is the documented spelling and
/// the others are here so that a reasonable guess does not silently
/// create a directory named `0`.
const DISABLE_WORDS: [&str; 4] = ["off", "0", "false", "none"];

/// The CubeCL global config was already set up before this helper ran,
/// so the override can no longer be installed.
#[derive(Debug, thiserror::Error)]
#[error(
    "CubeCL global config already initialized. Call install_compilation_cache() before any Denoiser::create"
)]
pub struct CacheAlreadyInitialisedError;

/// Where compiled kernels go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheLocation {
    /// Nothing is cached and every run recompiles.
    Disabled,
    /// Compiled kernels are written under this directory.
    Dir(PathBuf),
}

/// Decides where compiled kernels go, from the environment alone.
///
/// Kept separate from [`install_compilation_cache`] because installing
/// the choice writes to a global that can only be set once per process,
/// while the choice itself is worth testing over many inputs.
///
/// `env` is the raw value of [`COMPILATION_CACHE_ENV`]. An unset or
/// empty value takes the default, one of [`DISABLE_WORDS`] turns caching
/// off, and anything else is used as the directory.
///
/// The default is `$XDG_CACHE_HOME/av-denoise`, or the platform's cache
/// directory under `$HOME` when `XDG_CACHE_HOME` is unset. With no home
/// directory to fall back on there is nowhere sensible to write, so
/// caching is off.
pub(crate) fn resolve_cache_location(
    env: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
    is_macos: bool,
) -> CacheLocation {
    if let Some(raw) = env {
        let text = raw.to_string_lossy();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if DISABLE_WORDS.iter().any(|w| trimmed.eq_ignore_ascii_case(w)) {
                return CacheLocation::Disabled;
            }
            return CacheLocation::Dir(PathBuf::from(raw));
        }
    }

    if let Some(xdg) = xdg_cache_home.filter(|x| !x.is_empty()) {
        return CacheLocation::Dir(PathBuf::from(xdg).join(CACHE_DIR_NAME));
    }

    let Some(home) = home.filter(|h| !h.is_empty()) else {
        return CacheLocation::Disabled;
    };
    let base = if is_macos {
        PathBuf::from(home).join("Library").join("Caches")
    } else {
        PathBuf::from(home).join(".cache")
    };
    CacheLocation::Dir(base.join(CACHE_DIR_NAME))
}

/// Points CubeCL's compilation and autotune caches at a directory.
///
/// Returns `Ok(Some(path))` with the directory in use, or `Ok(None)`
/// when caching is off. Caching is off when
/// [`COMPILATION_CACHE_ENV`] says so, when there is no home directory to
/// derive a default from, or when the directory cannot be created.
///
/// A directory that cannot be created is reported through `tracing` and
/// then ignored. Denoising works without a cache, so failing to write
/// one is not a reason to refuse to run.
///
/// Returns `Err` if something else has already read the global config,
/// which usually means a CubeCL client was created first.
pub fn install_compilation_cache() -> Result<Option<PathBuf>, CacheAlreadyInitialisedError> {
    let location = resolve_cache_location(
        std::env::var_os(COMPILATION_CACHE_ENV).as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
        cfg!(target_os = "macos"),
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn os(text: &str) -> OsString {
        OsString::from(text)
    }

    fn resolve(env: Option<&str>, xdg: Option<&str>, home: Option<&str>) -> CacheLocation {
        let env = env.map(os);
        let xdg = xdg.map(os);
        let home = home.map(os);
        resolve_cache_location(env.as_deref(), xdg.as_deref(), home.as_deref(), false)
    }

    /// With nothing set, the cache lands under the XDG cache directory,
    /// which is where a compiler cache belongs and where a container can
    /// mount over it.
    #[test]
    fn the_default_is_the_xdg_cache_directory() {
        assert_eq!(
            resolve(None, Some("/home/u/.cache"), Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    /// `XDG_CACHE_HOME` is frequently unset on machines that still have
    /// a perfectly good home directory, so the fallback matters more
    /// than the primary path does.
    #[test]
    fn without_xdg_the_default_sits_under_the_home_directory() {
        assert_eq!(
            resolve(None, None, Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    /// macOS keeps caches somewhere else, and this crate builds there
    /// through its `metal` feature.
    #[test]
    fn macos_uses_its_own_cache_directory() {
        let home = os("/Users/u");
        assert_eq!(
            resolve_cache_location(None, None, Some(home.as_os_str()), true),
            CacheLocation::Dir(PathBuf::from("/Users/u/Library/Caches/av-denoise")),
        );
    }

    /// An explicit path wins over both defaults. This is the case CI
    /// runs and containers use.
    #[test]
    fn an_explicit_path_overrides_every_default() {
        assert_eq!(
            resolve(Some("/mnt/cache"), Some("/home/u/.cache"), Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/mnt/cache")),
        );
    }

    /// Benchmarking needs the compilation cost a first run pays, and a
    /// warm cache hides it.
    #[test]
    fn the_disable_words_turn_caching_off() {
        for word in ["off", "OFF", "Off", "0", "false", "FALSE", "none", " off "] {
            assert_eq!(
                resolve(Some(word), Some("/home/u/.cache"), Some("/home/u")),
                CacheLocation::Disabled,
                "{word} should disable caching",
            );
        }
    }

    /// A path that merely looks like a disable word is still a path.
    /// Nothing here should turn `/tmp/offsite` into "off".
    #[test]
    fn a_path_containing_a_disable_word_is_still_a_path() {
        assert_eq!(
            resolve(Some("/tmp/offsite"), None, Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/tmp/offsite")),
        );
    }

    /// An empty variable reads as "not set" rather than as "off",
    /// matching what an unset variable does.
    #[test]
    fn an_empty_variable_takes_the_default() {
        assert_eq!(
            resolve(Some(""), Some("/home/u/.cache"), Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
        assert_eq!(
            resolve(Some("   "), Some("/home/u/.cache"), Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    /// An empty `XDG_CACHE_HOME` is not a usable directory, so the home
    /// directory answers instead.
    #[test]
    fn an_empty_xdg_falls_through_to_the_home_directory() {
        assert_eq!(
            resolve(None, Some(""), Some("/home/u")),
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    /// With no home directory there is nowhere sensible to write, and
    /// picking the working directory would scatter caches wherever the
    /// binary happened to run.
    #[test]
    fn no_home_directory_means_no_cache() {
        assert_eq!(resolve(None, None, None), CacheLocation::Disabled);
        assert_eq!(resolve(None, None, Some("")), CacheLocation::Disabled);
    }
}
