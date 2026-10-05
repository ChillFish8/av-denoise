//! Where CubeCL keeps its compiled kernels
//!
//! Compiling the kernels was measured at about ten seconds, and CubeCL caches nothing by default.
//! [install_compilation_cache] points it at [default_cache_dir], or at the directory named by the
//! `AV_DENOISE_COMPILATION_CACHE` environment variable, so that cost is paid once per machine rather
//! than once per run. Setting that variable to `off` disables caching,
//! which benchmarks want because a warm cache hides the compile cost a first run pays.
//!
//! Each build of the kernels gets its own subdirectory, and an install removes other builds'
//! subdirectories once they have gone unused for a week.
//!
//! The cache has to be installed before the first [HostDenoiser](crate::HostDenoiser) is created,
//! because building a CubeCL client locks the global config. Library callers can pick the directory
//! themselves with [install_compilation_cache_at].
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
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};
use std::time::{Duration, SystemTime};

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use etcetera::base_strategy::{BaseStrategy, choose_base_strategy};

/// The environment variable that overrides where compiled kernels are cached, or turns caching off.
pub const COMPILATION_CACHE_ENV: &str = "AV_DENOISE_COMPILATION_CACHE";

/// Where compiled kernels are cached, once an install has settled it.
static CACHE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

const CACHE_DIR_NAME: &str = "av-denoise";

/// A hash of the kernel sources that names this build's subdirectory of the cache root.
///
/// CubeCL keys cached kernels by their signature rather than their body, so each build of the
/// kernels needs a directory of its own.
pub(crate) const KERNEL_HASH: &str = av_denoise_core::KERNEL_HASH;

/// How long another build's subdirectory can go unused before an install removes it.
const STALE_BUILD_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// The values of [COMPILATION_CACHE_ENV] that turn caching off, compared without regard to case.
///
/// `off` is the documented spelling. The others stop a reasonable guess from silently creating a
/// directory named `0`.
const DISABLE_WORDS: [&str; 4] = ["off", "0", "false", "none"];

/// Something went wrong installing the kernel cache.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The CubeCL global config was set up before the cache could be installed.
    #[error("CubeCL global config already initialized. Install the cache before any HostDenoiser::create")]
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
/// [Disabled](CacheLocation::Disabled) is reachable only when [COMPILATION_CACHE_ENV] names one of
/// [DISABLE_WORDS]. Every platform has a default directory, so nothing else produces it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheLocation {
    Disabled,
    Dir(PathBuf),
}

/// The directory compiled kernels are cached in when nothing overrides it.
pub fn default_cache_dir() -> PathBuf {
    let platform_cache = choose_base_strategy().ok().map(|strategy| strategy.cache_dir());
    if platform_cache.is_none() {
        tracing::warn!("no platform cache directory available, falling back to the temporary directory");
    }

    let temp_dir = std::env::temp_dir();
    resolve_default_dir(platform_cache, temp_dir)
}

fn resolve_default_dir(platform_cache: Option<PathBuf>, temp_dir: PathBuf) -> PathBuf {
    platform_cache.unwrap_or(temp_dir).join(CACHE_DIR_NAME)
}

/// Decides where compiled kernels go from a default and the environment override alone.
pub(crate) fn resolve_cache_location(
    env: Option<&OsStr>,
    default: impl FnOnce() -> PathBuf,
) -> CacheLocation {
    if let Some(raw) = env {
        let text = raw.to_string_lossy();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if DISABLE_WORDS
                .iter()
                .any(|word| trimmed.eq_ignore_ascii_case(word))
            {
                return CacheLocation::Disabled;
            }

            // A value that is not UTF-8 cannot be trimmed portably, so it is used unchanged.
            let dir = match raw.to_str() {
                Some(utf8) => PathBuf::from(utf8.trim()),
                None => PathBuf::from(raw),
            };
            return CacheLocation::Dir(dir);
        }
    }

    let dir = default();
    CacheLocation::Dir(dir)
}

/// Points CubeCL's compilation and autotune caches at this build's subdirectory of `dir`.
///
/// The subdirectory is created if it does not exist.
pub fn install_compilation_cache_at(dir: &Path) -> Result<(), CacheError> {
    let build_dir = build_cache_dir(dir);
    if let Err(source) = std::fs::create_dir_all(&build_dir) {
        return Err(CacheError::Create {
            path: build_dir,
            source,
        });
    }

    set_runtime_config(&build_dir)?;
    tidy_cache_root(dir, &build_dir);

    let _ = CACHE_DIR.set(Some(build_dir));
    Ok(())
}

/// Installs the cache at [default_cache_dir], or the directory [COMPILATION_CACHE_ENV] names.
///
/// Returns this build's subdirectory of that root, or `Ok(None)` when the variable disables caching.
///
/// A directory that cannot be created is logged through `tracing` and then ignored, because denoising
/// works without a cache. A caller that wants that failure reported should use
/// [install_compilation_cache_at].
pub fn install_compilation_cache() -> Result<Option<PathBuf>, CacheError> {
    let env_override = std::env::var_os(COMPILATION_CACHE_ENV);
    let location = resolve_cache_location(env_override.as_deref(), default_cache_dir);

    let CacheLocation::Dir(root) = location else {
        return Ok(None);
    };

    let path = build_cache_dir(&root);
    if let Err(err) = std::fs::create_dir_all(&path) {
        tracing::warn!(
            ?path,
            %err,
            "cannot create the kernel cache directory, continuing without a cache"
        );
        return Ok(None);
    }

    set_runtime_config(&path)?;
    tidy_cache_root(&root, &path);

    // Only a successful install is recorded, so a failed one never latches `None` for the rest
    // of the process.
    let _ = CACHE_DIR.set(Some(path.clone()));

    Ok(Some(path))
}

/// Points CubeCL at a cache the first time it runs, and reports where.
///
/// This suits callers without a `main`, such as the VapourSynth plugin, where filter creation is
/// the earliest hook and runs once per filter.
///
/// A failure to install is logged through `tracing` and then ignored, because a plugin that refuses
/// to denoise is worse than one that recompiles. Failing means something else configured CubeCL
/// first, possibly with a cache of its own, so this answers `None` rather than guessing where it is.
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

/// This build's directory of cached kernels.
///
/// `None` until an install succeeds, and `None` for good when caching is off.
pub fn compilation_cache_dir() -> Option<&'static Path> {
    CACHE_DIR.get()?.as_deref()
}

fn set_runtime_config(path: &Path) -> Result<(), CacheError> {
    let cache_dir = path.to_path_buf();
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = Some(CacheConfig::File(cache_dir.clone()));
    config.autotune.cache = CacheConfig::File(cache_dir);

    // `RuntimeConfig::set` panics if the singleton is already set up, and CubeCL has no fallible
    // version, so the panic is caught and turned into a typed error.
    let install = std::panic::AssertUnwindSafe(|| {
        CubeClRuntimeConfig::set(config);
    });
    std::panic::catch_unwind(install).map_err(|_| CacheError::AlreadyInitialised)
}

fn build_cache_dir(root: &Path) -> PathBuf {
    root.join(KERNEL_HASH)
}

/// Marks this build's subdirectory as used and removes stale ones from other builds.
///
/// Both steps are best effort, because a cache that is tidied late still works.
fn tidy_cache_root(root: &Path, build_dir: &Path) {
    let now = SystemTime::now();

    if let Ok(handle) = File::open(build_dir) {
        let _ = handle.set_modified(now);
    }

    prune_stale_builds(root, KERNEL_HASH, now);
}

/// Removes other builds' subdirectories of `root` that have gone unused for longer than
/// [STALE_BUILD_AGE].
///
/// Only directories named like a build hash are candidates, so anything else in the root is left
/// alone. Every error is ignored.
fn prune_stale_builds(root: &Path, current_hash: &str, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };

        if name == current_hash || !is_build_hash(name) {
            continue;
        }

        // `DirEntry::metadata` does not follow symlinks, so a link is never mistaken for a build
        // directory.
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };

        let age = now.duration_since(modified).unwrap_or_default();
        if metadata.is_dir() && age > STALE_BUILD_AGE {
            let stale_path = entry.path();
            let _ = std::fs::remove_dir_all(stale_path);
        }
    }
}

fn is_build_hash(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
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
        let location = resolve(None, "/home/u/.cache/av-denoise");

        assert_eq!(
            location,
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    #[test]
    fn an_explicit_path_overrides_the_default() {
        let location = resolve(Some("/mnt/cache"), "/home/u/.cache/av-denoise");

        assert_eq!(location, CacheLocation::Dir(PathBuf::from("/mnt/cache")));
    }

    #[test]
    fn the_disable_words_turn_caching_off() {
        for word in ["off", "OFF", "Off", "0", "false", "FALSE", "none", " off "] {
            let location = resolve(Some(word), "/home/u/.cache/av-denoise");
            assert_eq!(location, CacheLocation::Disabled, "{word} should disable caching");
        }
    }

    #[test]
    fn a_path_containing_a_disable_word_is_still_a_path() {
        let location = resolve(Some("/tmp/offsite"), "/home/u/.cache/av-denoise");

        assert_eq!(location, CacheLocation::Dir(PathBuf::from("/tmp/offsite")));
    }

    #[test]
    fn an_empty_variable_takes_the_default() {
        let empty = resolve(Some(""), "/home/u/.cache/av-denoise");
        let blank = resolve(Some("   "), "/home/u/.cache/av-denoise");

        assert_eq!(
            empty,
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
        assert_eq!(
            blank,
            CacheLocation::Dir(PathBuf::from("/home/u/.cache/av-denoise")),
        );
    }

    #[test]
    fn a_padded_explicit_path_is_trimmed() {
        let location = resolve(Some(" /mnt/cache "), "/home/u/.cache/av-denoise");

        assert_eq!(location, CacheLocation::Dir(PathBuf::from("/mnt/cache")));
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_override_is_carried_through_untrimmed() {
        use std::os::unix::ffi::OsStringExt;

        // `0xFF` is not valid UTF-8 in any position, so `bytes` never decodes to a `&str`.
        let bytes = vec![b'/', b'm', b'n', b't', b'/', 0xFF, b'x'];
        let env = OsString::from_vec(bytes.clone());
        let expected_os = OsString::from_vec(bytes);
        let expected = PathBuf::from(expected_os);

        let location = resolve_cache_location(Some(&env), || PathBuf::from("/home/u/.cache/av-denoise"));

        assert_eq!(location, CacheLocation::Dir(expected));
    }

    #[test]
    fn resolve_default_dir_joins_the_platform_cache_directory() {
        let platform_cache = Some(PathBuf::from("/home/u/.cache"));
        let temp_dir = PathBuf::from("/tmp");

        let resolved = resolve_default_dir(platform_cache, temp_dir);

        assert_eq!(resolved, PathBuf::from("/home/u/.cache/av-denoise"));
    }

    #[test]
    fn resolve_default_dir_falls_back_to_the_temporary_directory() {
        let temp_dir = PathBuf::from("/tmp");

        let resolved = resolve_default_dir(None, temp_dir);

        assert_eq!(resolved, PathBuf::from("/tmp/av-denoise"));
    }

    #[test]
    fn the_build_cache_dir_is_the_root_joined_with_the_kernel_hash() {
        let root = PathBuf::from("/home/u/.cache/av-denoise");
        let expected = root.join(KERNEL_HASH);

        let build_dir = build_cache_dir(&root);

        assert_eq!(build_dir, expected);
    }

    #[test]
    fn the_kernel_hash_is_sixteen_lowercase_hex_chars() {
        let looks_like_hash = is_build_hash(KERNEL_HASH);

        assert!(looks_like_hash, "{KERNEL_HASH} is not a build hash");
    }

    #[test]
    fn is_build_hash_rejects_other_names() {
        for name in [
            "vulkan",
            "0123456789ABCDEF",
            "0123456789abcde",
            "0123456789abcdef0",
            "warm-0123456789ab",
        ] {
            let looks_like_hash = is_build_hash(name);
            assert!(!looks_like_hash, "{name} should not look like a build hash");
        }
    }

    /// Opening a directory as a `File` to set its times only works on Unix.
    #[cfg(unix)]
    #[test]
    fn pruning_removes_only_stale_builds_of_other_hashes() {
        let root = tempfile::tempdir().expect("temp dir");
        let now = SystemTime::now();
        let long_ago = now - STALE_BUILD_AGE - Duration::from_secs(60);

        let current = "aaaaaaaaaaaaaaaa";
        let stale = "bbbbbbbbbbbbbbbb";
        let fresh = "cccccccccccccccc";
        let non_hash_dirs = ["vulkan", "hip", "warm-1234"];

        let old_dirs = [current, stale].into_iter().chain(non_hash_dirs);
        for name in old_dirs {
            let path = root.path().join(name);
            std::fs::create_dir(&path).expect("create dir");

            let handle = File::open(&path).expect("open dir");
            handle.set_modified(long_ago).expect("set mtime");
        }

        let fresh_path = root.path().join(fresh);
        std::fs::create_dir(&fresh_path).expect("create dir");

        let stale_file = root.path().join("dddddddddddddddd");
        let handle = File::create(&stale_file).expect("create file");
        handle.set_modified(long_ago).expect("set mtime");

        prune_stale_builds(root.path(), current, now);

        assert!(!root.path().join(stale).exists(), "stale build should be removed");
        assert!(root.path().join(current).exists(), "current build should be kept");
        assert!(fresh_path.exists(), "fresh build should be kept");
        assert!(stale_file.exists(), "a file named like a hash should be kept");
        for name in non_hash_dirs {
            assert!(root.path().join(name).exists(), "{name} should be kept");
        }
    }

    #[test]
    fn pruning_a_missing_root_does_nothing() {
        let root = tempfile::tempdir().expect("temp dir");
        let missing = root.path().join("missing");
        let now = SystemTime::now();

        prune_stale_builds(&missing, KERNEL_HASH, now);
    }

    #[cfg(unix)]
    #[test]
    fn tidying_marks_the_current_build_as_recently_used() {
        let root = tempfile::tempdir().expect("temp dir");
        let build_dir = build_cache_dir(root.path());
        std::fs::create_dir(&build_dir).expect("create dir");

        let long_ago = SystemTime::now() - STALE_BUILD_AGE - Duration::from_secs(60);
        let handle = File::open(&build_dir).expect("open dir");
        handle.set_modified(long_ago).expect("set mtime");

        tidy_cache_root(root.path(), &build_dir);

        let metadata = std::fs::metadata(&build_dir).expect("build dir metadata");
        let modified = metadata.modified().expect("mtime");
        let age = SystemTime::now().duration_since(modified).unwrap_or_default();
        assert!(age < Duration::from_secs(60), "build dir mtime is {age:?} old");
    }
}
