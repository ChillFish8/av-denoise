//! Lets one process fill a cold kernel cache while the others wait
//!
//! CubeCL compiles a kernel the first time it is dispatched and writes it to the cache that
//! [install_compilation_cache](crate::install_compilation_cache) points it at. The cache's table of
//! contents is a snapshot taken when the GPU client is built, so a process that starts while another
//! is still compiling shares nothing with it, pays the full compilation cost again and writes its
//! own copy of the same kernels.
//!
//! [WarmUp] holds a lock file next to the cache while the first process compiles, and the rest
//! block on it, so each waiting process builds its own client after the wait, when the cache
//! already holds every kernel. A finished run leaves a stamp file behind, so later processes skip
//! the lock entirely.

use std::collections::HashSet;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use cubecl::hash::StableHasher;

use crate::cache::compilation_cache_dir;
use crate::planar::{FrameLayout, PlaneOptions};

/// How long a process waits for the one ahead of it before compiling for itself.
///
/// Compiling the kernels was measured at about ten seconds on a quiet machine, and a machine running
/// an encode is not quiet, so the limit is generous. Waiting longer is worse than duplicating the
/// work, because the encoder has nothing to do until a frame arrives.
const WAIT_LIMIT: Duration = Duration::from_secs(180);

const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The keys this process already holds a place for.
static CLAIMED_KEYS: LazyLock<Mutex<HashSet<u128>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Records that this process is taking the place for `key`, and reports whether it was free.
fn claim_key(key: u128) -> bool {
    CLAIMED_KEYS
        .lock()
        .expect("warm-up key mutex poisoned")
        .insert(key)
}

/// Gives `key` back, so a later filter in this process can queue for it again.
fn release_key(key: u128) {
    CLAIMED_KEYS
        .lock()
        .expect("warm-up key mutex poisoned")
        .remove(&key);
}

/// Identifies the set of kernels a denoiser compiles.
///
/// Radii, channel mode, depth and algorithm are baked into the kernels at compile time, so processes
/// compiling different sets never wait for each other and a stamp for one set never vouches for another.
///
/// The key hashes the `Debug` rendering of both inputs rather than a hand-written field list, which
/// would silently stop covering a newly added field and let a process trust a stamp for kernels it
/// never compiled. Fields that only reach the GPU at runtime, such as strength, make the key finer
/// than needed, which costs an extra warm-up and is the safe direction to err in.
///
/// The crate version is part of the key because a release can upgrade CubeCL, which files its cache
/// under the CubeCL version, and a stamp from before the upgrade would call the emptied cache warm.
/// A rebuild of the kernel sources needs nothing here, because the stamps live in that build's own
/// cache directory.
pub fn kernel_key(options: &PlaneOptions, layout: FrameLayout) -> u128 {
    let version = env!("CARGO_PKG_VERSION");
    let rendered = format!("{version}|{options:?}|{layout:?}");
    StableHasher::hash_one(&rendered)
}

/// A held place in the queue to fill a cold cache.
///
/// Obtained from [WarmUp::begin] and given up with [WarmUp::finish] once the kernels are compiled.
/// Dropping one without calling `finish` releases the lock without leaving a stamp, so a run that
/// failed part way through does not convince the next process that the cache is warm.
#[derive(Debug)]
pub struct WarmUp {
    lock: File,
    stamp: PathBuf,
    key: u128,
}

impl WarmUp {
    /// Takes a place in the queue for the kernels `key` identifies.
    ///
    /// Returns `Some` while holding the lock, and the caller compiles under it. Returns `None` when
    /// the cache is already warm for these kernels, caching is off, or the lock could not be taken
    /// within the three minute wait limit. In every one of those the caller compiles as usual.
    ///
    /// Blocks for as long as the process ahead takes to compile, so it belongs on the path that
    /// builds a denoiser rather than the path that renders a frame.
    pub fn begin(key: u128) -> Option<Self> {
        let dir = compilation_cache_dir()?;
        Self::begin_in(dir, key, WAIT_LIMIT)
    }

    /// [WarmUp::begin] against an explicit directory and wait limit.
    fn begin_in(dir: &Path, key: u128, wait_limit: Duration) -> Option<Self> {
        // A file lock is held by the process rather than the handle, so a second filter in this
        // process asking for the same kernels would wait out `wait_limit` on its own lock. One script
        // can easily build two filters, so the place is taken at most once per key per process and
        // the second caller carries on.
        if !claim_key(key) {
            return None;
        }

        let held = Self::acquire(dir, key, wait_limit);

        if held.is_none() {
            release_key(key);
        }

        held
    }

    /// [WarmUp::begin_in] without the in-process bookkeeping, which its caller handles.
    fn acquire(dir: &Path, key: u128, wait_limit: Duration) -> Option<Self> {
        let stamp_name = format!("warm-{key:032x}.stamp");
        let stamp = dir.join(stamp_name);

        if stamp.exists() {
            return None;
        }

        let lock_name = format!("warm-{key:032x}.lock");
        let lock_path = dir.join(lock_name);
        let lock = open_lock_file(&lock_path)?;

        if !wait_for_lock(&lock, wait_limit) {
            return None;
        }

        // Whoever held the lock has finished compiling by now, and checking the stamp again is what
        // turns the queue into a single warm-up rather than one per waiting process.
        if stamp.exists() {
            let _ = lock.unlock();
            return None;
        }

        tracing::debug!(?stamp, "compiling kernels for a cold cache");
        Some(Self { lock, stamp, key })
    }

    /// Records that the kernels are compiled and lets the next process through.
    pub fn finish(self) {
        if let Err(err) = std::fs::write(&self.stamp, b"") {
            // A missing stamp reads as a cold cache, which is slower but still correct.
            tracing::debug!(stamp = ?self.stamp, %err, "cannot write the kernel warm-up stamp");
        }
    }
}

impl Drop for WarmUp {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
        release_key(self.key);
    }
}

/// Opens the lock file, creating it when this is the first process to ask for these kernels.
///
/// A directory that cannot be written is logged and then ignored, because denoising works without
/// the queue and only compiles more than once.
fn open_lock_file(path: &Path) -> Option<File> {
    let opened = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);

    match opened {
        Ok(file) => Some(file),
        Err(err) => {
            tracing::debug!(?path, %err, "cannot open the kernel warm-up lock, compiling unqueued");
            None
        },
    }
}

/// Blocks until the lock is held, giving up after `wait_limit`.
///
/// Returns whether the lock is held. It is a real advisory file lock rather than a file whose presence
/// means "taken", so the operating system releases it when Av1an kills a worker mid-compile and the
/// next process in line wakes up straight away.
fn wait_for_lock(lock: &File, wait_limit: Duration) -> bool {
    let start = Instant::now();

    loop {
        match lock.try_lock() {
            Ok(()) => return true,
            Err(TryLockError::WouldBlock) => {},
            Err(TryLockError::Error(err)) => {
                tracing::debug!(%err, "cannot take the kernel warm-up lock, compiling unqueued");
                return false;
            },
        }

        if start.elapsed() >= wait_limit {
            tracing::warn!(
                "waited {:?} for another process to compile kernels, compiling for ourselves",
                wait_limit,
            );
            return false;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key no other test shares, so one test's lock and stamp files are never seen by the next.
    fn key(index: u128) -> u128 {
        0xa5a5_0000_0000_0000_0000_0000_0000_0000 + index
    }

    /// Short enough that a contended lock fails the test quickly rather than after three minutes.
    const BRIEFLY: Duration = Duration::from_millis(50);

    /// The lock is taken directly because a file lock is held per process, and a second `begin_in`
    /// would be answered by the in-process registry instead of the lock.
    #[test]
    fn a_lock_held_elsewhere_keeps_this_process_out() {
        let dir = tempfile::tempdir().unwrap();
        let lock_key = key(1);
        let lock_name = format!("warm-{lock_key:032x}.lock");
        let path = dir.path().join(lock_name);
        let elsewhere = open_lock_file(&path).unwrap();
        elsewhere.lock().unwrap();

        let place = WarmUp::begin_in(dir.path(), lock_key, BRIEFLY);

        assert!(
            place.is_none(),
            "a caller gives up rather than compiling alongside the process ahead",
        );
    }

    #[test]
    fn one_process_takes_one_place_per_key() {
        let dir = tempfile::tempdir().unwrap();
        let shared_key = key(6);

        let first = WarmUp::begin_in(dir.path(), shared_key, BRIEFLY);
        assert!(first.is_some(), "the first caller compiles");

        let second = WarmUp::begin_in(dir.path(), shared_key, BRIEFLY);
        assert!(
            second.is_none(),
            "the second caller carries on rather than waiting for itself",
        );
    }

    #[test]
    fn a_released_place_can_be_taken_again() {
        let dir = tempfile::tempdir().unwrap();
        let place_key = key(7);

        let released = WarmUp::begin_in(dir.path(), place_key, BRIEFLY);
        drop(released);

        let retaken = WarmUp::begin_in(dir.path(), place_key, BRIEFLY);
        assert!(
            retaken.is_some(),
            "the key is free again once the place is given up",
        );
    }

    #[test]
    fn a_finished_warm_up_lets_the_next_process_straight_through() {
        let dir = tempfile::tempdir().unwrap();
        let place_key = key(2);

        WarmUp::begin_in(dir.path(), place_key, BRIEFLY).unwrap().finish();

        let next = WarmUp::begin_in(dir.path(), place_key, BRIEFLY);
        assert!(next.is_none(), "a warm cache needs no queue");
    }

    #[test]
    fn an_abandoned_warm_up_leaves_the_cache_cold() {
        let dir = tempfile::tempdir().unwrap();
        let place_key = key(3);

        let abandoned = WarmUp::begin_in(dir.path(), place_key, BRIEFLY);
        drop(abandoned);

        let next = WarmUp::begin_in(dir.path(), place_key, BRIEFLY);
        assert!(next.is_some(), "no stamp means the kernels still need compiling");
    }

    /// A `PlaneOptions` with no accelerator named, so this module builds with any backend feature.
    fn options() -> PlaneOptions {
        PlaneOptions {
            accelerators: Vec::new(),
            device: crate::Device::Default,
            intent: crate::ChannelIntent::LumaChroma,
            mode: crate::DenoisingMode::Temporal { radius: 2 },
            algorithm: crate::Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        }
    }

    fn layout() -> FrameLayout {
        FrameLayout {
            width: 1920,
            height: 1080,
            subsampling: crate::Subsampling::Yuv420,
            depth: crate::Depth::Eight,
        }
    }

    #[test]
    fn the_same_settings_give_the_same_key() {
        let first_options = options();
        let first_layout = layout();
        let first = kernel_key(&first_options, first_layout);

        let second_options = options();
        let second_layout = layout();
        let second = kernel_key(&second_options, second_layout);

        assert_eq!(first, second);
    }

    #[test]
    fn a_different_depth_gives_a_different_key() {
        let ten_bit = FrameLayout {
            depth: crate::Depth::Ten,
            ..layout()
        };

        let test_options = options();
        let eight_bit = layout();
        let eight_bit_key = kernel_key(&test_options, eight_bit);
        let ten_bit_key = kernel_key(&test_options, ten_bit);

        assert_ne!(eight_bit_key, ten_bit_key);
    }

    #[test]
    fn a_different_radius_gives_a_different_key() {
        let wider = PlaneOptions {
            mode: crate::DenoisingMode::Temporal { radius: 3 },
            ..options()
        };

        let default_options = options();
        let test_layout = layout();
        let default_key = kernel_key(&default_options, test_layout);
        let wider_key = kernel_key(&wider, test_layout);

        assert_ne!(default_key, wider_key);
    }

    #[test]
    fn grain_export_gives_a_different_key() {
        let exporting_options = crate::Nl4dOptions {
            grain_export: true,
            ..crate::Nl4dOptions::default()
        };
        let exporting = PlaneOptions {
            algorithm: crate::Algorithm::Nl4d(exporting_options),
            ..options()
        };

        let plain_options = crate::Nl4dOptions::default();
        let plain = PlaneOptions {
            algorithm: crate::Algorithm::Nl4d(plain_options),
            ..options()
        };

        let test_layout = layout();
        let plain_key = kernel_key(&plain, test_layout);
        let exporting_key = kernel_key(&exporting, test_layout);

        assert_ne!(plain_key, exporting_key);
    }

    #[test]
    fn different_kernels_do_not_wait_for_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let first_key = key(4);
        let second_key = key(5);

        let first = WarmUp::begin_in(dir.path(), first_key, BRIEFLY);
        let second = WarmUp::begin_in(dir.path(), second_key, BRIEFLY);

        assert!(
            first.is_some() && second.is_some(),
            "separate keys queue separately"
        );
    }
}
