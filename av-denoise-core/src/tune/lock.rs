use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::time::{Duration, Instant};

use super::config::{CacheHook, CachePhase};

/// The lock file that keeps tuning to one process at a time.
const TUNE_LOCK_NAME: &str = "tune.lock";

/// How long a process waits for another to finish tuning before running with stored results.
///
/// Explicit tuning can take ten minutes, so the wait is generous.
const TUNE_WAIT_LIMIT: Duration = Duration::from_secs(15 * 60);

/// How often a waiting process checks whether the lock came free.
pub const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The one place on this machine to tune kernels at a time.
///
/// Tuning times kernels, so two processes tuning at once would measure each other. The lock is
/// released when this is dropped.
#[derive(Debug)]
pub struct TuneLock {
    lock: File,
}

/// What asking for the tune lock gave.
#[derive(Debug)]
pub enum TuneLockOutcome {
    Held(TuneLock),
    /// There is no store or its lock file cannot be opened, so the caller tunes without a lock.
    Unlocked,
    /// Another process held the lock for the whole wait, so the caller should use stored results.
    TimedOut,
}

impl TuneLock {
    /// Waits for the tune lock in the `store` directory, creating the directory if it is missing.
    ///
    /// Once the lock is held, `hook` is told [CachePhase::LockTaken].
    pub fn acquire(store: Option<&Path>, hook: Option<&CacheHook>) -> TuneLockOutcome {
        let Some(dir) = store else {
            return TuneLockOutcome::Unlocked;
        };

        Self::acquire_in(dir, TUNE_WAIT_LIMIT, hook)
    }

    pub(crate) fn acquire_in(dir: &Path, wait_limit: Duration, hook: Option<&CacheHook>) -> TuneLockOutcome {
        if let Err(error) = std::fs::create_dir_all(dir) {
            tracing::debug!(?dir, %error, "cannot create the tuning store, carrying on without a lock");
            return TuneLockOutcome::Unlocked;
        }

        let path = dir.join(TUNE_LOCK_NAME);
        let Some(lock) = open_lock_file(&path) else {
            return TuneLockOutcome::Unlocked;
        };

        if !wait_for_lock(&lock, wait_limit) {
            return TuneLockOutcome::TimedOut;
        }

        // A process that tuned before this one merged its kernels before it let the lock go.
        if let Some(hook) = hook {
            hook(CachePhase::LockTaken);
        }

        TuneLockOutcome::Held(TuneLock { lock })
    }
}

impl Drop for TuneLock {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
    }
}

/// Opens the lock file, creating it when this is the first process to ask.
///
/// A file that cannot be opened is logged and then ignored, because tuning works without the queue.
pub fn open_lock_file(path: &Path) -> Option<File> {
    let opened = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);

    match opened {
        Ok(file) => Some(file),
        Err(error) => {
            tracing::debug!(?path, %error, "cannot open the lock file, carrying on without it");
            None
        },
    }
}

/// Blocks until the lock is held, giving up after `wait_limit`.
///
/// Returns whether the lock is held. It is an advisory file lock, so the operating system releases it
/// when the holder dies and the next process in line wakes up straight away.
pub fn wait_for_lock(lock: &File, wait_limit: Duration) -> bool {
    let start = Instant::now();

    loop {
        match lock.try_lock() {
            Ok(()) => return true,
            Err(TryLockError::WouldBlock) => {},
            Err(TryLockError::Error(error)) => {
                tracing::debug!(%error, "cannot take the lock file, carrying on without it");
                return false;
            },
        }

        if start.elapsed() >= wait_limit {
            tracing::warn!(
                "waited {wait_limit:?} for another process to release the lock file, carrying on without it",
            );
            return false;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}
