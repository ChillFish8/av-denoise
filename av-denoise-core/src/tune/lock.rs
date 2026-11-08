use std::fs::File;
use std::path::Path;
use std::time::Duration;

use crate::lockfile::{open_lock_file, wait_for_lock};

/// The lock file that keeps tuning to one process at a time.
const TUNE_LOCK_NAME: &str = "tune.lock";

/// How long a process waits for another to finish tuning before running with stored results.
///
/// Explicit tuning can take ten minutes, so the wait is generous.
const TUNE_WAIT_LIMIT: Duration = Duration::from_secs(15 * 60);

/// The one place on this machine to tune kernels at a time.
///
/// Tuning times kernels, so two processes tuning at once would measure each other. The lock is
/// released when this is dropped.
#[derive(Debug)]
pub(crate) struct TuneLock {
    lock: File,
}

/// What asking for the tune lock gave.
#[derive(Debug)]
pub(crate) enum TuneLockOutcome {
    Held(#[expect(dead_code, reason = "only dropping it is needed, which releases the lock")] TuneLock),
    /// There is no store or its lock file cannot be opened, so the caller tunes without a lock.
    Unlocked,
    /// Another process held the lock for the whole wait, so the caller should use stored results.
    TimedOut,
}

impl TuneLock {
    /// Waits for the tune lock in the `store` directory, creating the directory if it is missing.
    pub(crate) fn acquire(store: Option<&Path>) -> TuneLockOutcome {
        let Some(dir) = store else {
            return TuneLockOutcome::Unlocked;
        };

        Self::acquire_in(dir, TUNE_WAIT_LIMIT)
    }

    pub(crate) fn acquire_in(dir: &Path, wait_limit: Duration) -> TuneLockOutcome {
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

        TuneLockOutcome::Held(TuneLock { lock })
    }
}

impl Drop for TuneLock {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
    }
}
