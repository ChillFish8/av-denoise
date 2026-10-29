//! Advisory lock files that queue processes on this machine
//!
//! - [open_lock_file] opens a lock file, creating it when missing
//! - [wait_for_lock] blocks until the lock is held or a wait limit runs out

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::time::{Duration, Instant};

/// How often a waiting process checks whether the lock came free.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Opens the lock file, creating it when this is the first process to ask.
///
/// A file that cannot be opened is logged as a warning and gives `None`.
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
            tracing::warn!(?path, %error, "cannot open the lock file, carrying on without it");
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
