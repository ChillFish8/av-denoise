use std::path::Path;
use std::time::Duration;

use cubecl_environment::persistence::{Access, Database};

use crate::lockfile::{open_lock_file, wait_for_lock};

/// How long a process waits for another to finish the first open before carrying on without it.
const FIRST_OPEN_WAIT: Duration = Duration::from_secs(30);

/// Opens the shared kernel cache once, one process at a time, before CubeCL does.
///
/// A new cache file is switched to write-ahead logging by whichever process opens it first. Two
/// processes doing that together can make the loser give up on the cache for its whole lifetime.
/// Opening it here under a lock leaves the file ready, so later opens do not race.
///
/// This is best effort. Any failure is logged and the caller carries on.
pub fn prepare_kernel_cache(database: &Path) {
    let Some(directory) = database.parent() else {
        return;
    };

    let lock_path = directory.join("first-open.lock");
    let Some(lock) = open_lock_file(&lock_path) else {
        return;
    };

    if !wait_for_lock(&lock, FIRST_OPEN_WAIT) {
        return;
    }

    match Database::open_at(database, Access::ReadWrite) {
        Ok(handle) => drop(handle),
        Err(error) => {
            tracing::warn!(path = ?database, %error, "cannot open the kernel cache");
        },
    }

    drop(lock);
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use cubecl_environment::persistence::Origin;

    use super::*;

    #[test]
    fn preparing_a_new_cache_leaves_it_in_wal_mode() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("default.db");

        prepare_kernel_cache(&database);

        let mut header = [0u8; 20];
        std::fs::File::open(&database)
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(header[18], 2);
        assert_eq!(header[19], 2);
    }

    #[test]
    fn preparing_an_existing_cache_keeps_its_entries() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("default.db");

        {
            let handle = Database::open_at(&database, Access::ReadWrite).unwrap();
            handle.insert("namespace", b"key", b"value", Origin::Local);
        }

        prepare_kernel_cache(&database);

        let handle = Database::open_at(&database, Access::ReadWrite).unwrap();
        let stored = handle.get("namespace", b"key").unwrap();
        assert_eq!(&stored[..], b"value");
    }

    #[test]
    fn preparing_without_a_writable_directory_returns_quietly() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");
        let database = missing.join("default.db");

        prepare_kernel_cache(&database);

        assert!(!missing.exists());
    }
}
