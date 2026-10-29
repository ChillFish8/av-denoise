use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::tune::lock::{TuneLock, TuneLockOutcome, open_lock_file};
use crate::tune::{CacheHook, CachePhase};

const BRIEFLY: Duration = Duration::from_millis(50);

#[test]
fn the_lock_calls_the_hook_once_it_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let hook: CacheHook = Arc::new(move |phase| {
        assert_eq!(phase, CachePhase::LockTaken);
        counter.fetch_add(1, Ordering::SeqCst);
    });

    let outcome = TuneLock::acquire_in(dir.path(), BRIEFLY, Some(&hook));

    assert!(matches!(outcome, TuneLockOutcome::Held(_)), "{outcome:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn no_store_means_no_lock() {
    let outcome = TuneLock::acquire(None, None);

    assert!(matches!(outcome, TuneLockOutcome::Unlocked), "{outcome:?}");
}

#[test]
fn a_missing_store_directory_is_created() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("tune");

    let outcome = TuneLock::acquire(Some(&store), None);

    assert!(matches!(outcome, TuneLockOutcome::Held(_)), "{outcome:?}");
    assert!(store.join("tune.lock").exists());
}

#[test]
fn a_tune_lock_held_elsewhere_keeps_this_process_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tune.lock");
    let elsewhere = open_lock_file(&path).unwrap();
    elsewhere.lock().unwrap();

    let outcome = TuneLock::acquire_in(dir.path(), BRIEFLY, None);

    assert!(matches!(outcome, TuneLockOutcome::TimedOut), "{outcome:?}");
}

#[test]
fn a_free_tune_lock_is_taken_and_released() {
    let dir = tempfile::tempdir().unwrap();

    let first = TuneLock::acquire_in(dir.path(), BRIEFLY, None);
    assert!(matches!(first, TuneLockOutcome::Held(_)));
    drop(first);

    let second = TuneLock::acquire_in(dir.path(), BRIEFLY, None);
    assert!(matches!(second, TuneLockOutcome::Held(_)));
}
