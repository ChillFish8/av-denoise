use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, RandomState};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, Once, PoisonError};
use std::time::{Duration, SystemTime};

use cubecl::config::CubeClRuntimeConfig;
use cubecl::config::cache::CacheConfig;
use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{Access, Database, Origin};

/// The lock file that guards the shared cache file.
const MERGE_LOCK_NAME: &str = "merge.lock";

/// The directory next to the shared cache file that holds each process's private file.
const PRIVATE_DIR_NAME: &str = "private";

/// Namespaces that describe how one process built its cache, rather than holding cached kernels.
const SKIPPED_PREFIX: &str = "records/";

/// The files the database engine writes next to a database file.
const SIDECARS: [&str; 3] = ["-wal", "-tshm", "-shm"];

/// How long an unlocked private file must sit unused before another process deletes it.
///
/// The lock a running process holds is what keeps its file safe. The age only covers the moment
/// between a new process creating its lock file and locking it.
const STALE_PRIVATE_AGE: Duration = Duration::from_secs(5 * 60);

/// The private file this process works on, once [PrivateKernelCache::start] has filled it.
static ACTIVE: Mutex<Option<ActiveCache>> = Mutex::new(None);

/// Set once a failure to take the shared cache's lock has been logged as a warning.
static LOCK_WARNED: AtomicBool = AtomicBool::new(false);

type EntryKey = (String, Vec<u8>);

/// A private copy of a shared kernel cache, used by one process only.
///
/// The database engine CubeCL caches kernels in crashes and corrupts its file when several processes
/// write to one file at once. So each process points CubeCL at a private file of its own. The
/// private file starts as a copy of the shared file, and the kernels the process compiles are merged
/// back into the shared file at [merge_kernel_cache] and [finish_kernel_cache].
///
/// A lock file next to the shared file guards it. Copies take it shared, so processes can start in
/// parallel. Merges, refills and checkpoints take it exclusively. Each closes the shared file before
/// it lets the lock go. When the lock cannot be taken, the shared file is left alone, and the process
/// starts cold or keeps its kernels private.
///
/// Starting a process copies the whole shared file. File systems that clone files, such as XFS,
/// btrfs and APFS, make that copy almost free. Elsewhere it reads and writes every byte, so it takes
/// time in proportion to the size of the cache. A merge that has new kernels reads the whole private
/// file, which also takes time in proportion to the size of the cache.
#[derive(Debug)]
pub struct PrivateKernelCache {
    shared: PathBuf,
    private_dir: PathBuf,
    name: String,
}

impl PrivateKernelCache {
    /// Points `config` at a new private file for this process, next to the shared cache in `dir`.
    ///
    /// The shared file keeps the environment name `config` already had. Call [start](Self::start)
    /// once `config` is installed, before any CubeCL store or client opens the private file.
    pub fn configure(dir: &Path, config: &mut CubeClRuntimeConfig) -> Self {
        let shared_file_name = cubecl::environment::file_name(&config.environment.name);
        let shared = dir.join(shared_file_name);
        let private_dir = dir.join(PRIVATE_DIR_NAME);
        let name = private_name();

        config.environment.path = CacheConfig::Directory(private_dir.clone());
        config.environment.name = name.clone();

        Self {
            shared,
            private_dir,
            name,
        }
    }

    fn private_file(&self) -> PathBuf {
        let file_name = cubecl::environment::file_name(&self.name);
        self.private_dir.join(file_name)
    }

    /// Fills the private file from the shared cache and makes it the one later merges read.
    ///
    /// It must run before any CubeCL store or client opens the private file, because the fill writes
    /// the file in place. It also deletes private files that exited processes left behind. Every
    /// failure is logged and the process carries on, with a cold private file at worst.
    pub fn start(self) {
        if let Err(err) = std::fs::create_dir_all(&self.private_dir) {
            tracing::warn!(dir = ?self.private_dir, %err, "cannot create the private kernel cache directory");
        }

        let liveness_path = self.private_dir.join(lock_file_name(&self.name));
        let liveness = hold_liveness_lock(&liveness_path);
        prune_stale_private_files(&self.private_dir, &self.name, SystemTime::now());

        let private = self.private_file();
        let merge_lock = merge_lock_path(&self.shared);
        let filled = guarded(|| fill_private_file(&self.shared, &private, &merge_lock));
        let shared_seen = filled.flatten();

        let written = write_state(&private);
        let active = ActiveCache {
            shared: self.shared,
            private,
            merge_lock,
            liveness_path,
            liveness,
            shared_seen,
            written,
            merged: HashSet::new(),
        };

        let mut slot = active_slot();
        *slot = Some(active);
    }
}

/// Copies the kernels this process compiled since the last merge into the shared cache.
///
/// Does nothing when no private file is in use or nothing was written since the last merge. A failed
/// merge is logged and the process carries on, because the cache only saves compile time.
pub fn merge_kernel_cache() {
    let mut slot = active_slot();
    if let Some(active) = slot.as_mut() {
        guarded(|| active.merge());
    }
}

/// Copies into the private file what other processes merged into the shared cache since this process
/// last read it.
///
/// Does nothing when the shared file has not changed, or when no private file is in use.
pub fn refill_kernel_cache() {
    let mut slot = active_slot();
    if let Some(active) = slot.as_mut() {
        guarded(|| active.refill_if_changed());
    }
}

/// Merges a last time and deletes this process's private file.
///
/// Later merges do nothing. A private file that cannot be deleted while it is open is left for a
/// later process to delete once this one has exited.
pub fn finish_kernel_cache() {
    let taken = active_slot().take();
    let Some(mut active) = taken else {
        return;
    };

    guarded(|| active.merge());
    active.delete();
}

/// Runs [finish_kernel_cache] when the process exits normally.
///
/// It suits processes without a single place to finish from, such as test binaries.
pub fn finish_kernel_cache_at_exit() {
    static REGISTER: Once = Once::new();

    REGISTER.call_once(|| {
        // SAFETY: `finish_at_exit` takes no arguments, never unwinds, and stays valid for the life
        // of the process.
        let failed = unsafe { atexit(finish_at_exit) };
        if failed != 0 {
            tracing::warn!("cannot finish the kernel cache at exit, its kernels stay private");
        }
    });
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

/// Finishes the kernel cache on a new thread.
///
/// The exiting thread may already have dropped its thread locals, which the database engine needs.
extern "C" fn finish_at_exit() {
    let name = String::from("kernel-cache-exit");
    let spawned = std::thread::Builder::new().name(name).spawn(finish_kernel_cache);

    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
}

/// The slot holding the private file in use.
///
/// A panic is never let out while the slot is held, so a poisoned slot still holds a usable value.
fn active_slot() -> MutexGuard<'static, Option<ActiveCache>> {
    ACTIVE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs cache work, turning a panic into a logged failure.
///
/// The database engine panics on some corrupt files, and a cache that cannot be read is only a
/// slower start, never a reason to stop the process.
fn guarded<T>(work: impl FnOnce() -> T) -> Option<T> {
    let caught = std::panic::catch_unwind(AssertUnwindSafe(work));
    if caught.is_err() {
        tracing::warn!("the kernel cache work panicked, carrying on without it");
    }

    caught.ok()
}

/// The private file in use and what is known about its contents.
struct ActiveCache {
    shared: PathBuf,
    private: PathBuf,
    merge_lock: PathBuf,
    liveness_path: PathBuf,
    liveness: Option<File>,
    /// The shared file's modification time when this process last read or wrote it.
    shared_seen: Option<SystemTime>,
    /// The private file as of the last fill or merge.
    written: WriteState,
    /// Entries already merged, so a later merge skips them.
    merged: HashSet<EntryKey>,
}

impl ActiveCache {
    fn merge(&mut self) {
        let written = write_state(&self.private);
        if written == self.written {
            return;
        }

        let opened = Database::open_at(&self.private, Access::ReadWrite);
        let private = match opened {
            Ok(private) => private,
            Err(err) => {
                let path = &self.private;
                tracing::warn!(?path, %err, "cannot open the private kernel cache to merge it");
                return;
            },
        };

        let entries = new_entries(&private, &self.merged);
        drop(private);

        if !entries.is_empty() {
            let Some(lock) = lock_merge(&self.merge_lock, LockKind::Exclusive) else {
                return;
            };

            let merged = write_shared(&self.shared, &entries);
            self.shared_seen = modified(&self.shared);
            drop(lock);

            if !merged {
                return;
            }

            let keys = entries.into_iter().map(|(namespace, key, _)| (namespace, key));
            self.merged.extend(keys);
        }

        self.written = written;
    }

    fn refill_if_changed(&mut self) {
        let current = modified(&self.shared);
        if current.is_none() || current == self.shared_seen {
            return;
        }

        let Some(lock) = lock_merge(&self.merge_lock, LockKind::Exclusive) else {
            return;
        };

        copy_shared_entries(&self.shared, &self.private);
        self.shared_seen = modified(&self.shared);
        drop(lock);

        self.written = write_state(&self.private);
    }

    fn delete(self) {
        remove_database_files(&self.private);
        drop(self.liveness);

        let _ = std::fs::remove_file(&self.liveness_path);
    }
}

/// The size of a write-ahead log header, which the engine writes when it opens a file, before any
/// write.
const WAL_HEADER_BYTES: u64 = 32;

/// The size and modification time of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileState {
    len: u64,
    modified: Option<SystemTime>,
}

fn file_state(path: &Path) -> Option<FileState> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata.modified().ok();

    Some(FileState {
        len: metadata.len(),
        modified,
    })
}

/// A database file and its write-ahead log as they sit on disk, which change on every write.
///
/// A log holding only its header counts as no log, and only the database file's length is kept,
/// because opening a file rewrites its header. A process that only reads therefore never counts as
/// written. New entries either add log frames or, once checkpointed, lengthen the database file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriteState {
    database_len: Option<u64>,
    log: Option<FileState>,
}

fn write_state(database: &Path) -> WriteState {
    let log_path = sidecar(database, SIDECARS[0]);
    let log = file_state(&log_path).filter(|state| state.len > WAL_HEADER_BYTES);
    let database = file_state(database);
    let database_len = database.map(|state| state.len);

    WriteState { database_len, log }
}

fn has_log(database: &Path) -> bool {
    let state = write_state(database);
    state.log.is_some()
}

fn modified(path: &Path) -> Option<SystemTime> {
    let metadata = std::fs::metadata(path).ok()?;
    metadata.modified().ok()
}

/// A process id plus a random suffix, so no two live processes share a name.
fn private_name() -> String {
    let process = std::process::id();
    let now = SystemTime::now();
    let random = RandomState::new().hash_one(now);

    format!("{process}-{random:016x}")
}

fn lock_file_name(name: &str) -> String {
    format!("{name}.lock")
}

fn merge_lock_path(shared: &Path) -> PathBuf {
    let current_dir = Path::new(".");
    let dir = shared.parent().unwrap_or(current_dir);
    dir.join(MERGE_LOCK_NAME)
}

fn sidecar(database: &Path, suffix: &str) -> PathBuf {
    let mut path = database.as_os_str().to_owned();
    path.push(suffix);
    PathBuf::from(path)
}

/// Copies the shared file over the private one and returns the shared file's modification time.
///
/// The copy runs under the shared lock, so other processes can copy at the same time while no merge
/// or checkpoint can. A shared file with a write-ahead log is first checkpointed under the exclusive
/// lock, so the copy holds every entry without the log. Nothing is copied when the shared file does
/// not exist yet or the lock cannot be taken, and the private file then starts empty.
fn fill_private_file(shared: &Path, private: &Path, merge_lock: &Path) -> Option<SystemTime> {
    let lock = lock_merge(merge_lock, LockKind::Shared)?;
    if !shared.exists() {
        return None;
    }

    if !has_log(shared) {
        let seen = copy_shared_file(shared, private);
        drop(lock);
        return seen;
    }

    // A file lock cannot be raised from shared to exclusive in one step, so the log is checked again
    // once the exclusive lock is held.
    drop(lock);
    let lock = lock_merge(merge_lock, LockKind::Exclusive)?;
    if has_log(shared) && !make_standalone(shared) {
        return None;
    }

    let seen = copy_shared_file(shared, private);
    drop(lock);
    seen
}

fn copy_shared_file(shared: &Path, private: &Path) -> Option<SystemTime> {
    let seen = modified(shared);
    if let Err(err) = std::fs::copy(shared, private) {
        tracing::warn!(?shared, ?private, %err, "cannot copy the shared kernel cache, starting cold");
        remove_database_files(private);
        return None;
    }

    seen
}

/// Folds the shared file's write-ahead log into it and closes it, reporting whether that worked.
fn make_standalone(shared: &Path) -> bool {
    let database = match Database::open_at(shared, Access::ReadWrite) {
        Ok(database) => database,
        Err(err) => {
            tracing::warn!(?shared, %err, "cannot open the shared kernel cache");
            return false;
        },
    };

    match database.into_standalone() {
        Ok(()) => true,
        Err(err) => {
            tracing::warn!(?shared, %err, "cannot checkpoint the shared kernel cache");
            false
        },
    }
}

/// The entries this process wrote to `private` that are not in `merged` yet.
///
/// Entries copied from the shared file are marked imported, so only local ones are new.
fn new_entries(private: &Database, merged: &HashSet<EntryKey>) -> Vec<(String, Vec<u8>, Vec<u8>)> {
    let mut entries = Vec::new();

    for namespace in private.namespaces() {
        if namespace.starts_with(SKIPPED_PREFIX) {
            continue;
        }

        private.scan_with_origin(&namespace, &mut |key, value, origin| {
            if origin != Origin::Local {
                return;
            }

            let entry_key = (namespace.clone(), key.to_vec());
            if !merged.contains(&entry_key) {
                entries.push((namespace.clone(), key.to_vec(), value.to_vec()));
            }
        });
    }

    entries
}

/// Inserts `entries` into the shared file and closes it, reporting whether the merge landed.
///
/// They go in as imported entries, so a later fill marks them as not this process's own.
fn write_shared(shared: &Path, entries: &[(String, Vec<u8>, Vec<u8>)]) -> bool {
    let database = match Database::open_at(shared, Access::ReadWrite) {
        Ok(database) => database,
        Err(err) => {
            tracing::warn!(?shared, %err, "cannot open the shared kernel cache to merge into it");
            return false;
        },
    };

    let mut failed = 0;
    let namespaces: HashSet<&String> = entries.iter().map(|(namespace, _, _)| namespace).collect();
    for namespace in namespaces {
        let mut batch = entries
            .iter()
            .filter(|(entry_namespace, _, _)| entry_namespace == namespace)
            .map(|(_, key, value)| to_bytes_pair(key, value));
        let summary = database.insert_many(namespace, &mut batch, Origin::Imported);
        failed += summary.failed;
    }

    if failed > 0 {
        tracing::warn!(
            ?shared,
            failed,
            "some kernels could not be merged into the shared cache"
        );
    }

    if let Err(err) = database.into_standalone() {
        tracing::warn!(?shared, %err, "cannot checkpoint the shared kernel cache after a merge");
    }

    failed == 0
}

/// Inserts the shared file's entries that `private` lacks, without overwriting any.
fn copy_shared_entries(shared: &Path, private: &Path) {
    if !shared.exists() {
        return;
    }

    let source = match Database::open_at(shared, Access::ReadWrite) {
        Ok(source) => source,
        Err(err) => {
            tracing::warn!(?shared, %err, "cannot open the shared kernel cache to refill from it");
            return;
        },
    };

    let target = match Database::open_at(private, Access::ReadWrite) {
        Ok(target) => target,
        Err(err) => {
            tracing::warn!(?private, %err, "cannot open the private kernel cache to refill it");
            return;
        },
    };

    for namespace in source.namespaces() {
        if namespace.starts_with(SKIPPED_PREFIX) {
            continue;
        }

        let mut entries = Vec::new();
        source.scan(&namespace, &mut |key, value| {
            let pair = to_bytes_pair(key, value);
            entries.push(pair);
        });

        let mut batch = entries.into_iter();
        target.insert_many(&namespace, &mut batch, Origin::Imported);
    }

    drop(target);
    if let Err(err) = source.into_standalone() {
        tracing::warn!(?shared, %err, "cannot checkpoint the shared kernel cache after a refill");
    }
}

fn to_bytes_pair(key: &[u8], value: &[u8]) -> (Bytes, Bytes) {
    let key = Bytes::from_bytes_vec(key.to_vec());
    let value = Bytes::from_bytes_vec(value.to_vec());
    (key, value)
}

/// Deletes a database file and its sidecars, ignoring files that are missing or still open.
fn remove_database_files(database: &Path) {
    let _ = std::fs::remove_file(database);
    for suffix in SIDECARS {
        let path = sidecar(database, suffix);
        let _ = std::fs::remove_file(path);
    }
}

/// Takes the lock that marks a private file as owned by a running process.
///
/// The operating system releases it when the process exits, however it exits.
fn hold_liveness_lock(path: &Path) -> Option<File> {
    let file = open_lock_file(path)?;
    match file.lock() {
        Ok(()) => Some(file),
        Err(err) => {
            tracing::warn!(?path, %err, "cannot lock the private kernel cache");
            None
        },
    }
}

#[derive(Debug, Clone, Copy)]
enum LockKind {
    Shared,
    Exclusive,
}

/// The shared cache's lock, released when dropped.
struct MergeLock {
    file: File,
}

impl Drop for MergeLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Blocks until this process holds the shared cache's lock in `kind`.
///
/// Returns `None` when the lock cannot be taken, for example on a read-only or network file system,
/// and the caller then leaves the shared file alone.
fn lock_merge(path: &Path, kind: LockKind) -> Option<MergeLock> {
    let file = match open_merge_lock(path) {
        Ok(file) => file,
        Err(err) => {
            warn_lock_failure(path, &err);
            return None;
        },
    };

    let locked = match kind {
        LockKind::Shared => file.lock_shared(),
        LockKind::Exclusive => file.lock(),
    };

    if let Err(err) = locked {
        warn_lock_failure(path, &err);
        return None;
    }

    Some(MergeLock { file })
}

/// Opens the lock file, read-only when it already exists in a directory this process cannot write.
fn open_merge_lock(path: &Path) -> std::io::Result<File> {
    let writable = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);

    match writable {
        Ok(file) => Ok(file),
        Err(_) => File::open(path),
    }
}

/// Logs the first failure to take the shared cache's lock as a warning, and later ones quietly.
fn warn_lock_failure(path: &Path, err: &std::io::Error) {
    let warned = LOCK_WARNED.swap(true, Ordering::Relaxed);
    if warned {
        tracing::debug!(?path, %err, "cannot take the kernel cache lock");
        return;
    }

    tracing::warn!(?path, %err, "cannot take the kernel cache lock, the shared cache is left alone");
}

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
            tracing::warn!(?path, %err, "cannot open the kernel cache lock file");
            None
        },
    }
}

/// Deletes the private files in `dir` whose process has exited and that have gone unused for longer
/// than [STALE_PRIVATE_AGE].
///
/// A process holds a lock on its own private file for as long as it runs, so a file whose lock can
/// be taken has no running owner. Only files named like a private file are candidates.
fn prune_stale_private_files(dir: &Path, own_name: &str, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    let mut names = HashSet::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };

        let stem = file_name.split('.').next().unwrap_or_default();
        if stem != own_name && is_private_name(stem) {
            names.insert(stem.to_string());
        }
    }

    for name in names {
        prune_if_stale(dir, &name, now);
    }
}

fn prune_if_stale(dir: &Path, name: &str, now: SystemTime) {
    let lock_path = dir.join(lock_file_name(name));
    let lock = OpenOptions::new().read(true).write(true).open(&lock_path);
    let lock = match lock {
        Ok(lock) => Some(lock),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return,
    };

    let owned = lock.as_ref().is_some_and(|lock| lock.try_lock().is_err());
    if owned {
        return;
    }

    let file_name = cubecl::environment::file_name(name);
    let database = dir.join(file_name);
    let last_used = last_modified(&database, &lock_path);
    let age = now.duration_since(last_used).unwrap_or_default();
    if age <= STALE_PRIVATE_AGE {
        return;
    }

    remove_database_files(&database);
    drop(lock);

    let _ = std::fs::remove_file(lock_path);
}

/// The latest modification time among a private file, its sidecars and its lock file.
fn last_modified(database: &Path, lock_path: &Path) -> SystemTime {
    let sidecars = SIDECARS.iter().map(|suffix| sidecar(database, suffix));
    let paths = [database.to_path_buf(), lock_path.to_path_buf()]
        .into_iter()
        .chain(sidecars);

    paths
        .filter_map(|path| std::fs::metadata(path).ok())
        .filter_map(|metadata| metadata.modified().ok())
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Whether `name` looks like a [private_name], which is a process id, a dash and sixteen hex characters.
fn is_private_name(name: &str) -> bool {
    let Some((process, random)) = name.split_once('-') else {
        return false;
    };

    let is_process = !process.is_empty() && process.bytes().all(|byte| byte.is_ascii_digit());
    let is_random = random.len() == 16
        && random
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));

    is_process && is_random
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMESPACE: &str = "cubecl/test/kernels";

    fn insert(database: &Path, key: &str, origin: Origin) {
        let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
        let value = format!("value of {key}");
        opened.insert(NAMESPACE, key.as_bytes(), value.as_bytes(), origin);
    }

    /// Every entry of [NAMESPACE] in `database`, by key, with its origin.
    fn entries(database: &Path) -> Vec<(String, Origin)> {
        let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
        let mut found = Vec::new();
        opened.scan_with_origin(NAMESPACE, &mut |key, _, origin| {
            let key = String::from_utf8_lossy(key).to_string();
            found.push((key, origin));
        });

        found.sort_by(|left, right| left.0.cmp(&right.0));
        found
    }

    fn active_cache(dir: &Path) -> ActiveCache {
        let private_dir = dir.join(PRIVATE_DIR_NAME);
        std::fs::create_dir_all(&private_dir).expect("create the private dir");

        let name = private_name();
        let file_name = cubecl::environment::file_name(&name);
        let private = private_dir.join(file_name);
        let shared = dir.join("default.db");
        let merge_lock = merge_lock_path(&shared);
        let liveness_path = private_dir.join(lock_file_name(&name));
        let liveness = hold_liveness_lock(&liveness_path);
        let written = write_state(&private);

        ActiveCache {
            shared,
            private,
            merge_lock,
            liveness_path,
            liveness,
            shared_seen: None,
            written,
            merged: HashSet::new(),
        }
    }

    fn fill(cache: &mut ActiveCache) {
        cache.shared_seen = fill_private_file(&cache.shared, &cache.private, &cache.merge_lock);
        cache.written = write_state(&cache.private);
    }

    fn standalone(database: &Path) {
        let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
        opened.into_standalone().expect("checkpoint the database");
    }

    /// Fails when this process still has `database` or one of its sidecars open or mapped.
    #[cfg(target_os = "linux")]
    fn assert_closed(database: &Path) {
        let name = database.to_string_lossy().to_string();
        let descriptors = std::fs::read_dir("/proc/self/fd").unwrap();
        for descriptor in descriptors.flatten() {
            let Ok(target) = std::fs::read_link(descriptor.path()) else {
                continue;
            };

            let target = target.to_string_lossy().to_string();
            assert!(!target.starts_with(&name), "{target} is still open");
        }

        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains(&name), "{name} is still mapped");
    }

    #[test]
    fn a_fill_copies_the_shared_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "a", Origin::Imported);
        insert(&cache.shared, "b", Origin::Imported);
        standalone(&cache.shared);

        fill(&mut cache);

        let copied = entries(&cache.private);
        let expected = vec![
            ("a".to_string(), Origin::Imported),
            ("b".to_string(), Origin::Imported),
        ];
        assert_eq!(copied, expected);
        assert!(cache.shared_seen.is_some());
    }

    #[test]
    fn a_fill_checkpoints_a_shared_file_with_a_log_first() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "logged", Origin::Imported);
        assert!(
            has_log(&cache.shared),
            "the insert should leave a write-ahead log"
        );

        fill(&mut cache);

        let copied = entries(&cache.private);
        let expected = vec![("logged".to_string(), Origin::Imported)];
        assert_eq!(copied, expected);
        assert!(
            !has_log(&cache.shared),
            "the fill should checkpoint the shared file"
        );

        #[cfg(target_os = "linux")]
        assert_closed(&cache.shared);
    }

    #[test]
    fn a_fill_without_a_shared_file_leaves_the_private_file_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());

        fill(&mut cache);

        assert!(!cache.private.exists());
    }

    #[test]
    fn a_fill_without_the_lock_leaves_the_private_file_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "shared", Origin::Imported);
        standalone(&cache.shared);
        cache.merge_lock = dir.path().join("missing").join(MERGE_LOCK_NAME);

        fill(&mut cache);

        assert!(!cache.private.exists());
    }

    #[test]
    fn a_merge_copies_only_the_entries_this_process_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "shared", Origin::Imported);
        standalone(&cache.shared);
        fill(&mut cache);
        insert(&cache.private, "compiled", Origin::Local);

        cache.merge();

        let merged = entries(&cache.shared);
        let expected = vec![
            ("compiled".to_string(), Origin::Imported),
            ("shared".to_string(), Origin::Imported),
        ];
        assert_eq!(merged, expected);

        let merged_key = (NAMESPACE.to_string(), b"compiled".to_vec());
        assert!(cache.merged.contains(&merged_key));
        assert_eq!(cache.merged.len(), 1);
    }

    #[test]
    fn reading_the_private_file_leaves_it_unwritten() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "shared", Origin::Imported);
        standalone(&cache.shared);
        fill(&mut cache);

        let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
        let found = private.get(NAMESPACE, b"shared");
        drop(private);

        assert!(found.is_some());
        assert_eq!(write_state(&cache.private), cache.written);
    }

    #[test]
    fn writing_the_private_file_is_noticed() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "shared", Origin::Imported);
        standalone(&cache.shared);
        fill(&mut cache);

        insert(&cache.private, "compiled", Origin::Local);

        assert_ne!(write_state(&cache.private), cache.written);
    }

    #[test]
    fn a_merge_without_the_lock_leaves_the_shared_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.private, "compiled", Origin::Local);
        cache.merge_lock = dir.path().join("missing").join(MERGE_LOCK_NAME);

        cache.merge();

        assert!(!cache.shared.exists(), "nothing should have been merged");
        assert!(cache.merged.is_empty());
    }

    #[test]
    fn a_merge_skips_the_records_namespaces() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
        private.insert("records/v1/sessions", b"session", b"value", Origin::Local);
        drop(private);

        let found = {
            let private = Database::open_at(&cache.private, Access::ReadWrite).unwrap();
            new_entries(&private, &cache.merged)
        };

        cache.merge();

        assert!(found.is_empty(), "{found:?}");
        assert!(!cache.shared.exists(), "nothing should have been merged");
    }

    #[test]
    fn a_merged_shared_file_has_no_write_ahead_log_left() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.private, "compiled", Origin::Local);

        cache.merge();

        assert!(cache.shared.exists());
        assert!(!has_log(&cache.shared));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_shared_file_is_closed_after_a_merge_and_a_refill() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "shared", Origin::Imported);
        standalone(&cache.shared);
        insert(&cache.private, "compiled", Origin::Local);

        cache.merge();
        assert_closed(&cache.shared);

        cache.shared_seen = None;
        cache.refill_if_changed();
        assert_closed(&cache.shared);
    }

    #[test]
    fn a_refill_adds_what_others_merged_without_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "first", Origin::Imported);
        standalone(&cache.shared);
        fill(&mut cache);
        insert(&cache.private, "mine", Origin::Local);

        insert(&cache.shared, "second", Origin::Imported);
        standalone(&cache.shared);
        cache.refill_if_changed();

        let refilled = entries(&cache.private);
        let expected = vec![
            ("first".to_string(), Origin::Imported),
            ("mine".to_string(), Origin::Local),
            ("second".to_string(), Origin::Imported),
        ];
        assert_eq!(refilled, expected);
    }

    #[test]
    fn a_refill_skips_an_unchanged_shared_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = active_cache(dir.path());
        insert(&cache.shared, "first", Origin::Imported);
        standalone(&cache.shared);
        fill(&mut cache);
        let lock_dir = dir.path().join("missing");
        cache.merge_lock = lock_dir.join(MERGE_LOCK_NAME);

        cache.refill_if_changed();

        assert!(
            !LOCK_WARNED.load(Ordering::Relaxed),
            "an unchanged file needs no lock"
        );
    }

    #[test]
    fn a_panic_in_cache_work_is_caught_and_leaves_the_slot_usable() {
        let caught = guarded(|| panic!("a corrupt cache"));
        assert!(caught.is_none());

        let poisoner = std::thread::spawn(|| {
            let _slot = ACTIVE.lock();
            panic!("poison the slot");
        });
        let _ = poisoner.join();
        assert!(ACTIVE.is_poisoned());

        merge_kernel_cache();
        refill_kernel_cache();
        finish_kernel_cache();
    }

    #[test]
    fn deleting_removes_the_private_file_and_its_lock() {
        let dir = tempfile::tempdir().unwrap();
        let cache = active_cache(dir.path());
        insert(&cache.private, "compiled", Origin::Local);
        let private = cache.private.clone();
        let lock = cache.liveness_path.clone();

        cache.delete();

        assert!(!private.exists());
        assert!(!lock.exists());
    }

    #[test]
    fn a_private_name_is_recognised() {
        let name = private_name();

        assert!(is_private_name(&name), "{name}");
        for other in [
            "default",
            "123",
            "abc-0123456789abcdef",
            "123-0123456789ABCDEF",
            "123-0123",
        ] {
            assert!(!is_private_name(other), "{other}");
        }
    }

    fn make_private_files(dir: &Path, name: &str, modified: SystemTime) {
        let file_name = cubecl::environment::file_name(name);
        let database = dir.join(file_name);
        let wal = sidecar(&database, "-wal");
        let lock = dir.join(lock_file_name(name));
        for path in [database, wal, lock] {
            let file = File::create(&path).unwrap();
            file.set_modified(modified).unwrap();
        }
    }

    fn has_files(dir: &Path, name: &str) -> bool {
        let file_name = cubecl::environment::file_name(name);
        let database = dir.join(file_name);
        let lock = dir.join(lock_file_name(name));
        database.exists() || lock.exists()
    }

    #[test]
    fn pruning_removes_only_stale_files_of_exited_processes() {
        let dir = tempfile::tempdir().unwrap();
        let now = SystemTime::now();
        let long_ago = now - STALE_PRIVATE_AGE - Duration::from_secs(60);

        let own = "1-aaaaaaaaaaaaaaaa";
        let stale = "2-bbbbbbbbbbbbbbbb";
        let live = "3-cccccccccccccccc";
        let fresh = "4-dddddddddddddddd";
        for name in [own, stale, live] {
            make_private_files(dir.path(), name, long_ago);
        }

        make_private_files(dir.path(), fresh, now);

        let foreign = dir.path().join("default.db");
        let foreign_file = File::create(&foreign).unwrap();
        foreign_file.set_modified(long_ago).unwrap();

        let live_lock = dir.path().join(lock_file_name(live));
        let held = File::options().read(true).write(true).open(live_lock).unwrap();
        held.lock().unwrap();

        prune_stale_private_files(dir.path(), own, now);

        assert!(!has_files(dir.path(), stale), "stale files should be removed");
        assert!(has_files(dir.path(), own), "this process's files should be kept");
        assert!(
            has_files(dir.path(), live),
            "a running process's files should be kept"
        );
        assert!(has_files(dir.path(), fresh), "fresh files should be kept");
        assert!(
            foreign.exists(),
            "files not named like a private file should be kept"
        );
    }

    #[test]
    fn fills_hold_the_lock_together_and_merges_wait_for_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(MERGE_LOCK_NAME);

        let first = lock_merge(&path, LockKind::Shared).expect("a first reader");
        let second = lock_merge(&path, LockKind::Shared).expect("a second reader");

        let writer = open_merge_lock(&path).unwrap();
        let blocked = writer.try_lock();
        assert!(blocked.is_err(), "a writer must wait for the readers");

        drop(first);
        drop(second);
        let free = writer.try_lock();
        assert!(free.is_ok(), "a writer gets the lock once the readers are gone");
    }
}
