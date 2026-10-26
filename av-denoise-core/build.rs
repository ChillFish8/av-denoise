#[path = "src/kernel_hash.rs"]
mod kernel_hash;

use std::fs;
use std::path::{Path, PathBuf};

use self::kernel_hash::{FNV_OFFSET_BASIS, fnv1a};

/// The workspace lockfile, which pins the CubeCL version a build links.
const LOCKFILE: &str = "../Cargo.lock";
/// Stands in for the CubeCL version when the crate builds outside its workspace.
const UNLOCKED_VERSION: &str = "unlocked";

/// Hashes every source file and the CubeCL version, so the kernel cache is keyed by the exact
/// kernel sources and compiler a build contains.
///
/// The bench hash also covers `benches`, which defines kernels of its own.
fn main() {
    let source_root = Path::new("src");
    let source_hash = hash_tree(source_root, FNV_OFFSET_BASIS);
    let cubecl_version = locked_cubecl_version();
    let kernel_hash = kernel_hash::kernel_hash(source_hash, &cubecl_version);
    println!("cargo:rustc-env=AV_DENOISE_KERNEL_HASH={kernel_hash:016x}");
    println!("cargo:rerun-if-changed=src");
    if Path::new(LOCKFILE).is_file() {
        println!("cargo:rerun-if-changed={LOCKFILE}");
    }

    let bench_root = Path::new("benches");
    if bench_root.is_dir() {
        let bench_hash = hash_tree(bench_root, kernel_hash);
        println!("cargo:rustc-env=AV_DENOISE_BENCH_KERNEL_HASH={bench_hash:016x}");
        println!("cargo:rerun-if-changed=benches");
    }
}

/// Every `cubecl` version the workspace lockfile pins, joined with `,`.
///
/// A packaged build has no workspace lockfile, so it falls back to [UNLOCKED_VERSION], as does a
/// lockfile with no `cubecl` entry.
fn locked_cubecl_version() -> String {
    let Ok(lockfile) = fs::read_to_string(LOCKFILE) else {
        return UNLOCKED_VERSION.to_string();
    };

    let mut versions = Vec::new();
    let mut lines = lockfile.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "name = \"cubecl\"" {
            continue;
        }

        let version_line = lines.next().unwrap_or_default();
        let version = version_line
            .trim()
            .strip_prefix("version = \"")
            .and_then(|rest| rest.strip_suffix('"'));
        if let Some(version) = version {
            versions.push(version);
        }
    }

    if versions.is_empty() {
        return UNLOCKED_VERSION.to_string();
    }

    versions.join(",")
}

/// Folds every file below `root` into `hash`, in path order.
fn hash_tree(root: &Path, mut hash: u64) -> u64 {
    let mut files = Vec::new();
    collect_files(root, &mut files);

    let mut relative_paths: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| (relative_path(root, &path), path))
        .collect();
    relative_paths.sort();

    for (relative, path) in relative_paths {
        let contents = fs::read(&path).expect("source file is readable");
        let length = contents.len() as u64;

        hash = fnv1a(hash, relative.as_bytes());
        hash = fnv1a(hash, &[0]);
        hash = fnv1a(hash, &length.to_le_bytes());
        hash = fnv1a(hash, &contents);
    }

    hash
}

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).expect("source directory is readable");
    for entry in entries {
        let path = entry.expect("source directory entry is readable").path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

/// The path of `path` below `root`, joined with `/` on every platform.
fn relative_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).expect("path is inside the source root");
    let components: Vec<String> = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();
    components.join("/")
}
