use std::fs;
use std::path::{Path, PathBuf};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hashes every source file so the kernel cache is keyed by the exact
/// kernel sources a build contains.
fn main() {
    let source_root = Path::new("src");

    let mut files = Vec::new();
    collect_files(source_root, &mut files);

    let mut relative_paths: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| (relative_path(source_root, &path), path))
        .collect();
    relative_paths.sort();

    let mut hash = FNV_OFFSET_BASIS;
    for (relative, path) in relative_paths {
        let contents = fs::read(&path).expect("source file is readable");
        let length = contents.len() as u64;

        hash = fnv1a(hash, relative.as_bytes());
        hash = fnv1a(hash, &[0]);
        hash = fnv1a(hash, &length.to_le_bytes());
        hash = fnv1a(hash, &contents);
    }

    println!("cargo:rustc-env=AV_DENOISE_KERNEL_HASH={hash:016x}");
    println!("cargo:rerun-if-changed=src");
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

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
