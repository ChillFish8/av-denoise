use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::Context;
use etcetera::BaseStrategy;

#[cfg(windows)]
const LIBDEVICE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/nvidia/windows/libdevice.10.bc"
));

#[cfg(not(windows))]
const LIBDEVICE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/nvidia/linux/libdevice.10.bc"
));

const FILE_NAME: &str = "libdevice.10.bc";

/// The temporary file name for this process, so concurrent installs never share one.
fn temp_name() -> String {
    format!("{FILE_NAME}.{}.partial", std::process::id())
}

/// The directory the bundled libdevice is installed into, used as a `CUDA_PATH`.
fn bundled_root(cache_dir: &Path) -> PathBuf {
    cache_dir.join("av-denoise").join("cuda")
}

/// Writes libdevice under `root/nvvm/libdevice`, unless a copy of the right size is there.
///
/// The file is written to a temporary name first, so an interrupted write never leaves a
/// truncated copy under the real name.
fn install_libdevice(root: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let directory = root.join("nvvm").join("libdevice");
    let target = directory.join(FILE_NAME);

    let existing = std::fs::metadata(&target);
    if existing.is_ok_and(|metadata| metadata.len() == bytes.len() as u64) {
        return Ok(());
    }

    std::fs::create_dir_all(&directory)?;

    let partial = directory.join(temp_name());
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, &target)
}

/// Whether the bundled libdevice is needed, which is when the user has not set `CUDA_PATH`.
fn should_install(cuda_path: Option<&OsStr>) -> bool {
    cuda_path.is_none()
}

/// Installs the bundled libdevice into the cache and points `CUDA_PATH` at it.
///
/// A `CUDA_PATH` that is already set is left alone.
///
/// # Safety
///
/// The same safety rules as [std::env::set_var] apply.
pub unsafe fn point_cuda_path_at_bundle() -> anyhow::Result<()> {
    let cuda_path = std::env::var_os("CUDA_PATH");
    if !should_install(cuda_path.as_deref()) {
        return Ok(());
    }

    let strategy = etcetera::choose_base_strategy().context("no cache directory")?;
    let cache_dir = strategy.cache_dir();
    let root = bundled_root(cache_dir.as_path());
    install_libdevice(&root, LIBDEVICE).context("cannot write libdevice")?;

    // SAFETY: forwarded from this function's own precondition.
    unsafe { std::env::set_var("CUDA_PATH", root) };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundle_lives_under_av_denoise_cuda() {
        let root = bundled_root(Path::new("/cache"));
        assert_eq!(root, Path::new("/cache/av-denoise/cuda"));
    }

    #[test]
    fn install_writes_under_nvvm_libdevice() {
        let dir = tempfile::tempdir().unwrap();
        install_libdevice(dir.path(), b"bitcode").unwrap();

        let file = dir.path().join("nvvm/libdevice/libdevice.10.bc");
        let written = std::fs::read(file).unwrap();
        assert_eq!(written, b"bitcode");
    }

    // Same length, different bytes. The size check treats it as complete.
    #[test]
    fn a_complete_copy_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        install_libdevice(dir.path(), b"bitcode").unwrap();
        install_libdevice(dir.path(), b"BITCODE").unwrap();

        let file = dir.path().join("nvvm/libdevice/libdevice.10.bc");
        let kept = std::fs::read(file).unwrap();
        assert_eq!(kept, b"bitcode");
    }

    #[test]
    fn a_truncated_copy_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        install_libdevice(dir.path(), b"bit").unwrap();
        install_libdevice(dir.path(), b"bitcode").unwrap();

        let file = dir.path().join("nvvm/libdevice/libdevice.10.bc");
        let replaced = std::fs::read(file).unwrap();
        assert_eq!(replaced, b"bitcode");
    }

    #[test]
    fn a_stale_temp_file_does_not_block_install() {
        let dir = tempfile::tempdir().unwrap();
        let libdevice_dir = dir.path().join("nvvm/libdevice");
        std::fs::create_dir_all(&libdevice_dir).unwrap();
        std::fs::write(libdevice_dir.join("libdevice.10.bc.0.partial"), b"half").unwrap();
        std::fs::write(libdevice_dir.join(temp_name()), b"half").unwrap();
        install_libdevice(dir.path(), b"bitcode").unwrap();

        let written = std::fs::read(libdevice_dir.join("libdevice.10.bc")).unwrap();
        assert_eq!(written, b"bitcode");
    }

    #[test]
    fn a_copy_from_a_concurrent_writer_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let libdevice_dir = dir.path().join("nvvm/libdevice");
        std::fs::create_dir_all(&libdevice_dir).unwrap();
        std::fs::write(libdevice_dir.join("libdevice.10.bc"), b"bitcode").unwrap();

        let outcome = install_libdevice(dir.path(), b"bitcode");
        assert!(outcome.is_ok());
    }

    #[test]
    fn a_set_cuda_path_is_kept() {
        assert!(!should_install(Some(OsStr::new("/opt/my-toolkit"))));
        assert!(should_install(None));
    }
}
