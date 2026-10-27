use std::path::PathBuf;

use av_denoise_core::PrivateKernelCache;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

const CACHE_DIR_NAME: &str = "bench-kernel-cache";

/// A hash of the crate's sources and benches, which names this build's cache directory.
const BENCH_KERNEL_HASH: &str = env!("AV_DENOISE_BENCH_KERNEL_HASH");

/// Points CubeCL's compilation cache at a directory every run of this build shares.
///
/// A rerun of the same build reuses the kernels an earlier run compiled. The directory is named by a
/// hash of the crate's sources and benches, so each build has its own. The run works on a private
/// copy of the shared cache and merges its new kernels back when it exits. It panics if CubeCL has
/// already read its config.
pub fn install_bench_compilation_cache() {
    let cache_dir = cache_dir();
    let _ = std::fs::create_dir_all(&cache_dir);

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    let private_cache = PrivateKernelCache::configure(&cache_dir, &mut config);
    CubeClRuntimeConfig::set(config);

    private_cache.start();
    av_denoise_core::finish_kernel_cache_at_exit();
}

fn cache_dir() -> PathBuf {
    let target_dir = match option_env!("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target"),
    };

    target_dir.join(CACHE_DIR_NAME).join(BENCH_KERNEL_HASH)
}
