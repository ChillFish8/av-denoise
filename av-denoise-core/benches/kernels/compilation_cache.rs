use std::path::PathBuf;

use av_denoise_core::KERNEL_HASH;
use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

const CACHE_DIR_NAME: &str = "bench-kernel-cache";

/// Points CubeCL's compilation cache at a directory every run of this build shares.
///
/// A rerun of the same build reuses the kernels an earlier run compiled. CubeCL keys cached kernels
/// by their signature rather than their body, so the directory is named by
/// [KERNEL_HASH](av_denoise_core::KERNEL_HASH). It runs before the first client is made.
pub fn install_bench_compilation_cache() {
    let cache_dir = cache_dir();
    let _ = std::fs::create_dir_all(&cache_dir);

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = Some(CacheConfig::File(cache_dir));
    CubeClRuntimeConfig::set(config);
}

fn cache_dir() -> PathBuf {
    let target_dir = match option_env!("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target"),
    };

    target_dir.join(CACHE_DIR_NAME).join(KERNEL_HASH)
}
