use std::path::PathBuf;
use std::sync::Once;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

use crate::KERNEL_HASH;

const CACHE_DIR_NAME: &str = "test-kernel-cache";

/// Points CubeCL's compilation cache at a directory every test process of this build shares.
///
/// Each test runs in its own process, so without it every process recompiles every kernel it
/// launches. The directory is named by [KERNEL_HASH](crate::KERNEL_HASH), so each build has its
/// own. It runs once per process, before the first client is made.
pub(crate) fn install_test_compilation_cache() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(install);
}

fn install() {
    let cache_dir = cache_dir();
    let _ = std::fs::create_dir_all(&cache_dir);

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(cache_dir);

    // `RuntimeConfig::set` panics once the config has been read. A test that reaches CubeCL
    // another way first keeps compiling without the cache, which is slower but still correct.
    let set = std::panic::AssertUnwindSafe(|| {
        CubeClRuntimeConfig::set(config);
    });
    let _ = std::panic::catch_unwind(set);
}

fn cache_dir() -> PathBuf {
    let target_dir = match option_env!("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target"),
    };

    target_dir.join(CACHE_DIR_NAME).join(KERNEL_HASH)
}
