use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::thread::available_parallelism;

use av_denoise_core::tune::{
    CompileHelpers,
    PRECOMPILE_THREADS,
    TuneLedger,
    compilation_cache_enabled,
    vulkan_compile_helpers,
};
use cubecl::wgpu::WgpuDevice;

use super::describe::find_adapter;

/// The compile helpers made so far in this process, one set per adapter.
///
/// CubeCL never frees a client, so every ledger on an adapter shares one set instead of making its
/// own. A set whose helpers all failed in a stage is replaced by a new one on the next request, and the
/// replaced set's devices stay alive until the process exits. A set that could not be made is
/// remembered as `None` and never tried again in this process.
static SHARED_HELPERS: LazyLock<Mutex<HashMap<String, Option<Arc<CompileHelpers>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Gives `ledger` a factory for Vulkan clients on `device`'s adapter that compile tuner candidates
/// in parallel.
///
/// Nothing is installed when the ledger already has a factory or the on-disk compilation cache is
/// off. The clients are made, or found, only when the first stage that can use them compiles.
pub(crate) fn install_compile_helpers(ledger: &TuneLedger, device: &WgpuDevice) {
    if ledger.has_compile_helpers() || !compilation_cache_enabled() || helpers_turned_off() {
        return;
    }

    let device = device.clone();
    let factory = Box::new(move || shared_helpers(&device));
    ledger.set_compile_helpers(factory);
}

/// This process's helpers for `device`'s adapter, made on the first call.
fn shared_helpers(device: &WgpuDevice) -> Option<Arc<CompileHelpers>> {
    let found = find_adapter(device, wgpu::Backend::Vulkan);
    let Some((instance, adapter)) = found else {
        tracing::debug!(
            ?device,
            "no compile helpers, so tuner candidates compile on the tuned client"
        );
        return None;
    };

    let info = adapter.get_info();
    let adapter_key = format!(
        "{}|{}|{:#06x}:{:#06x}",
        info.device_pci_bus_id, info.name, info.vendor, info.device
    );
    let mut shared = SHARED_HELPERS.lock().unwrap_or_else(PoisonError::into_inner);
    let previous = shared.get(&adapter_key);
    let failed = matches!(previous, Some(Some(helpers)) if helpers.has_failed());
    if failed {
        tracing::debug!("the compile helpers failed before, so a new set replaces them");
        shared.remove(&adapter_key);
    }

    let helpers = shared
        .entry(adapter_key)
        .or_insert_with(|| make_helpers(&instance, &adapter));

    helpers.clone()
}

/// [PRECOMPILE_THREADS] helpers, or fewer on a machine with fewer threads, and none below two.
fn make_helpers(instance: &wgpu::Instance, adapter: &wgpu::Adapter) -> Option<Arc<CompileHelpers>> {
    let threads = available_parallelism().map_or(1, NonZeroUsize::get);
    let count = threads.min(PRECOMPILE_THREADS);
    if count < 2 {
        tracing::debug!(threads, "too few threads for compile helpers");
        return None;
    }

    let made = vulkan_compile_helpers(instance, adapter, count);
    let Some(helpers) = made else {
        tracing::debug!(count, "cannot open the compile helpers' devices");
        return None;
    };

    tracing::debug!(count, "compile helpers ready");
    Some(Arc::new(helpers))
}

/// Set in a run of the library's unit tests to turn the compile helpers off, so a test can compare
/// with and without them.
#[cfg(test)]
const TEST_NO_HELPERS_ENV: &str = "AV_DENOISE_TEST_NO_COMPILE_HELPERS";

#[cfg(test)]
fn helpers_turned_off() -> bool {
    std::env::var_os(TEST_NO_HELPERS_ENV).is_some()
}

#[cfg(not(test))]
fn helpers_turned_off() -> bool {
    false
}
