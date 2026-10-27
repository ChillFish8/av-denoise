use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::prelude::*;

#[cfg(feature = "vulkan")]
pub use self::vulkan_helpers::vulkan_compile_helpers;

/// Makes, or finds, the compile helpers the first time a batch asks for them.
pub type HelperFactory = Box<dyn FnOnce() -> Option<Arc<CompileHelpers>> + Send>;

/// Extra clients on the tuned device's adapter that compile candidates while the tuned client waits.
///
/// Each helper compiles on a device thread of its own and stores the kernel in CubeCL's on-disk
/// compilation cache. The tuned client then loads every kernel from that cache, which costs only
/// the driver's pipeline creation.
pub struct CompileHelpers {
    clients: Box<dyn Any + Send + Sync>,
    count: usize,
    failed: AtomicBool,
}

impl CompileHelpers {
    pub fn new(clients: Vec<Client>) -> Self {
        let count = clients.len();

        Self {
            clients: Box::new(clients),
            count,
            failed: AtomicBool::new(false),
        }
    }

    /// Whether the helpers failed candidates the tuned client then compiled, after which the set
    /// goes unused.
    pub fn has_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    pub(crate) fn mark_failed(&self) {
        self.failed.store(true, Ordering::Relaxed);
    }

    /// The helpers, when they run on `R`.
    pub(crate) fn clients(&self) -> &[Client] {
        let clients = self.clients.downcast_ref::<Vec<Client>>();

        match clients {
            Some(clients) => clients,
            None => &[],
        }
    }
}

impl fmt::Debug for CompileHelpers {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompileHelpers")
            .field("count", &self.count)
            .field("failed", &self.has_failed())
            .finish()
    }
}

/// Where a ledger gets its compile helpers.
///
/// The factory runs once, when the first batch asks, so a run that compiles nothing never makes a
/// helper device.
#[derive(Default)]
pub(crate) struct HelperSource {
    factory: Mutex<Option<HelperFactory>>,
    helpers: OnceLock<Option<Arc<CompileHelpers>>>,
}

impl HelperSource {
    /// Keeps `factory` unless one was set before.
    pub(crate) fn set_factory(&self, factory: HelperFactory) {
        let mut slot = self.factory.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() && self.helpers.get().is_none() {
            *slot = Some(factory);
        }
    }

    pub(crate) fn is_set(&self) -> bool {
        let factory = self.factory.lock().unwrap_or_else(PoisonError::into_inner);
        factory.is_some() || self.helpers.get().is_some()
    }

    /// The helpers that run on `R`, made on the first call, and empty when there are none or they
    /// have failed.
    pub(crate) fn clients(&self) -> &[Client] {
        let helpers = self.helpers.get_or_init(|| self.make());

        match helpers {
            Some(helpers) if !helpers.has_failed() => helpers.clients(),
            _ => &[],
        }
    }

    /// Stops using the helpers, and lets whoever shares them see that they failed.
    pub(crate) fn mark_failed(&self) {
        if let Some(Some(helpers)) = self.helpers.get() {
            helpers.mark_failed();
        }
    }

    fn make(&self) -> Option<Arc<CompileHelpers>> {
        let factory = self.factory.lock().unwrap_or_else(PoisonError::into_inner).take();
        factory.and_then(|make| make())
    }
}

impl fmt::Debug for HelperSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HelperSource")
            .field("helpers", &self.helpers.get())
            .finish()
    }
}

/// Whether CubeCL keeps compiled kernels on disk, which compiling on helpers needs.
pub fn compilation_cache_enabled() -> bool {
    let config = CubeClRuntimeConfig::get();
    config.compilation.cache
}

#[cfg(feature = "vulkan")]
mod vulkan_helpers {
    use cubecl::future::block_on;
    use cubecl::wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, init_device, vulkan};

    use super::CompileHelpers;

    /// `count` helpers on `adapter`, each with a Vulkan device of its own.
    ///
    /// Every device is requested before any helper is registered, so a failure partway leaves no
    /// helper behind. CubeCL never unregisters a client, so the helpers live until the process ends.
    pub fn vulkan_compile_helpers(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        count: usize,
    ) -> Option<CompileHelpers> {
        let mut devices = Vec::with_capacity(count);

        for _ in 0..count {
            let request = vulkan::request_vulkan_device(adapter);
            let device = block_on(request)?;
            devices.push(device);
        }

        let clients = devices
            .into_iter()
            .map(|(device, queue)| {
                let setup = WgpuSetup {
                    instance: instance.clone(),
                    adapter: adapter.clone(),
                    device,
                    queue,
                    backend: wgpu::Backend::Vulkan,
                };
                let options = RuntimeOptions::default();
                let helper: WgpuDevice = init_device(setup, options);
                let device = cubecl::Device::from(helper);
                device.client()
            })
            .collect();

        let helpers = CompileHelpers::new(clients);
        Some(helpers)
    }
}
