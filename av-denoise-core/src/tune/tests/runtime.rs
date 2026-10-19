use cubecl::prelude::*;

use crate::collab::needs_warp_uniform_search;
use crate::tune::{DeviceLimits, Legality};

/// The runtime GPU tests in this module run on, picked from the enabled backend.
#[cfg(feature = "cuda")]
pub(crate) type TestRuntime = cubecl::cuda::CudaRuntime;

/// The runtime GPU tests in this module run on, picked from the enabled backend.
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub(crate) type TestRuntime = cubecl::hip::HipRuntime;

/// Keeps the items it wraps only when the tests run on the wgpu runtime.
macro_rules! wgpu_only {
    ($($item:item)*) => {
        $(
            #[cfg(all(
                any(feature = "vulkan", feature = "metal"),
                not(any(feature = "cuda", feature = "rocm"))
            ))]
            $item
        )*
    };
}

pub(crate) use wgpu_only;

wgpu_only! {
    /// The runtime GPU tests in this module run on, picked from the enabled backend.
    pub(crate) type TestRuntime = cubecl::wgpu::WgpuRuntime;
}

wgpu_only! {
    /// The adapter the test client runs on, with the instance it was found on.
    ///
    /// Tests pin their device with `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'`, so that form picks
    /// the `N`-th discrete adapter. Without it, the high-performance adapter is used, as CubeCL does.
    pub(crate) fn test_adapter() -> (wgpu::Instance, wgpu::Adapter) {
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        };
        let instance = wgpu::Instance::new(descriptor);
        let pinned = std::env::var("CUBECL_WGPU_DEFAULT_DEVICE").ok();
        let index = pinned
            .as_deref()
            .and_then(|text| text.strip_prefix("DiscreteGpu("))
            .and_then(|rest| rest.strip_suffix(')'))
            .and_then(|digits| digits.parse::<usize>().ok());

        let adapter = match index {
            Some(index) => {
                let listing = instance.enumerate_adapters(wgpu::Backends::VULKAN);
                let adapters = cubecl::future::block_on(listing);
                adapters
                    .into_iter()
                    .filter(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
                    .nth(index)
                    .expect("the pinned discrete adapter exists")
            },
            None => {
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                };
                let request = instance.request_adapter(&options);
                let requested = cubecl::future::block_on(request);
                requested.expect("a high-performance adapter exists")
            },
        };

        (instance, adapter)
    }
}

pub(crate) fn test_client() -> ComputeClient<TestRuntime> {
    crate::tests::compilation_cache::install_test_compilation_cache();
    let device = <TestRuntime as Runtime>::Device::default();
    TestRuntime::client(&device)
}

/// Every rule relaxed except what the test backend needs for correct output.
///
/// Equivalence tests run every candidate the device can execute, including those a real device
/// class is not offered.
pub(crate) fn test_legality(client: &ComputeClient<TestRuntime>) -> Legality {
    let limits = DeviceLimits::from_client(client);
    let on_cuda = needs_warp_uniform_search(client);

    Legality {
        uniform_walk_only: on_cuda,
        f16_search: true,
        early_exit: true,
        packed_atomics: limits.int64_atomic_add,
    }
}
