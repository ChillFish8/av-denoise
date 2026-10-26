use cubecl::prelude::*;

use crate::collab::needs_warp_uniform_search;
use crate::tune::{DeviceLimits, Legality};

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
    /// The adapter the test client runs on, with the instance it was found on.
    ///
    /// `AVD_TEST_DEVICE` picks the adapter, so `discrete:N` is the `N`-th discrete adapter. With the
    /// `default` selector, `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'` still picks the `N`-th
    /// discrete adapter. Otherwise the high-performance adapter is used, as CubeCL does.
    pub(crate) fn test_adapter() -> (wgpu::Instance, wgpu::Adapter) {
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        };
        let instance = wgpu::Instance::new(descriptor);

        let adapter = match pinned_adapter() {
            Some((kind, index)) => {
                let listing = instance.enumerate_adapters(wgpu::Backends::VULKAN);
                let adapters = cubecl::future::block_on(listing);
                adapters
                    .into_iter()
                    .filter(|adapter| adapter.get_info().device_type == kind)
                    .nth(index)
                    .expect("the pinned adapter exists")
            },
            None => {
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                };
                let request = instance.request_adapter(&options);
                let requested = cubecl::future::block_on(request);
                requested.expect("a high-performance adapter exists")
            },
        };

        (instance, adapter)
    }

    /// The adapter type and index the test environment pins, if it pins one.
    fn pinned_adapter() -> Option<(wgpu::DeviceType, usize)> {
        use std::str::FromStr;

        use crate::bench_api::Device;

        let label = crate::testing::test_device_label();
        let selector = Device::from_str(&label).expect("the test device selector parses");

        match selector {
            Device::Default => {
                let pinned = std::env::var("CUBECL_WGPU_DEFAULT_DEVICE").ok()?;
                let digits = pinned.strip_prefix("DiscreteGpu(")?.strip_suffix(')')?;
                let index = digits.parse::<usize>().ok()?;
                Some((wgpu::DeviceType::DiscreteGpu, index))
            },
            Device::Discrete(index) => Some((wgpu::DeviceType::DiscreteGpu, index)),
            Device::Integrated(index) => Some((wgpu::DeviceType::IntegratedGpu, index)),
            Device::Virtual(index) => Some((wgpu::DeviceType::VirtualGpu, index)),
            Device::Cpu => Some((wgpu::DeviceType::Cpu, 0)),
        }
    }
}

pub(crate) fn test_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

pub(crate) fn packed_supported() -> bool {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);

    limits.int64_atomic_add
}

/// Every rule relaxed except what the test backend needs for correct output.
///
/// Equivalence tests run every candidate the device can execute, including those a real device
/// class is not offered.
pub(crate) fn test_legality(client: &Client) -> Legality {
    let limits = DeviceLimits::from_client(client);
    let on_cuda = needs_warp_uniform_search(client);

    Legality {
        uniform_walk_only: on_cuda,
        f16_search: true,
        packed_atomics: limits.int64_atomic_add,
    }
}
