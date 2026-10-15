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
