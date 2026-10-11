use cubecl::prelude::*;

/// The runtime GPU tests in this module run on, picked from the enabled backend.
#[cfg(feature = "cuda")]
pub(crate) type TestRuntime = cubecl::cuda::CudaRuntime;

/// The runtime GPU tests in this module run on, picked from the enabled backend.
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub(crate) type TestRuntime = cubecl::hip::HipRuntime;

/// The runtime GPU tests in this module run on, picked from the enabled backend.
#[cfg(all(
    any(feature = "vulkan", feature = "metal"),
    not(any(feature = "cuda", feature = "rocm"))
))]
pub(crate) type TestRuntime = cubecl::wgpu::WgpuRuntime;

pub(crate) fn test_client() -> ComputeClient<TestRuntime> {
    let device = <TestRuntime as Runtime>::Device::default();
    TestRuntime::client(&device)
}
