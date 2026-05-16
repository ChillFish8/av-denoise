use cubecl::prelude::*;

use crate::models::Accelerator;

/// Probe each enabled accelerator and return the first one whose client
/// can be created and synchronised. cubecl 0.10 kernels are fully
/// asynchronous, so a successful `client.sync()` is sufficient to
/// confirm the backend is usable — no test kernel needed.
#[allow(dead_code)]
pub fn sniff_best_accelerator(enable: &[Accelerator]) -> Option<Accelerator> {
    for accelerator in enable {
        let is_enabled = match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => probe_runtime::<cubecl::cuda::CudaRuntime>("CUDA"),
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => probe_runtime::<cubecl::hip::HipRuntime>("ROCM"),
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => probe_runtime::<cubecl::wgpu::WgpuRuntime>("VULKAN"),
            #[cfg(feature = "metal")]
            Accelerator::Metal => probe_runtime::<cubecl::wgpu::WgpuRuntime>("METAL"),
            #[cfg(feature = "cpu")]
            Accelerator::Cpu => probe_runtime::<cubecl::cpu::CpuRuntime>("CPU"),
        };

        if is_enabled {
            return Some(*accelerator);
        }
    }

    None
}

#[allow(dead_code)]
fn probe_runtime<R: Runtime>(name: &'static str) -> bool {
    let device = <R::Device as Default>::default();
    let client = R::client(&device);
    match cubecl::future::block_on(client.sync()) {
        Ok(()) => true,
        Err(err) => {
            tracing::debug!(err = ?err, "could not use {name} runtime");
            false
        },
    }
}
