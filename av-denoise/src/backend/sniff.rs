//! Picking a backend that actually works on this machine

use cubecl::prelude::*;

use super::accelerate::Accelerator;
use super::device::Device;
use super::probe::open_client;

/// Returns the first accelerator, in order of preference, whose client can be built and synchronised
/// on `device`.
///
/// cubecl kernels are fully asynchronous, so a successful `client.sync()` is enough to prove the
/// backend works and no test kernel is needed. The probe runs on `device` rather than the backend's
/// default, because opening a client on another card tests the wrong hardware and pays that card's
/// first-time driver initialisation.
///
/// An accelerator that cannot express `device` at all is treated as unavailable, which is the answer
/// building on it would give one step earlier. So is a backend whose driver libraries are missing,
/// however loudly it fails.
///
/// ```no_run
/// use av_denoise::Device;
/// use av_denoise::accelerate::get_default_accelerators;
/// use av_denoise::sniff::sniff_best_accelerator;
///
/// let preferred = get_default_accelerators();
/// match sniff_best_accelerator(&preferred, &Device::Default) {
///     Some(accelerator) => println!("running on {accelerator}"),
///     None => println!("no usable backend on this machine"),
/// }
/// ```
pub fn sniff_best_accelerator(enable: &[Accelerator], device: &Device) -> Option<Accelerator> {
    for accelerator in enable {
        let is_enabled = match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => match device.to_cuda() {
                Ok(cuda_device) => probe_runtime::<cubecl::cuda::CudaRuntime>(*accelerator, &cuda_device),
                Err(_) => false,
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => match device.to_amd() {
                Ok(amd_device) => probe_runtime::<cubecl::hip::HipRuntime>(*accelerator, &amd_device),
                Err(_) => false,
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => match device.to_wgpu() {
                Ok(wgpu_device) => probe_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &wgpu_device),
                Err(_) => false,
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => match device.to_wgpu() {
                Ok(wgpu_device) => probe_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &wgpu_device),
                Err(_) => false,
            },
            // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)` widens the `Accelerator` enum to
            // include variants whose backend feature is not enabled. Never reached at runtime.
            #[cfg(docsrs)]
            #[expect(
                unreachable_patterns,
                reason = "the arm only keeps the match exhaustive on docs.rs"
            )]
            _ => unreachable!(),
        };

        if is_enabled {
            return Some(*accelerator);
        }
    }

    None
}

fn probe_runtime<R: Runtime>(accelerator: Accelerator, device: &R::Device) -> bool {
    open_client::<R>(accelerator, device).is_some()
}
