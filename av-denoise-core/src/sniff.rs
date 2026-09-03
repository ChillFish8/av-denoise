//! Picking a backend that actually works on this machine.
//!
//! A build can enable several accelerators, but only some of them will
//! run on any given machine. A CUDA build on a machine with no Nvidia
//! driver, for instance, compiles fine and then fails at runtime.
//!
//! [`sniff_best_accelerator`] takes a list in order of preference and
//! returns the first one that really starts on the device the caller
//! asked for, so callers can list a fast backend followed by a safe
//! fallback.
//!
//! The probe runs on that device rather than on the backend's default
//! one. Opening a client is what proves a backend works, and opening it
//! on a card the caller did not choose both tests the wrong hardware and
//! pays that card's first-time driver initialisation.
//!
//! [`Denoiser::create`](crate::Denoiser::create) calls this for you, so
//! reach for it directly only when you want to know the answer without
//! building a denoiser.
//!
//! ```no_run
//! use av_denoise_core::Device;
//! use av_denoise_core::accelerate::get_default_accelerators;
//! use av_denoise_core::sniff::sniff_best_accelerator;
//!
//! match sniff_best_accelerator(&get_default_accelerators(), &Device::Default) {
//!     Some(accelerator) => println!("running on {accelerator}"),
//!     None => println!("no usable backend on this machine"),
//! }
//! ```

use cubecl::prelude::*;

use crate::accelerate::Accelerator;
use crate::device::Device;
use crate::probe::open_client;

/// Tries each accelerator in turn and returns the first one whose client
/// can be built and synchronised on `device`.
///
/// cubecl kernels are fully asynchronous, so a successful
/// `client.sync()` is enough to prove the backend works. No test kernel
/// is needed.
///
/// An accelerator that cannot express `device` at all is treated as
/// unavailable and the search moves on. That is the same answer the
/// caller would get from trying to build on it, one step earlier. So is
/// a backend whose driver libraries are missing, however loudly it
/// fails, see [`probe`](crate::probe).
pub fn sniff_best_accelerator(enable: &[Accelerator], device: &Device) -> Option<Accelerator> {
    for accelerator in enable {
        let is_enabled = match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => match device.to_cuda() {
                Ok(dev) => probe_runtime::<cubecl::cuda::CudaRuntime>(*accelerator, &dev),
                Err(_) => false,
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => match device.to_amd() {
                Ok(dev) => probe_runtime::<cubecl::hip::HipRuntime>(*accelerator, &dev),
                Err(_) => false,
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => match device.to_wgpu() {
                Ok(dev) => probe_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &dev),
                Err(_) => false,
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => match device.to_wgpu() {
                Ok(dev) => probe_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &dev),
                Err(_) => false,
            },
            // docs.rs widens the `Accelerator` variants behind
            // `cfg(docsrs)` so they all appear in the rendered enum, even
            // when the matching backend feature is off. This arm keeps
            // the match exhaustive there and is never reached at
            // runtime.
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

/// Reports whether `accelerator` can really run on `device`.
fn probe_runtime<R: Runtime>(accelerator: Accelerator, device: &R::Device) -> bool {
    open_client::<R>(accelerator, device).is_some()
}
