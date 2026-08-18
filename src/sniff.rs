//! Picking a backend that actually works on this machine.
//!
//! A build can enable several accelerators, but only some of them will
//! run on any given machine. A CUDA build on a machine with no Nvidia
//! driver, for instance, compiles fine and then fails at runtime.
//!
//! [`sniff_best_accelerator`] takes a list in order of preference and
//! returns the first one that really starts, so callers can list a fast
//! backend followed by a safe fallback.
//!
//! [`Denoiser::create`](crate::Denoiser::create) calls this for you, so
//! reach for it directly only when you want to know the answer without
//! building a denoiser.
//!
//! ```no_run
//! use av_denoise::accelerate::get_default_accelerators;
//! use av_denoise::sniff::sniff_best_accelerator;
//!
//! match sniff_best_accelerator(&get_default_accelerators()) {
//!     Some(accelerator) => println!("running on {accelerator}"),
//!     None => println!("no usable backend on this machine"),
//! }
//! ```

use cubecl::prelude::*;

use crate::accelerate::Accelerator;

/// Tries each accelerator in turn and returns the first one whose client
/// can be built and synchronised.
///
/// cubecl kernels are fully asynchronous, so a successful
/// `client.sync()` is enough to prove the backend works. No test kernel
/// is needed.
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
            // docs.rs widens the `Accelerator` variants behind
            // `cfg(docsrs)` so they all appear in the rendered enum, even
            // when the matching backend feature is off. This arm keeps
            // the match exhaustive there and is never reached at
            // runtime.
            #[cfg(docsrs)]
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };

        if is_enabled {
            return Some(*accelerator);
        }
    }

    None
}

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
