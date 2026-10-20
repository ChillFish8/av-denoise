//! Picking a backend that actually works on this machine

use std::fmt;

use cubecl::prelude::*;

use super::accelerate::Accelerator;
#[cfg(feature = "cuda")]
use super::cuda_toolkit::check_cuda_toolkit;
use super::device::Device;
use super::probe::open_client;

/// Why an accelerator in the priority list was passed over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendUnavailable {
    pub accelerator: Accelerator,
    pub reason: String,
}

impl fmt::Display for BackendUnavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.accelerator, self.reason)
    }
}

/// Returns the first accelerator, in order of preference, that can run a test kernel on `device`.
///
/// [select_accelerator] does the same and also reports why each skipped accelerator failed.
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
    select_accelerator(enable, device).ok()
}

/// Returns the first accelerator, in order of preference, that can run a test kernel on `device`,
/// or why each one could not.
///
/// The probe runs on `device` rather than the backend's default, because opening a client on
/// another card tests the wrong hardware and pays that card's first-time driver initialisation.
/// CUDA also needs its toolkit's headers and NVRTC, its kernel compiler, which ship with the CUDA
/// toolkit rather than the driver.
///
/// An accelerator that cannot express `device` at all is skipped, which is the answer building on
/// it would give one step earlier. So is a backend whose driver libraries are missing, however
/// loudly it fails.
pub fn select_accelerator(
    enable: &[Accelerator],
    device: &Device,
) -> Result<Accelerator, Vec<BackendUnavailable>> {
    let mut skipped = Vec::new();

    for accelerator in enable {
        let probed = probe(*accelerator, device);
        match probed {
            Ok(()) => return Ok(*accelerator),
            Err(reason) => skipped.push(BackendUnavailable {
                accelerator: *accelerator,
                reason,
            }),
        }
    }

    Err(skipped)
}

fn probe(accelerator: Accelerator, device: &Device) -> Result<(), String> {
    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let cuda_device = device.to_cuda().map_err(|err| err.to_string())?;
            check_cuda_toolkit()?;
            probe_runtime::<cubecl::cuda::CudaRuntime>(accelerator, &cuda_device)
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let amd_device = device.to_amd().map_err(|err| err.to_string())?;
            probe_runtime::<cubecl::hip::HipRuntime>(accelerator, &amd_device)
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let wgpu_device = device.to_wgpu().map_err(|err| err.to_string())?;
            probe_runtime::<cubecl::wgpu::WgpuRuntime>(accelerator, &wgpu_device)
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let wgpu_device = device.to_wgpu().map_err(|err| err.to_string())?;
            probe_runtime::<cubecl::wgpu::WgpuRuntime>(accelerator, &wgpu_device)
        },
        // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)` widens the `Accelerator` enum to
        // include variants whose backend feature is not enabled. Never reached at runtime.
        #[cfg(docsrs)]
        #[expect(
            unreachable_patterns,
            reason = "the arm only keeps the match exhaustive on docs.rs"
        )]
        _ => unreachable!(),
    }
}

fn probe_runtime<R: Runtime>(accelerator: Accelerator, device: &R::Device) -> Result<(), String> {
    open_client::<R>(accelerator, device)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_list_selects_nothing_and_has_no_reasons() {
        let selected = select_accelerator(&[], &Device::Default);

        assert_eq!(selected, Err(Vec::new()));
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn a_working_backend_is_selected() {
        let selected = select_accelerator(&[Accelerator::Vulkan], &Device::Default);

        assert_eq!(selected, Ok(Accelerator::Vulkan));
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn a_device_the_backend_cannot_find_reports_why() {
        let missing_device = Device::Discrete { index: 99 };

        let selected = select_accelerator(&[Accelerator::Vulkan], &missing_device);

        let reasons = selected.expect_err("there is no 100th discrete GPU");
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0].accelerator, Accelerator::Vulkan);
        assert!(
            reasons[0].reason.contains("No Discrete GPU device found"),
            "{}",
            reasons[0].reason
        );
    }
}
