#[cfg(feature = "rocm")]
use cubecl::device::AmdDevice;
#[cfg(feature = "cuda")]
use cubecl::device::CudaDevice;
#[cfg(feature = "vulkan")]
use cubecl::device::WgpuBackend;
#[cfg(any(feature = "vulkan", feature = "metal"))]
use cubecl::device::{WgpuDevice, WgpuDeviceKind};

use super::accelerate::Accelerator;
use super::device::Device;

/// A device selector the backend cannot name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BackendError {
    /// The backend has no device of this kind.
    #[cfg_attr(
        not(any(feature = "cuda", feature = "rocm")),
        expect(dead_code, reason = "only CUDA and ROCm reject a device kind")
    )]
    #[error(
        "device {device:?} is not supported on the {runtime} runtime, use `default` or `discrete[:N]`",
        runtime = runtime_name(.accelerator)
    )]
    UnsupportedDevice {
        accelerator: Accelerator,
        device: Device,
    },
}

/// The CubeCL device `device` names on `accelerator`.
///
/// No driver is touched, so whether the device exists is only known once a client opens on it.
pub(crate) fn resolve(accelerator: Accelerator, device: Device) -> Result<cubecl::Device, BackendError> {
    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let index = gpu_index(accelerator, device)?;
            let cuda_device = CudaDevice::new(index);
            Ok(cubecl::Device::from(cuda_device))
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let index = gpu_index(accelerator, device)?;
            let amd_device = AmdDevice::new(index);
            Ok(cubecl::Device::from(amd_device))
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let kind = wgpu_kind(device);
            let any_backend = WgpuDevice::new(kind);
            let vulkan_device = any_backend.on(WgpuBackend::Vulkan);
            Ok(cubecl::Device::from(vulkan_device))
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            // wgpu picks the graphics API itself, which is Metal on Apple targets.
            let kind = wgpu_kind(device);
            let metal_device = WgpuDevice::new(kind);
            Ok(cubecl::Device::from(metal_device))
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

/// The wgpu device inside `selected`, when [resolve] named it for Vulkan.
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_device(selected: &cubecl::Device) -> Option<&WgpuDevice> {
    match selected {
        cubecl::Device::Wgpu(wgpu_device) if wgpu_device.backend == WgpuBackend::Vulkan => Some(wgpu_device),
        _ => None,
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_index(accelerator: Accelerator, device: Device) -> Result<usize, BackendError> {
    match device {
        Device::Default => Ok(0),
        Device::Discrete { index } => Ok(index),
        other => Err(BackendError::UnsupportedDevice {
            accelerator,
            device: other,
        }),
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
fn wgpu_kind(device: Device) -> WgpuDeviceKind {
    match device {
        Device::Default => WgpuDeviceKind::DefaultDevice,
        Device::Discrete { index } => WgpuDeviceKind::DiscreteGpu(index),
        Device::Integrated { index } => WgpuDeviceKind::IntegratedGpu(index),
        Device::Virtual { index } => WgpuDeviceKind::VirtualGpu(index),
        Device::Cpu => WgpuDeviceKind::Cpu,
    }
}

/// The runtime's name as error messages spell it.
fn runtime_name(accelerator: &Accelerator) -> &'static str {
    match accelerator {
        #[cfg(any(feature = "cuda", docsrs))]
        Accelerator::Cuda => "CUDA",
        #[cfg(any(feature = "vulkan", docsrs))]
        Accelerator::Vulkan => "Vulkan",
        #[cfg(any(feature = "metal", docsrs))]
        Accelerator::Metal => "Metal",
        #[cfg(any(feature = "rocm", docsrs))]
        Accelerator::Rocm => "ROCm",
    }
}
