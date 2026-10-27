#[cfg(feature = "rocm")]
use cubecl::device::AmdDevice;
#[cfg(feature = "cuda")]
use cubecl::device::CudaDevice;
#[cfg(feature = "metal")]
use cubecl::device::MetalDevice;
#[cfg(feature = "vulkan")]
use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

use super::accelerate::Accelerator;
use super::device::Device;

/// A device selector the backend cannot name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BackendError {
    /// The backend has no device of this kind.
    #[cfg_attr(
        not(any(feature = "cuda", feature = "rocm", feature = "metal")),
        expect(dead_code, reason = "only CUDA, ROCm and Metal reject a device kind")
    )]
    #[error(
        "device {device:?} is not supported on the {runtime} runtime, use `default` or `discrete[:N]`",
        runtime = runtime_name(.accelerator)
    )]
    UnsupportedDevice {
        accelerator: Accelerator,
        device: Device,
    },
    /// The index is past the last GPU the backend sees.
    #[cfg(feature = "metal")]
    #[error(
        "device {device} is not on this machine, the {runtime} runtime sees {available} {noun}",
        runtime = runtime_name(.accelerator),
        noun = gpu_noun(*.available)
    )]
    MissingDevice {
        accelerator: Accelerator,
        device: Device,
        available: usize,
    },
}

/// The CubeCL device `device` names on `accelerator`.
///
/// Only Metal asks the driver, to count its GPUs. On every other backend whether the device exists
/// is only known once a client opens on it.
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
            let gpus = MetalGpus::count();
            let kind = metal_device(device, gpus)?;
            Ok(cubecl::Device::from(kind))
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

/// How many GPUs of each kind the native Metal runtime sees.
#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalGpus {
    pub discrete: usize,
    pub integrated: usize,
}

#[cfg(feature = "metal")]
impl MetalGpus {
    /// Counts the GPUs Metal lists.
    pub(crate) fn count() -> Self {
        let discrete = count_metal_gpus(MetalDevice::DiscreteGpu(0));
        let integrated = count_metal_gpus(MetalDevice::IntegratedGpu(0));
        Self { discrete, integrated }
    }

    fn total(self) -> usize {
        self.discrete + self.integrated
    }

    /// The selector of every GPU, in the order [metal_device] counts them.
    pub(crate) fn selectors(self) -> Vec<Device> {
        (0..self.total())
            .map(|index| Device::Discrete { index })
            .collect()
    }
}

/// How many Metal GPUs share the kind of `first`.
#[cfg(feature = "metal")]
fn count_metal_gpus(first: MetalDevice) -> usize {
    let device = cubecl::Device::from(first);
    let device_id = device.to_id();
    let listed = cubecl::Device::enumerate(device_id);
    listed.len()
}

/// The native Metal device `device` names, out of `gpus`.
///
/// An index counts the discrete GPUs first and the integrated ones after, so `discrete:0` is the
/// first GPU on every Mac.
#[cfg(feature = "metal")]
pub(crate) fn metal_device(device: Device, gpus: MetalGpus) -> Result<MetalDevice, BackendError> {
    let accelerator = Accelerator::Metal;

    match device {
        Device::Default => Ok(MetalDevice::DefaultDevice),
        Device::Discrete { index } if index < gpus.discrete => Ok(MetalDevice::DiscreteGpu(index)),
        Device::Discrete { index } if index < gpus.total() => {
            let integrated_index = index - gpus.discrete;
            Ok(MetalDevice::IntegratedGpu(integrated_index))
        },
        Device::Discrete { .. } => Err(BackendError::MissingDevice {
            accelerator,
            device,
            available: gpus.total(),
        }),
        other => Err(BackendError::UnsupportedDevice {
            accelerator,
            device: other,
        }),
    }
}

#[cfg(feature = "metal")]
fn gpu_noun(count: usize) -> &'static str {
    match count {
        1 => "GPU",
        _ => "GPUs",
    }
}

#[cfg(feature = "vulkan")]
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
