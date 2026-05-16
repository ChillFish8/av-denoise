//! Backend-agnostic device selector. Each cubecl runtime exposes a
//! different `Device` type — `CudaDevice { index }`, `AmdDevice { index }`,
//! `WgpuDevice` (a variant enum), and the unit `CpuDevice`. [`Device`]
//! is a unified description that the library translates into the
//! concrete type for the runtime it ends up using.

use std::str::FromStr;

/// Where to run the compute. The library maps each variant onto the
/// concrete `Device` type of whichever cubecl runtime was selected.
///
/// Some variants are only meaningful for certain backends. `Integrated`
/// and `Virtual` are wgpu-only; `Cpu` is a no-op on the `cpu` runtime
/// and selects `WgpuDevice::Cpu` on wgpu. Asking for a variant on a
/// runtime that can't honour it returns an error from the relevant
/// `to_*` conversion.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Device {
    /// Backend-chosen default device.
    #[default]
    Default,
    /// Discrete GPU at ordinal `index`.
    ///
    /// Maps to `CudaDevice { index }`, `AmdDevice { index }`, or
    /// `WgpuDevice::DiscreteGpu(index)`.
    Discrete { index: usize },
    /// Integrated GPU at ordinal `index`. wgpu-only.
    Integrated { index: usize },
    /// Virtual GPU at ordinal `index`. wgpu-only.
    Virtual { index: usize },
    /// Software/CPU device. Valid on the `cpu` runtime and on wgpu
    /// (where it picks the lavapipe / software adapter).
    Cpu,
}

/// `FromStr` accepts the same syntax as the bench CLI:
///
/// - `default`
/// - `discrete[:N]`, `integrated[:N]`, `virtual[:N]` (default `N = 0`)
/// - `cpu`
impl FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (kind, idx) = s.split_once(':').unwrap_or((s, "0"));
        let index: usize = idx
            .parse()
            .map_err(|_| format!("invalid device index '{idx}' in '{s}'"))?;
        match kind {
            "default" => Ok(Device::Default),
            "discrete" => Ok(Device::Discrete { index }),
            "integrated" => Ok(Device::Integrated { index }),
            "virtual" => Ok(Device::Virtual { index }),
            "cpu" => Ok(Device::Cpu),
            other => Err(format!(
                "unknown device kind '{other}'; expected default, discrete[:N], integrated[:N], virtual[:N], or cpu"
            )),
        }
    }
}

#[cfg(feature = "cuda")]
impl Device {
    pub fn to_cuda(&self) -> Result<cubecl::cuda::CudaDevice, anyhow::Error> {
        match self {
            Device::Default => Ok(cubecl::cuda::CudaDevice { index: 0 }),
            Device::Discrete { index } => Ok(cubecl::cuda::CudaDevice { index: *index }),
            other => Err(anyhow::anyhow!(
                "device {other:?} is not supported on the CUDA runtime; use `default` or `discrete[:N]`"
            )),
        }
    }
}

#[cfg(feature = "rocm")]
impl Device {
    pub fn to_amd(&self) -> Result<cubecl::hip::AmdDevice, anyhow::Error> {
        match self {
            Device::Default => Ok(cubecl::hip::AmdDevice { index: 0 }),
            Device::Discrete { index } => Ok(cubecl::hip::AmdDevice { index: *index }),
            other => Err(anyhow::anyhow!(
                "device {other:?} is not supported on the ROCm runtime; use `default` or `discrete[:N]`"
            )),
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
impl Device {
    pub fn to_wgpu(&self) -> Result<cubecl::wgpu::WgpuDevice, anyhow::Error> {
        use cubecl::wgpu::WgpuDevice;
        Ok(match self {
            Device::Default => WgpuDevice::DefaultDevice,
            Device::Discrete { index } => WgpuDevice::DiscreteGpu(*index),
            Device::Integrated { index } => WgpuDevice::IntegratedGpu(*index),
            Device::Virtual { index } => WgpuDevice::VirtualGpu(*index),
            Device::Cpu => WgpuDevice::Cpu,
        })
    }
}

#[cfg(feature = "cpu")]
impl Device {
    pub fn to_cpu(&self) -> Result<cubecl::cpu::CpuDevice, anyhow::Error> {
        match self {
            Device::Default | Device::Cpu => Ok(cubecl::cpu::CpuDevice),
            other => Err(anyhow::anyhow!(
                "device {other:?} is not supported on the CPU runtime; use `default` or `cpu`"
            )),
        }
    }
}
