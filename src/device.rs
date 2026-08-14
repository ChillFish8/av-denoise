//! Which physical device to run on.
//!
//! A [`Device`] picks the hardware inside a backend, while
//! [`Accelerator`](crate::accelerate::Accelerator) picks the backend
//! itself. The two are chosen separately because most backends expose
//! more than one device.
//!
//! [`Device`] implements `FromStr`, so it can be taken straight from a
//! command-line flag or a config file.
//!
//! ```
//! use av_denoise::Device;
//!
//! // Let the backend decide.
//! assert_eq!("default".parse::<Device>().unwrap(), Device::Default);
//!
//! // Or name the second discrete GPU in the machine.
//! assert_eq!(
//!     "discrete:1".parse::<Device>().unwrap(),
//!     Device::Discrete { index: 1 },
//! );
//! ```

use std::str::FromStr;

/// Where to run the compute.
///
/// Each variant maps onto the concrete `Device` type of whichever cubecl
/// runtime was selected.
///
/// Not every variant makes sense on every backend. `Integrated` and
/// `Virtual` are wgpu-only, and `Cpu` does nothing on the `cpu` runtime
/// while selecting `WgpuDevice::Cpu` on wgpu.
///
/// Asking for a variant a runtime cannot honour returns an error from
/// the matching `to_*` conversion.
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
    /// The software device. Valid on the `cpu` runtime, and on wgpu
    /// where it picks the lavapipe or software adapter.
    Cpu,
}

impl FromStr for Device {
    type Err = String;

    /// Accepts the same spellings as the bench CLI.
    ///
    /// - `default`
    /// - `discrete[:N]`, `integrated[:N]`, and `virtual[:N]`, where `N`
    ///   defaults to 0
    /// - `cpu`
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
                "unknown device kind '{other}', expected default, discrete[:N], integrated[:N], virtual[:N], or cpu"
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
                "device {other:?} is not supported on the CUDA runtime, use `default` or `discrete[:N]`"
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
                "device {other:?} is not supported on the ROCm runtime, use `default` or `discrete[:N]`"
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
                "device {other:?} is not supported on the CPU runtime, use `default` or `cpu`"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_default() {
        assert_eq!("default".parse::<Device>().unwrap(), Device::Default);
    }

    #[test]
    fn parse_discrete_with_and_without_index() {
        assert_eq!(
            "discrete".parse::<Device>().unwrap(),
            Device::Discrete { index: 0 },
        );
        assert_eq!(
            "discrete:3".parse::<Device>().unwrap(),
            Device::Discrete { index: 3 },
        );
    }

    #[test]
    fn parse_integrated_virtual_cpu() {
        assert_eq!(
            "integrated:1".parse::<Device>().unwrap(),
            Device::Integrated { index: 1 },
        );
        assert_eq!(
            "virtual:2".parse::<Device>().unwrap(),
            Device::Virtual { index: 2 },
        );
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
    }

    #[test]
    fn parse_rejects_unknown_kind() {
        assert!("unicorn".parse::<Device>().is_err());
    }

    #[test]
    fn parse_rejects_non_numeric_index() {
        assert!("discrete:abc".parse::<Device>().is_err());
    }

    #[test]
    fn default_is_default_variant() {
        assert_eq!(Device::default(), Device::Default);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_runtime_rejects_gpu_variants() {
        assert!(Device::Default.to_cpu().is_ok());
        assert!(Device::Cpu.to_cpu().is_ok());
        assert!(Device::Discrete { index: 0 }.to_cpu().is_err());
    }
}
