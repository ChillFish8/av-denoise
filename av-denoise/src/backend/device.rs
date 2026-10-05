//! Which physical device to run on

use std::fmt;
use std::str::FromStr;

/// Which physical device inside a backend runs the compute.
///
/// The backend itself is picked separately with [Accelerator](crate::accelerate::Accelerator),
/// because most backends expose more than one device. Each variant maps onto the concrete `Device`
/// type of whichever cubecl runtime was selected.
///
/// `Integrated`, `Virtual` and `Cpu` are wgpu-only. Asking for a variant a runtime cannot honour
/// returns an error from the matching `to_*` conversion.
///
/// ```
/// use av_denoise::Device;
///
/// // Let the backend decide.
/// assert_eq!("default".parse::<Device>().unwrap(), Device::Default);
///
/// // Or name the second discrete GPU in the machine.
/// assert_eq!(
///     "discrete:1".parse::<Device>().unwrap(),
///     Device::Discrete { index: 1 },
/// );
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Device {
    /// Backend-chosen default device.
    #[default]
    Default,
    /// Discrete GPU at ordinal `index`.
    ///
    /// Maps to `CudaDevice { index }`, `AmdDevice { index }`, or `WgpuDevice::DiscreteGpu(index)`.
    Discrete { index: usize },
    /// Integrated GPU at ordinal `index`. wgpu-only.
    Integrated { index: usize },
    /// Virtual GPU at ordinal `index`. wgpu-only.
    Virtual { index: usize },
    /// The software device, which picks the lavapipe or software adapter on wgpu.
    Cpu,
}

impl FromStr for Device {
    type Err = String;

    /// Parses a device selector.
    ///
    /// - `default`, which takes no index
    /// - `discrete[:N]`, `integrated[:N]`, and `virtual[:N]`, where `N` defaults to 0
    /// - `cpu`, which takes no index
    fn from_str(selector: &str) -> Result<Self, Self::Err> {
        let (kind, suffix) = match selector.split_once(':') {
            Some((kind, index_text)) => (kind, Some(index_text)),
            None => (selector, None),
        };

        if matches!(kind, "default" | "cpu") && suffix.is_some() {
            return Err(format!(
                "device kind '{kind}' takes no index, got '{selector}'. Only discrete, integrated, and virtual take an index"
            ));
        }

        let parse_index = |index_text: &str| -> Result<usize, String> {
            index_text
                .parse()
                .map_err(|_| format!("invalid device index '{index_text}' in '{selector}'"))
        };
        let index_text = suffix.unwrap_or("0");

        match kind {
            "default" => Ok(Device::Default),
            "cpu" => Ok(Device::Cpu),
            "discrete" => {
                let index = parse_index(index_text)?;
                Ok(Device::Discrete { index })
            },
            "integrated" => {
                let index = parse_index(index_text)?;
                Ok(Device::Integrated { index })
            },
            "virtual" => {
                let index = parse_index(index_text)?;
                Ok(Device::Virtual { index })
            },
            other => Err(format!(
                "unknown device kind '{other}', expected default, discrete[:N], integrated[:N], virtual[:N], or cpu"
            )),
        }
    }
}

impl fmt::Display for Device {
    /// Writes the selector spelling [FromStr] accepts, such as `discrete:1`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Default => formatter.write_str("default"),
            Device::Discrete { index } => write!(formatter, "discrete:{index}"),
            Device::Integrated { index } => write!(formatter, "integrated:{index}"),
            Device::Virtual { index } => write!(formatter, "virtual:{index}"),
            Device::Cpu => formatter.write_str("cpu"),
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
    fn parse_rejects_index_on_default_and_cpu() {
        assert!("default:0".parse::<Device>().is_err());
        assert!("default:1".parse::<Device>().is_err());
        assert!("cpu:2".parse::<Device>().is_err());
    }

    #[test]
    fn rejected_index_error_names_the_kind() {
        let default_error = "default:1".parse::<Device>().unwrap_err();
        assert!(default_error.contains("default"), "{default_error}");

        let cpu_error = "cpu:2".parse::<Device>().unwrap_err();
        assert!(cpu_error.contains("cpu"), "{cpu_error}");
    }

    #[test]
    fn display_writes_selector_spellings() {
        assert_eq!(Device::Default.to_string(), "default");
        assert_eq!(Device::Discrete { index: 1 }.to_string(), "discrete:1");
        assert_eq!(Device::Integrated { index: 0 }.to_string(), "integrated:0");
        assert_eq!(Device::Virtual { index: 2 }.to_string(), "virtual:2");
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn display_round_trips_through_from_str() {
        let devices = [
            Device::Default,
            Device::Discrete { index: 3 },
            Device::Integrated { index: 1 },
            Device::Virtual { index: 0 },
            Device::Cpu,
        ];
        for device in devices {
            let printed = device.to_string();
            assert_eq!(printed.parse::<Device>().unwrap(), device, "{printed}");
        }
    }

    #[test]
    fn default_is_default_variant() {
        assert_eq!(Device::default(), Device::Default);
    }
}
