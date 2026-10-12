use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use av_denoise_core::tune::DeviceIdentity;
use cubecl::hash::StableHasher;
use cubecl::prelude::*;

/// Identities already worked out in this process, keyed by backend and device selector.
static IDENTITIES: LazyLock<Mutex<HashMap<String, DeviceIdentity>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "cuda")]
const NVIDIA_VENDOR_ID: u32 = 0x10de;
#[cfg(feature = "rocm")]
const AMD_VENDOR_ID: u32 = 0x1002;

/// Returns the identity cached under `memo_key`, working it out with `describe` the first time.
fn remembered_identity(memo_key: String, describe: impl FnOnce() -> DeviceIdentity) -> DeviceIdentity {
    {
        let identities = IDENTITIES.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(identity) = identities.get(&memo_key) {
            return identity.clone();
        }
    }

    let identity = describe();
    let mut identities = IDENTITIES.lock().unwrap_or_else(PoisonError::into_inner);
    identities.insert(memo_key, identity.clone());
    identity
}

/// An identity built from the runtime's hardware properties, for when the driver cannot be asked.
///
/// The adapter field carries a stable hash of the properties, so two different devices on one
/// backend still get different store files.
pub(crate) fn fallback_identity<R: Runtime>(backend: &str, client: &ComputeClient<R>) -> DeviceIdentity {
    let hardware = &client.properties().hardware;
    let rendered = format!("{hardware:?}");
    let hash = StableHasher::hash_one(&rendered);

    DeviceIdentity {
        backend: backend.to_string(),
        adapter: format!("unknown-{hash:032x}"),
        vendor_id: 0,
        device_id: 0,
        driver: String::new(),
        os: std::env::consts::OS.to_string(),
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
mod wgpu_identity {
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    use super::*;

    /// The variable CubeCL reads to point its default device at a specific adapter.
    const DEFAULT_DEVICE_ENV: &str = "CUBECL_WGPU_DEFAULT_DEVICE";

    /// The identity of the adapter CubeCL picks for `device` on `backend`.
    ///
    /// The adapter is found the way CubeCL finds it, by kind and index, so the name matches the
    /// device the client runs on.
    pub(crate) fn describe_wgpu(
        device: &WgpuDevice,
        backend: wgpu::Backend,
        client: &ComputeClient<WgpuRuntime>,
    ) -> DeviceIdentity {
        let memo_key = format!("wgpu|{backend:?}|{device:?}");

        remembered_identity(memo_key, || {
            let backend_name = format!("{backend:?}").to_ascii_lowercase();
            let info = adapter_info(device, backend);

            match info {
                Some(info) => {
                    let driver = format!("{} {}", info.driver, info.driver_info);

                    DeviceIdentity {
                        backend: backend_name,
                        adapter: info.name,
                        vendor_id: info.vendor,
                        device_id: info.device,
                        driver: driver.trim().to_string(),
                        os: std::env::consts::OS.to_string(),
                    }
                },
                None => fallback_identity(&backend_name, client),
            }
        })
    }

    fn adapter_info(device: &WgpuDevice, backend: wgpu::Backend) -> Option<wgpu::AdapterInfo> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: backend.into(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        };
        let instance = wgpu::Instance::new(descriptor);
        let selected = default_override(device);

        let adapter = match selected {
            WgpuDevice::DiscreteGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::DiscreteGpu, index)
            },
            WgpuDevice::IntegratedGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::IntegratedGpu, index)
            },
            WgpuDevice::VirtualGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::VirtualGpu, index)
            },
            WgpuDevice::Cpu => pick_adapter(&instance, backend, wgpu::DeviceType::Cpu, 0),
            _ => {
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                };
                let request = instance.request_adapter(&options);
                let requested = cubecl::future::block_on(request);
                requested.ok()
            },
        }?;

        Some(adapter.get_info())
    }

    /// The `index`-th adapter of `kind`.
    ///
    /// Adapters of kind `Other` form a fallback list CubeCL also counts from, used when there are
    /// not enough adapters of `kind`.
    fn pick_adapter(
        instance: &wgpu::Instance,
        backend: wgpu::Backend,
        kind: wgpu::DeviceType,
        index: usize,
    ) -> Option<wgpu::Adapter> {
        let backends: wgpu::Backends = backend.into();
        let listing = instance.enumerate_adapters(backends);
        let adapters = cubecl::future::block_on(listing);
        let mut matching = Vec::new();
        let mut other = Vec::new();

        for adapter in adapters {
            let device_type = adapter.get_info().device_type;

            if device_type == wgpu::DeviceType::Other {
                other.push(adapter);
            } else if device_type == kind {
                matching.push(adapter);
            }
        }

        if index < matching.len() {
            return matching.into_iter().nth(index);
        }

        other.into_iter().nth(index)
    }

    /// `device`, or the adapter [DEFAULT_DEVICE_ENV] names when `device` is the default.
    fn default_override(device: &WgpuDevice) -> WgpuDevice {
        if !matches!(device, WgpuDevice::DefaultDevice) {
            return device.clone();
        }

        let Ok(text) = std::env::var(DEFAULT_DEVICE_ENV) else {
            return device.clone();
        };

        let parsed = parse_override(&text);
        parsed.unwrap_or_else(|| device.clone())
    }

    fn parse_override(text: &str) -> Option<WgpuDevice> {
        if text == "Cpu" {
            return Some(WgpuDevice::Cpu);
        }

        let (kind, rest) = text.split_once('(')?;
        let index_text = rest.strip_suffix(')')?;
        let index: usize = index_text.parse().ok()?;

        match kind {
            "DiscreteGpu" => Some(WgpuDevice::DiscreteGpu(index)),
            "IntegratedGpu" => Some(WgpuDevice::IntegratedGpu(index)),
            "VirtualGpu" => Some(WgpuDevice::VirtualGpu(index)),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn overrides_parse_the_cubecl_spellings() {
            assert_eq!(parse_override("DiscreteGpu(1)"), Some(WgpuDevice::DiscreteGpu(1)));
            assert_eq!(
                parse_override("IntegratedGpu(0)"),
                Some(WgpuDevice::IntegratedGpu(0))
            );
            assert_eq!(parse_override("Cpu"), Some(WgpuDevice::Cpu));
            assert_eq!(parse_override("Discrete(1)"), None);
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) use self::wgpu_identity::describe_wgpu;

/// The identity of CUDA device `index`, from the driver.
#[cfg(feature = "cuda")]
pub(crate) fn describe_cuda(
    index: usize,
    client: &ComputeClient<cubecl::cuda::CudaRuntime>,
) -> DeviceIdentity {
    let memo_key = format!("cuda|{index}");

    remembered_identity(memo_key, || {
        let details = cuda_details(index);
        details.unwrap_or_else(|error| {
            tracing::debug!(%error, "cannot read the CUDA device name, using its properties");
            fallback_identity("cuda", client)
        })
    })
}

#[cfg(feature = "cuda")]
fn cuda_details(index: usize) -> Result<DeviceIdentity, anyhow::Error> {
    use cudarc::driver::{result, sys};

    let ordinal = i32::try_from(index)?;

    // The identity can be read before any client initialises the driver. `cuInit` is idempotent.
    result::init()?;

    let device = result::device::get(ordinal)?;
    let name = result::device::get_name(device)?;
    let attribute = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;

    // SAFETY: `device` came from `result::device::get` just above.
    let pci_device = unsafe { result::device::get_attribute(device, attribute) }?;

    let mut version = 0i32;
    // SAFETY: the driver was initialised above, and `version` is a valid out pointer.
    let status = unsafe { sys::cuDriverGetVersion(&mut version) };
    if status != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("cuDriverGetVersion returned {status:?}");
    }

    let major = version / 1000;
    let minor = (version % 1000) / 10;

    Ok(DeviceIdentity {
        backend: "cuda".to_string(),
        adapter: name,
        vendor_id: NVIDIA_VENDOR_ID,
        device_id: pci_device as u32,
        driver: format!("cuda driver {major}.{minor}"),
        os: std::env::consts::OS.to_string(),
    })
}

/// The identity of ROCm device `index`, from HIP.
#[cfg(feature = "rocm")]
pub(crate) fn describe_rocm(index: usize, client: &ComputeClient<cubecl::hip::HipRuntime>) -> DeviceIdentity {
    let memo_key = format!("rocm|{index}");

    remembered_identity(memo_key, || {
        let details = rocm_details(index);
        details.unwrap_or_else(|error| {
            tracing::debug!(%error, "cannot read the ROCm device name, using its properties");
            fallback_identity("rocm", client)
        })
    })
}

#[cfg(feature = "rocm")]
fn rocm_details(index: usize) -> Result<DeviceIdentity, anyhow::Error> {
    use std::ffi::CStr;
    use std::mem::MaybeUninit;

    let device = i32::try_from(index)?;
    let mut properties = MaybeUninit::<cubecl_hip_sys::hipDeviceProp_tR0600>::uninit();

    // SAFETY: HIP fills `properties` on success, which is checked before it is read.
    let status = unsafe { cubecl_hip_sys::hipGetDevicePropertiesR0600(properties.as_mut_ptr(), device) };
    if status != cubecl_hip_sys::HIP_SUCCESS {
        anyhow::bail!("hipGetDevicePropertiesR0600 returned {status}");
    }

    // SAFETY: the call above succeeded, so every field is initialised.
    let properties = unsafe { properties.assume_init() };
    // SAFETY: HIP writes NUL-terminated strings into both fixed-size arrays.
    let name = unsafe { CStr::from_ptr(properties.name.as_ptr()) };
    // SAFETY: as above.
    let arch = unsafe { CStr::from_ptr(properties.gcnArchName.as_ptr()) };
    let name = name.to_string_lossy();
    let arch = arch.to_string_lossy();

    let mut version = 0i32;
    // SAFETY: `version` is a valid out pointer.
    let status = unsafe { cubecl_hip_sys::hipDriverGetVersion(&mut version) };
    if status != cubecl_hip_sys::HIP_SUCCESS {
        anyhow::bail!("hipDriverGetVersion returned {status}");
    }

    Ok(DeviceIdentity {
        backend: "rocm".to_string(),
        adapter: format!("{name} ({arch})"),
        vendor_id: AMD_VENDOR_ID,
        device_id: properties.pciDeviceID as u32,
        driver: format!("hip driver {version}"),
        os: std::env::consts::OS.to_string(),
    })
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    use super::*;

    #[test]
    fn the_default_vulkan_adapter_has_a_name_and_a_vendor() {
        let device = WgpuDevice::DefaultDevice;
        let client = <WgpuRuntime as Runtime>::client(&device);
        let backend = *client.info();

        let identity = describe_wgpu(&device, backend, &client);

        assert_eq!(identity.backend, "vulkan");
        assert!(!identity.adapter.is_empty());
        assert!(!identity.adapter.starts_with("unknown-"), "{identity:?}");
        assert_ne!(identity.vendor_id, 0);
    }

    #[test]
    fn describing_twice_gives_the_same_identity() {
        let device = WgpuDevice::DefaultDevice;
        let client = <WgpuRuntime as Runtime>::client(&device);
        let backend = *client.info();

        let first = describe_wgpu(&device, backend, &client);
        let second = describe_wgpu(&device, backend, &client);

        assert_eq!(first.identity_hash(), second.identity_hash());
    }

    #[test]
    fn the_fallback_identity_is_stable_for_one_device() {
        let device = WgpuDevice::DefaultDevice;
        let client = <WgpuRuntime as Runtime>::client(&device);

        let first = fallback_identity("vulkan", &client);
        let second = fallback_identity("vulkan", &client);

        assert_eq!(first, second);
        assert!(first.adapter.starts_with("unknown-"));
    }
}
