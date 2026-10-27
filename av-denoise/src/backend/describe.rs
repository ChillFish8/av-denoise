#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
use std::collections::HashMap;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
use std::sync::{LazyLock, Mutex, PoisonError};

#[cfg(feature = "rocm")]
use av_denoise_core::tune::AMD_VENDOR_ID;
use av_denoise_core::tune::DeviceIdentity;
#[cfg(feature = "cuda")]
use av_denoise_core::tune::NVIDIA_VENDOR_ID;
#[cfg(feature = "vulkan")]
use cubecl::device::WgpuBackend;
use cubecl::hash::StableHasher;
use cubecl::prelude::*;

/// Identities already worked out in this process, keyed by backend and device selector.
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
static IDENTITIES: LazyLock<Mutex<HashMap<String, DeviceIdentity>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The identity of the device `selected` names, with `client` open on it.
pub(crate) fn describe(selected: &cubecl::Device, client: &Client) -> DeviceIdentity {
    match selected {
        #[cfg(feature = "cuda")]
        cubecl::Device::Cuda(cuda_device) => describe_cuda(cuda_device.index, client),
        #[cfg(feature = "rocm")]
        cubecl::Device::Hip(amd_device) => describe_rocm(amd_device.index, client),
        #[cfg(feature = "vulkan")]
        cubecl::Device::Wgpu(wgpu_device) if wgpu_device.backend == WgpuBackend::Vulkan => {
            describe_wgpu(wgpu_device, wgpu::Backend::Vulkan, client)
        },
        #[cfg(feature = "metal")]
        cubecl::Device::Metal(_) => describe_metal(client),
        other => {
            let runtime = format!("{:?}", other.runtime());
            let backend = runtime.to_ascii_lowercase();
            fallback_identity(&backend, client)
        },
    }
}

/// Returns the identity cached under `memo_key`, working it out with `describe` the first time.
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
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
fn fallback_identity(backend: &str, client: &Client) -> DeviceIdentity {
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
        compute_capability: None,
    }
}

#[cfg(feature = "vulkan")]
mod wgpu_identity {
    use cubecl::wgpu::{WgpuDevice, WgpuDeviceKind};

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
        client: &Client,
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
                        compute_capability: None,
                    }
                },
                None => fallback_identity(&backend_name, client),
            }
        })
    }

    fn adapter_info(device: &WgpuDevice, backend: wgpu::Backend) -> Option<wgpu::AdapterInfo> {
        let (_, adapter) = find_adapter(device, backend)?;
        Some(adapter.get_info())
    }

    /// The adapter CubeCL picks for `device` on `backend`, with the instance it was found on.
    pub(crate) fn find_adapter(
        device: &WgpuDevice,
        backend: wgpu::Backend,
    ) -> Option<(wgpu::Instance, wgpu::Adapter)> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: backend.into(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        };
        let instance = wgpu::Instance::new(descriptor);
        let selected = default_override(device);

        let adapter = match selected {
            WgpuDeviceKind::DiscreteGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::DiscreteGpu, index)
            },
            WgpuDeviceKind::IntegratedGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::IntegratedGpu, index)
            },
            WgpuDeviceKind::VirtualGpu(index) => {
                pick_adapter(&instance, backend, wgpu::DeviceType::VirtualGpu, index)
            },
            WgpuDeviceKind::Cpu => pick_adapter(&instance, backend, wgpu::DeviceType::Cpu, 0),
            _ => {
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                };
                let request = instance.request_adapter(&options);
                let requested = cubecl::future::block_on(request);
                requested.ok()
            },
        }?;

        Some((instance, adapter))
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

    /// `device`'s kind, or the adapter [DEFAULT_DEVICE_ENV] names when `device` is the default.
    fn default_override(device: &WgpuDevice) -> WgpuDeviceKind {
        if !matches!(device.kind, WgpuDeviceKind::DefaultDevice) {
            return device.kind.clone();
        }

        let Ok(text) = std::env::var(DEFAULT_DEVICE_ENV) else {
            return device.kind.clone();
        };

        let parsed = parse_override(&text);
        parsed.unwrap_or_else(|| device.kind.clone())
    }

    fn parse_override(text: &str) -> Option<WgpuDeviceKind> {
        if text == "Cpu" {
            return Some(WgpuDeviceKind::Cpu);
        }

        let (kind, rest) = text.split_once('(')?;
        let index_text = rest.strip_suffix(')')?;
        let index: usize = index_text.parse().ok()?;

        match kind {
            "DiscreteGpu" => Some(WgpuDeviceKind::DiscreteGpu(index)),
            "IntegratedGpu" => Some(WgpuDeviceKind::IntegratedGpu(index)),
            "VirtualGpu" => Some(WgpuDeviceKind::VirtualGpu(index)),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn overrides_parse_the_cubecl_spellings() {
            assert_eq!(
                parse_override("DiscreteGpu(1)"),
                Some(WgpuDeviceKind::DiscreteGpu(1))
            );
            assert_eq!(
                parse_override("IntegratedGpu(0)"),
                Some(WgpuDeviceKind::IntegratedGpu(0))
            );
            assert_eq!(parse_override("Cpu"), Some(WgpuDeviceKind::Cpu));
            assert_eq!(parse_override("Discrete(1)"), None);
        }
    }
}

#[cfg(feature = "vulkan")]
use self::wgpu_identity::describe_wgpu;
#[cfg(feature = "vulkan")]
pub(crate) use self::wgpu_identity::find_adapter;

/// The file macOS keeps its own version in.
#[cfg(feature = "metal")]
const SYSTEM_VERSION_PLIST: &str = "/System/Library/CoreServices/SystemVersion.plist";

/// The identity of the Metal GPU `client` runs on.
#[cfg(feature = "metal")]
fn describe_metal(client: &Client) -> DeviceIdentity {
    let properties = client.properties();
    let name = &properties.identity.name;
    let plist = std::fs::read_to_string(SYSTEM_VERSION_PLIST).unwrap_or_default();
    let os_version = product_version(&plist).unwrap_or_default();
    metal_identity(name, &os_version)
}

/// The identity of a Metal GPU from its name and the macOS version.
///
/// The Metal shader compiler ships with macOS, so the OS version takes the driver's place.
#[cfg(feature = "metal")]
fn metal_identity(name: &str, os_version: &str) -> DeviceIdentity {
    DeviceIdentity {
        backend: "metal".to_string(),
        adapter: name.to_string(),
        vendor_id: 0,
        device_id: 0,
        driver: os_version.to_string(),
        os: std::env::consts::OS.to_string(),
        compute_capability: None,
    }
}

/// The `ProductVersion` string in the text of a `SystemVersion.plist`, such as `15.1`.
#[cfg(feature = "metal")]
fn product_version(plist: &str) -> Option<String> {
    let (_, after_key) = plist.split_once("<key>ProductVersion</key>")?;
    let (_, after_open) = after_key.split_once("<string>")?;
    let (version, _) = after_open.split_once("</string>")?;
    Some(version.trim().to_string())
}

/// The identity of CUDA device `index`, from the driver.
#[cfg(feature = "cuda")]
fn describe_cuda(index: usize, client: &Client) -> DeviceIdentity {
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

    let capability_major_attribute = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    let capability_minor_attribute = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;

    // SAFETY: `device` came from `result::device::get` above.
    let capability_major = unsafe { result::device::get_attribute(device, capability_major_attribute) }?;
    // SAFETY: as above.
    let capability_minor = unsafe { result::device::get_attribute(device, capability_minor_attribute) }?;
    let compute_capability = (capability_major * 10 + capability_minor) as u32;

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
        compute_capability: Some(compute_capability),
    })
}

/// The identity of ROCm device `index`, from HIP.
#[cfg(feature = "rocm")]
fn describe_rocm(index: usize, client: &Client) -> DeviceIdentity {
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
        compute_capability: None,
    })
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::backend::accelerate::Accelerator;
    use crate::backend::resolve::resolve;

    /// The test device on Vulkan, with a client on it.
    fn test_vulkan_client() -> (cubecl::Device, Client) {
        let selector = crate::testing::test_device();
        let selected = resolve(Accelerator::Vulkan, selector).expect("vulkan names every device kind");
        let client = selected.client();
        (selected, client)
    }

    #[test]
    fn the_default_vulkan_adapter_has_a_name_and_a_vendor() {
        let (selected, client) = test_vulkan_client();

        let identity = describe(&selected, &client);

        assert_eq!(identity.backend, "vulkan");
        assert!(!identity.adapter.is_empty());
        assert!(!identity.adapter.starts_with("unknown-"), "{identity:?}");
        assert_ne!(identity.vendor_id, 0);
    }

    #[test]
    fn describing_twice_gives_the_same_identity() {
        let (selected, client) = test_vulkan_client();

        let first = describe(&selected, &client);
        let second = describe(&selected, &client);

        assert_eq!(first.identity_hash(), second.identity_hash());
    }

    #[test]
    fn the_fallback_identity_is_stable_for_one_device() {
        let (_, client) = test_vulkan_client();

        let first = fallback_identity("vulkan", &client);
        let second = fallback_identity("vulkan", &client);

        assert_eq!(first, second);
        assert!(first.adapter.starts_with("unknown-"));
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;

    #[test]
    fn metal_identity_keys_on_the_name_and_os_version() {
        let first = metal_identity("Apple M2 Max", "15.1");
        let same = metal_identity("Apple M2 Max", "15.1");
        let upgraded = metal_identity("Apple M2 Max", "15.2");

        assert_eq!(first.backend, "metal");
        assert_eq!(first.adapter, "Apple M2 Max");
        assert_eq!(first.identity_hash(), same.identity_hash());
        assert_ne!(first.identity_hash(), upgraded.identity_hash());
    }

    #[test]
    fn product_version_reads_the_system_version_plist() {
        let plist = "<dict>\n\t<key>ProductName</key>\n\t<string>macOS</string>\n\t\
                     <key>ProductVersion</key>\n\t<string>15.1</string>\n</dict>";

        let version = product_version(plist);
        let missing = product_version("<dict></dict>");

        assert_eq!(version.as_deref(), Some("15.1"));
        assert_eq!(missing, None);
    }
}
