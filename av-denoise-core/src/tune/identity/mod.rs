#[cfg(feature = "vulkan")]
pub(crate) mod adapter;

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
use std::collections::HashMap;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
use std::sync::{LazyLock, Mutex, PoisonError};

#[cfg(feature = "vulkan")]
use cubecl::device::WgpuBackend;
use cubecl::hash::StableHasher;
use cubecl::prelude::*;

#[cfg(feature = "vulkan")]
use self::adapter::describe_wgpu;
#[cfg(feature = "vulkan")]
pub use self::adapter::find_adapter;
#[cfg(feature = "rocm")]
use crate::tune::AMD_VENDOR_ID;
use crate::tune::DeviceIdentity;
#[cfg(feature = "cuda")]
use crate::tune::NVIDIA_VENDOR_ID;

/// Identities already worked out in this process, keyed by backend and device selector.
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
static IDENTITIES: LazyLock<Mutex<HashMap<String, DeviceIdentity>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The identity of the device `selected` names, with `client` open on it.
pub fn describe(selected: &cubecl::Device, client: &Client) -> DeviceIdentity {
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
pub(crate) fn fallback_identity(backend: &str, client: &Client) -> DeviceIdentity {
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
pub(crate) fn metal_identity(name: &str, os_version: &str) -> DeviceIdentity {
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
pub(crate) fn product_version(plist: &str) -> Option<String> {
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

/// The identity of CUDA device `index`, read straight from the driver.
#[cfg(feature = "cuda")]
pub fn cuda_details(index: usize) -> Result<DeviceIdentity, anyhow::Error> {
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
