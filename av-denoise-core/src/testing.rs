use std::str::FromStr;

use cubecl::prelude::Client;

use crate::bench_api::Device;

const ACCELERATOR_ENV: &str = "AVD_TEST_ACCELERATOR";
const DEVICE_ENV: &str = "AVD_TEST_DEVICE";

/// Every accelerator name, in the order the default is picked from.
const ACCELERATORS: &[&str] = &["cuda", "vulkan", "metal", "rocm"];

/// Opens a client on the backend and device the test environment names.
///
/// `AVD_TEST_ACCELERATOR` picks the backend (`vulkan`, `metal`, `cuda` or `rocm`) and
/// `AVD_TEST_DEVICE` the device, in the CLI's `--device` syntax. See [test_accelerator] and
/// [test_device_label] for the defaults.
pub fn test_client() -> Client {
    let device = test_device();
    device.client()
}

/// The CubeCL device the test environment names.
pub fn test_device() -> cubecl::Device {
    let label = test_device_label();
    let parsed = Device::from_str(&label);
    let selector = parsed.unwrap_or_else(|message| panic!("{DEVICE_ENV}={label}: {message}"));
    let accelerator = test_accelerator();
    device_on(accelerator, selector)
}

/// The device selector the test environment names, such as `discrete:1`.
///
/// Without `AVD_TEST_DEVICE` it is `default`, which leaves the pick to the backend.
pub fn test_device_label() -> String {
    std::env::var(DEVICE_ENV).unwrap_or_else(|_| "default".to_string())
}

/// The accelerator the test environment names, or the first one this build has.
///
/// Without `AVD_TEST_ACCELERATOR` the first compiled backend is used, in the order cuda, vulkan,
/// metal, rocm. Panics when the name is unknown, not compiled in, or no backend is compiled at all.
pub fn test_accelerator() -> &'static str {
    let requested = std::env::var(ACCELERATOR_ENV).ok();
    let compiled = compiled_accelerators();
    let picked = pick_accelerator(requested.as_deref(), &compiled);
    picked.unwrap_or_else(|message| panic!("{message}"))
}

/// The CubeCL device `selector` names on `accelerator`.
///
/// Panics when the accelerator is not compiled in, or cannot express the selector.
#[cfg_attr(
    not(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm")),
    expect(
        unused_variables,
        reason = "every backend that reads the selector is compiled out"
    )
)]
pub fn device_on(accelerator: &str, selector: Device) -> cubecl::Device {
    match accelerator {
        #[cfg(feature = "vulkan")]
        "vulkan" => vulkan_device(selector),
        #[cfg(feature = "metal")]
        "metal" => metal_device(selector),
        #[cfg(feature = "cuda")]
        "cuda" => {
            let index = gpu_index(selector, accelerator);
            let cuda_device = cubecl::device::CudaDevice::new(index);
            cubecl::Device::from(cuda_device)
        },
        #[cfg(feature = "rocm")]
        "rocm" => {
            let index = gpu_index(selector, accelerator);
            let amd_device = cubecl::device::AmdDevice::new(index);
            cubecl::Device::from(amd_device)
        },
        other => panic!("{other} is not compiled in, build with `--features {other}`"),
    }
}

/// The GPU index a CUDA or ROCm device selector names.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_index(selector: Device, accelerator: &str) -> usize {
    match selector {
        Device::Default => 0,
        Device::Discrete(index) => index,
        other => panic!("{other:?} is not supported on {accelerator}, expected `default` or `discrete[:N]`"),
    }
}

#[cfg(feature = "vulkan")]
fn vulkan_device(selector: Device) -> cubecl::Device {
    use cubecl::wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

    let kind = match selector {
        Device::Default => WgpuDeviceKind::DefaultDevice,
        Device::Discrete(index) => WgpuDeviceKind::DiscreteGpu(index),
        Device::Integrated(index) => WgpuDeviceKind::IntegratedGpu(index),
        Device::Virtual(index) => WgpuDeviceKind::VirtualGpu(index),
        Device::Cpu => WgpuDeviceKind::Cpu,
    };
    let any_backend = WgpuDevice::new(kind);
    let wgpu_device = any_backend.on(WgpuBackend::Vulkan);
    cubecl::Device::from(wgpu_device)
}

/// The native Metal device a selector names.
///
/// An index counts the discrete GPUs first and the integrated ones after, so `discrete:0` is the
/// first GPU on every Mac.
#[cfg(feature = "metal")]
fn metal_device(selector: Device) -> cubecl::Device {
    use cubecl::device::MetalDevice;

    let kind = match selector {
        Device::Default => MetalDevice::DefaultDevice,
        Device::Discrete(index) => {
            let discrete_count = metal_gpu_count(MetalDevice::DiscreteGpu(0));
            let integrated_count = metal_gpu_count(MetalDevice::IntegratedGpu(0));
            let gpu_count = discrete_count + integrated_count;
            assert!(
                index < gpu_count,
                "discrete:{index} is past the last Metal GPU, this machine has {gpu_count}"
            );

            if index < discrete_count {
                MetalDevice::DiscreteGpu(index)
            } else {
                MetalDevice::IntegratedGpu(index - discrete_count)
            }
        },
        other => panic!("{other:?} is not supported on metal, expected `default` or `discrete[:N]`"),
    };

    cubecl::Device::from(kind)
}

/// How many Metal GPUs share the kind of `first`.
#[cfg(feature = "metal")]
fn metal_gpu_count(first: cubecl::device::MetalDevice) -> usize {
    let device = cubecl::Device::from(first);
    let device_id = device.to_id();
    let listed = cubecl::Device::enumerate(device_id);
    listed.len()
}

fn compiled_accelerators() -> Vec<&'static str> {
    let features = [
        ("cuda", cfg!(feature = "cuda")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("metal", cfg!(feature = "metal")),
        ("rocm", cfg!(feature = "rocm")),
    ];

    features
        .into_iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(name, _)| name)
        .collect()
}

/// Picks the accelerator `requested` names, or the first of `compiled` in [ACCELERATORS] order when
/// nothing is requested.
///
/// The error says why the name cannot be used, or that no backend is compiled in at all.
fn pick_accelerator(requested: Option<&str>, compiled: &[&str]) -> Result<&'static str, String> {
    let Some(name) = requested else {
        let first_compiled = ACCELERATORS.iter().find(|known| compiled.contains(known));
        let accepted = ACCELERATORS.join(", ");
        let message =
            format!("no accelerator is compiled in, build with `--features` naming one of {accepted}");
        return first_compiled.copied().ok_or(message);
    };

    let Some(known) = ACCELERATORS.iter().find(|known| **known == name) else {
        let accepted = ACCELERATORS.join(", ");
        let message =
            format!("{ACCELERATOR_ENV}={name} is not a known accelerator, expected one of {accepted}");
        return Err(message);
    };

    if !compiled.contains(known) {
        let message =
            format!("{ACCELERATOR_ENV}={name} is known but not compiled in, build with `--features {name}`");
        return Err(message);
    }

    Ok(known)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_first_compiled_in_accelerator_order() {
        let over_vulkan = pick_accelerator(None, &["rocm", "vulkan"]);
        let over_cuda = pick_accelerator(None, &["rocm", "cuda"]);
        let over_metal = pick_accelerator(None, &["rocm", "metal"]);
        let only_rocm = pick_accelerator(None, &["rocm"]);

        assert_eq!(over_vulkan, Ok("vulkan"));
        assert_eq!(over_cuda, Ok("cuda"));
        assert_eq!(over_metal, Ok("metal"));
        assert_eq!(only_rocm, Ok("rocm"));
    }

    #[test]
    fn default_without_a_compiled_backend_is_an_error() {
        let message = pick_accelerator(None, &[]).unwrap_err();

        assert!(message.contains("--features"));
    }

    #[test]
    fn requested_compiled_accelerator_is_used() {
        let picked = pick_accelerator(Some("rocm"), &["vulkan", "rocm"]);

        assert_eq!(picked, Ok("rocm"));
    }

    #[test]
    fn unknown_accelerator_lists_the_accepted_names() {
        let message = pick_accelerator(Some("opencl"), &["vulkan"]).unwrap_err();

        assert!(message.contains("opencl"));
        assert!(message.contains("cuda, vulkan, metal, rocm"));
    }

    #[test]
    fn known_but_missing_accelerator_names_the_feature() {
        let message = pick_accelerator(Some("cuda"), &["vulkan"]).unwrap_err();

        assert!(message.contains("`--features cuda`"));
    }
}
