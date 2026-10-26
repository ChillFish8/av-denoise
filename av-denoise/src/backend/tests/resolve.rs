#[cfg(feature = "rocm")]
use cubecl::device::AmdDevice;
#[cfg(feature = "vulkan")]
use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

use crate::backend::accelerate::Accelerator;
use crate::backend::device::Device;
#[cfg(feature = "rocm")]
use crate::backend::resolve::BackendError;
use crate::backend::resolve::resolve;

#[cfg(feature = "vulkan")]
#[test]
fn resolve_maps_each_vulkan_device_kind() {
    let cases = [
        (Device::Default, WgpuDeviceKind::DefaultDevice),
        (Device::Discrete { index: 1 }, WgpuDeviceKind::DiscreteGpu(1)),
        (Device::Integrated { index: 2 }, WgpuDeviceKind::IntegratedGpu(2)),
        (Device::Virtual { index: 3 }, WgpuDeviceKind::VirtualGpu(3)),
        (Device::Cpu, WgpuDeviceKind::Cpu),
    ];

    for (device, kind) in cases {
        let label = device.to_string();
        let resolved = resolve(Accelerator::Vulkan, device).expect("vulkan names every device kind");

        let expected_device = WgpuDevice::new(kind).on(WgpuBackend::Vulkan);
        let expected = cubecl::Device::from(expected_device);
        assert_eq!(resolved, expected, "{label}");
    }
}

#[cfg(feature = "rocm")]
#[test]
fn resolve_maps_a_rocm_index() {
    let discrete = resolve(Accelerator::Rocm, Device::Discrete { index: 2 }).expect("discrete:2");
    let default = resolve(Accelerator::Rocm, Device::Default).expect("default");

    let second_gpu = AmdDevice::new(2);
    let first_gpu = AmdDevice::new(0);
    let expected_discrete = cubecl::Device::from(second_gpu);
    let expected_default = cubecl::Device::from(first_gpu);
    assert_eq!(discrete, expected_discrete);
    assert_eq!(default, expected_default);
}

#[cfg(feature = "rocm")]
#[test]
fn resolve_rejects_a_device_the_backend_does_not_have() {
    let unsupported = [
        Device::Integrated { index: 0 },
        Device::Virtual { index: 1 },
        Device::Cpu,
    ];

    for device in unsupported {
        let expected = BackendError::UnsupportedDevice {
            accelerator: Accelerator::Rocm,
            device: device.clone(),
        };
        let error = resolve(Accelerator::Rocm, device).expect_err("rocm names only default and discrete");
        assert_eq!(error, expected);
    }

    let error = resolve(Accelerator::Rocm, Device::Cpu).expect_err("rocm has no cpu device");
    let message = error.to_string();
    assert_eq!(
        message,
        "device Cpu is not supported on the ROCm runtime, use `default` or `discrete[:N]`",
    );
}
