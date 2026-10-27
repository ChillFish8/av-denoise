#[cfg(feature = "rocm")]
use cubecl::device::AmdDevice;
#[cfg(feature = "metal")]
use cubecl::device::MetalDevice;
#[cfg(feature = "vulkan")]
use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

use crate::backend::accelerate::Accelerator;
use crate::backend::device::Device;
#[cfg(any(feature = "rocm", feature = "metal"))]
use crate::backend::resolve::BackendError;
use crate::backend::resolve::resolve;
#[cfg(feature = "metal")]
use crate::backend::resolve::{MetalGpus, metal_device};
#[cfg(any(feature = "rocm", feature = "metal"))]
use crate::backend::selected_device;

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

#[cfg(feature = "metal")]
#[test]
fn metal_rejects_wgpu_only_device_kinds() {
    let unsupported = [
        Device::Integrated { index: 0 },
        Device::Virtual { index: 1 },
        Device::Cpu,
    ];

    for device in unsupported {
        let expected = BackendError::UnsupportedDevice {
            accelerator: Accelerator::Metal,
            device: device.clone(),
        };
        let error = resolve(Accelerator::Metal, device).expect_err("metal names only default and an index");
        assert_eq!(error, expected);
    }

    let error = resolve(Accelerator::Metal, Device::Cpu).expect_err("metal has no cpu device");
    let message = error.to_string();
    assert_eq!(
        message,
        "device Cpu is not supported on the Metal runtime, use `default` or `discrete[:N]`",
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_accepts_the_default_and_an_index() {
    let gpus = MetalGpus {
        discrete: 2,
        integrated: 1,
    };
    let cases = [
        (Device::Default, MetalDevice::DefaultDevice),
        (Device::Discrete { index: 0 }, MetalDevice::DiscreteGpu(0)),
        (Device::Discrete { index: 1 }, MetalDevice::DiscreteGpu(1)),
        (Device::Discrete { index: 2 }, MetalDevice::IntegratedGpu(0)),
    ];

    for (device, expected) in cases {
        let label = device.to_string();
        let mapped = metal_device(device, gpus).expect("metal names the default and every index");
        assert_eq!(mapped, expected, "{label}");
    }

    let apple_silicon = MetalGpus {
        discrete: 0,
        integrated: 1,
    };
    let first_gpu = metal_device(Device::Discrete { index: 0 }, apple_silicon);
    let past_the_end = metal_device(Device::Discrete { index: 1 }, apple_silicon);
    let missing = BackendError::MissingDevice {
        accelerator: Accelerator::Metal,
        device: Device::Discrete { index: 1 },
        available: 1,
    };
    assert_eq!(first_gpu, Ok(MetalDevice::IntegratedGpu(0)));
    assert_eq!(past_the_end, Err(missing));

    let resolved = resolve(Accelerator::Metal, Device::Default).expect("metal names the default");
    let default_device = cubecl::Device::from(MetalDevice::DefaultDevice);
    assert_eq!(resolved, default_device);
}

#[cfg(feature = "metal")]
#[test]
fn metal_rejects_an_index_past_its_gpus() {
    let gpus = MetalGpus {
        discrete: 0,
        integrated: 1,
    };

    let device = Device::Discrete { index: 1 };
    let error = metal_device(device, gpus).expect_err("one gpu has no index 1");
    let message = error.to_string();
    assert_eq!(
        message,
        "device discrete:1 is not on this machine, the Metal runtime sees 1 GPU",
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_lists_an_index_for_every_gpu() {
    let gpus = MetalGpus {
        discrete: 1,
        integrated: 1,
    };

    let listed = gpus.selectors();

    let expected = [Device::Discrete { index: 0 }, Device::Discrete { index: 1 }];
    assert_eq!(listed, expected);
    for device in listed {
        let label = device.to_string();
        let mapped = metal_device(device, gpus);
        assert!(mapped.is_ok(), "{label} is listed but does not resolve");
    }
}

#[cfg(feature = "rocm")]
#[test]
fn a_device_rocm_cannot_name_reports_the_accepted_forms() {
    let device = Device::Cpu;

    let error = selected_device(&[Accelerator::Rocm], &device).expect_err("rocm has no cpu device");

    let message = error.to_string();
    assert!(message.contains("use `default` or `discrete[:N]`"), "{message}");
}

#[cfg(feature = "metal")]
#[test]
fn a_device_metal_cannot_name_reports_the_accepted_forms() {
    let device = Device::Integrated { index: 0 };

    let error = selected_device(&[Accelerator::Metal], &device).expect_err("metal has no integrated kind");

    let message = error.to_string();
    assert!(message.contains("use `default` or `discrete[:N]`"), "{message}");
}
