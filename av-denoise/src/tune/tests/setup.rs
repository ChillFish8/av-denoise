use av_denoise_core::tune::{DeviceIdentity, DeviceLimits, DeviceProfile};

use crate::tune::driver::f32_profile_applies;
use crate::tune::{KernelRequest, KernelSetup, TuningMode, engine_kernels};

#[test]
fn tuning_off_still_applies_the_legality_rules() {
    let device = pascal_vulkan_profile();
    let setup = KernelSetup::default();
    let request = KernelRequest {
        tuning: TuningMode::Off,
        deterministic: false,
        profile_key: "nl4d",
        setup: &setup,
    };

    let kernels = engine_kernels(&request, &device);

    assert!(kernels.legality.uniform_walk_only);
    assert!(!kernels.legality.f16_search);
    assert!(kernels.plan.is_empty());
}

#[test]
fn the_f32_profile_does_not_apply_where_f16_is_illegal() {
    let device = pascal_vulkan_profile();

    let applies = f32_profile_applies(true, &device);

    assert!(!applies);
}

#[test]
fn the_f32_profile_applies_to_nl4d_where_f16_is_legal() {
    let mut device = pascal_vulkan_profile();
    device.identity.adapter = "NVIDIA GeForce RTX 4090".to_string();
    device.identity.device_id = 0x2684;

    let nl4d_applies = f32_profile_applies(true, &device);
    let nlmeans_applies = f32_profile_applies(false, &device);

    assert!(nl4d_applies);
    assert!(!nlmeans_applies);
}

/// A GTX 1070 under Vulkan whose driver reports f16 buffers and arithmetic.
fn pascal_vulkan_profile() -> DeviceProfile {
    let identity = DeviceIdentity {
        backend: "vulkan".to_string(),
        adapter: "NVIDIA GeForce GTX 1070".to_string(),
        vendor_id: 0x10DE,
        device_id: 0x1B81,
        driver: "nvidia 580.95".to_string(),
        os: "linux".to_string(),
        compute_capability: None,
    };
    let limits = DeviceLimits {
        plane_size_min: 32,
        plane_size_max: 32,
        max_shared_bytes: 48 * 1024,
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 64),
        compute_units: Some(15),
        f16_buffer: true,
        f16_arithmetic: true,
        int64_atomic_add: true,
    };

    DeviceProfile { identity, limits }
}
