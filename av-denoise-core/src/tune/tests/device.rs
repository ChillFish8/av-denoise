use super::runtime::test_client;
use crate::tune::{DeviceIdentity, DeviceLimits};

fn r9700_identity() -> DeviceIdentity {
    DeviceIdentity {
        backend: "vulkan".to_string(),
        adapter: "AMD Radeon AI PRO R9700 (RADV GFX1201)".to_string(),
        vendor_id: 0x1002,
        device_id: 0x7551,
        driver: "radv Mesa 25.2.4".to_string(),
        os: "linux".to_string(),
        compute_capability: None,
    }
}

/// A literal, so a change to the hash or its input text fails here instead of silently orphaning
/// every stored result.
#[test]
fn the_identity_hash_is_stable() {
    let identity = r9700_identity();
    let hash = identity.identity_hash();

    assert_eq!(hash, "81afbc01fae15a6c47ceddbcb0036d0a");
}

#[test]
fn every_identity_field_changes_the_hash() {
    let base = r9700_identity();
    let base_hash = base.identity_hash();

    let mut other_driver = base.clone();
    other_driver.driver = "radv Mesa 25.3.0".to_string();
    let mut other_os = base.clone();
    other_os.os = "windows".to_string();
    let mut other_device = base.clone();
    other_device.device_id = 0x7550;

    for changed in [other_driver, other_os, other_device] {
        let changed_hash = changed.identity_hash();
        assert_ne!(changed_hash, base_hash, "{changed:?}");
    }
}

#[test]
fn limits_come_from_the_runtime() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);

    assert!(limits.plane_size_min > 0);
    assert!(limits.plane_size_max >= limits.plane_size_min);
    assert!(limits.max_shared_bytes >= 16 * 1024);
    assert!(limits.max_units_per_cube >= 256);
    assert!(limits.max_cube_dim.0 >= 64);
}

#[test]
fn the_f16_search_needs_both_buffer_and_arithmetic_support() {
    let client = test_client();
    let mut limits = DeviceLimits::from_client(&client);

    limits.f16_buffer = true;
    limits.f16_arithmetic = false;
    assert!(!limits.supports_f16_search());

    limits.f16_arithmetic = true;
    assert!(limits.supports_f16_search());
}

#[test]
fn the_compute_capability_does_not_change_the_hash() {
    let identity = r9700_identity();
    let mut with_capability = identity.clone();
    with_capability.compute_capability = Some(86);

    let base_hash = identity.identity_hash();
    let capability_hash = with_capability.identity_hash();

    assert_eq!(base_hash, capability_hash);
}
