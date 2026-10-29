use super::legality::faked_profile;
use crate::tune::setup::{KernelSetup, device_priority, engine_kernels};
use crate::tune::{Legality, NVIDIA_VENDOR_ID, TuningConfig, TuningMode};

/// The GTX 1070, whose legality rules differ from the defaults.
const GTX_1070: u32 = 0x1B81;

#[test]
fn off_builds_baselines_under_legality_and_priority() {
    let device = faked_profile("vulkan", NVIDIA_VENDOR_ID, GTX_1070, None);
    let setup = KernelSetup::default();
    let config = TuningConfig {
        mode: TuningMode::Off,
        ..TuningConfig::default()
    };

    let kernels = engine_kernels(&setup, &config, "nl4d", &device);
    let legality = Legality::for_device(&device);
    let priority = device_priority(&device);

    assert!(kernels.plan.is_empty());
    assert!(kernels.ledger.is_none());
    assert_eq!(kernels.legality, legality);
    assert_ne!(kernels.legality, Legality::default());
    assert!(std::ptr::eq(kernels.priority, priority));
}
