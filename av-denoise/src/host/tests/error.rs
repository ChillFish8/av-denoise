use crate::backend::accelerate::Accelerator;
use crate::backend::sniff::BackendUnavailable;
use crate::host::DenoiserError;

#[test]
fn no_accelerator_lists_why_each_backend_was_skipped() {
    let reasons = vec![
        BackendUnavailable {
            accelerator: Accelerator::Vulkan,
            reason: "could not create a device".to_string(),
        },
        BackendUnavailable {
            accelerator: Accelerator::Vulkan,
            reason: "no adapter".to_string(),
        },
    ];
    let error = DenoiserError::NoAcceleratorAvailable { reasons };

    let message = error.to_string();

    assert_eq!(
        message,
        "no accelerator from the priority list is available\n  vulkan: could not create a device\n  vulkan: no adapter"
    );
}

#[test]
fn no_accelerator_without_reasons_is_one_line() {
    let error = DenoiserError::NoAcceleratorAvailable { reasons: Vec::new() };

    let message = error.to_string();

    assert_eq!(message, "no accelerator from the priority list is available");
}
