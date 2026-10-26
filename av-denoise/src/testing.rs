use std::str::FromStr;

use crate::Device;
use crate::accelerate::Accelerator;

/// The accelerator the test environment names, or the first one this build has.
///
/// It reads `AVD_TEST_ACCELERATOR` by the same rules as
/// [test_accelerator](av_denoise_core::testing::test_accelerator), so both crates pick one backend.
pub fn test_accelerator() -> Accelerator {
    let name = av_denoise_core::testing::test_accelerator();
    let parsed = Accelerator::from_str(name);
    parsed.unwrap_or_else(|_| panic!("{name} is compiled in but does not parse as an accelerator"))
}

/// The device `AVD_TEST_DEVICE` names, or [Device::Default] without it.
pub fn test_device() -> Device {
    let label = av_denoise_core::testing::test_device_label();
    let parsed = Device::from_str(&label);
    parsed.unwrap_or_else(|message| panic!("AVD_TEST_DEVICE={label}: {message}"))
}
