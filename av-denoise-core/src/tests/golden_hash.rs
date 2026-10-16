use std::sync::OnceLock;

use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, init_setup};

/// The adapter a golden table was recorded on.
pub(crate) struct RecordedDevice {
    pub(crate) name: &'static str,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    /// The wgpu backend, as its `Debug` form prints it.
    pub(crate) backend: &'static str,
}

/// The RADV R9700 every golden table in the crate was recorded on.
pub(crate) const RADV_R9700: RecordedDevice = RecordedDevice {
    name: "AMD Radeon AI PRO R9700 (RADV GFX1201)",
    vendor_id: 0x1002,
    device_id: 0x7551,
    backend: "Vulkan",
};

/// The adapter the default test client runs on.
#[derive(Debug)]
pub(crate) struct LiveDevice {
    pub(crate) name: String,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) backend: String,
}

/// Creates the default wgpu client and reports which adapter it runs on.
///
/// The client is made the way the first `client()` call makes it, with the default options, so
/// every later client in the process shares it. It has to run before any other client is made.
pub(crate) fn live_device() -> &'static LiveDevice {
    static LIVE: OnceLock<LiveDevice> = OnceLock::new();

    LIVE.get_or_init(|| {
        crate::tests::compilation_cache::install_test_compilation_cache();

        let device = WgpuDevice::DefaultDevice;
        let options = RuntimeOptions::default();
        let setup = init_setup::<AutoGraphicsApi>(&device, options);
        let info = setup.adapter.get_info();
        let backend = format!("{:?}", setup.backend);

        LiveDevice {
            name: info.name,
            vendor_id: info.vendor,
            device_id: info.device,
            backend,
        }
    })
}

/// Whether the test client runs on `recorded`, printing a skip notice when it does not.
pub(crate) fn runs_on_recording_device(recorded: &RecordedDevice) -> bool {
    let live = live_device();
    let matches = live.vendor_id == recorded.vendor_id
        && live.device_id == recorded.device_id
        && live.backend == recorded.backend;

    if !matches {
        let recorded_label = format!(
            "{} ({:#06x}:{:#06x}, {})",
            recorded.name, recorded.vendor_id, recorded.device_id, recorded.backend,
        );
        let live_label = format!(
            "{} ({:#06x}:{:#06x}, {})",
            live.name, live.vendor_id, live.device_id, live.backend,
        );
        eprintln!("skipping golden hashes: recorded on {recorded_label}, running on {live_label}");
    }

    matches
}

/// The FNV-1a hash of `words`, each fed as four little-endian bytes.
pub(crate) fn fnv1a_words(words: impl IntoIterator<Item = u32>) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;

    for word in words {
        let bytes = word.to_le_bytes();
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01B3);
        }
    }

    hash
}
