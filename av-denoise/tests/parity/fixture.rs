use std::collections::BTreeMap;
use std::path::PathBuf;

use wgpu::{Adapter, Backends, Instance, InstanceDescriptor};

pub type Entries = BTreeMap<(String, String), String>;

const ADAPTERS_PREFIX: &str = "# adapters: ";

/// The recorded adapter fingerprint and hashes.
pub struct Fixture {
    pub adapters: String,
    pub entries: Entries,
}

/// FNV-1a 64, chosen because its output never changes between Rust releases.
pub fn hash(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325u64;

    for &byte in bytes {
        state ^= byte as u64;
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }

    state
}

pub fn path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("tests/fixtures/parity.txt")
}

/// Names and drivers of every Vulkan adapter, sorted and joined with `; `.
pub fn adapter_fingerprint() -> String {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::VULKAN;
    let instance = Instance::new(descriptor);
    let enumeration = instance.enumerate_adapters(Backends::VULKAN);
    let adapters = cubecl::future::block_on(enumeration);

    let mut described: Vec<String> = adapters.iter().map(describe).collect();
    described.sort();
    described.join("; ")
}

/// Panics when the recorded adapters differ from this machine's.
pub fn check_adapters(recorded: &str) {
    let current = adapter_fingerprint();
    if recorded == current {
        return;
    }

    panic!(
        "parity fixture was recorded on different adapters\n recorded: {recorded}\n current: {current}\n \
         re-record with AVD_PARITY_RECORD=1 on this machine before starting a refactor, never mid-refactor"
    );
}

/// Reads the fixture.
///
/// Every line after the adapters header is `<config> <label> <hash as 16 hex digits>`.
pub fn read() -> Fixture {
    let fixture_path = path();
    let text = std::fs::read_to_string(fixture_path)
        .expect("parity fixture missing, record it with AVD_PARITY_RECORD=1");

    let mut lines = text.lines();
    let header = lines
        .next()
        .expect("parity fixture has no adapters header, re-record it with AVD_PARITY_RECORD=1");
    let adapters = header
        .strip_prefix(ADAPTERS_PREFIX)
        .expect("parity fixture has no adapters header, re-record it with AVD_PARITY_RECORD=1")
        .to_string();

    let mut entries = Entries::new();

    for line in lines {
        let mut parts = line.split_whitespace();
        let config = parts.next().expect("config").to_string();
        let label = parts.next().expect("label").to_string();
        let hash = parts.next().expect("hash").to_string();
        entries.insert((config, label), hash);
    }

    Fixture { adapters, entries }
}

pub fn write(entries: &Entries) {
    let adapters = adapter_fingerprint();
    let mut text = format!("{ADAPTERS_PREFIX}{adapters}\n");

    for ((config, label), hash) in entries {
        let line = format!("{config} {label} {hash}\n");
        text.push_str(&line);
    }

    let fixture_path = path();
    std::fs::write(fixture_path, text).expect("write parity fixture");
}

fn describe(adapter: &Adapter) -> String {
    let info = adapter.get_info();
    format!("{}|{}|{}", info.name, info.driver, info.driver_info)
}
