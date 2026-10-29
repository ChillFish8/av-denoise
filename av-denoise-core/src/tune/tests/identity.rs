use crate::tune::DeviceIdentity;

const EXPECTED_HASH: &str = "d89f107b8c7a6b5a17831feeb32045a3";

#[test]
fn a_known_identity_keeps_its_hash() {
    let identity = DeviceIdentity {
        backend: "vulkan".to_string(),
        adapter: "AMD Radeon RX 7900 XTX (RADV NAVI31)".to_string(),
        vendor_id: 0x1002,
        device_id: 0x744c,
        driver: "radv Mesa 25.2.0".to_string(),
        os: "linux".to_string(),
        compute_capability: None,
    };

    assert_eq!(identity.identity_hash(), EXPECTED_HASH);
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use std::str::FromStr;

    use cubecl::prelude::Client;
    use cubecl::wgpu::WgpuDeviceKind;

    use crate::bench_api::Device;
    use crate::testing::{device_on, test_device_label};
    use crate::tune::describe;
    use crate::tune::identity::adapter::parse_override;
    use crate::tune::identity::fallback_identity;

    /// The test device pinned to Vulkan, with a client on it.
    fn test_vulkan_client() -> (cubecl::Device, Client) {
        let label = test_device_label();
        let parsed = Device::from_str(&label);
        let selector = parsed.unwrap_or_else(|message| panic!("AVD_TEST_DEVICE={label}: {message}"));
        let selected = device_on("vulkan", selector);
        let client = selected.client();
        (selected, client)
    }

    #[test]
    fn overrides_parse_the_cubecl_spellings() {
        assert_eq!(
            parse_override("DiscreteGpu(1)"),
            Some(WgpuDeviceKind::DiscreteGpu(1))
        );
        assert_eq!(
            parse_override("IntegratedGpu(0)"),
            Some(WgpuDeviceKind::IntegratedGpu(0))
        );
        assert_eq!(parse_override("Cpu"), Some(WgpuDeviceKind::Cpu));
        assert_eq!(parse_override("Discrete(1)"), None);
    }

    #[test]
    fn the_default_vulkan_adapter_has_a_name_and_a_vendor() {
        let (selected, client) = test_vulkan_client();

        let identity = describe(&selected, &client);

        assert_eq!(identity.backend, "vulkan");
        assert!(!identity.adapter.is_empty());
        assert!(!identity.adapter.starts_with("unknown-"), "{identity:?}");
        assert_ne!(identity.vendor_id, 0);
    }

    #[test]
    fn describing_twice_gives_the_same_identity() {
        let (selected, client) = test_vulkan_client();

        let first = describe(&selected, &client);
        let second = describe(&selected, &client);

        assert_eq!(first.identity_hash(), second.identity_hash());
    }

    #[test]
    fn the_fallback_identity_is_stable_for_one_device() {
        let (_, client) = test_vulkan_client();

        let first = fallback_identity("vulkan", &client);
        let second = fallback_identity("vulkan", &client);

        assert_eq!(first, second);
        assert!(first.adapter.starts_with("unknown-"));
    }
}

#[cfg(feature = "metal")]
mod metal {
    use crate::tune::identity::{metal_identity, product_version};

    #[test]
    fn metal_identity_keys_on_the_name_and_os_version() {
        let first = metal_identity("Apple M2 Max", "15.1");
        let same = metal_identity("Apple M2 Max", "15.1");
        let upgraded = metal_identity("Apple M2 Max", "15.2");

        assert_eq!(first.backend, "metal");
        assert_eq!(first.adapter, "Apple M2 Max");
        assert_eq!(first.identity_hash(), same.identity_hash());
        assert_ne!(first.identity_hash(), upgraded.identity_hash());
    }

    #[test]
    fn product_version_reads_the_system_version_plist() {
        let plist = "<dict>\n\t<key>ProductName</key>\n\t<string>macOS</string>\n\t\
                     <key>ProductVersion</key>\n\t<string>15.1</string>\n</dict>";

        let version = product_version(plist);
        let missing = product_version("<dict></dict>");

        assert_eq!(version.as_deref(), Some("15.1"));
        assert_eq!(missing, None);
    }
}
