use super::fixtures::{CollabSpec, collab_site};
use super::probe::small_device_limits;
use super::runtime::test_client;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::sync;
use crate::tune::sites::{CollabVariant, Walk};
use crate::tune::{
    APPLE_VENDOR_ID,
    DeviceClass,
    DeviceIdentity,
    DeviceLimits,
    DeviceProfile,
    INTEL_VENDOR_ID,
    Legality,
    NVIDIA_VENDOR_ID,
    SiteCandidate,
    VariantSite,
    check_plane_size,
    legal_candidates,
};

/// The GTX 1070, a GP104 Pascal consumer part.
const GTX_1070: u32 = 0x1B81;
/// The GTX 970, a GM204 Maxwell part.
const GTX_970: u32 = 0x13C2;
/// The GTX 680, a GK104 Kepler part.
const GTX_680: u32 = 0x1180;
/// The Quadro GP100, a Pascal part with fast f16.
const QUADRO_GP100: u32 = 0x15F0;
/// The RTX 3050 Ti laptop, an Ampere part the table does not list.
const RTX_3050_TI_LAPTOP: u32 = 0x25A0;

/// A device profile with the given identity and 32-wide planes.
pub(crate) fn faked_profile(
    backend: &str,
    vendor_id: u32,
    device_id: u32,
    compute_capability: Option<u32>,
) -> DeviceProfile {
    let identity = DeviceIdentity {
        backend: backend.to_string(),
        adapter: "faked".to_string(),
        vendor_id,
        device_id,
        driver: "faked".to_string(),
        os: "linux".to_string(),
        compute_capability,
    };
    let mut limits = small_device_limits();
    limits.plane_size_min = 32;
    limits.plane_size_max = 32;
    limits.max_shared_bytes = 48 * 1024;
    limits.f16_buffer = true;
    limits.f16_arithmetic = true;
    limits.int64_atomic_add = true;

    DeviceProfile { identity, limits }
}

#[test]
fn pascal_maxwell_and_kepler_under_vulkan_are_pre_volta_with_slow_f16() {
    for device_id in [GTX_1070, GTX_970, GTX_680] {
        let profile = faked_profile("vulkan", NVIDIA_VENDOR_ID, device_id, None);
        let class = DeviceClass::of(&profile.identity);
        let legality = Legality::for_device(&profile);

        assert_eq!(
            class,
            DeviceClass::NvidiaPreVolta { fast_f16: false },
            "{device_id:#06x}"
        );
        assert!(legality.uniform_walk_only, "{device_id:#06x}");
        assert!(!legality.f16_search, "{device_id:#06x}");
    }
}

#[test]
fn pascal_and_maxwell_under_cuda_are_pre_volta_with_slow_f16() {
    for capability in [52, 61] {
        let profile = faked_profile("cuda", NVIDIA_VENDOR_ID, 0, Some(capability));
        let legality = Legality::for_device(&profile);

        assert!(legality.uniform_walk_only, "sm_{capability}");
        assert!(!legality.f16_search, "sm_{capability}");
    }
}

#[test]
fn gp100_keeps_the_f16_search_but_walks_uniformly() {
    let vulkan = faked_profile("vulkan", NVIDIA_VENDOR_ID, QUADRO_GP100, None);
    let cuda = faked_profile("cuda", NVIDIA_VENDOR_ID, QUADRO_GP100, Some(60));

    for profile in [vulkan, cuda] {
        let legality = Legality::for_device(&profile);

        assert!(legality.uniform_walk_only, "{}", profile.identity.backend);
        assert!(legality.f16_search, "{}", profile.identity.backend);
    }
}

#[test]
fn volta_and_later_keep_todays_behaviour() {
    let cuda = faked_profile("cuda", NVIDIA_VENDOR_ID, RTX_3050_TI_LAPTOP, Some(86));
    let vulkan = faked_profile("vulkan", NVIDIA_VENDOR_ID, RTX_3050_TI_LAPTOP, None);

    for profile in [cuda, vulkan] {
        let class = DeviceClass::of(&profile.identity);
        let legality = Legality::for_device(&profile);

        assert_eq!(class, DeviceClass::Nvidia, "{}", profile.identity.backend);
        assert!(!legality.uniform_walk_only, "{}", profile.identity.backend);
        assert!(legality.f16_search, "{}", profile.identity.backend);
    }
}

/// An NVIDIA id outside the tables, or an unread CUDA capability, is treated as Volta or later.
#[test]
fn an_unknown_nvidia_device_keeps_todays_behaviour() {
    let unknown_id = faked_profile("vulkan", NVIDIA_VENDOR_ID, 0xFFFF, None);
    let unread_capability = faked_profile("cuda", NVIDIA_VENDOR_ID, 0, None);

    for profile in [unknown_id, unread_capability] {
        let legality = Legality::for_device(&profile);
        let defaults = Legality::default();

        assert_eq!(legality.uniform_walk_only, defaults.uniform_walk_only);
        assert_eq!(legality.f16_search, defaults.f16_search);
    }
}

#[test]
fn intel_and_unknown_vendors_get_their_own_classes() {
    let intel = faked_profile("vulkan", INTEL_VENDOR_ID, 0xE20B, None);
    let unknown = faked_profile("vulkan", 0x1234, 0x0001, None);

    let intel_class = DeviceClass::of(&intel.identity);
    let unknown_class = DeviceClass::of(&unknown.identity);

    assert_eq!(intel_class, DeviceClass::Intel);
    assert_eq!(unknown_class, DeviceClass::Unknown);
}

#[test]
fn packed_atomics_need_64_bit_atomics_and_never_run_on_metal() {
    let capable = faked_profile("vulkan", INTEL_VENDOR_ID, 0xE20B, None);
    let mut incapable = capable.clone();
    incapable.limits.int64_atomic_add = false;
    let metal = faked_profile("metal", APPLE_VENDOR_ID, 0, None);

    assert!(Legality::for_device(&capable).packed_atomics);
    assert!(!Legality::for_device(&incapable).packed_atomics);
    assert!(!Legality::for_device(&metal).packed_atomics);
}

#[test]
fn defaults_are_todays_behaviour() {
    let defaults = Legality::default();

    assert!(!defaults.uniform_walk_only);
    assert!(defaults.f16_search);
    assert!(!defaults.packed_atomics);
}

#[test]
fn planes_narrower_than_eight_lanes_are_rejected() {
    let mut limits = small_device_limits();
    limits.plane_size_min = 4;

    let error = check_plane_size(&limits).expect_err("4 lanes cannot hold an 8-lane group");
    let message = error.to_string();

    assert_eq!(error.plane_size, 4);
    assert!(message.contains("4 lanes"), "{message}");

    limits.plane_size_min = 8;
    assert!(check_plane_size(&limits).is_ok());
}

#[test]
fn the_device_id_tables_are_sorted_and_disjoint() {
    let tables = crate::tune::legality::nvidia_tables_for_tests();

    for (name, ids) in tables {
        let sorted = ids.windows(2).all(|pair| pair[0] < pair[1]);
        assert!(sorted, "{name} must be strictly ascending for binary search");
    }

    for (index, (name, ids)) in tables.iter().enumerate() {
        for (other_name, other_ids) in &tables[index + 1..] {
            let shared = ids.iter().find(|id| other_ids.binary_search(id).is_ok());
            assert!(shared.is_none(), "{name} and {other_name} share {shared:?}");
        }
    }
}

fn luma_spec() -> CollabSpec {
    CollabSpec::new(64, 48, ChannelMode::Luma, 2)
}

#[test]
fn pre_volta_devices_are_offered_only_uniform_walks() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let devices = [
        faked_profile("vulkan", NVIDIA_VENDOR_ID, GTX_1070, None),
        faked_profile("vulkan", NVIDIA_VENDOR_ID, GTX_970, None),
        faked_profile("cuda", NVIDIA_VENDOR_ID, GTX_1070, Some(61)),
        faked_profile("cuda", NVIDIA_VENDOR_ID, GTX_970, Some(52)),
    ];

    for device in devices {
        let legality = Legality::for_device(&device);
        let mut site = collab_site(&client, &luma_spec());
        site.shape.uniform_walk_only = legality.uniform_walk_only;

        let offered = legal_candidates(&site, &limits, &legality).expect("the baseline fits");
        let baseline = site.baseline();

        assert!(!offered.is_empty());
        assert_eq!(baseline.walk, Walk::Uniform, "{}", device.identity.backend);
        assert!(
            offered.iter().all(|variant| variant.walk == Walk::Uniform),
            "{} offered a clipped walk",
            device.identity.backend,
        );
    }
}

#[test]
fn uniform_only_devices_keep_the_combined_uniform_variants() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let devices = [
        faked_profile("vulkan", NVIDIA_VENDOR_ID, GTX_1070, None),
        faked_profile("cuda", NVIDIA_VENDOR_ID, GTX_970, Some(52)),
    ];
    let uniform: Vec<CollabVariant> = CollabVariant::ALL
        .iter()
        .copied()
        .filter(|variant| variant.walk == Walk::Uniform)
        .collect();

    for device in devices {
        let legality = Legality::for_device(&device);
        let mut site = collab_site(&client, &luma_spec());
        site.shape.uniform_walk_only = legality.uniform_walk_only;

        let offered = legal_candidates(&site, &limits, &legality).expect("the baseline fits");
        let names: Vec<&str> = offered.iter().map(|variant| variant.name()).collect();
        let backend = &device.identity.backend;

        assert_eq!(offered, uniform, "{backend}");
        assert_eq!(offered.len(), 30, "{backend}");
        assert_eq!(names[0], "groups8_uniform", "{backend}");
        assert!(names.contains(&"groups4_uniform"), "{backend}");
        assert!(names.contains(&"groups16_uniform"), "{backend}");
        assert!(names.contains(&"groups4_uniform_tiled"), "{backend}");
        assert!(names.contains(&"groups16_uniform_nslide"), "{backend}");
        assert!(
            names.contains(&"groups16_uniform_tiled_batched_nslide"),
            "{backend}"
        );
    }

    // The HIP runtime crashes at process exit while an upload is still queued.
    sync(&client).expect("the uploads finish");
}

#[test]
fn an_unknown_nvidia_device_is_offered_both_walks_off_cuda() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let device = faked_profile("vulkan", NVIDIA_VENDOR_ID, 0xFFFF, None);
    let legality = Legality::for_device(&device);
    let site = collab_site(&client, &luma_spec());

    let offered = legal_candidates(&site, &limits, &legality).expect("the baseline fits");
    let has_clipped = offered.iter().any(|variant| variant.walk == Walk::Clipped);
    let runs_on_cuda = site.shape.uniform_walk_only;

    assert_eq!(has_clipped, !runs_on_cuda);
}
