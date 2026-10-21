use av_denoise_core::tune::{DeviceIdentity, DeviceLimits, DeviceProfile, PipelineProfile};
use av_denoise_core::{
    DEFAULT_PILOT_STRENGTH_SCALE,
    Nl4dOptions,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
};

use crate::tune::driver::{alternative_profiles_for, f32_profile_applies};
use crate::tune::{KernelRequest, KernelSetup, TuningMode, engine_kernels};
use crate::{Algorithm, Depth};

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

#[test]
fn nlmeans_is_offered_only_the_integer_search() {
    let device = r9700_profile();
    let nlmeans_algorithm = Algorithm::default();
    let nl4d_algorithm = nl4d();

    let nlmeans = alternative_profiles_for(&nlmeans_algorithm, Depth::Eight, &device);
    let nl4d = alternative_profiles_for(&nl4d_algorithm, Depth::Eight, &device);

    assert_eq!(nlmeans, vec![PipelineProfile::IntegerSearch]);
    assert!(nl4d.contains(&PipelineProfile::SplitSearch));
    assert!(nl4d.contains(&PipelineProfile::F32Search));
}

#[test]
fn a_prefiltered_nlmeans_is_not_offered_integer_search() {
    let device = r9700_profile();
    let pilot = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };
    let bilateral = PrefilterMode::Bilateral {
        sigma_s: 3.0,
        sigma_r: 0.02,
    };

    for prefilter in [pilot, bilateral] {
        let fast = nlmeans_with(prefilter);
        let hq = nlmeans_hq_with(prefilter);

        let fast_profiles = alternative_profiles_for(&fast, Depth::Eight, &device);
        let hq_profiles = alternative_profiles_for(&hq, Depth::Eight, &device);

        assert!(fast_profiles.is_empty(), "{prefilter:?}: {fast_profiles:?}");
        assert!(hq_profiles.is_empty(), "{prefilter:?}: {hq_profiles:?}");
    }

    let unfiltered_fast = nlmeans_with(PrefilterMode::None);
    let unfiltered_hq = nlmeans_hq_with(PrefilterMode::None);

    let fast_profiles = alternative_profiles_for(&unfiltered_fast, Depth::Eight, &device);
    let hq_profiles = alternative_profiles_for(&unfiltered_hq, Depth::Eight, &device);

    assert_eq!(fast_profiles, vec![PipelineProfile::IntegerSearch]);
    assert_eq!(hq_profiles, vec![PipelineProfile::IntegerSearch]);
}

#[test]
fn a_device_with_illegal_f16_is_not_offered_the_f32_profile() {
    let device = pascal_vulkan_profile();
    let algorithm = nl4d();

    let profiles = alternative_profiles_for(&algorithm, Depth::Eight, &device);

    assert!(!profiles.contains(&PipelineProfile::F32Search));
    assert!(profiles.contains(&PipelineProfile::SplitSearch));
}

#[test]
fn packed_atomics_are_offered_only_where_legal() {
    let capable = r9700_profile();
    let mut incapable = r9700_profile();
    incapable.limits.int64_atomic_add = false;
    let algorithm = nl4d();

    let offered = alternative_profiles_for(&algorithm, Depth::Eight, &capable);
    let withheld = alternative_profiles_for(&algorithm, Depth::Eight, &incapable);

    assert!(offered.contains(&PipelineProfile::PackedAtomics));
    assert!(!withheld.contains(&PipelineProfile::PackedAtomics));
}

#[test]
fn integer_search_is_offered_only_for_eight_and_ten_bit_sources() {
    let device = r9700_profile();
    let nl4d_algorithm = nl4d();
    let nlmeans_algorithm = Algorithm::default();

    for depth in [Depth::Eight, Depth::Ten] {
        let profiles = alternative_profiles_for(&nl4d_algorithm, depth, &device);
        let last = profiles.last().copied();

        assert_eq!(
            last,
            Some(PipelineProfile::IntegerSearch),
            "{depth:?}: {profiles:?}"
        );
    }

    let twelve_bit = alternative_profiles_for(&nl4d_algorithm, Depth::Twelve, &device);
    assert!(
        !twelve_bit.contains(&PipelineProfile::IntegerSearch),
        "{twelve_bit:?}"
    );

    for depth in [Depth::Eight, Depth::Ten] {
        let nlmeans = alternative_profiles_for(&nlmeans_algorithm, depth, &device);

        assert_eq!(nlmeans, vec![PipelineProfile::IntegerSearch], "{depth:?}");
    }

    let twelve_bit_nlmeans = alternative_profiles_for(&nlmeans_algorithm, Depth::Twelve, &device);
    assert!(twelve_bit_nlmeans.is_empty(), "{twelve_bit_nlmeans:?}");
}

#[test]
fn integer_search_needs_the_code_storage_type() {
    let mut device = r9700_profile();
    device.limits.u16_buffer = false;
    let algorithm = nl4d();

    for depth in [Depth::Eight, Depth::Ten] {
        let profiles = alternative_profiles_for(&algorithm, depth, &device);

        assert!(
            !profiles.contains(&PipelineProfile::IntegerSearch),
            "{depth:?}: {profiles:?}"
        );
        assert!(profiles.contains(&PipelineProfile::SplitSearch), "{depth:?}");
    }
}

fn nl4d() -> Algorithm {
    let options = Nl4dOptions::default();
    Algorithm::Nl4d(options)
}

fn nlmeans_with(prefilter: PrefilterMode) -> Algorithm {
    let options = NlmeansOptions {
        prefilter,
        ..NlmeansOptions::default()
    };

    Algorithm::Nlmeans(options)
}

fn nlmeans_hq_with(prefilter: PrefilterMode) -> Algorithm {
    let nlm = NlmeansOptions {
        prefilter,
        ..NlmeansOptions::default()
    };
    let options = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };

    Algorithm::NlmeansHq(options)
}

/// An AMD Radeon AI PRO R9700 under Vulkan.
fn r9700_profile() -> DeviceProfile {
    let mut device = pascal_vulkan_profile();
    device.identity.adapter = "AMD Radeon AI PRO R9700".to_string();
    device.identity.vendor_id = 0x1002;
    device.identity.device_id = 0x7551;
    device.limits.int64_atomic_add = true;
    device.limits.f16_matrix = true;
    device.limits.u16_buffer = true;

    device
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
        f16_matrix: false,
        u16_buffer: false,
    };

    DeviceProfile { identity, limits }
}
