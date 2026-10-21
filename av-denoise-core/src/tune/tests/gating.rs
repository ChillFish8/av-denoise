use cubecl::features::{EnumSet, Features, MmaConfig, TypeUsage};
use cubecl::ir::{
    DeviceProperties,
    ElemType,
    FloatKind,
    HardwareProperties,
    MemoryDeviceProperties,
    StorageType,
    UIntKind,
};
use cubecl::profile::TimingMethod;

use crate::engine::SampleFormat;
use crate::nlmeans::ChannelMode;
use crate::nlmeans::params::channel_scale_for;
use crate::tune::sites::{
    Centre,
    CollabShape,
    CollabVariant,
    FLOAT_SEARCH_COUNT,
    IntCentre,
    IntegerVariant,
    SearchVariant,
    Walk,
    WindowCandidate,
    WindowShape,
    search_candidate_fits,
    search_candidate_legal,
    search_shared_bytes,
    window_candidate_fits,
    window_candidate_legal,
    window_shared_bytes,
};
use crate::tune::{DeviceLimits, IntegerCodes, Legality, SiteCandidate, matrix_plane_width};

/// What a fake device offers, turned into the runtime properties a real client reports.
#[derive(Debug, Clone, Copy)]
struct FakeDevice {
    matrix_config: Option<MmaConfig>,
    plane_size_min: u32,
    plane_size_max: u32,
    shared_bytes: usize,
    u16_usage: EnumSet<TypeUsage>,
}

impl FakeDevice {
    /// An RDNA 4 card under RADV.
    fn rdna4() -> Self {
        let config = f16_matrix_config(FloatKind::F32, 16);

        Self {
            matrix_config: Some(config),
            plane_size_min: 32,
            plane_size_max: 64,
            shared_bytes: 64 * 1024,
            u16_usage: TypeUsage::all(),
        }
    }

    /// A Pascal card under Vulkan, which has no matrix units.
    fn pascal() -> Self {
        Self {
            matrix_config: None,
            plane_size_min: 32,
            plane_size_max: 32,
            shared_bytes: 48 * 1024,
            u16_usage: TypeUsage::all(),
        }
    }

    fn properties(&self) -> DeviceProperties {
        let mut features = Features::default();

        if let Some(config) = self.matrix_config {
            features.matmul.cmma.insert(config);
        }

        let memory = MemoryDeviceProperties {
            max_page_size: 1 << 30,
            alignment: 256,
        };
        let hardware = HardwareProperties {
            load_width: 128,
            plane_size_min: self.plane_size_min,
            plane_size_max: self.plane_size_max,
            max_bindings: 64,
            max_shared_memory_size: self.shared_bytes,
            max_cube_count: (65535, 65535, 65535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: Some(16),
            max_vector_size: 4,
        };
        let mut properties = DeviceProperties::new(features, memory, hardware, TimingMethod::Device);

        if !self.u16_usage.is_empty() {
            let u16_elem = ElemType::UInt(UIntKind::U16);
            properties.register_type_usage(u16_elem, self.u16_usage);
        }

        properties
    }

    fn limits(&self) -> DeviceLimits {
        let properties = self.properties();
        DeviceLimits::from_properties(&properties)
    }
}

/// A square f16-input matrix config of side `size`, accumulating in `accumulator`.
fn f16_matrix_config(accumulator: FloatKind, size: u32) -> MmaConfig {
    let half_elem = ElemType::Float(FloatKind::F16);
    let half_type = StorageType::Scalar(half_elem);
    let accumulator_elem = ElemType::Float(accumulator);
    let accumulator_type = StorageType::Scalar(accumulator_elem);

    MmaConfig {
        a_type: half_type,
        b_type: half_type,
        cd_type: accumulator_type,
        m: size,
        n: size,
        k: size,
    }
}

#[test]
fn limits_read_the_f16_matrix_config_from_the_properties() {
    assert!(FakeDevice::rdna4().limits().f16_matrix);
    assert!(!FakeDevice::pascal().limits().f16_matrix);
}

#[test]
fn near_miss_matrix_configs_are_not_the_f16_matrix() {
    let half_accumulator = f16_matrix_config(FloatKind::F16, 16);
    let longer_k = MmaConfig {
        k: 32,
        ..f16_matrix_config(FloatKind::F32, 16)
    };
    let narrow_m = MmaConfig {
        m: 32,
        n: 8,
        ..f16_matrix_config(FloatKind::F32, 16)
    };

    for config in [half_accumulator, longer_k, narrow_m] {
        let device = FakeDevice {
            matrix_config: Some(config),
            ..FakeDevice::rdna4()
        };
        let limits = device.limits();

        assert!(!limits.f16_matrix, "{config:?}");
    }
}

#[test]
fn the_matrix_plane_is_the_widest_plane_at_32_or_64() {
    let config = f16_matrix_config(FloatKind::F32, 16);
    let cases = [
        (16, Some(config), None),
        (32, Some(config), Some(32)),
        (64, Some(config), Some(64)),
        (128, Some(config), None),
        (64, None, None),
    ];

    for (plane_size_max, matrix_config, expected) in cases {
        let device = FakeDevice {
            matrix_config,
            plane_size_min: 16,
            plane_size_max,
            ..FakeDevice::rdna4()
        };
        let limits = device.limits();
        let width = matrix_plane_width(&limits);

        assert_eq!(width, expected, "{device:?}");
    }
}

#[test]
fn limits_read_code_storage_from_the_properties() {
    let with_storage = FakeDevice::rdna4();
    let without_storage = FakeDevice {
        u16_usage: EnumSet::empty(),
        ..with_storage
    };
    let without_buffers = FakeDevice {
        u16_usage: TypeUsage::Conversion | TypeUsage::Arithmetic,
        ..with_storage
    };

    assert!(with_storage.limits().u16_buffer);
    assert!(!without_storage.limits().u16_buffer);
    assert!(!without_buffers.limits().u16_buffer);
}

#[test]
fn integer_codes_cover_eight_to_ten_bit_sources() {
    let eight_bit = IntegerCodes::from_format(SampleFormat::U8);
    let nine_bit = IntegerCodes::from_format(SampleFormat::U16 { depth: 9 });
    let ten_bit = IntegerCodes::from_format(SampleFormat::U16 { depth: 10 });
    let twelve_bit = IntegerCodes::from_format(SampleFormat::U16 { depth: 12 });
    let float = IntegerCodes::from_format(SampleFormat::F32);

    assert_eq!(eight_bit, Some(IntegerCodes { depth: 8 }));
    assert_eq!(nine_bit, Some(IntegerCodes { depth: 9 }));
    assert_eq!(ten_bit, Some(IntegerCodes { depth: 10 }));
    assert_eq!(twelve_bit, None);
    assert_eq!(float, None);
}

#[test]
fn integer_codes_split_only_past_eight_bits() {
    let eight_bit = IntegerCodes { depth: 8 };
    let ten_bit = IntegerCodes { depth: 10 };

    assert_eq!(eight_bit.max_code(), 255);
    assert_eq!(ten_bit.max_code(), 1023);
    assert!(!eight_bit.wide());
    assert!(ten_bit.wide());
    assert_eq!(eight_bit.split_shift(), 0);
    assert_eq!(ten_bit.split_shift(), 2);
    assert_eq!(eight_bit.code_bytes(), 2);
    assert_eq!(ten_bit.code_bytes(), 2);
}

#[test]
fn integer_codes_need_their_storage_type() {
    let storing_device = FakeDevice::rdna4();
    let bare_device = FakeDevice {
        u16_usage: EnumSet::empty(),
        ..storing_device
    };
    let with_storage = storing_device.limits();
    let without_storage = bare_device.limits();

    for depth in [8, 10] {
        let codes = IntegerCodes { depth };

        assert!(codes.storable(&with_storage), "depth {depth}");
        assert!(!codes.storable(&without_storage), "depth {depth}");
    }
}

#[test]
fn matrix_sums_stay_exact_up_to_258_values() {
    assert!(IntegerCodes::matrix_exact(243));
    assert!(IntegerCodes::matrix_exact(258));
    assert!(!IntegerCodes::matrix_exact(259));
}

#[test]
fn the_inverse_scale_matches_the_kernel_distance_scale() {
    let eight_bit = IntegerCodes { depth: 8 };
    let ten_bit = IntegerCodes { depth: 10 };
    let cases = [
        (eight_bit, ChannelMode::Luma, 65025.0),
        (ten_bit, ChannelMode::Chroma, 1_046_529.0),
        (ten_bit, ChannelMode::Yuv, 1_046_529.0),
    ];

    for (codes, mode, squared_max) in cases {
        let channels = mode.count();
        let scale = channel_scale_for(channels);
        let inverse_scale = codes.inv_scale2(channels);

        assert_eq!(inverse_scale, scale / squared_max, "{codes:?} {mode:?}");
    }
}

/// A 1080p luma search shape at the library's default geometry.
fn search_shape(codes: Option<IntegerCodes>, matrix_plane: u32) -> CollabShape {
    CollabShape {
        f16_search: false,
        radius: 1,
        grid_frames: 2,
        refine: 2,
        mv_stride: 2 * 240 * 135,
        conf_stride: 240 * 135,
        blk_step: 8,
        blksize: 16,
        blocks_x: 240,
        blocks_y: 135,
        width: 1920,
        height: 1080,
        channels: 1,
        k_max: 8,
        stored_ch: 1,
        spatial_radius: 9,
        refs_x: 479,
        refs_y: 269,
        map_cols: 240,
        map_rows: 136,
        pooled: false,
        uniform_walk_only: false,
        packed: false,
        codes,
        matrix_plane,
    }
}

/// Every search variant legal on `shape` under `legality`, on an RDNA 4 card.
fn legal_searches(shape: &CollabShape, legality: &Legality) -> Vec<SearchVariant> {
    let limits = FakeDevice::rdna4().limits();

    SearchVariant::ALL
        .iter()
        .copied()
        .filter(|&candidate| search_candidate_legal(candidate, shape, &limits, legality))
        .collect()
}

fn integer_search(walk: Walk) -> SearchVariant {
    let variant = IntegerVariant {
        walk,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

fn integer_tiled_search() -> SearchVariant {
    let variant = IntegerVariant {
        centre: IntCentre::Tiled,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

#[test]
fn the_integer_tile_counts_four_bytes_per_code() {
    let eight_bit = Some(IntegerCodes { depth: 8 });
    let tiled = integer_tiled_search();

    // A yuv pixel stores 4 lanes, but the tile packs only the 3 the distance sums.
    let cases = [
        (ChannelMode::Luma, 5616),
        (ChannelMode::Chroma, 11232),
        (ChannelMode::Yuv, 16848),
    ];
    for (mode, expected) in cases {
        let mut shape = search_shape(eight_bit, 64);
        shape.channels = mode.count();
        shape.stored_ch = mode.storage_count();

        let float_bytes = size_of::<f32>();
        let bytes = search_shared_bytes(tiled, &shape, float_bytes);
        assert_eq!(bytes, expected, "{mode:?}");
    }
}

#[test]
fn integer_search_candidates_are_legal_only_with_codes() {
    let legality = Legality::default();
    let rdna4 = FakeDevice::rdna4().limits();
    let matrix_plane = matrix_plane_width(&rdna4).expect("RDNA 4 offers the matrix plane");
    let ten_bit = Some(IntegerCodes { depth: 10 });
    let without_codes = search_shape(None, 0);
    let with_codes = search_shape(ten_bit, matrix_plane);

    let float_only = legal_searches(&without_codes, &legality);
    let expected_floats: Vec<SearchVariant> = SearchVariant::ALL[..FLOAT_SEARCH_COUNT].to_vec();
    assert_eq!(float_only, expected_floats);

    let baseline = SearchVariant::Float(CollabVariant::BASELINE);
    let clipped = integer_search(Walk::Clipped);
    let uniform = integer_search(Walk::Uniform);
    let integer_tiled = integer_tiled_search();
    let integer_route = legal_searches(&with_codes, &legality);
    assert_eq!(integer_route, vec![baseline, clipped, uniform, integer_tiled]);

    let search_bytes = size_of::<f32>();
    for candidate in [baseline, clipped, uniform, integer_tiled] {
        let fits = search_candidate_fits(candidate, &with_codes, &rdna4, &legality, search_bytes);
        assert!(fits, "{candidate:?}");
    }

    // The integer placeholder is one i32 code. The untiled float baseline's is one f32 tile value.
    let integer_bytes = search_shared_bytes(clipped, &with_codes, search_bytes);
    let baseline_bytes = search_shared_bytes(baseline, &with_codes, search_bytes);
    assert_eq!(integer_bytes, size_of::<u32>());
    assert_eq!(baseline_bytes, search_bytes);

    let tiled = SearchVariant::Float(CollabVariant {
        centre: Centre::Tiled,
        ..CollabVariant::BASELINE
    });
    let tiled_fits = search_candidate_fits(tiled, &with_codes, &rdna4, &legality, search_bytes);
    assert!(
        !tiled_fits,
        "a float variant other than the baseline runs only without codes"
    );
}

#[test]
fn integer_search_offers_only_uniform_walks_where_the_device_needs_them() {
    let uniform_legality = Legality {
        uniform_walk_only: true,
        ..Legality::default()
    };
    let eight_bit = Some(IntegerCodes { depth: 8 });
    let shape = search_shape(eight_bit, 64);

    let legal = legal_searches(&shape, &uniform_legality);
    let uniform_variant = CollabVariant::baseline_for(true);
    let uniform_baseline = SearchVariant::Float(uniform_variant);
    let uniform = integer_search(Walk::Uniform);

    assert_eq!(legal, vec![uniform_baseline, uniform]);
}

/// A window shape of `channels` at patch radius 4 and search radius 2, the bench geometry.
fn window_shape(channels: ChannelMode, pair: bool, codes: Option<IntegerCodes>) -> WindowShape {
    WindowShape {
        pair,
        codes,
        has_reference: false,
        channels: channels.count(),
        stored_ch: channels.storage_count(),
        patch_radius: 4,
        search_radius: 2,
        matrix_plane: 0,
    }
}

/// Every window candidate legal on `shape`, on an RDNA 4 card.
fn legal_windows(shape: &WindowShape) -> Vec<WindowCandidate> {
    let limits = FakeDevice::rdna4().limits();

    WindowCandidate::ALL
        .iter()
        .copied()
        .filter(|&candidate| window_candidate_legal(candidate, shape, &limits))
        .collect()
}

#[test]
fn integer_windows_need_codes_and_an_unfiltered_reference() {
    let eight_bit = Some(IntegerCodes { depth: 8 });
    let float_shapes = vec![
        WindowCandidate::Block32x8,
        WindowCandidate::Block32x16,
        WindowCandidate::Block16x16,
        WindowCandidate::Separable32x8,
        WindowCandidate::Separable32x16,
        WindowCandidate::Separable16x16,
    ];
    let integer_shapes = vec![WindowCandidate::Block32x8, WindowCandidate::ScalarInt];
    let baseline_only = vec![WindowCandidate::Block32x8];

    for pair in [false, true] {
        let unfiltered = window_shape(ChannelMode::Luma, pair, eight_bit);
        let filtered = WindowShape {
            has_reference: true,
            ..unfiltered
        };
        let without_codes = window_shape(ChannelMode::Luma, pair, None);

        let integer_route = legal_windows(&unfiltered);
        let with_reference = legal_windows(&filtered);
        let float_route = legal_windows(&without_codes);

        assert_eq!(integer_route, integer_shapes, "pair {pair}");
        assert_eq!(with_reference, baseline_only, "pair {pair}");
        assert_eq!(float_route, float_shapes, "pair {pair}");
    }

    let mut no_storage = FakeDevice::rdna4();
    no_storage.u16_usage = EnumSet::empty();
    let no_storage_limits = no_storage.limits();
    let shape = window_shape(ChannelMode::Luma, false, eight_bit);
    let stored = window_candidate_legal(WindowCandidate::ScalarInt, &shape, &no_storage_limits);
    assert!(!stored, "the integer window needs u16 storage");
}

#[test]
fn the_scalar_int_window_counts_its_code_tile_and_integer_distance_tiles() {
    let ten_bit = Some(IntegerCodes { depth: 10 });
    let single = window_shape(ChannelMode::Luma, false, ten_bit);
    let pair = window_shape(ChannelMode::Luma, true, ten_bit);

    // Centre tile: (32 + 8 + 4) x (8 + 8 + 4) codes of 2 bytes = 1760 bytes.
    // One distance tile: (32 + 8) x (8 + 8) i32 and one spare word = 2564 bytes.
    let single_bytes = window_shared_bytes(WindowCandidate::ScalarInt, &single);
    let pair_bytes = window_shared_bytes(WindowCandidate::ScalarInt, &pair);

    assert_eq!(single_bytes, 1760 + 2564);
    assert_eq!(single_bytes, 4324);
    assert_eq!(pair_bytes, 1760 + 2 * 2564);
    assert_eq!(pair_bytes, 6888);

    let rdna4 = FakeDevice::rdna4().limits();
    let fits = window_candidate_fits(WindowCandidate::ScalarInt, &pair, &rdna4);
    assert!(fits);
}

#[test]
fn existing_window_byte_counts_are_unchanged_through_the_pure_function() {
    let single = WindowShape {
        patch_radius: 3,
        search_radius: 7,
        ..window_shape(ChannelMode::Yuv, false, None)
    };
    let pair = WindowShape { pair: true, ..single };

    // The values `window.rs` pins on a 64x48 yuv site at patch radius 3 and search radius 7.
    let cases = [
        (WindowCandidate::Block32x8, &pair, 23296 + 2 * 2128 + 2 * 4),
        (WindowCandidate::Block32x8, &single, 23296 + 2128 + 4),
        (WindowCandidate::Separable32x8, &pair, 23296 + 2 * 2128 + 2 * 1792),
        (WindowCandidate::Separable32x8, &single, 23296 + 2128 + 1792),
        (WindowCandidate::Block32x16, &pair, 36648),
        (WindowCandidate::Block16x16, &pair, 24616),
        (WindowCandidate::Separable32x16, &pair, 42272),
        (WindowCandidate::Separable16x16, &pair, 27424),
    ];

    for (candidate, shape, expected) in cases {
        let bytes = window_shared_bytes(candidate, shape);
        assert_eq!(bytes, expected, "{} pair {}", candidate.name(), shape.pair);
    }
}
