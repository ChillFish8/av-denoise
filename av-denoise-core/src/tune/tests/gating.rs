use cubecl::features::{EnumSet, Features, TypeUsage};
use cubecl::ir::{DeviceProperties, ElemType, HardwareProperties, MemoryDeviceProperties, UIntKind};
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
use crate::tune::{DeviceLimits, IntegerCodes, Legality, SiteCandidate};

/// What a fake device offers, turned into the runtime properties a real client reports.
#[derive(Debug, Clone)]
struct FakeDevice {
    plane_size_min: u32,
    plane_size_max: u32,
    shared_bytes: usize,
    u16_usage: EnumSet<TypeUsage>,
}

impl FakeDevice {
    /// An RDNA 4 card under RADV.
    fn rdna4() -> Self {
        Self {
            plane_size_min: 32,
            plane_size_max: 64,
            shared_bytes: 64 * 1024,
            u16_usage: TypeUsage::all(),
        }
    }

    fn properties(&self) -> DeviceProperties {
        let features = Features::default();

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

#[test]
fn limits_read_code_storage_from_the_properties() {
    let with_storage = FakeDevice::rdna4();
    let without_storage = FakeDevice {
        u16_usage: EnumSet::empty(),
        ..with_storage.clone()
    };
    let without_buffers = FakeDevice {
        u16_usage: TypeUsage::Conversion | TypeUsage::Arithmetic,
        ..with_storage.clone()
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
fn integer_codes_hold_the_top_code_and_two_bytes() {
    let eight_bit = IntegerCodes { depth: 8 };
    let ten_bit = IntegerCodes { depth: 10 };

    assert_eq!(eight_bit.max_code(), 255);
    assert_eq!(ten_bit.max_code(), 1023);
    assert_eq!(eight_bit.code_bytes(), 2);
    assert_eq!(ten_bit.code_bytes(), 2);
}

#[test]
fn integer_codes_need_their_storage_type() {
    let storing_device = FakeDevice::rdna4();
    let bare_device = FakeDevice {
        u16_usage: EnumSet::empty(),
        ..storing_device.clone()
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
fn search_shape(codes: Option<IntegerCodes>) -> CollabShape {
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
    }
}

/// Every search variant legal on `shape` under `legality`.
fn legal_searches(shape: &CollabShape, legality: &Legality) -> Vec<SearchVariant> {
    SearchVariant::ALL
        .iter()
        .copied()
        .filter(|&candidate| search_candidate_legal(candidate, shape, legality))
        .collect()
}

fn integer_search(walk: Walk) -> SearchVariant {
    let variant = IntegerVariant {
        walk,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

fn named_search(name: &str) -> SearchVariant {
    SearchVariant::from_name(name).expect("every named search variant is listed")
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
        let mut shape = search_shape(eight_bit);
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
    let ten_bit = Some(IntegerCodes { depth: 10 });
    let without_codes = search_shape(None);
    let with_codes = search_shape(ten_bit);

    let float_only = legal_searches(&without_codes, &legality);
    let expected_floats: Vec<SearchVariant> = SearchVariant::ALL[..FLOAT_SEARCH_COUNT].to_vec();
    assert_eq!(float_only, expected_floats);

    let baseline = SearchVariant::Float(CollabVariant::BASELINE);
    let clipped = integer_search(Walk::Clipped);
    let uniform = integer_search(Walk::Uniform);
    let integer_tiled = integer_tiled_search();
    let sliding7x7 = named_search("int_sliding7x7");
    let sliding15x7 = named_search("int_sliding15x7");
    let integer_route = legal_searches(&with_codes, &legality);
    let expected_route = vec![baseline, clipped, uniform, integer_tiled, sliding7x7, sliding15x7];
    assert_eq!(integer_route, expected_route);

    let search_bytes = size_of::<f32>();
    for candidate in expected_route {
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
    let shape = search_shape(eight_bit);

    let legal = legal_searches(&shape, &uniform_legality);
    let uniform_variant = CollabVariant::baseline_for(true);
    let uniform_baseline = SearchVariant::Float(uniform_variant);
    let uniform = integer_search(Walk::Uniform);

    assert_eq!(legal, vec![uniform_baseline, uniform]);
}

/// A 1080p search of `mode` at spatial radius 9 with codes of `depth` bits.
fn coded_search_shape(mode: ChannelMode, depth: u32) -> CollabShape {
    let codes = Some(IntegerCodes { depth });
    let mut shape = search_shape(codes);
    shape.channels = mode.count();
    shape.stored_ch = mode.storage_count();

    shape
}

/// An RDNA 4 card with `shared_kb` KB of shared memory and planes up to `plane_size_max`.
fn sized_rdna4(shared_kb: usize, plane_size_max: u32) -> DeviceLimits {
    let device = FakeDevice {
        shared_bytes: shared_kb * 1024,
        plane_size_max,
        ..FakeDevice::rdna4()
    };

    device.limits()
}

fn candidate_bytes(candidate: SearchVariant, shape: &CollabShape) -> usize {
    let search_bytes = size_of::<f32>();
    search_shared_bytes(candidate, shape, search_bytes)
}

fn candidate_fits(candidate: SearchVariant, shape: &CollabShape, limits: &DeviceLimits) -> bool {
    let legality = Legality::default();
    let search_bytes = size_of::<f32>();

    search_candidate_fits(candidate, shape, limits, &legality, search_bytes)
}

#[test]
fn the_sliding_centre_counts_its_block_sums() {
    let cases = [("int_sliding7x7", 512), ("int_sliding15x7", 1024)];
    let wide = sized_rdna4(16, 64);
    let narrow = sized_rdna4(16, 32);

    for (name, expected) in cases {
        let sliding = named_search(name);
        let shape = coded_search_shape(ChannelMode::Yuv, 10);

        let bytes = candidate_bytes(sliding, &shape);
        let fits_at_64 = candidate_fits(sliding, &shape, &wide);
        let fits_at_32 = candidate_fits(sliding, &shape, &narrow);

        assert_eq!(bytes, expected, "{name}");
        assert!(fits_at_64, "{name}");
        assert!(fits_at_32, "{name}");
    }
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
