use cubecl::features::{EnumSet, Features, MmaConfig, TypeUsage};
use cubecl::ir::{
    DeviceProperties,
    ElemType,
    FloatKind,
    HardwareProperties,
    IntKind,
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
use crate::tune::{
    DeviceLimits,
    HIP_RUNTIME,
    Int8Layout,
    Int8Matrix,
    IntegerCodes,
    Legality,
    SiteCandidate,
    int8_matrix_plane_width,
    matrix_plane_width,
};

/// What a fake device offers, turned into the runtime properties a real client reports.
#[derive(Debug, Clone)]
struct FakeDevice {
    matrix_config: Option<MmaConfig>,
    int8_configs: Vec<MmaConfig>,
    plane_size_min: u32,
    plane_size_max: u32,
    shared_bytes: usize,
    u16_usage: EnumSet<TypeUsage>,
    u8_usage: EnumSet<TypeUsage>,
    i8_usage: EnumSet<TypeUsage>,
}

impl FakeDevice {
    /// An RDNA 4 card under RADV.
    fn rdna4() -> Self {
        let config = f16_matrix_config(FloatKind::F32, 16);
        let unsigned = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 16);
        let signed = int8_matrix_config(SIGNED_ELEM, SIGNED_ELEM, I32_ELEM, 16, 16, 16);

        Self {
            matrix_config: Some(config),
            int8_configs: vec![unsigned, signed],
            plane_size_min: 32,
            plane_size_max: 64,
            shared_bytes: 64 * 1024,
            u16_usage: TypeUsage::all(),
            u8_usage: TypeUsage::all(),
            i8_usage: TypeUsage::all(),
        }
    }

    /// A Pascal card under Vulkan, which has no matrix units.
    fn pascal() -> Self {
        Self {
            matrix_config: None,
            int8_configs: Vec::new(),
            plane_size_min: 32,
            plane_size_max: 32,
            shared_bytes: 48 * 1024,
            u16_usage: TypeUsage::all(),
            u8_usage: TypeUsage::all(),
            i8_usage: TypeUsage::all(),
        }
    }

    fn properties(&self) -> DeviceProperties {
        let mut features = Features::default();

        if let Some(config) = self.matrix_config {
            features.matmul.cmma.insert(config);
        }

        for &config in &self.int8_configs {
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

        if !self.u8_usage.is_empty() {
            properties.register_type_usage(UNSIGNED_ELEM, self.u8_usage);
        }

        if !self.i8_usage.is_empty() {
            properties.register_type_usage(SIGNED_ELEM, self.i8_usage);
        }

        properties
    }

    fn limits(&self) -> DeviceLimits {
        let properties = self.properties();
        DeviceLimits::from_properties(&properties)
    }

    /// The limits on `runtime` for devices of `architectures`, after the matrix rule.
    fn limits_on(&self, runtime: &str, architectures: &[&str]) -> DeviceLimits {
        let properties = self.properties();
        let architectures: Vec<String> = architectures.iter().map(|name| name.to_string()).collect();

        DeviceLimits::for_runtime(&properties, runtime, &architectures)
    }
}

/// The name CubeCL's wgpu runtime reports over SPIR-V.
const WGPU_SPIRV_RUNTIME: &str = "wgpu<spirv>";

#[test]
fn hip_on_rdna4_offers_no_matrix_products() {
    let rdna4 = FakeDevice::rdna4();
    let cases: [&[&str]; 3] = [&["gfx1201"], &["gfx1200"], &["gfx1100", "gfx1201"]];

    for architectures in cases {
        let limits = rdna4.limits_on(HIP_RUNTIME, architectures);
        let f16_plane = matrix_plane_width(&limits);
        let int8_plane = int8_matrix_plane_width(&limits);

        assert!(!limits.f16_matrix, "{architectures:?}");
        assert_eq!(limits.int8_matrix, None, "{architectures:?}");
        assert!(!limits.int8_signed, "{architectures:?}");
        assert_eq!(f16_plane, None, "{architectures:?}");
        assert_eq!(int8_plane, None, "{architectures:?}");
    }
}

#[test]
fn hip_with_an_unknown_architecture_offers_no_matrix_products() {
    let limits = FakeDevice::rdna4().limits_on(HIP_RUNTIME, &[]);

    assert!(!limits.f16_matrix);
    assert_eq!(limits.int8_matrix, None);
}

#[test]
fn hip_on_rdna3_keeps_the_reported_f16_matrix() {
    let rdna3 = FakeDevice {
        int8_configs: Vec::new(),
        plane_size_max: 32,
        ..FakeDevice::rdna4()
    };

    let limits = rdna3.limits_on(HIP_RUNTIME, &["gfx1100"]);
    let f16_plane = matrix_plane_width(&limits);

    assert!(limits.f16_matrix);
    assert_eq!(limits.int8_matrix, None);
    assert_eq!(f16_plane, Some(32));
}

#[test]
fn vulkan_on_rdna4_keeps_both_matrix_products() {
    let limits = FakeDevice::rdna4().limits_on(WGPU_SPIRV_RUNTIME, &["gfx1201"]);
    let raw_limits = FakeDevice::rdna4().limits();
    let f16_plane = matrix_plane_width(&limits);
    let int8_plane = int8_matrix_plane_width(&limits);

    assert!(limits.f16_matrix);
    assert_eq!(limits.int8_matrix, Some(Int8Matrix::Unsigned));
    assert_eq!(f16_plane, Some(64));
    assert_eq!(int8_plane, Some(64));
    assert_eq!(limits, raw_limits);
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

const UNSIGNED_ELEM: ElemType = ElemType::UInt(UIntKind::U8);
const SIGNED_ELEM: ElemType = ElemType::Int(IntKind::I8);
const I32_ELEM: ElemType = ElemType::Int(IntKind::I32);

/// A matrix config of `a` times `b` accumulating in `cd`, at `m` x `n` x `k`.
fn int8_matrix_config(a: ElemType, b: ElemType, cd: ElemType, m: u32, n: u32, k: u32) -> MmaConfig {
    MmaConfig {
        a_type: StorageType::Scalar(a),
        b_type: StorageType::Scalar(b),
        cd_type: StorageType::Scalar(cd),
        m,
        n,
        k,
    }
}

/// An RDNA 4 card listing only `configs` as its int8 matrix products.
fn int8_device(configs: &[MmaConfig]) -> FakeDevice {
    FakeDevice {
        int8_configs: configs.to_vec(),
        ..FakeDevice::rdna4()
    }
}

#[test]
fn int8_gating_takes_only_the_exact_16x16x16_i32_configs() {
    let half_elem = ElemType::Float(FloatKind::F16);
    let u32_elem = ElemType::UInt(UIntKind::U32);
    let f32_elem = ElemType::Float(FloatKind::F32);
    let unsigned = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 16);
    let signed = int8_matrix_config(SIGNED_ELEM, SIGNED_ELEM, I32_ELEM, 16, 16, 16);

    let accepted = [
        (vec![unsigned], Int8Matrix::Unsigned),
        (vec![signed], Int8Matrix::Signed),
        (vec![signed, unsigned], Int8Matrix::Unsigned),
    ];
    for (configs, expected) in accepted {
        let device = int8_device(&configs);
        let limits = device.limits();
        assert_eq!(limits.int8_matrix, Some(expected), "{configs:?}");
    }

    let half_result = int8_matrix_config(SIGNED_ELEM, SIGNED_ELEM, half_elem, 16, 16, 16);
    let longer_k = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 32);
    let narrow_m = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 8, 32, 16);
    let unsigned_result = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, u32_elem, 16, 16, 16);
    let float_result = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, f32_elem, 16, 16, 16);
    let mixed = int8_matrix_config(UNSIGNED_ELEM, SIGNED_ELEM, I32_ELEM, 16, 16, 16);

    let refused = [
        vec![half_result],
        vec![longer_k],
        vec![narrow_m],
        vec![unsigned_result, float_result],
        vec![mixed],
    ];
    for configs in refused {
        let device = int8_device(&configs);
        let limits = device.limits();
        assert_eq!(limits.int8_matrix, None, "{configs:?}");
    }

    let without_int8 = FakeDevice {
        u8_usage: TypeUsage::Conversion | TypeUsage::Buffer,
        ..int8_device(&[unsigned])
    };
    let without_int8_limits = without_int8.limits();
    assert_eq!(
        without_int8_limits.int8_matrix, None,
        "a listed config needs u8 arithmetic"
    );
}

#[test]
fn int8_gating_falls_back_to_i8_when_only_i8_runs_arithmetic() {
    let unsigned = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 16);
    let signed = int8_matrix_config(SIGNED_ELEM, SIGNED_ELEM, I32_ELEM, 16, 16, 16);
    let signed_only = FakeDevice {
        u8_usage: TypeUsage::Conversion | TypeUsage::Buffer,
        ..int8_device(&[unsigned, signed])
    };

    let limits = signed_only.limits();

    assert_eq!(limits.int8_matrix, Some(Int8Matrix::Signed));
}

#[test]
fn the_int8_plane_follows_the_widest_plane_rule() {
    let unsigned = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 16);
    let f16_config = f16_matrix_config(FloatKind::F32, 16);
    let cases = [
        (32, vec![unsigned], Some(32)),
        (64, vec![unsigned], Some(64)),
        (16, vec![unsigned], None),
        (128, vec![unsigned], None),
        (64, Vec::new(), None),
    ];

    for (plane_size_max, configs, expected) in cases {
        for matrix_config in [Some(f16_config), None] {
            let device = FakeDevice {
                matrix_config,
                plane_size_min: 16,
                plane_size_max,
                ..int8_device(&configs)
            };
            let limits = device.limits();
            let width = int8_matrix_plane_width(&limits);

            assert_eq!(width, expected, "{device:?}");
        }
    }
}

#[test]
fn int8_layouts_split_codes_so_every_part_fits_its_operand() {
    let eight_bit = IntegerCodes { depth: 8 };
    let ten_bit = IntegerCodes { depth: 10 };
    let cases = [
        (Int8Matrix::Unsigned, eight_bit, 0, 1, 1, 1),
        (Int8Matrix::Unsigned, ten_bit, 2, 2, 4, 3),
        (Int8Matrix::Signed, eight_bit, 4, 2, 4, 3),
        (Int8Matrix::Signed, ten_bit, 5, 2, 4, 3),
    ];

    for (element, codes, low_bits, parts, products, accumulators) in cases {
        let layout = Int8Layout::for_codes(element, codes);
        let label = format!("{element:?} at {} bits", codes.depth);

        assert_eq!(layout.element, element, "{label}");
        assert_eq!(layout.low_bits, low_bits, "{label}");
        assert_eq!(layout.parts(), parts, "{label}");
        assert_eq!(layout.products(), products, "{label}");
        assert_eq!(layout.accumulators(), accumulators, "{label}");

        let operand_max = match element {
            Int8Matrix::Unsigned => 255,
            Int8Matrix::Signed => 127,
        };
        let low_mask = (1 << low_bits) - 1;

        for code in 0..=codes.max_code() {
            let high = code >> low_bits;
            let low = code & low_mask;
            let rebuilt = (high << low_bits) + low;

            assert!(high <= operand_max, "{label}: high part of {code}");
            assert!(low <= operand_max, "{label}: low part of {code}");
            assert_eq!(rebuilt, code, "{label}");
        }
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
        int8: None,
        int8_plane: 0,
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
    let refs16_matrix = named_search("int_refs16_matrix");
    let refs8x2_matrix = named_search("int_refs8x2_matrix");
    let integer_route = legal_searches(&with_codes, &legality);
    let expected_route = vec![
        baseline,
        clipped,
        uniform,
        integer_tiled,
        refs16_matrix,
        refs8x2_matrix,
    ];
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
    let shape = search_shape(eight_bit, 64);

    let legal = legal_searches(&shape, &uniform_legality);
    let uniform_variant = CollabVariant::baseline_for(true);
    let uniform_baseline = SearchVariant::Float(uniform_variant);
    let uniform = integer_search(Walk::Uniform);
    let refs16_matrix = named_search("int_refs16_matrix");
    let refs8x2_matrix = named_search("int_refs8x2_matrix");

    assert_eq!(
        legal,
        vec![uniform_baseline, uniform, refs16_matrix, refs8x2_matrix]
    );
}

/// A 1080p search of `mode` at spatial radius 9 with codes of `depth` bits and the f16 matrix
/// plane `matrix_plane`.
fn coded_search_shape(mode: ChannelMode, depth: u32, matrix_plane: u32) -> CollabShape {
    let codes = Some(IntegerCodes { depth });
    let mut shape = search_shape(codes, matrix_plane);
    shape.channels = mode.count();
    shape.stored_ch = mode.storage_count();

    shape
}

/// [coded_search_shape] on the int8 route of `element` at `int8_plane`.
fn int8_search_shape(mode: ChannelMode, depth: u32, element: Int8Matrix, int8_plane: u32) -> CollabShape {
    let shape = coded_search_shape(mode, depth, 64);

    CollabShape {
        int8: Some(element),
        int8_plane,
        ..shape
    }
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

fn matrix_bytes(candidate: SearchVariant, shape: &CollabShape) -> usize {
    let search_bytes = size_of::<f32>();
    search_shared_bytes(candidate, shape, search_bytes)
}

fn matrix_fits(candidate: SearchVariant, shape: &CollabShape, limits: &DeviceLimits) -> bool {
    let legality = Legality::default();
    let search_bytes = size_of::<f32>();

    search_candidate_fits(candidate, shape, limits, &legality, search_bytes)
}

/// The names of every matrix search variant legal on `shape` under `limits`.
fn legal_matrix_names(shape: &CollabShape, limits: &DeviceLimits) -> Vec<&'static str> {
    let legality = Legality::default();

    SearchVariant::ALL
        .iter()
        .copied()
        .filter(|&candidate| match candidate {
            SearchVariant::Integer(variant) => variant.is_matrix() || variant.is_int8(),
            SearchVariant::Float(_) => false,
        })
        .filter(|&candidate| search_candidate_legal(candidate, shape, limits, &legality))
        .map(|candidate| candidate.name())
        .collect()
}

#[test]
fn the_matrix_centre_fits_16_kb_only_for_8_bit_luma_at_plane_64() {
    let refs16 = named_search("int_refs16_matrix");
    let luma_at_64 = coded_search_shape(ChannelMode::Luma, 8, 64);
    let luma_at_32 = coded_search_shape(ChannelMode::Luma, 8, 32);
    let chroma_at_64 = coded_search_shape(ChannelMode::Chroma, 8, 64);
    let ten_bit_at_64 = coded_search_shape(ChannelMode::Luma, 10, 64);
    let wide = sized_rdna4(16, 64);
    let narrow = sized_rdna4(16, 32);

    let bytes_at_64 = matrix_bytes(refs16, &luma_at_64);
    let bytes_at_32 = matrix_bytes(refs16, &luma_at_32);
    let fits_at_64 = matrix_fits(refs16, &luma_at_64, &wide);
    let fits_at_32 = matrix_fits(refs16, &luma_at_32, &narrow);
    let chroma_fits = matrix_fits(refs16, &chroma_at_64, &wide);
    let ten_bit_fits = matrix_fits(refs16, &ten_bit_at_64, &wide);

    assert_eq!(bytes_at_64, 13548);
    assert_eq!(bytes_at_32, 16620);
    assert!(fits_at_64);
    assert!(!fits_at_32);
    assert!(!chroma_fits);
    assert!(!ten_bit_fits);
}

#[test]
fn ten_bit_luma_matrix_centre_fits_32_kb_at_plane_32() {
    let refs16 = named_search("int_refs16_matrix");
    let shape = coded_search_shape(ChannelMode::Luma, 10, 32);
    let limits = sized_rdna4(32, 32);

    let bytes = matrix_bytes(refs16, &shape);
    let fits = matrix_fits(refs16, &shape, &limits);

    assert_eq!(bytes, 31332);
    assert!(fits);
}

#[test]
fn ten_bit_yuv_matrix_centre_does_not_fit_32_kb() {
    let refs16 = named_search("int_refs16_matrix");
    let shape = coded_search_shape(ChannelMode::Yuv, 10, 32);
    let small = sized_rdna4(32, 32);
    let large = sized_rdna4(64, 32);

    let bytes = matrix_bytes(refs16, &shape);
    let fits_small = matrix_fits(refs16, &shape, &small);
    let fits_large = matrix_fits(refs16, &shape, &large);

    assert_eq!(bytes, 49220);
    assert!(!fits_small);
    assert!(fits_large);
}

#[test]
fn the_two_row_matrix_centre_counts_its_smaller_union() {
    let refs8x2 = named_search("int_refs8x2_matrix");
    let shape = coded_search_shape(ChannelMode::Luma, 8, 64);
    let bytes = matrix_bytes(refs8x2, &shape);

    assert_eq!(bytes, 10636);
}

#[test]
fn the_int8_matrix_centre_counts_one_byte_per_part() {
    let refs16_i8 = named_search("int_refs16_i8");
    let refs16_f16 = named_search("int_refs16_matrix");
    let small_wide = sized_rdna4(16, 64);
    let small_narrow = sized_rdna4(16, 32);
    let medium_wide = sized_rdna4(32, 64);
    let medium_narrow = sized_rdna4(32, 32);

    let whole_at_64 = int8_search_shape(ChannelMode::Luma, 8, Int8Matrix::Unsigned, 64);
    let whole_at_32 = int8_search_shape(ChannelMode::Luma, 8, Int8Matrix::Unsigned, 32);
    let f16_at_32 = coded_search_shape(ChannelMode::Luma, 8, 32);
    let whole_bytes_at_64 = matrix_bytes(refs16_i8, &whole_at_64);
    let whole_bytes_at_32 = matrix_bytes(refs16_i8, &whole_at_32);
    let whole_fits_at_64 = matrix_fits(refs16_i8, &whole_at_64, &small_wide);
    let whole_fits_at_32 = matrix_fits(refs16_i8, &whole_at_32, &small_narrow);
    let f16_fits_at_32 = matrix_fits(refs16_f16, &f16_at_32, &small_narrow);

    assert_eq!(whole_bytes_at_64, 10800);
    assert_eq!(whole_bytes_at_32, 13360);
    assert!(whole_fits_at_64);
    assert!(whole_fits_at_32);
    assert!(!f16_fits_at_32);

    // Both split layouts take 2 parts and 3 accumulators.
    let split_cases = [(Int8Matrix::Signed, 8), (Int8Matrix::Unsigned, 10)];
    for (element, depth) in split_cases {
        let label = format!("{element:?} {depth}-bit");
        let split_at_64 = int8_search_shape(ChannelMode::Luma, depth, element, 64);
        let split_at_32 = int8_search_shape(ChannelMode::Luma, depth, element, 32);

        let bytes_at_64 = matrix_bytes(refs16_i8, &split_at_64);
        let bytes_at_32 = matrix_bytes(refs16_i8, &split_at_32);
        let medium_fits_at_64 = matrix_fits(refs16_i8, &split_at_64, &medium_wide);
        let medium_fits_at_32 = matrix_fits(refs16_i8, &split_at_32, &medium_narrow);
        let small_fits_at_64 = matrix_fits(refs16_i8, &split_at_64, &small_wide);
        let small_fits_at_32 = matrix_fits(refs16_i8, &split_at_32, &small_narrow);

        assert_eq!(bytes_at_64, 17644, "{label}");
        assert_eq!(bytes_at_32, 24812, "{label}");
        assert!(medium_fits_at_64, "{label}");
        assert!(medium_fits_at_32, "{label}");
        assert!(!small_fits_at_64, "{label}");
        assert!(!small_fits_at_32, "{label}");
    }

    let signed_yuv = int8_search_shape(ChannelMode::Yuv, 10, Int8Matrix::Signed, 32);
    let signed_yuv_bytes = matrix_bytes(refs16_i8, &signed_yuv);
    let signed_yuv_fits = matrix_fits(refs16_i8, &signed_yuv, &medium_narrow);

    assert_eq!(signed_yuv_bytes, 33756);
    assert!(!signed_yuv_fits);
}

#[test]
fn int8_candidates_need_the_int8_config_and_its_plane() {
    let without_int8 = FakeDevice {
        int8_configs: Vec::new(),
        ..FakeDevice::rdna4()
    };
    let without_limits = without_int8.limits();
    let rdna4 = FakeDevice::rdna4().limits();
    let shape = int8_search_shape(ChannelMode::Luma, 8, Int8Matrix::Unsigned, 64);
    let narrow_plane = CollabShape {
        int8_plane: 32,
        ..shape
    };
    let no_product = CollabShape { int8: None, ..shape };

    let none_offered = legal_matrix_names(&shape, &without_limits);
    let offered = legal_matrix_names(&shape, &rdna4);
    let wrong_plane = legal_matrix_names(&narrow_plane, &rdna4);
    let unnamed = legal_matrix_names(&no_product, &rdna4);

    let f16_centres = vec!["int_refs16_matrix", "int_refs8x2_matrix"];
    let with_int8 = vec!["int_refs16_matrix", "int_refs8x2_matrix", "int_refs16_i8"];
    assert_eq!(none_offered, f16_centres);
    assert_eq!(offered, with_int8);
    assert_eq!(wrong_plane, f16_centres);
    assert_eq!(unnamed, f16_centres);
}

#[test]
fn an_int8_shape_needs_the_device_to_run_its_own_product() {
    let unsigned_config = int8_matrix_config(UNSIGNED_ELEM, UNSIGNED_ELEM, I32_ELEM, 16, 16, 16);
    let unsigned_only = FakeDevice {
        int8_configs: vec![unsigned_config],
        ..FakeDevice::rdna4()
    };
    let unsigned_limits = unsigned_only.limits();
    let rdna4 = FakeDevice::rdna4().limits();
    let signed_shape = int8_search_shape(ChannelMode::Luma, 8, Int8Matrix::Signed, 64);
    let unsigned_shape = int8_search_shape(ChannelMode::Luma, 8, Int8Matrix::Unsigned, 64);

    let signed_on_unsigned = legal_matrix_names(&signed_shape, &unsigned_limits);
    let unsigned_on_unsigned = legal_matrix_names(&unsigned_shape, &unsigned_limits);
    let signed_on_both = legal_matrix_names(&signed_shape, &rdna4);

    let f16_centres = vec!["int_refs16_matrix", "int_refs8x2_matrix"];
    let with_int8 = vec!["int_refs16_matrix", "int_refs8x2_matrix", "int_refs16_i8"];
    assert_eq!(signed_on_unsigned, f16_centres);
    assert_eq!(unsigned_on_unsigned, with_int8);
    assert_eq!(signed_on_both, with_int8);
    assert_eq!(rdna4.int8_matrix, Some(Int8Matrix::Unsigned));
    assert!(rdna4.int8_signed);
    assert!(!unsigned_limits.int8_signed);
}

#[test]
fn matrix_candidates_need_the_config_and_the_widest_plane() {
    let without_matrix = FakeDevice {
        matrix_config: None,
        int8_configs: Vec::new(),
        ..FakeDevice::rdna4()
    };
    let without_limits = without_matrix.limits();
    let rdna4 = FakeDevice::rdna4().limits();
    let at_64 = coded_search_shape(ChannelMode::Luma, 8, 64);
    let at_32 = coded_search_shape(ChannelMode::Luma, 8, 32);

    let none_offered = legal_matrix_names(&at_64, &without_limits);
    let narrower = legal_matrix_names(&at_32, &rdna4);
    let widest = legal_matrix_names(&at_64, &rdna4);

    assert!(none_offered.is_empty(), "{none_offered:?}");
    assert!(narrower.is_empty(), "{narrower:?}");
    assert_eq!(widest, vec!["int_refs16_matrix", "int_refs8x2_matrix"]);
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
