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
use crate::tune::{DeviceLimits, IntegerCodes, matrix_plane_width};

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
