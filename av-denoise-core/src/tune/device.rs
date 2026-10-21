use cubecl::hash::StableHasher;
use cubecl::ir::features::{AtomicUsage, TypeUsage};
use cubecl::ir::{DeviceProperties, ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::prelude::*;

use super::legality::offers_f16_matrix;

/// Which physical device and driver a tuning result was measured on.
///
/// The host fills this in, because the runtime's properties carry no adapter name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    /// The backend, such as `vulkan`, `metal`, `cuda` or `rocm`.
    pub backend: String,
    pub adapter: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// The driver name and version, as the backend reports them.
    pub driver: String,
    pub os: String,
    /// The CUDA compute capability as `major * 10 + minor`, when the backend reports one.
    ///
    /// The device id already names the chip, so the capability is left out of the hash.
    pub compute_capability: Option<u32>,
}

impl DeviceIdentity {
    /// A stable hash of every identity field except the compute capability, as 32 lowercase hex
    /// characters.
    ///
    /// Stable across runs and builds.
    pub fn identity_hash(&self) -> String {
        let rendered = format!(
            "{}|{}|{:08x}|{:08x}|{}|{}",
            self.backend, self.adapter, self.vendor_id, self.device_id, self.driver, self.os,
        );
        let hash = StableHasher::hash_one(&rendered);

        format!("{hash:032x}")
    }
}

/// The hardware limits a candidate's launch has to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    pub plane_size_min: u32,
    pub plane_size_max: u32,
    pub max_shared_bytes: usize,
    pub max_units_per_cube: u32,
    pub max_cube_dim: (u32, u32, u32),
    /// Streaming multiprocessors or compute units, when the runtime reports them.
    pub compute_units: Option<u32>,
    pub f16_buffer: bool,
    pub f16_arithmetic: bool,
    /// Whether 64-bit integer atomic adds work on storage buffers.
    pub int64_atomic_add: bool,
    /// Whether the device runs 16x16x16 f16 matrix products accumulating in f32.
    pub f16_matrix: bool,
    /// Whether u16 storage buffers work.
    pub u16_buffer: bool,
}

impl DeviceLimits {
    pub fn from_client<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let properties = client.properties();
        Self::from_properties(properties)
    }

    pub fn from_properties(properties: &DeviceProperties) -> Self {
        let hardware = &properties.hardware;

        let f16_elem = ElemType::Float(FloatKind::F16);
        let f16_type = StorageType::Scalar(f16_elem);
        let f16_usage = properties.type_usage(f16_type);

        let i64_elem = ElemType::Int(IntKind::I64);
        let i64_scalar = StorageType::Scalar(i64_elem);
        let i64_usage = properties.type_usage(i64_scalar);
        let i64_atomic = StorageType::Atomic(i64_elem);
        let i64_atomic_type = Type::new(i64_atomic);
        let i64_atomic_usage = properties.atomic_type_usage(i64_atomic_type);
        let i64_arithmetic = i64_usage.contains(TypeUsage::Arithmetic);
        let i64_atomic_add = i64_atomic_usage.contains(AtomicUsage::Add);

        let u16_elem = ElemType::UInt(UIntKind::U16);
        let u16_type = StorageType::Scalar(u16_elem);
        let u16_usage = properties.type_usage(u16_type);

        let f16_matrix = offers_f16_matrix(&properties.features.matmul.cmma);

        Self {
            plane_size_min: hardware.plane_size_min,
            plane_size_max: hardware.plane_size_max,
            max_shared_bytes: hardware.max_shared_memory_size,
            max_units_per_cube: hardware.max_units_per_cube,
            max_cube_dim: hardware.max_cube_dim,
            compute_units: hardware.num_streaming_multiprocessors,
            f16_buffer: f16_usage.contains(TypeUsage::Buffer),
            f16_arithmetic: f16_usage.contains(TypeUsage::Arithmetic),
            int64_atomic_add: i64_arithmetic && i64_atomic_add,
            f16_matrix,
            u16_buffer: u16_usage.contains(TypeUsage::Buffer),
        }
    }

    /// Whether the f16 search ring can run, which needs f16 buffers and f16 arithmetic.
    pub fn supports_f16_search(&self) -> bool {
        self.f16_buffer && self.f16_arithmetic
    }
}

/// A device's identity and its limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProfile {
    pub identity: DeviceIdentity,
    pub limits: DeviceLimits,
}

impl DeviceProfile {
    pub fn identity_hash(&self) -> String {
        self.identity.identity_hash()
    }
}
