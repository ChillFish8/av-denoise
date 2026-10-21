use std::collections::BTreeSet;

use cubecl::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};

use super::codes::IntegerCodes;
use super::device::{DeviceIdentity, DeviceLimits, DeviceProfile};
use crate::matrix::MATRIX_TILE;

/// The PCI vendor id of NVIDIA.
pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;
/// The PCI vendor id of AMD.
pub const AMD_VENDOR_ID: u32 = 0x1002;
/// The PCI vendor id of Intel.
pub const INTEL_VENDOR_ID: u32 = 0x8086;
/// The vendor id Apple GPUs report.
pub const APPLE_VENDOR_ID: u32 = 0x106B;

/// The narrowest plane the 8-lane groups of the nl4d kernels work on.
pub const MIN_PLANE_SIZE: u32 = 8;

/// The plane widths matrix candidates run at.
pub const MATRIX_PLANE_WIDTHS: [u32; 2] = [32, 64];

/// The int8 products in the order they are preferred when a device lists both.
pub(crate) const INT8_PREFERENCE: [Int8Matrix; 2] = [Int8Matrix::Unsigned, Int8Matrix::Signed];

/// The name CubeCL's HIP runtime reports.
pub const HIP_RUNTIME: &str = "hip";

/// The gfx architecture prefix of RDNA 4 chips.
const RDNA4_ARCH_PREFIX: &str = "gfx12";

/// The bits a u8 operand holds.
const UNSIGNED_OPERAND_BITS: u32 = 8;

/// The bits an i8 operand holds without reaching its sign bit.
const SIGNED_OPERAND_BITS: u32 = 7;

/// The first CUDA compute capability with independent thread scheduling.
const VOLTA_COMPUTE_CAPABILITY: u32 = 70;

/// Pre-Volta compute capabilities with full-rate f16. These are Tegra X1, GP100 and Tegra X2.
const FAST_F16_PRE_VOLTA: [u32; 3] = [53, 60, 62];

// NVIDIA device ids by chip, from the PCI ID database (https://pci-ids.ucw.cz), as shipped in
// hwdata's `pci.ids` version 2026.09.03. Every entry under vendor 10de whose name starts with the
// chip code is listed, except the audio functions.

/// Kepler chips, GK104 to GK210.
const KEPLER_DEVICE_IDS: &[u16] = &[
    0x0FA0, 0x0FA5, 0x0FA7, 0x0FC0, 0x0FC1, 0x0FC2, 0x0FC4, 0x0FC5, 0x0FC6, 0x0FC8, 0x0FC9, 0x0FCB, 0x0FCC,
    0x0FCD, 0x0FCE, 0x0FCF, 0x0FD0, 0x0FD1, 0x0FD2, 0x0FD3, 0x0FD4, 0x0FD5, 0x0FD6, 0x0FD7, 0x0FD8, 0x0FD9,
    0x0FDA, 0x0FDB, 0x0FDC, 0x0FDD, 0x0FDE, 0x0FDF, 0x0FE0, 0x0FE1, 0x0FE2, 0x0FE3, 0x0FE4, 0x0FE5, 0x0FE6,
    0x0FE7, 0x0FE8, 0x0FE9, 0x0FEA, 0x0FEC, 0x0FED, 0x0FEE, 0x0FEF, 0x0FF0, 0x0FF1, 0x0FF2, 0x0FF3, 0x0FF5,
    0x0FF6, 0x0FF7, 0x0FF8, 0x0FF9, 0x0FFA, 0x0FFB, 0x0FFC, 0x0FFD, 0x0FFE, 0x0FFF, 0x1001, 0x1003, 0x1004,
    0x1005, 0x1007, 0x1008, 0x100A, 0x100C, 0x101E, 0x101F, 0x1020, 0x1021, 0x1022, 0x1023, 0x1024, 0x1026,
    0x1027, 0x1028, 0x1029, 0x102A, 0x102D, 0x102E, 0x102F, 0x103A, 0x103C, 0x103F, 0x1180, 0x1182, 0x1183,
    0x1184, 0x1185, 0x1186, 0x1187, 0x1188, 0x1189, 0x118A, 0x118B, 0x118C, 0x118D, 0x118E, 0x118F, 0x1191,
    0x1193, 0x1194, 0x1195, 0x1198, 0x1199, 0x119A, 0x119D, 0x119E, 0x119F, 0x11A0, 0x11A1, 0x11A2, 0x11A3,
    0x11A4, 0x11A5, 0x11A7, 0x11A8, 0x11A9, 0x11AA, 0x11AC, 0x11AF, 0x11B0, 0x11B1, 0x11B4, 0x11B6, 0x11B7,
    0x11B8, 0x11B9, 0x11BA, 0x11BB, 0x11BC, 0x11BD, 0x11BE, 0x11BF, 0x11C0, 0x11C1, 0x11C2, 0x11C3, 0x11C4,
    0x11C5, 0x11C6, 0x11C7, 0x11C8, 0x11CB, 0x11D0, 0x11D1, 0x11D2, 0x11D3, 0x11E0, 0x11E1, 0x11E2, 0x11E3,
    0x11E7, 0x11FA, 0x11FC, 0x11FF, 0x1280, 0x1281, 0x1282, 0x1283, 0x1284, 0x1285, 0x1286, 0x1287, 0x1288,
    0x1289, 0x128A, 0x128B, 0x128C, 0x1290, 0x1291, 0x1292, 0x1293, 0x1294, 0x1295, 0x1296, 0x1298, 0x1299,
    0x129A, 0x12A0, 0x12AD, 0x12AE, 0x12AF, 0x12B0, 0x12B1, 0x12B9, 0x12BA, 0x130B,
];

/// Maxwell chips, GM107 to GM206.
const MAXWELL_DEVICE_IDS: &[u16] = &[
    0x1340, 0x1341, 0x1344, 0x1346, 0x1347, 0x1348, 0x1349, 0x134B, 0x134D, 0x134E, 0x134F, 0x137A, 0x137B,
    0x137D, 0x1380, 0x1381, 0x1382, 0x1389, 0x1390, 0x1391, 0x1392, 0x1393, 0x1398, 0x1399, 0x139A, 0x139B,
    0x139C, 0x139D, 0x13AD, 0x13AE, 0x13B0, 0x13B1, 0x13B2, 0x13B3, 0x13B4, 0x13B6, 0x13B9, 0x13BA, 0x13BB,
    0x13BC, 0x13BD, 0x13BE, 0x13BF, 0x13C0, 0x13C1, 0x13C2, 0x13C3, 0x13C4, 0x13D7, 0x13D8, 0x13D9, 0x13DA,
    0x13E4, 0x13E7, 0x13F0, 0x13F1, 0x13F2, 0x13F3, 0x13F8, 0x13F9, 0x13FA, 0x13FB, 0x1401, 0x1402, 0x1404,
    0x1406, 0x1407, 0x1427, 0x1430, 0x1431, 0x1436, 0x1613, 0x1617, 0x1618, 0x1619, 0x161A, 0x1667, 0x1676,
    0x174D, 0x174E, 0x1789, 0x179C, 0x17C2, 0x17C8, 0x17F0, 0x17F1, 0x17FD,
];

/// Pascal consumer chips, GP102 to GP108, with f16 at 1/64 of the f32 rate.
const PASCAL_CONSUMER_DEVICE_IDS: &[u16] = &[
    0x1B00, 0x1B01, 0x1B02, 0x1B04, 0x1B06, 0x1B07, 0x1B30, 0x1B38, 0x1B39, 0x1B70, 0x1B78, 0x1B80, 0x1B81,
    0x1B82, 0x1B83, 0x1B84, 0x1B87, 0x1BA0, 0x1BA1, 0x1BA2, 0x1BA9, 0x1BAA, 0x1BAD, 0x1BB0, 0x1BB1, 0x1BB3,
    0x1BB4, 0x1BB5, 0x1BB6, 0x1BB7, 0x1BB8, 0x1BB9, 0x1BBB, 0x1BC7, 0x1BE0, 0x1BE1, 0x1C00, 0x1C01, 0x1C02,
    0x1C03, 0x1C04, 0x1C06, 0x1C07, 0x1C09, 0x1C20, 0x1C21, 0x1C22, 0x1C23, 0x1C2D, 0x1C30, 0x1C31, 0x1C35,
    0x1C36, 0x1C60, 0x1C61, 0x1C62, 0x1C70, 0x1C81, 0x1C82, 0x1C83, 0x1C8C, 0x1C8D, 0x1C8E, 0x1C8F, 0x1C90,
    0x1C91, 0x1C92, 0x1C94, 0x1C96, 0x1CA7, 0x1CA8, 0x1CAA, 0x1CB1, 0x1CB2, 0x1CB3, 0x1CB6, 0x1CBA, 0x1CBB,
    0x1CBC, 0x1CBD, 0x1CCC, 0x1CCD, 0x1CFA, 0x1CFB, 0x1D01, 0x1D02, 0x1D10, 0x1D11, 0x1D12, 0x1D13, 0x1D16,
    0x1D33, 0x1D34, 0x1D52, 0x1D56,
];

/// GP100, the Pascal chip with full-rate f16.
const PASCAL_GP100_DEVICE_IDS: &[u16] = &[
    0x15C2, 0x15F0, 0x15F1, 0x15F7, 0x15F8, 0x15F9, 0x15FA, 0x15FB, 0x15FC, 0x15FF, 0x1725, 0x172E, 0x172F,
];

/// A family of devices the legality rules tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    Amd,
    Apple,
    Intel,
    /// An NVIDIA part from Kepler, Maxwell or Pascal, without independent thread scheduling.
    NvidiaPreVolta {
        fast_f16: bool,
    },
    /// An NVIDIA part from Volta on, or one the table does not list.
    Nvidia,
    Unknown,
}

impl DeviceClass {
    /// The class of the device `identity` describes.
    ///
    /// CUDA reads the compute capability, ROCm and Metal are known by backend, and every other
    /// backend reads the PCI vendor and device ids.
    pub fn of(identity: &DeviceIdentity) -> Self {
        match identity.backend.as_str() {
            "cuda" => cuda_class(identity.compute_capability),
            "rocm" => DeviceClass::Amd,
            "metal" => DeviceClass::Apple,
            _ => vendor_class(identity.vendor_id, identity.device_id),
        }
    }
}

fn cuda_class(compute_capability: Option<u32>) -> DeviceClass {
    let Some(capability) = compute_capability else {
        return DeviceClass::Nvidia;
    };

    if capability >= VOLTA_COMPUTE_CAPABILITY {
        return DeviceClass::Nvidia;
    }

    let fast_f16 = FAST_F16_PRE_VOLTA.contains(&capability);
    DeviceClass::NvidiaPreVolta { fast_f16 }
}

fn vendor_class(vendor_id: u32, device_id: u32) -> DeviceClass {
    match vendor_id {
        AMD_VENDOR_ID => DeviceClass::Amd,
        INTEL_VENDOR_ID => DeviceClass::Intel,
        APPLE_VENDOR_ID => DeviceClass::Apple,
        NVIDIA_VENDOR_ID => nvidia_class(device_id),
        _ => DeviceClass::Unknown,
    }
}

fn nvidia_class(device_id: u32) -> DeviceClass {
    let Ok(short_id) = u16::try_from(device_id) else {
        return DeviceClass::Nvidia;
    };

    if PASCAL_GP100_DEVICE_IDS.binary_search(&short_id).is_ok() {
        return DeviceClass::NvidiaPreVolta { fast_f16: true };
    }

    let slow_tables = [KEPLER_DEVICE_IDS, MAXWELL_DEVICE_IDS, PASCAL_CONSUMER_DEVICE_IDS];
    let listed = slow_tables
        .iter()
        .any(|table| table.binary_search(&short_id).is_ok());

    if listed {
        DeviceClass::NvidiaPreVolta { fast_f16: false }
    } else {
        DeviceClass::Nvidia
    }
}

/// What a device may run beyond the fit check.
///
/// The default keeps the kernels a device runs without any rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Legality {
    /// Only warp-uniform search walks may run, because shuffles inside divergent control flow are
    /// undefined without independent thread scheduling.
    pub uniform_walk_only: bool,
    /// The f16 search ring may run.
    pub f16_search: bool,
    /// Packed 64-bit accumulators may run.
    pub packed_atomics: bool,
}

impl Default for Legality {
    fn default() -> Self {
        Self {
            uniform_walk_only: false,
            f16_search: true,
            packed_atomics: false,
        }
    }
}

impl Legality {
    pub fn for_device(device: &DeviceProfile) -> Self {
        let class = DeviceClass::of(&device.identity);
        let pre_volta = matches!(class, DeviceClass::NvidiaPreVolta { .. });
        let slow_f16 = class == DeviceClass::NvidiaPreVolta { fast_f16: false };
        let is_apple = class == DeviceClass::Apple;
        let packed_atomics = device.limits.int64_atomic_add && !is_apple;

        Self {
            uniform_walk_only: pre_volta,
            f16_search: !slow_f16,
            packed_atomics,
        }
    }
}

/// A device whose planes are narrower than the nl4d kernels' 8-lane groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "this device runs subgroups of {plane_size} lanes, and nl4d needs at least {MIN_PLANE_SIZE}. Use \
     nlmeans or another accelerator"
)]
pub struct PlaneSizeError {
    pub plane_size: u32,
}

/// Checks that every plane on the device holds at least [MIN_PLANE_SIZE] lanes.
pub fn check_plane_size(limits: &DeviceLimits) -> Result<(), PlaneSizeError> {
    if limits.plane_size_min >= MIN_PLANE_SIZE {
        return Ok(());
    }

    Err(PlaneSizeError {
        plane_size: limits.plane_size_min,
    })
}

/// Whether `configs` holds a 16x16x16 f16 matrix product accumulating in f32.
pub fn offers_f16_matrix(configs: &BTreeSet<MmaConfig>) -> bool {
    let half_elem = ElemType::Float(FloatKind::F16);
    let half_type = StorageType::Scalar(half_elem);
    let float_elem = ElemType::Float(FloatKind::F32);
    let float_type = StorageType::Scalar(float_elem);
    let config = MmaConfig {
        a_type: half_type,
        b_type: half_type,
        cd_type: float_type,
        m: MATRIX_TILE,
        n: MATRIX_TILE,
        k: MATRIX_TILE,
    };

    configs.contains(&config)
}

/// The plane width matrix candidates run at, or `None` when the device runs none.
///
/// Matrix candidates run at the widest plane the device offers, when that is one of
/// [MATRIX_PLANE_WIDTHS].
pub fn matrix_plane_width(limits: &DeviceLimits) -> Option<u32> {
    let width = limits.plane_size_max;
    let supported = MATRIX_PLANE_WIDTHS.contains(&width);

    if limits.f16_matrix && supported {
        Some(width)
    } else {
        None
    }
}

/// The 16x16x16 int8 matrix product a device runs, accumulating in i32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Int8Matrix {
    /// u8 times u8.
    Unsigned,
    /// i8 times i8.
    Signed,
}

impl Int8Matrix {
    /// The operand element of the product.
    pub fn element(self) -> ElemType {
        match self {
            Int8Matrix::Unsigned => ElemType::UInt(UIntKind::U8),
            Int8Matrix::Signed => ElemType::Int(IntKind::I8),
        }
    }

    /// The bits one operand value holds without overflow.
    fn value_bits(self) -> u32 {
        match self {
            Int8Matrix::Unsigned => UNSIGNED_OPERAND_BITS,
            Int8Matrix::Signed => SIGNED_OPERAND_BITS,
        }
    }

    /// The 16x16x16 config of the product, accumulating in i32.
    fn config(self) -> MmaConfig {
        let operand_elem = self.element();
        let operand_type = StorageType::Scalar(operand_elem);
        let accumulator_elem = ElemType::Int(IntKind::I32);
        let accumulator_type = StorageType::Scalar(accumulator_elem);

        MmaConfig {
            a_type: operand_type,
            b_type: operand_type,
            cd_type: accumulator_type,
            m: MATRIX_TILE,
            n: MATRIX_TILE,
            k: MATRIX_TILE,
        }
    }
}

/// How codes of one depth split into parts that fit an int8 operand, as a comptime kernel argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Int8Layout {
    pub element: Int8Matrix,
    /// Bits in the low part, or 0 when codes go in whole.
    pub low_bits: u32,
}

impl Int8Layout {
    /// The layout `element` takes for `codes`.
    ///
    /// Codes that fit the operand go in whole. Wider codes split, with the low part taking the bits
    /// the high part cannot hold. Signed operands split codes in half, so both parts hold at most 5
    /// bits at 10-bit depth.
    pub fn for_codes(element: Int8Matrix, codes: IntegerCodes) -> Int8Layout {
        let value_bits = element.value_bits();
        let low_bits = if codes.depth <= value_bits {
            0
        } else {
            match element {
                Int8Matrix::Unsigned => codes.depth - value_bits,
                Int8Matrix::Signed => codes.depth.div_ceil(2),
            }
        };

        Int8Layout { element, low_bits }
    }

    /// The parts each code splits into, 1 or 2.
    pub fn parts(self) -> u32 {
        if self.low_bits == 0 { 1 } else { 2 }
    }

    /// The matrix products one patch pair takes, 1 or 4.
    pub fn products(self) -> u32 {
        let parts = self.parts();
        parts * parts
    }

    /// The i32 accumulators one patch pair takes, 1 or 3.
    ///
    /// Both cross products share one accumulator.
    pub fn accumulators(self) -> u32 {
        if self.low_bits == 0 { 1 } else { 3 }
    }
}

/// The int8 product `configs` offer, preferring u8 when both are listed.
pub fn offers_int8_matrix(configs: &BTreeSet<MmaConfig>) -> Option<Int8Matrix> {
    INT8_PREFERENCE
        .into_iter()
        .find(|&element| lists_int8_matrix(configs, element))
}

/// Whether `configs` hold the exact 16x16x16 config of `element`, accumulating in i32.
pub fn lists_int8_matrix(configs: &BTreeSet<MmaConfig>, element: Int8Matrix) -> bool {
    let config = element.config();
    configs.contains(&config)
}

/// The plane width int8 matrix candidates run at, under the same rule as [matrix_plane_width].
pub fn int8_matrix_plane_width(limits: &DeviceLimits) -> Option<u32> {
    let width = limits.plane_size_max;
    let supported = MATRIX_PLANE_WIDTHS.contains(&width);

    if limits.int8_matrix.is_some() && supported {
        Some(width)
    } else {
        None
    }
}

/// Whether CubeCL compiles matrix products on `runtime` for devices of `architectures`.
///
/// CubeCL 0.10's HIP compiler emits RDNA 3 WMMA builtins for gfx12 chips, and LLVM aborts the
/// process when it cannot select them. HIP devices of unknown architecture are refused too.
pub fn compiles_matrix_products(runtime: &str, architectures: &[String]) -> bool {
    if runtime != HIP_RUNTIME {
        return true;
    }

    let known = !architectures.is_empty();
    let rdna4 = architectures
        .iter()
        .any(|architecture| architecture.starts_with(RDNA4_ARCH_PREFIX));

    known && !rdna4
}

/// Every device-id table with its name, for the ordering test.
#[cfg(test)]
pub(crate) fn nvidia_tables_for_tests() -> [(&'static str, &'static [u16]); 4] {
    [
        ("kepler", KEPLER_DEVICE_IDS),
        ("maxwell", MAXWELL_DEVICE_IDS),
        ("pascal consumer", PASCAL_CONSUMER_DEVICE_IDS),
        ("pascal gp100", PASCAL_GP100_DEVICE_IDS),
    ]
}
