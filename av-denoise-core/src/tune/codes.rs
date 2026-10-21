use super::device::DeviceLimits;
use super::plan::PipelineProfile;
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::integer::sliding::SLIDING_BLOCK;
use crate::engine::SampleFormat;
use crate::nlmeans::params::channel_scale_for;

/// The deepest source whose codes the integer route stores.
pub const MAX_INTEGER_DEPTH: u32 = 10;

/// The bound every exact f32 integer sum stays below.
///
/// f32 holds every integer up to 2^24 exactly, so a sum of integer products below it never rounds.
pub(crate) const F32_EXACT_LIMIT: u32 = 1 << 24;

/// The most code values one exact f16 matrix product may sum.
///
/// Every partial sum of products of 8-bit parts then stays below 2^24, where f32 is still exact.
pub const MATRIX_EXACT_VALUES: u32 = 258;

// Clippy rejects a runtime `assert!` on constants, so the bound is checked at compile time.
const _: () = assert!(
    MATRIX_EXACT_VALUES * 255 * 255 < F32_EXACT_LIMIT
        && (MATRIX_EXACT_VALUES + 1) * 255 * 255 >= F32_EXACT_LIMIT,
    "MATRIX_EXACT_VALUES must be the largest count of 8-bit products that sums exactly in f32"
);

/// The largest squared difference of two codes at [MAX_INTEGER_DEPTH].
const MAX_SQUARED_DIFFERENCE: u32 = ((1 << MAX_INTEGER_DEPTH) - 1) * ((1 << MAX_INTEGER_DEPTH) - 1);

// A scalar search lane sums one channel of one patch column in f32 before converting to i32.
const _: () = assert!(
    PATCH_SIZE * MAX_SQUARED_DIFFERENCE < F32_EXACT_LIMIT,
    "one channel of an 8-pixel column must sum exactly in f32 at the deepest codes"
);

// A sliding block sums one channel of its 4x4 pixels in f32 before converting to i32.
const _: () = assert!(
    SLIDING_BLOCK * SLIDING_BLOCK * MAX_SQUARED_DIFFERENCE < F32_EXACT_LIMIT,
    "one channel of a sliding block must sum exactly in f32 at the deepest codes"
);

/// The bits each matrix part holds.
const PART_BITS: u32 = 8;

/// The integer codes a source's samples arrive as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerCodes {
    /// The significant bits per sample.
    pub depth: u32,
}

impl IntegerCodes {
    /// The codes of `format`, or `None` when the integer route cannot store them.
    ///
    /// 8-bit sources and 16-bit words up to [MAX_INTEGER_DEPTH] bits have codes. Deeper and float
    /// sources have none.
    pub fn from_format(format: SampleFormat) -> Option<IntegerCodes> {
        match format {
            SampleFormat::U8 => Some(IntegerCodes { depth: 8 }),
            SampleFormat::U16 { depth } => {
                let depth = u32::from(depth);
                let storable = depth <= MAX_INTEGER_DEPTH;
                storable.then_some(IntegerCodes { depth })
            },
            SampleFormat::F32 => None,
        }
    }

    /// The codes an engine under `profile` keeps an integer ring of, or `None` for no ring.
    ///
    /// Only [IntegerSearch](crate::tune::PipelineProfile::IntegerSearch) keeps one, and only for a
    /// source with codes the device can store.
    pub fn for_profile(
        profile: PipelineProfile,
        format: SampleFormat,
        limits: &DeviceLimits,
    ) -> Option<IntegerCodes> {
        if profile != PipelineProfile::IntegerSearch {
            return None;
        }

        let codes = IntegerCodes::from_format(format)?;
        codes.storable(limits).then_some(codes)
    }

    pub fn max_code(self) -> u32 {
        (1 << self.depth) - 1
    }

    /// Whether codes split into a high and a low 8-bit part for matrix products.
    pub fn wide(self) -> bool {
        self.depth > PART_BITS
    }

    /// The shift that takes a code's high part down to 8 bits, 0 for 8-bit codes.
    pub fn split_shift(self) -> u32 {
        self.depth.saturating_sub(PART_BITS)
    }

    /// The bytes one stored code takes, which is 2 at every depth.
    pub fn code_bytes(self) -> usize {
        size_of::<u16>()
    }

    /// Whether the device can bind the buffers the codes are stored in.
    pub fn storable(self, limits: &DeviceLimits) -> bool {
        limits.u16_buffer
    }

    /// The factor that turns an integer SSD over `channels` channels into the kernels' distance.
    ///
    /// It is the channel scale over the squared top code, divided once on the host.
    pub fn inv_scale2(self, channels: u32) -> f32 {
        let scale = channel_scale_for(channels);
        let max_code = self.max_code();
        let squared_max = (max_code * max_code) as f32;

        scale / squared_max
    }

    /// Whether a matrix product over `values` code values sums exactly.
    pub fn matrix_exact(values: u32) -> bool {
        values <= MATRIX_EXACT_VALUES
    }
}
