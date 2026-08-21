//! The non-local means denoiser that sits behind [`crate::Denoiser`].
//!
//! Non-local means cleans a pixel by finding patches elsewhere that look
//! like the patch around it, then averaging them. Similar patches get a
//! large weight and dissimilar ones get almost none, so flat areas
//! smooth out while edges survive.
//!
//! The search can reach across neighbouring frames as well as within one
//! frame, which is what the temporal radius controls.
//!
//! # Layout
//!
//! `params` holds the tuning values and the calibrated defaults, and
//! [`NlmParams`] is the single struct everything else is built from.
//!
//! [`NlmDenoiser`] owns the GPU buffers and the frame ring, and
//! `dispatch` turns one set of parameters into the sequence of kernel
//! launches that produces a frame.
//!
//! [`kernels`] holds the GPU code itself. `noise` measures how noisy a
//! frame is, [`motion`] tracks movement between frames, and
//! [`prefilter`] builds the cleaner reference image that patches are
//! compared against.

pub mod kernels;
pub mod motion;
pub mod prefilter;

mod align;
mod denoiser;
mod dispatch;
mod noise;
mod params;
mod pending;

// Every test in this tree runs against a real GPU runtime, see
// `tests::helpers::R`, so it only builds when a wgpu-backed feature is
// enabled. A cpu-only build skips it entirely, and the
// `cpu_smoke_tests` module in `src/denoiser.rs` covers that backend
// instead.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

pub(crate) use denoiser::RingView;
pub use denoiser::{GpuOutput, NlmDenoiser};
pub use motion::{MotionCompensationMode, MotionEstimation, MotionSearch};
pub use params::{
    ChannelMode,
    HqParams,
    MAX_PATCH_RADIUS,
    MAX_SEARCH_RADIUS,
    MAX_TEMPORAL_RADIUS,
    MIN_FRAME_DIM,
    NlmParams,
    hq_default_strength,
    validate_dimensions,
};
pub use pending::Pending;
pub use prefilter::{DEFAULT_PILOT_STRENGTH_SCALE, PrefilterMode};

/// Cube X dimension for tile-heavy fused/separable kernels.
pub const BLOCK_X: u32 = 32;
/// Cube Y dimension for tile-heavy fused/separable kernels.
pub const BLOCK_Y: u32 = 8;

/// Cube shape for the per-pixel `nlm_accumulate` kernel, which has no
/// shared-memory tile.
///
/// On RDNA-class GPUs this shape benchmarks 10 to 25% faster than the
/// tile-heavy default. The kernel waits on memory rather than compute,
/// so the extra threads hide the load latency.
pub const BLOCK_X_THIN: u32 = 32;
pub const BLOCK_Y_THIN: u32 = 16;

/// Largest 1D grid a dispatch may ask for, set by the WebGPU and Vulkan
/// limits.
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for 1D utility kernels (copy, zero).
pub(crate) const BLOCK_1D: u32 = 256;

/// Bit depth of a source's samples.
///
/// Normalisation divides by [`Depth::max_value`], so a value in
/// normalised units means the same thing at every depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    Eight,
    Ten,
    Twelve,
}

/// Returned when a source declares a bit depth the denoiser does not handle.
#[derive(Debug, thiserror::Error)]
#[error("unsupported bit depth {0}, av-denoise supports 8, 10, and 12-bit")]
pub struct UnsupportedDepthError(pub usize);

impl Depth {
    /// Maps a declared bit depth onto a [`Depth`].
    pub fn from_bits(bits: usize) -> Result<Self, UnsupportedDepthError> {
        match bits {
            8 => Ok(Depth::Eight),
            10 => Ok(Depth::Ten),
            12 => Ok(Depth::Twelve),
            other => Err(UnsupportedDepthError(other)),
        }
    }

    /// Bits per sample.
    pub fn bits(self) -> usize {
        match self {
            Depth::Eight => 8,
            Depth::Ten => 10,
            Depth::Twelve => 12,
        }
    }

    /// Bytes each sample takes up on the wire.
    ///
    /// Depths above 8 use a little-endian 16-bit word.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Depth::Eight => 1,
            Depth::Ten | Depth::Twelve => 2,
        }
    }

    /// The largest sample value this depth can hold, which is also the
    /// normalisation divisor.
    pub fn max_value(self) -> f32 {
        ((1u32 << self.bits()) - 1) as f32
    }

    /// The sample value that means neutral chroma at this depth.
    pub fn neutral_chroma(self) -> u16 {
        1 << (self.bits() - 1)
    }
}

/// Scales native-depth samples into normalised `[0, 1]` f32.
pub fn normalize(input: &[u16], depth: Depth) -> Vec<f32> {
    let max = depth.max_value();
    input.iter().map(|&v| v as f32 / max).collect()
}

/// Reverse of [`normalize`].
///
/// Values outside `[0, 1]` are clamped, and `NaN` becomes 0.
pub fn denormalize(input: &[f32], depth: Depth) -> Vec<u16> {
    let max = depth.max_value();
    input
        .iter()
        .map(|&v| (v * max).round().clamp(0.0, max) as u16)
        .collect()
}

#[cfg(test)]
mod depth_tests {
    use super::*;

    #[test]
    fn from_bits_accepts_supported_depths() {
        assert_eq!(Depth::from_bits(8).unwrap(), Depth::Eight);
        assert_eq!(Depth::from_bits(10).unwrap(), Depth::Ten);
        assert_eq!(Depth::from_bits(12).unwrap(), Depth::Twelve);
    }

    #[test]
    fn from_bits_rejects_unsupported_depths() {
        for bits in [0, 9, 14, 16] {
            let err = Depth::from_bits(bits).expect_err("expected rejection");
            assert!(
                err.to_string().contains(&bits.to_string()),
                "error should name the depth, got {err}"
            );
        }
    }

    #[test]
    fn depth_properties_match_the_format() {
        assert_eq!(Depth::Eight.bytes_per_sample(), 1);
        assert_eq!(Depth::Ten.bytes_per_sample(), 2);
        assert_eq!(Depth::Twelve.bytes_per_sample(), 2);

        assert_eq!(Depth::Eight.max_value(), 255.0);
        assert_eq!(Depth::Ten.max_value(), 1023.0);
        assert_eq!(Depth::Twelve.max_value(), 4095.0);

        assert_eq!(Depth::Eight.neutral_chroma(), 128);
        assert_eq!(Depth::Ten.neutral_chroma(), 512);
        assert_eq!(Depth::Twelve.neutral_chroma(), 2048);
    }

    /// Limited-range black and white land on matching normalised values
    /// at every depth, which is what lets every calibrated constant in
    /// the library stay depth-independent.
    ///
    /// The match is within one 8-bit code level rather than exact. ITU
    /// defines the limited-range endpoints as exact multiples, so 235
    /// becomes 940 and then 3760, but full scale is not a multiple,
    /// because 255 becomes 1023 and then 4095.
    ///
    /// That leaves 235/255 and 940/1023 differing by 0.0027, roughly
    /// 0.69 of an 8-bit step. Agreement below one step is the real
    /// property here.
    #[test]
    fn normalized_scale_is_identical_across_depths() {
        /// One 8-bit code level, the precision the endpoints agree to.
        const TOL: f32 = 1.0 / 255.0;

        let eight = normalize(&[16, 235], Depth::Eight);
        let ten = normalize(&[64, 940], Depth::Ten);
        let twelve = normalize(&[256, 3760], Depth::Twelve);

        for (a, b) in eight.iter().zip(ten.iter()) {
            assert!((a - b).abs() < TOL, "8-bit {a} vs 10-bit {b}");
        }
        for (a, b) in eight.iter().zip(twelve.iter()) {
            assert!((a - b).abs() < TOL, "8-bit {a} vs 12-bit {b}");
        }
    }

    #[test]
    fn normalization_round_trips_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let original: Vec<u16> = vec![0, 1, 16, 64, 128, 235, max / 2, max - 1, max];
            let restored = denormalize(&normalize(&original, depth), depth);
            assert_eq!(original, restored, "round trip failed at {depth:?}");
        }
    }

    #[test]
    fn denormalize_clamps_out_of_range_input() {
        let out = denormalize(&[-0.5, 0.0, 1.0, 1.5], Depth::Ten);
        assert_eq!(out, vec![0, 0, 1023, 1023]);
    }
}
