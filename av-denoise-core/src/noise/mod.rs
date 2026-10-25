mod correlation;
mod curve;
mod derived;
mod estimator;
mod fold;
#[cfg(test)]
mod host;
pub(crate) mod kernels;
mod line_ring;
mod reading;
pub(crate) mod select;
mod slot;
#[cfg(test)]
mod snapshot;
mod spatial;
mod state;
#[cfg(test)]
mod stats;
mod strength_map;
mod temporal;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::correlation::{build_spatial_offset_lut, correlation_factor, spatial_offset_factor};
pub(crate) use self::correlation::{spatial_offset_index, spatial_offset_lut_len};
pub use self::curve::NOISE_CURVE_BINS;
#[cfg(test)]
pub(crate) use self::curve::NoiseCurve;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::curve::build_noise_curve;
pub use self::derived::DerivedValues;
pub(crate) use self::derived::{
    DERIVED_CONFIDENCE_SAD_FLOOR,
    DERIVED_CURVE_VALID,
    DERIVED_H2_INV_NORM,
    DERIVED_MAP_MODE,
    DERIVED_NOISE_OFFSET,
    DERIVED_PILOT_H2_INV_NORM,
    DERIVED_SAD_FLOOR,
    DERIVED_SIGMA,
    DERIVED_WEIGHT_SCALE,
    DERIVED_WORDS,
    DerivedCoefficients,
    NoiseConsumer,
};
pub(crate) use self::fold::{Fold, FoldSettings, NoiseSettings};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::snapshot::NoiseSnapshot;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::spatial::{NoiseCtx, run_noise_estimate};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::spatial::{partials_len, sigma_from_abs_sum};
pub(crate) use self::state::{
    NoiseShape,
    NoiseState,
    ReadingSettings,
    StrengthMapKind,
    reading_buffer_sizes,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::strength_map::QuarterClasses;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::strength_map::classify_quarters;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::strength_map::{QuarterClass, QuarterTensor};
pub(crate) use self::strength_map::{QuarterSettings, StrengthMapParams};
pub(crate) use self::temporal::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    UNMEASURED_FLATNESS,
    temporal_stats_blocks,
    temporal_stats_record_len,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::temporal::{
    TEMPORAL_NOISE_BLOCK,
    TemporalStatsCtx,
    accepted_static_blocks,
    aggregate_temporal_noise_stats,
    run_temporal_noise_stats,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::temporal::{temporal_noise_reading, temporal_stats_buf_bytes};

/// Cube X dimension for the Immerkær noise partials kernel.
pub(crate) const IMMERKAER_BLOCK_X: u32 = 32;
/// Cube Y dimension for the Immerkær noise partials kernel.
pub(crate) const IMMERKAER_BLOCK_Y: u32 = 8;

/// Block size for the 1D noise reduce and zero kernels.
pub(crate) const NOISE_BLOCK_1D: u32 = 256;

/// Every threshold is left as it is.
pub(crate) const STRENGTH_MAP_OFF: u32 = 0;
/// The map scales channel 0's noise-curve ratio.
pub(crate) const STRENGTH_MAP_LUMA: u32 = 1;
/// The map scales every channel's threshold, for a denoiser with no curve.
pub(crate) const STRENGTH_MAP_ALL: u32 = 2;
