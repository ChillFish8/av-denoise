pub(crate) mod colour;
mod correlation;
mod curve;
mod derived;
mod estimator;
mod fold;
mod history;
pub(crate) mod kernels;
mod reading;
pub(crate) mod select;
mod slot;
#[cfg(test)]
mod snapshot;
mod spatial;
mod state;
#[cfg(test)]
mod stats;
mod temporal;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::correlation::{build_spatial_offset_lut, correlation_factor, spatial_offset_factor};
pub(crate) use self::correlation::{spatial_offset_index, spatial_offset_lut_len};
pub use self::curve::NOISE_CURVE_BINS;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::curve::NoiseCurve;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::curve::build_noise_curve;
pub use self::derived::DerivedValues;
pub(crate) use self::derived::{
    DERIVED_CONFIDENCE_SAD_FLOOR,
    DERIVED_CURVE_VALID,
    DERIVED_H2_INV_NORM,
    DERIVED_NOISE_OFFSET,
    DERIVED_PILOT_H2_INV_NORM,
    DERIVED_SAD_FLOOR,
    DERIVED_SIGMA,
    DERIVED_WEIGHT_SCALE,
    DERIVED_WORDS,
    DerivedCoefficients,
    NoiseConsumer,
};
pub(crate) use self::fold::{FoldSettings, NoiseSettings};
pub(crate) use self::history::{NOISE_WINDOW_LEN, NOISE_WINDOW_RADIUS, WindowBounds, history_capacity};
pub(crate) use self::reading::{TAG_QUARTER, TAG_RHO};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::snapshot::NoiseSnapshot;
pub(crate) use self::spatial::IMMERKAER_LANES;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::spatial::{NoiseCtx, run_noise_estimate};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::spatial::{partials_len, sigma_from_abs_sum};
pub(crate) use self::state::{
    NoiseShape,
    NoiseState,
    ReadingSettings,
    SAMPLE_RUN_QUARTERS,
    SAMPLE_RUN_RHO,
    reading_buffer_sizes,
};
pub(crate) use self::temporal::{
    BLOCK_MOVED_LANE,
    BLOCK_TEXTURE_LANE,
    COMPENSATION_GAIN_MAX,
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_SUM_D,
    QUARTER_SUM_D2,
    StatsMotion,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTERS,
    UNMEASURED_FLATNESS,
    temporal_stats_blocks,
    temporal_stats_record_len,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::temporal::{
    TEMPORAL_QUARTER_SIZE,
    TemporalStatsCtx,
    accepted_static_blocks,
    aggregate_temporal_noise_stats,
    run_temporal_noise_stats,
    texture_lane,
};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::temporal::{temporal_noise_reading, temporal_stats_buf_bytes};

/// The most channels a noise reading holds, which sizes every per-channel slot.
pub(crate) const NOISE_CHANNELS: u32 = 2;

/// Cube X dimension for the Immerkær noise partials kernel.
pub(crate) const IMMERKAER_BLOCK_X: u32 = 32;
/// Cube Y dimension for the Immerkær noise partials kernel.
pub(crate) const IMMERKAER_BLOCK_Y: u32 = 8;

/// Block size for the 1D noise reduce and zero kernels.
pub(crate) const NOISE_BLOCK_1D: u32 = 256;
