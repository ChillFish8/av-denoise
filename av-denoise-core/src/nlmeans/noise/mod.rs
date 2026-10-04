mod correlation;
mod curve;
mod estimator;
mod spatial;
mod stats;
mod strength_map;
mod temporal;
#[cfg(test)]
mod tests;

pub(super) use self::correlation::{
    build_spatial_offset_lut,
    correlation_factor,
    spatial_offset_factor,
    spatial_offset_lut_len,
};
pub use self::curve::NOISE_CURVE_BINS;
pub(crate) use self::curve::NoiseCurve;
#[cfg(test)]
pub(super) use self::curve::build_noise_curve;
pub(super) use self::estimator::{EMA_ALPHA, NoiseEstimator};
pub(super) use self::spatial::{
    NoiseCtx,
    noise_partials_slot_stride_bytes,
    partials_len,
    run_noise_estimate,
    sigma_block_p25_from_partials,
    sigma_from_abs_sum,
};
#[cfg(test)]
pub(super) use self::strength_map::classify_quarters;
#[cfg(test)]
pub(crate) use self::strength_map::{QuarterClass, QuarterTensor};
pub(crate) use self::strength_map::{QuarterClasses, StrengthMapParams};
pub(super) use self::temporal::{
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
    TEMPORAL_QUARTERS,
    TemporalNoiseReading,
    TemporalNoiseSample,
    TemporalStatsCtx,
    read_temporal_stats_slot,
    run_temporal_noise_stats,
    temporal_noise_reading,
    temporal_stats_buf_bytes,
    zero_temporal_stats_slot,
};
#[cfg(test)]
pub(super) use self::temporal::{
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_SIZE,
    accepted_static_blocks,
    aggregate_temporal_noise_stats,
    temporal_stats_blocks,
    temporal_stats_record_len,
};
