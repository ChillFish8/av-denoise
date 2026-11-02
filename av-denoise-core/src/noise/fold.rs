use std::sync::atomic::Ordering;

use anyhow::ensure;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::correlation::{CORRELATION_FACTOR_TABLE, build_spatial_offset_lut, spatial_offset_lut_len};
use super::derived::{DERIVED_WORDS, DerivedCoefficients, DerivedValues, NoiseConsumer};
use super::history::{NOISE_WINDOW_LEN, ReadingHistory, WindowBounds};
use super::kernels::noise_window_fold;
use super::reading::READING_WORDS;
use super::spatial::{RESULTS_PER_SLOT, immerkaer_sigma_scale};
use super::state::{NoiseShape, NoiseState, ReadingSettings};
use super::{NOISE_BLOCK_1D, NOISE_CHANNELS, NOISE_CURVE_BINS};
use crate::tune::gpu::bind;

/// The factor each frame's combined reading is scaled by before the median.
pub(crate) const FOLD_SIGMA_SCALE: u32 = 0;
/// 1.0 when the consumer applies the curve.
pub(crate) const FOLD_CURVE_APPLIES: u32 = 1;
pub(crate) const FOLD_WEIGHT_G_MAX: u32 = 2;
pub(crate) const FOLD_WEIGHT_FLOOR: u32 = 3;
/// 1.0 when `h2` follows the median chain, 0.0 for a fixed strength.
pub(crate) const FOLD_H2_AUTO: u32 = 4;
pub(crate) const FOLD_H2_PER_INVERSE_VARIANCE: u32 = 5;
pub(crate) const FOLD_H2_FIXED: u32 = 6;
pub(crate) const FOLD_PILOT_H2_FACTOR: u32 = 7;
pub(crate) const FOLD_OFFSET_PER_VARIANCE: u32 = 8;
/// 1.0 when the main pass subtracts the offset.
pub(crate) const FOLD_MAIN_OFFSET: u32 = 9;
pub(crate) const FOLD_SAD_PER_SIGMA: u32 = 10;
pub(crate) const FOLD_CONFIDENCE_SAD_PER_SIGMA: u32 = 11;
/// The factor that turns one channel's Immerkær abs sum into its sigma.
pub(crate) const FOLD_IMMERKAER_FACTOR: u32 = 12;
/// The reciprocal width of each correlation-table segment, 3 words.
pub(crate) const FOLD_SEGMENT_RECIPROCALS: u32 = 13;
/// The `f32` words of the fold settings buffer.
pub(crate) const FOLD_WORDS: u32 = 16;

/// The words of one chain, its valid flag then one sigma per channel.
pub(crate) const STATE_CHAIN_WORDS: u32 = 1 + NOISE_CHANNELS;
/// The median chain, which drives the strength.
pub(crate) const STATE_MEDIAN: u32 = 0;
/// The low chain, which drives the noise offsets.
pub(crate) const STATE_LOW: u32 = STATE_MEDIAN + STATE_CHAIN_WORDS;
/// The low chain without the correlation boost.
pub(crate) const STATE_LOW_UNBOOSTED: u32 = STATE_LOW + STATE_CHAIN_WORDS;
/// The temporal median alone, each channel over the frames with a sample of that channel.
///
/// The flag is set when any channel has a sampled frame. A channel with none holds its unboosted
/// low value.
pub(crate) const STATE_TEMPORAL_ONLY: u32 = STATE_LOW_UNBOOSTED + STATE_CHAIN_WORDS;
/// 1.0 when a frame of the window has a sample of channel 0, which sets the correlation.
pub(crate) const STATE_RHO_VALID: u32 = STATE_TEMPORAL_ONLY + STATE_CHAIN_WORDS;
pub(crate) const STATE_RHO: u32 = STATE_RHO_VALID + 1;
/// 1.0 while the state holds a curve.
pub(crate) const STATE_CURVE_VALID: u32 = STATE_RHO + 1;
pub(crate) const STATE_QUARTER_MEDIAN: u32 = STATE_CURVE_VALID + 1;
/// The current curve's ratio per luma bin, 16 words.
pub(crate) const STATE_RATIOS: u32 = STATE_QUARTER_MEDIAN + 1;
/// The `f32` words of the stream state.
pub(crate) const STATE_WORDS: u32 = STATE_RATIOS + NOISE_CURVE_BINS as u32;

/// How a stream folds its readings.
///
/// The default folds with `sigma_scale` 1.0, zero coefficients and no offset tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FoldSettings {
    pub sigma_scale: f32,
    pub consumer: NoiseConsumer,
    pub coefficients: DerivedCoefficients,
    /// The search radius of the NLM offset tables the fold writes, `None` for no tables.
    pub offset_tables: Option<u32>,
}

impl Default for FoldSettings {
    fn default() -> Self {
        Self {
            sigma_scale: 1.0,
            consumer: NoiseConsumer::default(),
            coefficients: DerivedCoefficients::default(),
            offset_tables: None,
        }
    }
}

/// The NLM noise-floor tables of one search window, one entry per candidate.
pub(super) struct OffsetTables {
    pub(super) radius: u32,
    /// The main pass's noise offset scaled by the spatial offset factor at the folded rho.
    pub(super) spatial: Handle,
    /// The input noise offset with a zero centre, which the NLM pilot reads.
    pub(super) pilot: Handle,
}

impl OffsetTables {
    /// The tables of a `radius` window built on the host from `derived` at a rho of 0.
    pub(super) fn from_host(client: &Client, radius: u32, derived: DerivedValues) -> Self {
        let spatial_entries = build_spatial_offset_lut(radius, 0.0, derived.noise_offset);
        let spatial_bytes = f32::as_bytes(&spatial_entries);
        let pilot_entries = build_spatial_offset_lut(radius, 0.0, derived.input_noise_offset);
        let pilot_bytes = f32::as_bytes(&pilot_entries);

        Self {
            radius,
            spatial: client.create_from_slice(spatial_bytes),
            pilot: client.create_from_slice(pilot_bytes),
        }
    }
}

/// The settings a noise state is built with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct NoiseSettings {
    pub reading: ReadingSettings,
    pub fold: FoldSettings,
}

/// The fold settings buffer's words for `settings` on a plane of `shape`.
pub(super) fn fold_words(shape: NoiseShape, settings: NoiseSettings) -> [f32; FOLD_WORDS as usize] {
    let fold = settings.fold;
    let coefficients = fold.coefficients;
    let mut words = [0.0f32; FOLD_WORDS as usize];

    let interior = ((shape.width - 2) as f32) * ((shape.height - 2) as f32);
    let immerkaer_divisor = 6.0 * interior;

    words[FOLD_SIGMA_SCALE as usize] = fold.sigma_scale;
    words[FOLD_CURVE_APPLIES as usize] = flag(fold.consumer.curve_applies);
    words[FOLD_WEIGHT_G_MAX as usize] = fold.consumer.weight_g_max;
    words[FOLD_WEIGHT_FLOOR as usize] = fold.consumer.weight_floor;
    words[FOLD_H2_AUTO as usize] = flag(coefficients.h2_per_inverse_variance.is_some());
    words[FOLD_H2_PER_INVERSE_VARIANCE as usize] = coefficients.h2_per_inverse_variance.unwrap_or(0.0);
    words[FOLD_H2_FIXED as usize] = coefficients.h2_fixed;
    words[FOLD_PILOT_H2_FACTOR as usize] = coefficients.pilot_h2_factor;
    words[FOLD_OFFSET_PER_VARIANCE as usize] = coefficients.offset_per_variance;
    words[FOLD_MAIN_OFFSET as usize] = flag(coefficients.main_offset);
    words[FOLD_SAD_PER_SIGMA as usize] = coefficients.sad_per_sigma;
    words[FOLD_CONFIDENCE_SAD_PER_SIGMA as usize] = coefficients.confidence_sad_per_sigma;
    words[FOLD_IMMERKAER_FACTOR as usize] = immerkaer_sigma_scale() / immerkaer_divisor;

    for (index, segment) in CORRELATION_FACTOR_TABLE.windows(2).enumerate() {
        let width = segment[1].0 - segment[0].0;
        words[FOLD_SEGMENT_RECIPROCALS as usize + index] = 1.0 / width;
    }

    words
}

fn flag(on: bool) -> f32 {
    if on { 1.0 } else { 0.0 }
}

impl NoiseState {
    /// Queues one fold of the recorded frames `bounds` names into the stream state, the derived
    /// values, the current curve and the offset tables, in one launch.
    ///
    /// A state that pools colour cells also sums the same frames' cells, in one more launch. A
    /// state that holds fixed values queues nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when `bounds` reaches past the frames recorded since the last reset.
    pub(crate) fn queue_window_fold(&self, bounds: WindowBounds) -> Result<(), anyhow::Error> {
        let Some(measurement) = self.measurement.as_ref() else {
            return Ok(());
        };

        self.check_recorded(bounds)?;
        self.fold_history(&measurement.history, bounds)?;
        self.queue_colour_window(bounds);

        Ok(())
    }

    /// Queues one fold of the history frames `bounds` names into the stream state, the derived
    /// values, the current curve and the offset tables, in one launch.
    ///
    /// A state that holds fixed values queues nothing.
    pub(super) fn fold_history(
        &self,
        history: &ReadingHistory,
        bounds: WindowBounds,
    ) -> Result<(), anyhow::Error> {
        let Some(measurement) = self.measurement.as_ref() else {
            return Ok(());
        };

        let frames = bounds.len();
        ensure!(
            frames <= NOISE_WINDOW_LEN,
            "a window of {frames} frames is past the {NOISE_WINDOW_LEN} a fold reads"
        );

        let capacity = history.capacity();
        ensure!(
            frames <= capacity,
            "a window of {frames} frames is past the {capacity} history slots"
        );

        let records_len = (capacity * READING_WORDS) as usize;
        let results_len = capacity as usize * RESULTS_PER_SLOT;
        let records = history.records();
        let results = history.results();
        let table_args = self.table_args();

        unsafe {
            noise_window_fold::launch_unchecked(
                &self.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(NOISE_BLOCK_1D),
                bind(records, records_len),
                bind(results, results_len),
                bind(&measurement.fold_settings, FOLD_WORDS as usize),
                bind(&measurement.fold_state, STATE_WORDS as usize),
                bind(&self.derived, DERIVED_WORDS as usize),
                bind(&self.curve_ratios, NOISE_CURVE_BINS),
                table_args.spatial,
                table_args.pilot,
                bounds.first(),
                bounds.last(),
                capacity,
                self.shape.channels,
                table_args.radius,
            );
        }

        Ok(())
    }

    /// Starts a new stream, so the next fold reads only frames recorded after this call.
    pub(crate) fn reset_stream(&self) {
        let Some(measurement) = self.measurement.as_ref() else {
            return;
        };

        measurement.recorded.store(0, Ordering::Relaxed);
    }

    /// The offset tables and their search radius.
    ///
    /// Without tables the kernel never touches them, so the derived buffer stands in.
    fn table_args(&self) -> TableArgs {
        let Some(tables) = self.offset_tables.as_ref() else {
            return TableArgs {
                spatial: bind(&self.derived, DERIVED_WORDS as usize),
                pilot: bind(&self.derived, DERIVED_WORDS as usize),
                radius: 0,
            };
        };

        let table_len = spatial_offset_lut_len(tables.radius);

        TableArgs {
            spatial: bind(&tables.spatial, table_len),
            pilot: bind(&tables.pilot, table_len),
            radius: tables.radius,
        }
    }
}

/// The fold's offset table bindings and their search radius, 0 without tables.
struct TableArgs {
    spatial: BufferArg,
    pilot: BufferArg,
    radius: u32,
}
