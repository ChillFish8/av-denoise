use anyhow::ensure;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::correlation::{CORRELATION_FACTOR_TABLE, build_spatial_offset_lut, spatial_offset_lut_len};
use super::derived::{DERIVED_WORDS, DerivedCoefficients, DerivedValues, NoiseConsumer};
use super::kernels::{noise_fold, noise_reset};
use super::reading::READING_WORDS;
use super::spatial::{RESULTS_PER_SLOT, immerkaer_sigma_scale};
use super::state::{NoiseShape, NoiseState, ReadingSettings, StrengthMapKind, quarter_count};
use super::{NOISE_BLOCK_1D, NOISE_CURVE_BINS, STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};
use crate::engine::MAX_GRID_1D;
use crate::tune::gpu::bind;

/// The factor applied after combining and before smoothing.
pub(crate) const FOLD_SIGMA_SCALE: u32 = 0;
/// 1.0 when a centre with no sample borrows the first sample ahead.
pub(crate) const FOLD_BORROW_AHEAD: u32 = 1;
/// 1.0 when the consumer applies the curve.
pub(crate) const FOLD_CURVE_APPLIES: u32 = 2;
pub(crate) const FOLD_WEIGHT_G_MAX: u32 = 3;
pub(crate) const FOLD_WEIGHT_FLOOR: u32 = 4;
/// 1.0 when `h2` follows the median chain, 0.0 for a fixed strength.
pub(crate) const FOLD_H2_AUTO: u32 = 5;
pub(crate) const FOLD_H2_PER_INVERSE_VARIANCE: u32 = 6;
pub(crate) const FOLD_H2_FIXED: u32 = 7;
pub(crate) const FOLD_PILOT_H2_FACTOR: u32 = 8;
pub(crate) const FOLD_OFFSET_PER_VARIANCE: u32 = 9;
/// 1.0 when the main pass subtracts the offset.
pub(crate) const FOLD_MAIN_OFFSET: u32 = 10;
pub(crate) const FOLD_SAD_PER_SIGMA: u32 = 11;
pub(crate) const FOLD_CONFIDENCE_SAD_PER_SIGMA: u32 = 12;
/// The factor that turns one channel's Immerkær abs sum into its sigma.
pub(crate) const FOLD_IMMERKAER_FACTOR: u32 = 13;
/// The strength map the reading builds, as the `STRENGTH_MAP_*` mode it can reach.
pub(crate) const FOLD_MAP_KIND: u32 = 14;
/// The reciprocal width of each correlation-table segment, 3 words.
pub(crate) const FOLD_SEGMENT_RECIPROCALS: u32 = 15;
/// The `f32` words of the fold settings buffer.
pub(crate) const FOLD_WORDS: u32 = 18;

/// The median chain, which drives the strength.
pub(crate) const STATE_MEDIAN: u32 = 0;
/// The low chain, which drives the noise offsets.
pub(crate) const STATE_LOW: u32 = 4;
/// The low chain without the correlation boost.
pub(crate) const STATE_LOW_UNBOOSTED: u32 = 8;
/// The temporal median alone, updated only by a fold with a sample.
pub(crate) const STATE_TEMPORAL_ONLY: u32 = 12;
/// 1.0 once a sample has set the smoothed correlation.
pub(crate) const STATE_RHO_VALID: u32 = 16;
pub(crate) const STATE_RHO: u32 = 17;
/// 1.0 while the state holds a curve.
pub(crate) const STATE_CURVE_VALID: u32 = 18;
pub(crate) const STATE_QUARTER_MEDIAN: u32 = 19;
/// 1.0 while the state holds quarter classes, which it does exactly when it holds a curve.
pub(crate) const STATE_CLASSES_PRESENT: u32 = 20;
/// The current curve's ratio per luma bin, 16 words.
pub(crate) const STATE_RATIOS: u32 = 21;
/// The `f32` words of the stream state.
pub(crate) const STATE_WORDS: u32 = STATE_RATIOS + NOISE_CURVE_BINS as u32;

/// How a stream folds its readings.
///
/// The default folds with `sigma_scale` 1.0, no borrowing, zero coefficients and no offset tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FoldSettings {
    pub sigma_scale: f32,
    /// Whether a centre with no sample borrows the first sample ahead, as shifted edges need.
    pub borrow_ahead: bool,
    pub consumer: NoiseConsumer,
    pub coefficients: DerivedCoefficients,
    /// The search radius of the NLM offset tables the fold writes, `None` for no tables.
    pub offset_tables: Option<u32>,
}

impl Default for FoldSettings {
    fn default() -> Self {
        Self {
            sigma_scale: 1.0,
            borrow_ahead: false,
            consumer: NoiseConsumer::default(),
            coefficients: DerivedCoefficients::default(),
            offset_tables: None,
        }
    }
}

/// The NLM noise-floor tables of one search window, one entry per candidate.
pub(super) struct OffsetTables {
    pub(super) radius: u32,
    /// The main pass's noise offset scaled by the spatial offset factor at the smoothed rho.
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

/// Which reading a fold takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fold {
    /// A stream's first frame, folded from its Immerkær reading alone, the curve untouched.
    Seed { slot: u32 },
    /// The reading at logical position `centre` of a ring whose oldest frame sits in `ring_start`.
    Centre { ring_start: u32, centre: u32 },
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
    let map_kind = match settings.reading.map {
        StrengthMapKind::Off => STRENGTH_MAP_OFF,
        StrengthMapKind::Luma(_) => STRENGTH_MAP_LUMA,
        StrengthMapKind::Chroma { .. } => STRENGTH_MAP_ALL,
    };

    words[FOLD_SIGMA_SCALE as usize] = fold.sigma_scale;
    words[FOLD_BORROW_AHEAD as usize] = flag(fold.borrow_ahead);
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
    words[FOLD_MAP_KIND as usize] = map_kind as f32;

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
    /// Queues one fold of the readings `fold` names into the stream state, the derived values,
    /// the current curve, the current map and the offset tables, in one launch.
    ///
    /// With a map the launch spans one cube per 256 map quarters, so the copy runs across the
    /// grid. A state that holds fixed values queues nothing.
    pub(crate) fn queue_fold(&self, fold: Fold) -> Result<(), anyhow::Error> {
        let Some(measurement) = self.measurement.as_ref() else {
            return Ok(());
        };

        let shape = self.shape;
        let frames = shape.frames;
        let (ring_start, centre, seed) = match fold {
            Fold::Seed { slot } => (slot, 0, true),
            Fold::Centre { ring_start, centre } => (ring_start, centre, false),
        };

        ensure!(
            ring_start < frames,
            "ring start {ring_start} is past the {frames} slots"
        );
        ensure!(centre < frames, "centre {centre} is past the {frames} slots");

        let readings_len = (frames * READING_WORDS) as usize;
        let results_len = frames as usize * RESULTS_PER_SLOT;
        let map_len = self.current_map_len();
        let map_args = self.map_args(map_len);
        let map_cubes = map_len.div_ceil(NOISE_BLOCK_1D);
        let grid = map_cubes.clamp(1, MAX_GRID_1D);
        let table_args = self.table_args();

        unsafe {
            noise_fold::launch_unchecked(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(NOISE_BLOCK_1D),
                bind(&measurement.readings, readings_len),
                bind(&measurement.results, results_len),
                bind(&measurement.fold_settings, FOLD_WORDS as usize),
                bind(&measurement.fold_state, STATE_WORDS as usize),
                bind(&self.derived, DERIVED_WORDS as usize),
                bind(&self.curve_ratios, NOISE_CURVE_BINS),
                map_args.slot_maps,
                map_args.current_map,
                table_args.spatial,
                table_args.pilot,
                ring_start,
                centre,
                u32::from(seed),
                frames,
                shape.channels,
                map_len,
                table_args.radius,
            );
        }

        Ok(())
    }

    /// Queues the clear of every validity flag of the stream state, in one launch.
    ///
    /// A state that holds fixed values queues nothing.
    pub(crate) fn queue_reset(&self) {
        let Some(measurement) = self.measurement.as_ref() else {
            return;
        };

        unsafe {
            noise_reset::launch_unchecked(
                &self.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                bind(&measurement.fold_state, STATE_WORDS as usize),
            );
        }
    }

    /// The map quarters the fold copies, 0 without a strength map.
    fn current_map_len(&self) -> u32 {
        match self.measurement().classes.as_ref() {
            Some(_) => quarter_count(self.shape),
            None => 0,
        }
    }

    /// The slot maps and the current map, bound over `map_len` quarters per map.
    ///
    /// Without a map the kernel never touches them, so the derived buffer stands in.
    fn map_args(&self, map_len: u32) -> MapArgs {
        let Some(slot_maps) = self.slot_maps() else {
            return MapArgs {
                slot_maps: bind(&self.derived, DERIVED_WORDS as usize),
                current_map: bind(&self.derived, DERIVED_WORDS as usize),
            };
        };

        let slot_maps_len = map_len as usize * self.shape.frames as usize;

        MapArgs {
            slot_maps: bind(slot_maps, slot_maps_len),
            current_map: bind(&self.current_map, map_len as usize),
        }
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

/// The fold's map bindings.
struct MapArgs {
    slot_maps: BufferArg,
    current_map: BufferArg,
}

/// The fold's offset table bindings and their search radius, 0 without tables.
struct TableArgs {
    spatial: BufferArg,
    pilot: BufferArg,
    radius: u32,
}
