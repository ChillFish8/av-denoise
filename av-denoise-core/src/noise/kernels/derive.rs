use cubecl::prelude::*;

use crate::noise::correlation::CORRELATION_FACTOR_TABLE;
use crate::noise::derived::{
    DERIVED_CONFIDENCE_SAD_FLOOR,
    DERIVED_CURVE_VALID,
    DERIVED_H2_INV_NORM,
    DERIVED_INPUT_NOISE_OFFSET,
    DERIVED_NOISE_OFFSET,
    DERIVED_PILOT_H2_INV_NORM,
    DERIVED_RHO,
    DERIVED_SAD_FLOOR,
    DERIVED_WEIGHT_SCALE,
};
use crate::noise::fold::{
    FOLD_CONFIDENCE_SAD_PER_SIGMA,
    FOLD_H2_AUTO,
    FOLD_H2_FIXED,
    FOLD_H2_PER_INVERSE_VARIANCE,
    FOLD_MAIN_OFFSET,
    FOLD_OFFSET_PER_VARIANCE,
    FOLD_PILOT_H2_FACTOR,
    FOLD_SAD_PER_SIGMA,
    FOLD_SEGMENT_RECIPROCALS,
    FOLD_WEIGHT_FLOOR,
    FOLD_WEIGHT_G_MAX,
};

const _: () = assert!(
    CORRELATION_FACTOR_TABLE.len() == 4,
    "the fold interpolates exactly three correlation segments"
);

const RHO_0: f32 = CORRELATION_FACTOR_TABLE[0].0;
const RHO_1: f32 = CORRELATION_FACTOR_TABLE[1].0;
const RHO_2: f32 = CORRELATION_FACTOR_TABLE[2].0;
const RHO_3: f32 = CORRELATION_FACTOR_TABLE[3].0;
const FACTOR_0: f32 = CORRELATION_FACTOR_TABLE[0].1;
const FACTOR_1: f32 = CORRELATION_FACTOR_TABLE[1].1;
const FACTOR_2: f32 = CORRELATION_FACTOR_TABLE[2].1;
const FACTOR_3: f32 = CORRELATION_FACTOR_TABLE[3].1;
const FACTOR_SPAN_0: f32 = FACTOR_1 - FACTOR_0;
const FACTOR_SPAN_1: f32 = FACTOR_2 - FACTOR_1;
const FACTOR_SPAN_2: f32 = FACTOR_3 - FACTOR_2;
/// The shared words thread 0 hands the offset tables, the noise offset, the input noise offset
/// and the rho.
pub(super) const TABLE_INPUTS: u32 = 3;

/// The factor that scales a temporal sigma up for grain correlation `rho`.
///
/// It interpolates the correlation table linearly, with `rho` clamped to the table's range. Each
/// segment's fraction multiplies by the host's reciprocal of the segment width.
#[cube]
pub(super) fn correlation_factor(rho: f32, fold_settings: &[f32]) -> f32 {
    let x = f32::clamp(rho, RHO_0, RHO_3);
    let first_reciprocal = fold_settings[FOLD_SEGMENT_RECIPROCALS as usize];
    let second_reciprocal = fold_settings[(FOLD_SEGMENT_RECIPROCALS + 1) as usize];
    let third_reciprocal = fold_settings[(FOLD_SEGMENT_RECIPROCALS + 2) as usize];

    let first_fraction = (x - RHO_0) * first_reciprocal;
    let second_fraction = (x - RHO_1) * second_reciprocal;
    let third_fraction = (x - RHO_2) * third_reciprocal;
    let first = FACTOR_0 + first_fraction * FACTOR_SPAN_0;
    let second = FACTOR_1 + second_fraction * FACTOR_SPAN_1;
    let third = FACTOR_2 + third_fraction * FACTOR_SPAN_2;

    let later = select(x <= RHO_2, second, third);
    select(x <= RHO_1, first, later)
}

/// The share of the noise floor that applies at the candidate `dx`, `dy` under grain correlation
/// `rho`.
///
/// The centre gives 0. Every other candidate gives 1 with `rho` at or below 0, and otherwise
/// `1 - rho^distance`.
#[cube]
fn spatial_offset_factor(dx: i32, dy: i32, rho: f32) -> f32 {
    let squared = dx * dx + dy * dy;
    let distance = f32::sqrt(f32::cast_from(squared));
    let log_rho = f32::ln(rho);
    let shared = f32::exp(distance * log_rho);
    let correlated = 1.0f32 - shared;
    let off_centre = select(rho <= 0.0f32, 1.0f32, correlated);

    select(squared == 0, 0.0f32, off_centre)
}

/// Writes both offset tables of a `radius` search window across cube 0's threads.
///
/// Thread 0 of cube 0 puts the noise offset, the input noise offset and the rho in
/// `table_inputs` first. Only cube 0 holds them, and every thread of a cube takes the same branch,
/// so the barrier stays uniform within each cube. Entry `index` holds the candidate at
/// `dx = index % side - radius` and `dy = index / side - radius`, with `side = 2 * radius + 1`.
/// The spatial table scales the noise offset by the factor at the rho, and the pilot table scales
/// the input noise offset by the factor at a rho of 0.
#[cube]
pub(super) fn write_offset_tables(
    table_inputs: &Shared<[f32]>,
    spatial_table: &mut [f32],
    pilot_table: &mut [f32],
    #[comptime] radius: u32,
) {
    if CUBE_POS_X == 0 {
        sync_cube();

        let noise_offset = table_inputs[0];
        let input_offset = table_inputs[1];
        let rho = table_inputs[2];
        let side = comptime!(2 * radius + 1);
        let entries = comptime!(side * side);
        let centre = comptime!(radius as i32);
        let mut index = UNIT_POS_X;

        while index < entries {
            let column = i32::cast_from(index % side);
            let row = i32::cast_from(index / side);
            let dx = column - centre;
            let dy = row - centre;
            let spatial_factor = spatial_offset_factor(dx, dy, rho);
            let pilot_factor = spatial_offset_factor(dx, dy, 0.0f32);

            spatial_table[index as usize] = noise_offset * spatial_factor;
            pilot_table[index as usize] = input_offset * pilot_factor;
            index += CUBE_DIM_X;
        }
    }
}

/// The folded values every downstream derived value is computed from.
#[derive(CubeType, Clone, Copy)]
pub(super) struct FoldedNoise {
    /// Channel 0's derived sigma.
    pub(super) sigma0: f32,
    /// Channel 0's median chain.
    pub(super) median0: f32,
    /// The sum of the squared median chain over the active channels.
    pub(super) median_sum: f32,
    /// The sum of the squared low chain over the active channels.
    pub(super) low_sum: f32,
    /// Whether the curve exists and the consumer applies it.
    pub(super) curve_valid: bool,
    /// The grain correlation, 0 while unset.
    pub(super) rho: f32,
}

/// Writes every derived value past the sigmas from `folded`.
///
/// With a `table_radius` above 0 it also puts the noise offsets and the rho in `table_inputs`
/// for [write_offset_tables].
#[cube]
pub(super) fn write_derived(
    folded: &FoldedNoise,
    fold_settings: &[f32],
    derived: &mut [f32],
    table_inputs: &mut Shared<[f32]>,
    #[comptime] channels: u32,
    #[comptime] table_radius: u32,
) {
    let sigma0 = folded.sigma0;
    let g_max = fold_settings[FOLD_WEIGHT_G_MAX as usize];
    let weight_floor = fold_settings[FOLD_WEIGHT_FLOOR as usize];
    let norm = sigma0 * sigma0 * g_max * g_max;
    let usable = !norm.is_nan() && !norm.is_inf() && norm > weight_floor;

    let median0 = folded.median0;
    let h2_auto = fold_settings[FOLD_H2_AUTO as usize] != 0.0f32;
    let per_inverse_variance = fold_settings[FOLD_H2_PER_INVERSE_VARIANCE as usize];
    let channel_count = comptime!(channels as f32);
    let h2_measured = per_inverse_variance * channel_count / folded.median_sum;
    let h2 = select(h2_auto, h2_measured, fold_settings[FOLD_H2_FIXED as usize]);
    let input_offset = fold_settings[FOLD_OFFSET_PER_VARIANCE as usize] * folded.low_sum;
    let main_offset = fold_settings[FOLD_MAIN_OFFSET as usize] != 0.0f32;
    let noise_offset = select(main_offset, input_offset, 0.0f32);
    let rho = folded.rho;

    derived[DERIVED_WEIGHT_SCALE as usize] = select(usable, norm, weight_floor);
    derived[DERIVED_CURVE_VALID as usize] = select(folded.curve_valid, 1.0f32, 0.0f32);
    derived[DERIVED_SAD_FLOOR as usize] = fold_settings[FOLD_SAD_PER_SIGMA as usize] * median0;
    derived[DERIVED_CONFIDENCE_SAD_FLOOR as usize] =
        fold_settings[FOLD_CONFIDENCE_SAD_PER_SIGMA as usize] * median0;
    derived[DERIVED_H2_INV_NORM as usize] = h2;
    derived[DERIVED_PILOT_H2_INV_NORM as usize] = h2 * fold_settings[FOLD_PILOT_H2_FACTOR as usize];
    derived[DERIVED_NOISE_OFFSET as usize] = noise_offset;
    derived[DERIVED_INPUT_NOISE_OFFSET as usize] = input_offset;
    derived[DERIVED_RHO as usize] = rho;

    if comptime!(table_radius > 0) {
        table_inputs[0] = noise_offset;
        table_inputs[1] = input_offset;
        table_inputs[2] = rho;
    }
}
