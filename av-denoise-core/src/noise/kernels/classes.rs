use cubecl::prelude::*;

use super::sample::static_quarter_sigma;
use crate::collab::geometry::strength_map_dims;
use crate::noise::curve::{CLIP_HIGH, CLIP_LOW, NOISE_CURVE_BINS};
use crate::noise::line_ring::{FULL_QUARTER_WINDOWS, LINE_CONTRAST_FLOOR, LINE_SIGMA_FACTOR};
use crate::noise::reading::{READING_CURVE_VALID, READING_QUARTER_MEDIAN, READING_RATIOS, READING_WORDS};
use crate::noise::state::{CLASS_FLAT, CLASS_LINE, CLASS_PRESENT, QUARTER_TENSOR_WORDS};
use crate::noise::strength_map::{MOTION_FACTOR, QUARTER_FLAT_FACTOR, SHADOW_HIGH, SHADOW_LOW};
use crate::noise::temporal::{
    QUARTER_FLATNESS,
    QUARTER_LUMA_MAX,
    QUARTER_LUMA_MIN,
    QUARTER_LUMA_SUM,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    temporal_stats_record_len,
};

const CURVE_BINS: f32 = NOISE_CURVE_BINS as f32;
const LAST_BIN: f32 = (NOISE_CURVE_BINS - 1) as f32;
const LAST_LOWER_BIN: u32 = (NOISE_CURVE_BINS - 2) as u32;
const SHADOW_SPAN: f32 = SHADOW_HIGH - SHADOW_LOW;

/// The noise sigma the slot's curve predicts at `luma`.
///
/// The ratio is linear between bin centres and is scaled by the curve's quarter median. `base` is the slot's first reading word.
#[cube]
fn curve_sigma_at(readings: &[f32], base: u32, luma: f32) -> f32 {
    let position = f32::clamp(luma * CURVE_BINS - 0.5f32, 0.0f32, LAST_BIN);
    let truncated = u32::cast_from(position);
    let lower = u32::min(truncated, LAST_LOWER_BIN);
    let fraction = position - f32::cast_from(lower);
    let ratios = base + READING_RATIOS;
    let lower_ratio = readings[(ratios + lower) as usize];
    let upper_ratio = readings[(ratios + lower + 1) as usize];
    let ratio = lower_ratio + (upper_ratio - lower_ratio) * fraction;
    let quarter_median = readings[(base + READING_QUARTER_MEDIAN) as usize];

    ratio * quarter_median
}

/// Whether a full quarter holds a strong line.
///
/// Its RMS gradient must reach [LINE_SIGMA_FACTOR](crate::noise::line_ring::LINE_SIGMA_FACTOR)
/// times `sigma`, the curve's sigma at its luma, and its luma range
/// [LINE_CONTRAST_FLOOR](crate::noise::line_ring::LINE_CONTRAST_FLOOR) codes.
#[cube]
fn is_line(luma_range: f32, tensor_trace: f32, sigma: f32) -> bool {
    let mean_square_gradient = f32::max(tensor_trace / FULL_QUARTER_WINDOWS, 0.0f32);
    let rms_gradient = f32::sqrt(mean_square_gradient);
    let strong = rms_gradient >= LINE_SIGMA_FACTOR * sigma;
    let contrasty = luma_range * 255.0f32 >= LINE_CONTRAST_FLOOR;

    strong && contrasty
}

/// Classes every map quarter of a slot from its records and the slot's curve.
///
/// Thread `i` writes quarter `i` of the row-major map. A quarter with pixels is present. It is flat
/// when it passes the curve's static gates, reads under
/// [MOTION_FACTOR](crate::noise::strength_map::MOTION_FACTOR) times the curve at its luma, is
/// unclipped, and its flatness is under
/// [QUARTER_FLAT_FACTOR](crate::noise::strength_map::QUARTER_FLAT_FACTOR) of its own variance. A
/// full 8x8 quarter is a line when its gradient and luma range are strong. Every quarter also gets
/// its mean luma and its structure tensor. A quarter past the frame edge has no flags.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, the slot or a comptime shape"
)]
pub fn noise_classify(
    records: &[f32],
    readings: &[f32],
    classes: &mut [u32],
    lumas: &mut [f32],
    tensors: &mut [f32],
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
) {
    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let cols = comptime!(strength_map_dims(width, height).0);
    let rows = comptime!(strength_map_dims(width, height).1);
    let quarters = comptime!(cols * rows);
    let record_len = comptime!(temporal_stats_record_len(stored_ch));
    let quarters_base = comptime!(2 * stored_ch + TEMPORAL_QUARTER_BASE);

    let index = ABSOLUTE_POS_X;

    if index < quarters {
        let col = index % cols;
        let row = index / cols;
        let block_x = col / 2;
        let block_y = row / 2;
        let block = block_y * blocks_x + block_x;
        let quarter = (row % 2) * 2 + col % 2;
        let block_width = u32::min(TEMPORAL_NOISE_BLOCK, width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = u32::min(TEMPORAL_NOISE_BLOCK, height - block_y * TEMPORAL_NOISE_BLOCK);

        let offset_x = (col % 2) * TEMPORAL_QUARTER_SIZE;
        let offset_y = (row % 2) * TEMPORAL_QUARTER_SIZE;
        let inside_x = block_width - u32::min(block_width, offset_x);
        let inside_y = block_height - u32::min(block_height, offset_y);
        let quarter_width = u32::min(inside_x, TEMPORAL_QUARTER_SIZE);
        let quarter_height = u32::min(inside_y, TEMPORAL_QUARTER_SIZE);
        let pixels = f32::cast_from(quarter_width * quarter_height);
        let start = block * record_len + quarters_base + quarter * TEMPORAL_QUARTER_FIELDS;

        let (passes, sigma) = static_quarter_sigma(records, start, pixels);

        let present = pixels > 0.0f32;
        let divisor = select(present, pixels, 1.0f32);
        let luma_sum = records[(start + QUARTER_LUMA_SUM) as usize];
        let luma = luma_sum / divisor;
        let base = slot * READING_WORDS;
        let expected = curve_sigma_at(readings, base, luma);

        let luma_min = records[(start + QUARTER_LUMA_MIN) as usize];
        let luma_max = records[(start + QUARTER_LUMA_MAX) as usize];
        let flatness = records[(start + QUARTER_FLATNESS) as usize];
        let not_motion = sigma < MOTION_FACTOR * expected;
        let unclipped = luma_min >= CLIP_LOW && luma_max <= CLIP_HIGH;
        let smooth = flatness < QUARTER_FLAT_FACTOR * sigma * sigma;
        let flat = passes && not_motion && unclipped && smooth;

        let tensor_xx = records[(start + QUARTER_TENSOR_XX) as usize];
        let tensor_yy = records[(start + QUARTER_TENSOR_YY) as usize];
        let tensor_xy = records[(start + QUARTER_TENSOR_XY) as usize];
        let full = quarter_width == TEMPORAL_QUARTER_SIZE && quarter_height == TEMPORAL_QUARTER_SIZE;
        let luma_range = luma_max - luma_min;
        let trace = tensor_xx + tensor_yy;
        let line = full && is_line(luma_range, trace, expected);

        let present_bit = select(present, CLASS_PRESENT, 0u32);
        let flat_bit = select(flat, CLASS_FLAT, 0u32);
        let line_bit = select(line, CLASS_LINE, 0u32);
        let tensor_start = index * QUARTER_TENSOR_WORDS;

        classes[index as usize] = present_bit | flat_bit | line_bit;
        lumas[index as usize] = luma;
        tensors[tensor_start as usize] = tensor_xx;
        tensors[(tensor_start + 1) as usize] = tensor_yy;
        tensors[(tensor_start + 2) as usize] = tensor_xy;
    }
}

/// Whether any line quarter lies within `radius` quarters of `col`, `row`, clamped at the grid.
#[cube]
fn in_line_ring(
    classes: &[u32],
    col: u32,
    row: u32,
    #[comptime] radius: u32,
    #[comptime] cols: u32,
    #[comptime] rows: u32,
) -> bool {
    let mut ring = false;

    if comptime!(radius > 0) {
        let row_start = row - u32::min(row, radius);
        let row_end = u32::min(row + radius, comptime!(rows - 1)) + 1;
        let col_start = col - u32::min(col, radius);
        let col_end = u32::min(col + radius, comptime!(cols - 1)) + 1;

        for neighbour_row in row_start..row_end {
            for neighbour_col in col_start..col_end {
                let flags = classes[(neighbour_row * cols + neighbour_col) as usize];
                let line = (flags & CLASS_LINE) != 0u32;
                ring = ring || line;
            }
        }
    }

    ring
}

/// The coherence of the pooled tensor of the present quarters in the 3x3 around `col`, `row`.
///
/// The neighbourhood is clamped at the grid and summed in row-major order. A pooled tensor with no
/// gradient energy reads 0.
#[cube]
fn pooled_coherence(
    classes: &[u32],
    tensors: &[f32],
    col: u32,
    row: u32,
    #[comptime] cols: u32,
    #[comptime] rows: u32,
) -> f32 {
    let row_start = row - u32::min(row, 1u32);
    let row_end = u32::min(row + 1, comptime!(rows - 1)) + 1;
    let col_start = col - u32::min(col, 1u32);
    let col_end = u32::min(col + 1, comptime!(cols - 1)) + 1;

    let mut pooled_xx = 0.0f32;
    let mut pooled_yy = 0.0f32;
    let mut pooled_xy = 0.0f32;

    for neighbour_row in row_start..row_end {
        for neighbour_col in col_start..col_end {
            let neighbour = neighbour_row * cols + neighbour_col;
            let flags = classes[neighbour as usize];
            let tensor_start = neighbour * QUARTER_TENSOR_WORDS;

            if (flags & CLASS_PRESENT) != 0u32 {
                pooled_xx += tensors[tensor_start as usize];
                pooled_yy += tensors[(tensor_start + 1) as usize];
                pooled_xy += tensors[(tensor_start + 2) as usize];
            }
        }
    }

    let trace = pooled_xx + pooled_yy;
    let difference = pooled_xx - pooled_yy;
    let spread = f32::sqrt(difference * difference + 4.0f32 * pooled_xy * pooled_xy);
    let has_energy = trace > 0.0f32;
    let divisor = select(has_energy, trace, 1.0f32);
    let coherence = spread / divisor;

    select(has_energy, coherence, 0.0f32)
}

/// The shadow soften at `luma`, in full at and below luma 128 of 255 and fading to 1.0 at 160.
#[cube]
fn shadow_multiplier(luma: f32, shadow_soften: f32) -> f32 {
    let offset = luma - SHADOW_LOW;
    let progress = f32::clamp(offset / SHADOW_SPAN, 0.0f32, 1.0f32);

    shadow_soften + (1.0f32 - shadow_soften) * progress
}

/// Writes every map quarter's threshold multiplier into the strength map of `slot`.
///
/// A quarter lies in the line ring when a line quarter sits within `ring_radius` quarters of it,
/// and a radius of 0 turns the ring off. With a `texture_cut` below 1.0, a flat quarter outside the
/// ring whose pooled 3x3 tensor reads a coherence at or above the cut is not flat. With `luma_map`,
/// flat quarters get `flat_boost`, other quarters in the ring 1.0 and the rest the shadow soften at
/// their luma. Without it, flat quarters get `flat_boost` and the rest 1.0. A quarter past the frame
/// edge, and every quarter of a slot without a curve, gets 1.0.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a per-slot scalar or a comptime shape"
)]
pub fn noise_strength_map(
    readings: &[f32],
    classes: &[u32],
    lumas: &[f32],
    tensors: &[f32],
    slot_maps: &mut [f32],
    slot: u32,
    flat_boost: f32,
    shadow_soften: f32,
    texture_cut: f32,
    #[comptime] luma_map: bool,
    #[comptime] ring_radius: u32,
    #[comptime] cols: u32,
    #[comptime] rows: u32,
) {
    let quarters = comptime!(cols * rows);
    let index = ABSOLUTE_POS_X;

    if index < quarters {
        let col = index % cols;
        let row = index / cols;
        let flags = classes[index as usize];
        let present = (flags & CLASS_PRESENT) != 0u32;
        let flat = (flags & CLASS_FLAT) != 0u32;

        let ring = in_line_ring(classes, col, row, ring_radius, cols, rows);
        let coherence = pooled_coherence(classes, tensors, col, row, cols, rows);
        let veto_on = texture_cut < 1.0f32;
        let vetoed = veto_on && !ring && coherence >= texture_cut;
        let kept_flat = flat && !vetoed;

        let mut multiplier = select(kept_flat, flat_boost, 1.0f32);

        if comptime!(luma_map) {
            let luma = lumas[index as usize];
            let shadow = shadow_multiplier(luma, shadow_soften);
            let textured = select(ring, 1.0f32, shadow);
            let classed = select(kept_flat, flat_boost, textured);

            multiplier = select(present, classed, 1.0f32);
        }

        let base = slot * READING_WORDS;
        let curve_valid = readings[(base + READING_CURVE_VALID) as usize] != 0.0f32;
        let map_start = slot * quarters;

        slot_maps[(map_start + index) as usize] = select(curve_valid, multiplier, 1.0f32);
    }
}
