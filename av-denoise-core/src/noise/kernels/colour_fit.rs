use std::f32::consts::{LN_2, PI, TAU};

use cubecl::prelude::*;

use crate::noise::DERIVED_CURVE_VALID;
use crate::noise::colour::{
    COLOUR_BRIGHTNESS_BINS,
    COLOUR_BRIGHTNESS_STEP,
    COLOUR_CALIBRATION_RECORDS,
    COLOUR_CELL_CB_SUM,
    COLOUR_CELL_COUNT,
    COLOUR_CELL_CR_SUM,
    COLOUR_CELL_HISTOGRAM,
    COLOUR_CELL_LUMA_SUM,
    COLOUR_CELL_WORDS,
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    COLOUR_CHROMA_GAIN_MAX,
    COLOUR_CHROMA_GAIN_MIN,
    COLOUR_FIT_BANDWIDTH,
    COLOUR_FIT_CB,
    COLOUR_FIT_CELL_FIELDS,
    COLOUR_FIT_CR,
    COLOUR_FIT_KERNEL_LANES,
    COLOUR_FIT_KERNEL_THREADS,
    COLOUR_FIT_KNOT_PENALTY,
    COLOUR_FIT_KNOTS,
    COLOUR_FIT_LUMA,
    COLOUR_FIT_LUMA_DEAD_ZONE,
    COLOUR_FIT_MIN_BLOCKS,
    COLOUR_FIT_PLANE_BLOCKS,
    COLOUR_FIT_PLANE_LUMA_HIGH,
    COLOUR_FIT_PLANE_LUMA_LOW,
    COLOUR_FIT_PLANE_POPULATED,
    COLOUR_FIT_PLANE_PRIOR,
    COLOUR_FIT_PLANE_WORDS,
    COLOUR_FIT_PRIOR_SHARE,
    COLOUR_FIT_PULL,
    COLOUR_FIT_QUERIES,
    COLOUR_FIT_RESIDUAL,
    COLOUR_FIT_RIDGE,
    COLOUR_FIT_THREADS,
    COLOUR_FIT_WEIGHT,
    COLOUR_GAIN_BLOCK_RAMP,
    COLOUR_GAIN_NEUTRAL_RAMP,
    COLOUR_HISTOGRAM_BINS,
    COLOUR_HISTOGRAM_LOG_MIN,
    COLOUR_HISTOGRAM_LOG_STEP,
    COLOUR_HUE_BINS,
    COLOUR_KERNEL_CHROMA_STEP,
    COLOUR_KERNEL_LUMA_STEP,
    COLOUR_LEVEL_FIELDS,
    COLOUR_SATURATION_CENTRES,
};

const LAST_KNOT: u32 = COLOUR_FIT_KNOTS - 1;
/// The codes between two knots of the brightness base, which spans 0 to 255.
const KNOT_STEP: f32 = 255.0 / LAST_KNOT as f32;
const HUE_SECTORS: f32 = COLOUR_HUE_BINS as f32;
const SATURATION_LEVELS: u32 = COLOUR_SATURATION_CENTRES.len() as u32;
const LAST_BRIGHTNESS: f32 = (COLOUR_BRIGHTNESS_BINS - 1) as f32;
const LAST_SATURATION: f32 = (SATURATION_LEVELS - 1) as f32;
const SATURATION_CENTRE_0: f32 = COLOUR_SATURATION_CENTRES[0];
const SATURATION_CENTRE_1: f32 = COLOUR_SATURATION_CENTRES[1];
const SATURATION_CENTRE_2: f32 = COLOUR_SATURATION_CENTRES[2];
const SATURATION_CENTRE_3: f32 = COLOUR_SATURATION_CENTRES[3];
const SATURATION_CENTRE_4: f32 = COLOUR_SATURATION_CENTRES[4];
const SATURATION_CENTRE_5: f32 = COLOUR_SATURATION_CENTRES[5];
/// The kernel's exponent factor, `-1 / (2 h²)`.
const KERNEL_EXPONENT: f32 = -0.5 / (COLOUR_FIT_BANDWIDTH * COLOUR_FIT_BANDWIDTH);
// The base fit's sums, in order: the normal equations' diagonal, first off-diagonal and right-hand
// side, then the summed square root of the counts, the populated cells and the kept blocks.
const SUM_DIAGONAL: u32 = 0;
const SUM_OFF_DIAGONAL: u32 = SUM_DIAGONAL + COLOUR_FIT_KNOTS;
const SUM_RIGHT: u32 = SUM_OFF_DIAGONAL + LAST_KNOT;
const SUM_ROOT_COUNT: u32 = SUM_RIGHT + COLOUR_FIT_KNOTS;
const SUM_POPULATED: u32 = SUM_ROOT_COUNT + 1;
const SUM_BLOCKS: u32 = SUM_POPULATED + 1;
const SUM_QUANTITIES: u32 = SUM_BLOCKS + 1;
/// The bin edges of every histogram the gain normaliser reads, which are its breakpoints.
const HISTOGRAM_EDGES: u32 = COLOUR_HISTOGRAM_BINS + 1;
/// The histograms the gain normaliser reads, one per brightness bin.
const NORMALISER_RECORDS: u32 = COLOUR_BRIGHTNESS_BINS;
const NORMALISER_VALUES: u32 = NORMALISER_RECORDS * COLOUR_HISTOGRAM_BINS;
const NORMALISER_POINTS: u32 = NORMALISER_RECORDS * HISTOGRAM_EDGES;

/// The position of `value` between the centres `low` and `high`, as a fraction.
#[cube]
fn between(value: f32, low: f32, high: f32) -> f32 {
    (value - low) / (high - low)
}

/// The fractional index of chroma magnitude `saturation` along the saturation centres.
///
/// Magnitudes outside the centres clamp to the first or last.
#[cube]
fn saturation_position(saturation: f32) -> f32 {
    let step_1 = between(saturation, SATURATION_CENTRE_0, SATURATION_CENTRE_1);
    let step_2 = 1.0f32 + between(saturation, SATURATION_CENTRE_1, SATURATION_CENTRE_2);
    let step_3 = 2.0f32 + between(saturation, SATURATION_CENTRE_2, SATURATION_CENTRE_3);
    let step_4 = 3.0f32 + between(saturation, SATURATION_CENTRE_3, SATURATION_CENTRE_4);
    let step_5 = 4.0f32 + between(saturation, SATURATION_CENTRE_4, SATURATION_CENTRE_5);

    let upper_4 = select(saturation < SATURATION_CENTRE_4, step_4, step_5);
    let upper_3 = select(saturation < SATURATION_CENTRE_3, step_3, upper_4);
    let upper_2 = select(saturation < SATURATION_CENTRE_2, step_2, upper_3);
    let position = select(saturation < SATURATION_CENTRE_1, step_1, upper_2);

    f32::clamp(position, 0.0f32, LAST_SATURATION)
}

/// The table index of brightness centre `brightness`, saturation centre `level` and hue sector
/// `sector`.
///
/// The neutral level holds one value for every hue.
#[cube]
fn table_index(brightness: u32, level: u32, sector: u32) -> u32 {
    let coloured = 1u32 + (u32::max(level, 1u32) - 1u32) * COLOUR_HUE_BINS + sector;
    let ring = select(level == 0u32, 0u32, coloured);

    brightness * COLOUR_CELLS_PER_BRIGHTNESS + ring
}

/// The value of the table that starts at word `base` of `table`, at mean luma `luma` and chroma
/// `cb` and `cr`, centred on 0.
///
/// Every value is on the 8-bit scale. The table holds one value per colour cell, at the cell's
/// centre. The value is interpolated between the 8 nearest centres in brightness, saturation and
/// hue. Hue wraps around, and colours past the outer centres clamp to them.
#[cube]
pub(crate) fn colour_table_value(table: &[f32], base: u32, luma: f32, cb: f32, cr: f32) -> f32 {
    let raw_brightness = luma / COLOUR_BRIGHTNESS_STEP - 0.5f32;
    let brightness_position = f32::clamp(raw_brightness, 0.0f32, LAST_BRIGHTNESS);
    let brightness_floor = f32::floor(brightness_position);
    let brightness_low = u32::min(u32::cast_from(brightness_floor), COLOUR_BRIGHTNESS_BINS - 2u32);
    let brightness_fraction = brightness_position - f32::cast_from(brightness_low);

    let saturation = f32::sqrt(cb * cb + cr * cr);
    let level_position = saturation_position(saturation);
    let level_floor = f32::floor(level_position);
    let level_low = u32::min(u32::cast_from(level_floor), SATURATION_LEVELS - 2u32);
    let level_fraction = level_position - f32::cast_from(level_low);

    let hue = f32::atan2(cr, cb);
    let hue_position = (hue + PI) / TAU * HUE_SECTORS - 0.5f32;
    let hue_floor = f32::floor(hue_position);
    let hue_fraction = hue_position - hue_floor;
    let wrapped_floor = hue_floor + HUE_SECTORS;
    let sector_low = u32::cast_from(wrapped_floor) % COLOUR_HUE_BINS;

    let mut value = 0.0f32;
    #[unroll]
    for corner in 0..8u32 {
        let brightness_step = comptime!(corner / 4u32);
        let level_step = comptime!((corner / 2u32) % 2u32);
        let sector_step = comptime!(corner % 2u32);

        let brightness_weight = corner_weight(brightness_fraction, comptime!(brightness_step == 1));
        let level_weight = corner_weight(level_fraction, comptime!(level_step == 1));
        let sector_weight = corner_weight(hue_fraction, comptime!(sector_step == 1));
        let sector = (sector_low + sector_step) % COLOUR_HUE_BINS;
        let brightness = brightness_low + brightness_step;
        let level = level_low + level_step;
        let index = table_index(brightness, level, sector);
        let weight = brightness_weight * level_weight * sector_weight;

        value += weight * table[(base + index) as usize];
    }

    value
}

/// The gain the log-gain table that starts at word `base` of `table` gives a patch of mean luma
/// `luma` and chroma `cb` and `cr`, centred on 0, all on the 8-bit scale.
#[cube]
pub(crate) fn colour_gain(table: &[f32], base: u32, luma: f32, cb: f32, cr: f32) -> f32 {
    let log_gain = colour_table_value(table, base, luma, cb, cr);

    f32::exp(log_gain)
}

/// The mean luma, Cb and Cr of the colour at the centre of cell `cell`.
///
/// Neutral centres sit at no colour. Other centres sit at the middle of their hue sector.
#[cube]
fn cell_centre(cell: u32) -> (f32, f32, f32) {
    let brightness = cell / COLOUR_CELLS_PER_BRIGHTNESS;
    let ring = cell % COLOUR_CELLS_PER_BRIGHTNESS;
    let luma = (f32::cast_from(brightness) + 0.5f32) * COLOUR_BRIGHTNESS_STEP;

    let coloured_index = u32::max(ring, 1u32) - 1u32;
    let level = coloured_index / COLOUR_HUE_BINS + 1u32;
    let sector = coloured_index % COLOUR_HUE_BINS;
    let level_1 = select(level == 1u32, SATURATION_CENTRE_1, 0.0f32);
    let level_2 = select(level == 2u32, SATURATION_CENTRE_2, 0.0f32);
    let level_3 = select(level == 3u32, SATURATION_CENTRE_3, 0.0f32);
    let level_4 = select(level == 4u32, SATURATION_CENTRE_4, 0.0f32);
    let level_5 = select(level == 5u32, SATURATION_CENTRE_5, 0.0f32);
    let coloured_saturation = level_1 + level_2 + level_3 + level_4 + level_5;
    let saturation = select(ring == 0u32, 0.0f32, coloured_saturation);

    let angle = -PI + (f32::cast_from(sector) + 0.5f32) * (TAU / HUE_SECTORS);
    let cb = saturation * f32::cos(angle);
    let cr = saturation * f32::sin(angle);

    (luma, cb, cr)
}

/// The fractional knot position of mean luma `luma` on the brightness base, clamped to its knots.
#[cube]
fn knot_position(luma: f32) -> f32 {
    let position = luma / KNOT_STEP;

    f32::clamp(position, 0.0f32, LAST_KNOT as f32)
}

/// The lower knot of a knot position, which is never the last knot.
#[cube]
fn knot_low(position: f32) -> u32 {
    let floor = f32::floor(position);

    u32::min(u32::cast_from(floor), LAST_KNOT - 1u32)
}

/// The brightness base of plane header `header` in `planes` at mean luma `luma`.
///
/// The base holds flat past the populated cells' luma span, as the pipeline's curve does past its
/// last measured bin.
#[cube]
fn brightness_base(planes: &[f32], header: u32, luma: f32) -> f32 {
    let span_low = planes[(header + COLOUR_FIT_PLANE_LUMA_LOW) as usize];
    let span_high = planes[(header + COLOUR_FIT_PLANE_LUMA_HIGH) as usize];
    let held = f32::clamp(luma, span_low, span_high);
    let position = knot_position(held);
    let low = knot_low(position);
    let fraction = position - f32::cast_from(low);
    let low_value = planes[(header + low) as usize];
    let high_value = planes[(header + low + 1u32) as usize];

    low_value + (high_value - low_value) * fraction
}

/// `value` moved towards 0 by `dead_zone`, stopping at 0.
#[cube]
fn shrink(value: f32, dead_zone: f32) -> f32 {
    let magnitude = f32::abs(value);
    let size = f32::max(magnitude - dead_zone, 0.0f32);

    select(value < 0.0f32, -size, size)
}

/// Fits each plane's brightness base to its cell levels and writes every cell's fit record.
///
/// Cube `p` takes plane `p`. A cell is populated when it keeps at least 8 blocks. The base is a
/// 17-knot piecewise-linear curve over luma 0 to 255, fitted to the populated cells' levels at
/// their mean luma by least squares. Each cell weighs the square root of its count, and a
/// second-difference penalty keeps empty brightness bins defined. The plane header holds the knots,
/// the kernel prior, whether any cell is populated, the kept blocks and the populated cells' luma
/// span, past which the base holds flat. Each cell's record holds its mean colour, its count when
/// populated and 0 otherwise, and its level less the base. Every sum runs in a fixed order, so the
/// fit is the same on every run.
#[cube(launch_unchecked)]
pub fn noise_colour_fit_base(window: &[f32], levels: &[f32], fit_cells: &mut [f32], fit_planes: &mut [f32]) {
    let plane = CUBE_POS_X;
    let plane_cells = plane * COLOUR_CELLS;
    let header = plane * COLOUR_FIT_PLANE_WORDS;

    let mut lows = Shared::<[u32]>::new_slice(COLOUR_CELLS as usize);
    let mut fractions = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut weights = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut cell_levels = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut root_partials = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut populated_partials = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut block_partials = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut low_partials = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut high_partials = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut sums = Shared::<[f32]>::new_slice(SUM_QUANTITIES as usize);
    let mut knots = Shared::<[f32]>::new_slice(COLOUR_FIT_KNOTS as usize);

    let thread = UNIT_POS_X as usize;
    let mut root_sum = 0.0f32;
    let mut populated_sum = 0.0f32;
    let mut block_sum = 0.0f32;
    let mut span_low = f32::MAX.runtime();
    let mut span_high = f32::MIN.runtime();

    let mut cell = UNIT_POS_X;
    while cell < COLOUR_CELLS {
        let record = (plane_cells + cell) * COLOUR_CELL_WORDS;
        let count = window[(record + COLOUR_CELL_COUNT) as usize];
        let divisor = f32::max(count, 1.0f32);
        let luma = window[(record + COLOUR_CELL_LUMA_SUM) as usize] / divisor;
        let cb = window[(record + COLOUR_CELL_CB_SUM) as usize] / divisor;
        let cr = window[(record + COLOUR_CELL_CR_SUM) as usize] / divisor;

        let level_base = (plane_cells + cell) * COLOUR_LEVEL_FIELDS;
        let kept_blocks = levels[level_base as usize];
        let level = levels[(level_base + 1u32) as usize];
        let populated = kept_blocks >= COLOUR_FIT_MIN_BLOCKS;
        let position = knot_position(luma);
        let low = knot_low(position);

        let weight = select(populated, kept_blocks, 0.0f32);
        let index = cell as usize;
        lows[index] = low;
        fractions[index] = position - f32::cast_from(low);
        weights[index] = weight;
        cell_levels[index] = level;

        root_sum += f32::sqrt(weight);
        populated_sum += select(populated, 1.0f32, 0.0f32);
        block_sum += kept_blocks;
        if populated {
            span_low = f32::min(span_low, luma);
            span_high = f32::max(span_high, luma);
        }

        let fit_record = (plane_cells + cell) * COLOUR_FIT_CELL_FIELDS;
        fit_cells[(fit_record + COLOUR_FIT_LUMA) as usize] = luma;
        fit_cells[(fit_record + COLOUR_FIT_CB) as usize] = cb;
        fit_cells[(fit_record + COLOUR_FIT_CR) as usize] = cr;
        fit_cells[(fit_record + COLOUR_FIT_WEIGHT) as usize] = weight;

        cell += COLOUR_FIT_THREADS;
    }

    root_partials[thread] = root_sum;
    populated_partials[thread] = populated_sum;
    block_partials[thread] = block_sum;
    low_partials[thread] = span_low;
    high_partials[thread] = span_high;

    sync_cube();

    let half_threads = COLOUR_FIT_THREADS / 2u32;
    let mut half = half_threads.runtime();
    while half > 0u32 {
        if UNIT_POS_X < half {
            let partner = thread + half as usize;
            let root_pair = root_partials[thread] + root_partials[partner];
            let populated_pair = populated_partials[thread] + populated_partials[partner];
            let block_pair = block_partials[thread] + block_partials[partner];
            let low_pair = f32::min(low_partials[thread], low_partials[partner]);
            let high_pair = f32::max(high_partials[thread], high_partials[partner]);
            root_partials[thread] = root_pair;
            populated_partials[thread] = populated_pair;
            block_partials[thread] = block_pair;
            low_partials[thread] = low_pair;
            high_partials[thread] = high_pair;
        }

        sync_cube();
        half /= 2u32;
    }

    // A cell of brightness bin `b` has its mean luma inside the bin, so its lower knot is `b` or
    // `b + 1`. Knot `k` therefore only gathers the cells of bins `k - 2` to `k`.
    let knot = UNIT_POS_X;
    if knot < COLOUR_FIT_KNOTS {
        let first_bin = u32::max(knot, 2u32) - 2u32;
        let last_bin = u32::min(knot + 1u32, COLOUR_BRIGHTNESS_BINS);
        let first_cell = first_bin * COLOUR_CELLS_PER_BRIGHTNESS;
        let end_cell = last_bin * COLOUR_CELLS_PER_BRIGHTNESS;

        let mut diagonal = 0.0f32;
        let mut off_diagonal = 0.0f32;
        let mut right = 0.0f32;
        let mut index = first_cell;
        while index < end_cell {
            let low = lows[index as usize];
            let fraction = fractions[index as usize];
            let weight = weights[index as usize];
            let low_share = 1.0f32 - fraction;
            let is_low = low == knot;
            let is_high = low + 1u32 == knot;

            let low_square = select(is_low, low_share * low_share, 0.0f32);
            let high_square = select(is_high, fraction * fraction, 0.0f32);
            let low_tent = select(is_low, low_share, 0.0f32);
            let high_tent = select(is_high, fraction, 0.0f32);
            let cross = select(is_low, low_share * fraction, 0.0f32);

            diagonal += weight * (low_square + high_square);
            off_diagonal += weight * cross;
            right += weight * cell_levels[index as usize] * (low_tent + high_tent);
            index += 1u32;
        }

        sums[(SUM_DIAGONAL + knot) as usize] = diagonal;
        sums[(SUM_RIGHT + knot) as usize] = right;
        if knot < LAST_KNOT {
            sums[(SUM_OFF_DIAGONAL + knot) as usize] = off_diagonal;
        }
    }

    if UNIT_POS_X == 0u32 {
        sums[SUM_ROOT_COUNT as usize] = root_partials[0];
        sums[SUM_POPULATED as usize] = populated_partials[0];
        sums[SUM_BLOCKS as usize] = block_partials[0];
    }

    sync_cube();

    if UNIT_POS_X == 0u32 {
        let populated = sums[SUM_POPULATED as usize];
        let has_cells = populated > 0.0f32;
        let populated_divisor = f32::max(populated, 1.0f32);
        let mean_root = sums[SUM_ROOT_COUNT as usize] / populated_divisor;
        let penalty = COLOUR_FIT_KNOT_PENALTY * COLOUR_FIT_KNOT_PENALTY * mean_root;

        solve_knots(&sums, &mut knots, penalty);

        #[unroll]
        for knot in 0..COLOUR_FIT_KNOTS {
            let value = select(has_cells, knots[knot as usize], 0.0f32);
            knots[knot as usize] = value;
            fit_planes[(header + knot) as usize] = value;
        }

        let blocks = sums[SUM_BLOCKS as usize];
        let held_low = select(has_cells, low_partials[0], 0.0f32);
        let held_high = select(has_cells, high_partials[0], 255.0f32);
        fit_planes[(header + COLOUR_FIT_PLANE_PRIOR) as usize] = COLOUR_FIT_PRIOR_SHARE * blocks;
        fit_planes[(header + COLOUR_FIT_PLANE_POPULATED) as usize] = select(has_cells, 1.0f32, 0.0f32);
        fit_planes[(header + COLOUR_FIT_PLANE_BLOCKS) as usize] = blocks;
        fit_planes[(header + COLOUR_FIT_PLANE_LUMA_LOW) as usize] = held_low;
        fit_planes[(header + COLOUR_FIT_PLANE_LUMA_HIGH) as usize] = held_high;
    }

    sync_cube();

    let mut cell = UNIT_POS_X;
    while cell < COLOUR_CELLS {
        let index = cell as usize;
        let low = lows[index];
        let fraction = fractions[index];
        let low_value = knots[low as usize];
        let high_value = knots[(low + 1u32) as usize];
        let base = low_value + (high_value - low_value) * fraction;
        let residual = select(weights[index] > 0.0f32, cell_levels[index] - base, 0.0f32);

        let fit_record = (plane_cells + cell) * COLOUR_FIT_CELL_FIELDS;
        fit_cells[(fit_record + COLOUR_FIT_RESIDUAL) as usize] = residual;

        cell += COLOUR_FIT_THREADS;
    }
}

/// Solves the brightness base's normal equations from `sums` into `knots`.
///
/// The matrix is the data's banded part plus `penalty` times the second-difference operator's
/// square, which is pentadiagonal. It is scaled to a unit diagonal, given a small ridge so a fit
/// with too few brightness levels stays defined, and solved by a banded Cholesky factorisation with
/// one step of refinement.
#[cube]
fn solve_knots(sums: &Shared<[f32]>, knots: &mut Shared<[f32]>, penalty: f32) {
    let mut diagonal = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut first = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut second = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut right = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);

    #[unroll]
    for knot in 0..COLOUR_FIT_KNOTS {
        let index = knot as usize;
        diagonal[index] = sums[(SUM_DIAGONAL + knot) as usize];
        right[index] = sums[(SUM_RIGHT + knot) as usize];
        first[index] = 0.0f32;
        second[index] = 0.0f32;
        if comptime!(knot < LAST_KNOT) {
            first[index] = sums[(SUM_OFF_DIAGONAL + knot) as usize];
        }
    }

    // Each penalty row is 1, -2, 1 over three neighbouring knots.
    #[unroll]
    for row in 0..COLOUR_FIT_KNOTS - 2u32 {
        let index = row as usize;
        diagonal[index] += penalty;
        diagonal[index + 1] += 4.0f32 * penalty;
        diagonal[index + 2] += penalty;
        first[index] -= 2.0f32 * penalty;
        first[index + 1] -= 2.0f32 * penalty;
        second[index] += penalty;
    }

    let mut scale = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    #[unroll]
    for knot in 0..COLOUR_FIT_KNOTS {
        let index = knot as usize;
        let positive = f32::max(diagonal[index], 1.0e-30f32);
        scale[index] = 1.0f32 / f32::sqrt(positive);
    }

    let mut low_diagonal = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut low_first = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut low_second = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    #[unroll]
    for knot in 0..COLOUR_FIT_KNOTS {
        let index = knot as usize;
        let ridged_unit = 1.0f32 + COLOUR_FIT_RIDGE;
        let mut pivot = ridged_unit.runtime();
        if comptime!(knot >= 1) {
            pivot -= low_first[index - 1] * low_first[index - 1];
        }
        if comptime!(knot >= 2) {
            pivot -= low_second[index - 2] * low_second[index - 2];
        }

        let positive_pivot = f32::max(pivot, 1.0e-12f32);
        let root = f32::sqrt(positive_pivot);
        low_diagonal[index] = root;
        low_first[index] = 0.0f32;
        low_second[index] = 0.0f32;

        if comptime!(knot + 1 < COLOUR_FIT_KNOTS) {
            let mut below = first[index] * scale[index] * scale[index + 1];
            if comptime!(knot >= 1) {
                below -= low_second[index - 1] * low_first[index - 1];
            }
            low_first[index] = below / root;
        }
        if comptime!(knot + 2 < COLOUR_FIT_KNOTS) {
            low_second[index] = second[index] * scale[index] * scale[index + 2] / root;
        }
    }

    let mut solution = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    let mut residual = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
    #[unroll]
    for knot in 0..COLOUR_FIT_KNOTS {
        let index = knot as usize;
        solution[index] = 0.0f32;
        residual[index] = right[index];
    }

    // The first pass solves for the knots and the second for the correction its residual leaves.
    #[unroll]
    for _pass in 0..2u32 {
        let mut forward = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
        #[unroll]
        for knot in 0..COLOUR_FIT_KNOTS {
            let index = knot as usize;
            let mut value = residual[index] * scale[index];
            if comptime!(knot >= 1) {
                value -= low_first[index - 1] * forward[index - 1];
            }
            if comptime!(knot >= 2) {
                value -= low_second[index - 2] * forward[index - 2];
            }
            forward[index] = value / low_diagonal[index];
        }

        let mut backward = Array::<f32>::new(COLOUR_FIT_KNOTS as usize);
        #[unroll]
        for step in 0..COLOUR_FIT_KNOTS {
            let knot = comptime!(LAST_KNOT - step);
            let index = knot as usize;
            let mut value = forward[index];
            if comptime!(knot + 1 < COLOUR_FIT_KNOTS) {
                value -= low_first[index] * backward[index + 1];
            }
            if comptime!(knot + 2 < COLOUR_FIT_KNOTS) {
                value -= low_second[index] * backward[index + 2];
            }
            backward[index] = value / low_diagonal[index];
        }

        #[unroll]
        for knot in 0..COLOUR_FIT_KNOTS {
            let index = knot as usize;
            solution[index] += backward[index] * scale[index];
        }

        #[unroll]
        for knot in 0..COLOUR_FIT_KNOTS {
            let index = knot as usize;
            let ridge = COLOUR_FIT_RIDGE * diagonal[index];
            let mut product = (diagonal[index] + ridge) * solution[index];
            if comptime!(knot >= 1) {
                product += first[index - 1] * solution[index - 1];
            }
            if comptime!(knot >= 2) {
                product += second[index - 2] * solution[index - 2];
            }
            if comptime!(knot + 1 < COLOUR_FIT_KNOTS) {
                product += first[index] * solution[index + 1];
            }
            if comptime!(knot + 2 < COLOUR_FIT_KNOTS) {
                product += second[index] * solution[index + 2];
            }
            residual[index] = right[index] - product;
        }
    }

    #[unroll]
    for knot in 0..COLOUR_FIT_KNOTS {
        knots[knot as usize] = solution[knot as usize];
    }
}

/// Writes each plane's wide-kernel base at every cell's mean colour, then at every cell's centre.
///
/// Cube row `p` takes plane `p`. Each query is shared by [COLOUR_FIT_KERNEL_LANES] neighbouring
/// threads, which each take every lanes-th cell. The base is the brightness base plus a
/// Gaussian-weighted average of the populated cells' residuals, each weighted by its count.
/// Distances count 32 luma codes and 12 chroma codes as one unit. The plane's prior joins the
/// weights' sum, so the base falls back to the brightness base where cells are far away. Every sum
/// runs in a fixed order, so the base is the same on every run.
#[cube(launch_unchecked)]
pub fn noise_colour_fit_kernel(fit_cells: &[f32], fit_planes: &[f32], bases: &mut [f32]) {
    let plane = CUBE_POS_Y;
    let lane = UNIT_POS_X % COLOUR_FIT_KERNEL_LANES;
    let query = (CUBE_POS_X * COLOUR_FIT_KERNEL_THREADS + UNIT_POS_X) / COLOUR_FIT_KERNEL_LANES;
    let plane_cells = plane * COLOUR_CELLS;
    let header = plane * COLOUR_FIT_PLANE_WORDS;

    let mut numerators = Shared::<[f32]>::new_slice(COLOUR_FIT_KERNEL_THREADS as usize);
    let mut denominators = Shared::<[f32]>::new_slice(COLOUR_FIT_KERNEL_THREADS as usize);

    // Every cube stages the plane's cells in kernel units, with each cell's weighted residual.
    let mut cell_lumas = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut cell_cbs = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut cell_crs = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut cell_weights = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut cell_lifts = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);

    let mut staged = UNIT_POS_X;
    while staged < COLOUR_CELLS {
        let record = (plane_cells + staged) * COLOUR_FIT_CELL_FIELDS;
        let weight = fit_cells[(record + COLOUR_FIT_WEIGHT) as usize];
        let residual = fit_cells[(record + COLOUR_FIT_RESIDUAL) as usize];
        let index = staged as usize;

        cell_lumas[index] = fit_cells[(record + COLOUR_FIT_LUMA) as usize] / COLOUR_KERNEL_LUMA_STEP;
        cell_cbs[index] = fit_cells[(record + COLOUR_FIT_CB) as usize] / COLOUR_KERNEL_CHROMA_STEP;
        cell_crs[index] = fit_cells[(record + COLOUR_FIT_CR) as usize] / COLOUR_KERNEL_CHROMA_STEP;
        cell_weights[index] = weight;
        cell_lifts[index] = weight * residual;

        staged += COLOUR_FIT_KERNEL_THREADS;
    }

    sync_cube();

    let in_range = query < COLOUR_FIT_QUERIES;
    let is_centre = query >= COLOUR_CELLS;
    let query_cell = query % COLOUR_CELLS;
    let fit_record = (plane_cells + query_cell) * COLOUR_FIT_CELL_FIELDS;
    let (centre_luma, centre_cb, centre_cr) = cell_centre(query_cell);
    let mut mean_luma = 0.0f32;
    let mut mean_cb = 0.0f32;
    let mut mean_cr = 0.0f32;
    if in_range {
        mean_luma = fit_cells[(fit_record + COLOUR_FIT_LUMA) as usize];
        mean_cb = fit_cells[(fit_record + COLOUR_FIT_CB) as usize];
        mean_cr = fit_cells[(fit_record + COLOUR_FIT_CR) as usize];
    }
    let luma = select(is_centre, centre_luma, mean_luma);
    let cb = select(is_centre, centre_cb, mean_cb);
    let cr = select(is_centre, centre_cr, mean_cr);

    let query_luma = luma / COLOUR_KERNEL_LUMA_STEP;
    let query_cb = cb / COLOUR_KERNEL_CHROMA_STEP;
    let query_cr = cr / COLOUR_KERNEL_CHROMA_STEP;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    let mut cell = lane;
    while cell < COLOUR_CELLS {
        let index = cell as usize;
        let weight = cell_weights[index];

        if weight > 0.0f32 {
            let luma_gap = query_luma - cell_lumas[index];
            let cb_gap = query_cb - cell_cbs[index];
            let cr_gap = query_cr - cell_crs[index];
            let distance = luma_gap * luma_gap + cb_gap * cb_gap + cr_gap * cr_gap;
            let kernel = f32::exp(KERNEL_EXPONENT * distance);

            numerator += kernel * cell_lifts[index];
            denominator += kernel * weight;
        }

        cell += COLOUR_FIT_KERNEL_LANES;
    }

    numerators[UNIT_POS_X as usize] = numerator;
    denominators[UNIT_POS_X as usize] = denominator;

    sync_cube();

    if in_range && lane == 0u32 {
        let mut query_numerator = 0.0f32;
        let mut query_denominator = 0.0f32;
        for offset in 0..COLOUR_FIT_KERNEL_LANES {
            let partial = (UNIT_POS_X + offset) as usize;
            query_numerator += numerators[partial];
            query_denominator += denominators[partial];
        }

        let prior = fit_planes[(header + COLOUR_FIT_PLANE_PRIOR) as usize];
        let total = query_denominator + prior;
        let divisor = f32::max(total, 1.0e-30f32);
        let lift = select(total > 0.0f32, query_numerator / divisor, 0.0f32);
        let brightness = brightness_base(fit_planes, header, luma);

        bases[(plane * COLOUR_FIT_QUERIES + query) as usize] = brightness + lift;
    }
}

/// Writes each plane's log grain table, one value at every cell's centre.
///
/// Cube `p` takes plane `p`. A populated cell moves from the base at its mean colour towards its own
/// level by `n / (n + 15)` of the gap, with `n` its kept blocks. With `luma` the gap first shrinks
/// towards 0 by a small dead zone. An empty cell takes the mean move of its populated neighbours one
/// step along brightness, saturation and hue, or 0 with none. A neutral cell's saturation neighbour
/// is the mean of the populated cells of the first saturation level. Each centre's value is the
/// base there plus its move.
#[cube(launch_unchecked)]
pub fn noise_colour_fit_table(
    levels: &[f32],
    fit_cells: &[f32],
    bases: &[f32],
    table: &mut [f32],
    #[comptime] luma: bool,
) {
    let plane = CUBE_POS_X;
    let plane_cells = plane * COLOUR_CELLS;
    let dead_zone = comptime!(if luma { COLOUR_FIT_LUMA_DEAD_ZONE } else { 0.0f32 });

    let mut moves = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);
    let mut populated = Shared::<[f32]>::new_slice(COLOUR_CELLS as usize);

    let mut cell = UNIT_POS_X;
    while cell < COLOUR_CELLS {
        let fit_record = (plane_cells + cell) * COLOUR_FIT_CELL_FIELDS;
        let weight = fit_cells[(fit_record + COLOUR_FIT_WEIGHT) as usize];
        let level = levels[((plane_cells + cell) * COLOUR_LEVEL_FIELDS + 1u32) as usize];
        let base = bases[(plane * COLOUR_FIT_QUERIES + cell) as usize];
        let pull = weight / (weight + COLOUR_FIT_PULL);
        let gap = shrink(level - base, dead_zone);
        let has_cell = weight > 0.0f32;

        moves[cell as usize] = select(has_cell, pull * gap, 0.0f32);
        populated[cell as usize] = select(has_cell, 1.0f32, 0.0f32);

        cell += COLOUR_FIT_THREADS;
    }

    sync_cube();

    let mut cell = UNIT_POS_X;
    while cell < COLOUR_CELLS {
        let brightness = cell / COLOUR_CELLS_PER_BRIGHTNESS;
        let ring = cell % COLOUR_CELLS_PER_BRIGHTNESS;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;

        if brightness > 0u32 {
            let darker = cell - COLOUR_CELLS_PER_BRIGHTNESS;
            sum += moves[darker as usize];
            count += populated[darker as usize];
        }
        if brightness + 1u32 < COLOUR_BRIGHTNESS_BINS {
            let brighter = cell + COLOUR_CELLS_PER_BRIGHTNESS;
            sum += moves[brighter as usize];
            count += populated[brighter as usize];
        }

        let row = brightness * COLOUR_CELLS_PER_BRIGHTNESS;
        if ring == 0u32 {
            let mut ring_sum = 0.0f32;
            let mut ring_count = 0.0f32;
            for sector in 0..COLOUR_HUE_BINS {
                let neighbour = row + 1u32 + sector;
                ring_sum += moves[neighbour as usize];
                ring_count += populated[neighbour as usize];
            }

            if ring_count > 0.0f32 {
                sum += ring_sum / ring_count;
                count += 1.0f32;
            }
        } else {
            let coloured_index = ring - 1u32;
            let level = coloured_index / COLOUR_HUE_BINS;
            let sector = coloured_index % COLOUR_HUE_BINS;
            let less_saturated = select(level == 0u32, row, cell - COLOUR_HUE_BINS);
            sum += moves[less_saturated as usize];
            count += populated[less_saturated as usize];

            if level + 2u32 < SATURATION_LEVELS {
                let more_saturated = cell + COLOUR_HUE_BINS;
                sum += moves[more_saturated as usize];
                count += populated[more_saturated as usize];
            }

            let level_start = row + 1u32 + level * COLOUR_HUE_BINS;
            let next_sector = (sector + 1u32) % COLOUR_HUE_BINS;
            let previous_sector = (sector + COLOUR_HUE_BINS - 1u32) % COLOUR_HUE_BINS;
            let next = level_start + next_sector;
            let previous = level_start + previous_sector;
            sum += moves[next as usize] + moves[previous as usize];
            count += populated[next as usize] + populated[previous as usize];
        }

        let divisor = f32::max(count, 1.0f32);
        let filled = select(count > 0.0f32, sum / divisor, 0.0f32);
        let has_cell = populated[cell as usize] > 0.0f32;
        let cell_move = select(has_cell, moves[cell as usize], filled);
        let centre_base = bases[(plane * COLOUR_FIT_QUERIES + COLOUR_CELLS + cell) as usize];

        table[(plane_cells + cell) as usize] = centre_base + cell_move;

        cell += COLOUR_FIT_THREADS;
    }
}

/// Writes every cell's outlier ceiling, the table at the cell's mean colour plus log 2.
///
/// Thread `i` takes cell `i` of every plane's cells. A cell with no blocks keeps every block.
#[cube(launch_unchecked)]
pub fn noise_colour_fit_ceilings(
    window: &[f32],
    table: &[f32],
    ceilings: &mut [f32],
    #[comptime] cells: u32,
) {
    let index = ABSOLUTE_POS_X;

    if index < cells {
        let plane = index / COLOUR_CELLS;
        let record = index * COLOUR_CELL_WORDS;
        let count = window[(record + COLOUR_CELL_COUNT) as usize];
        let divisor = f32::max(count, 1.0f32);
        let luma = window[(record + COLOUR_CELL_LUMA_SUM) as usize] / divisor;
        let cb = window[(record + COLOUR_CELL_CB_SUM) as usize] / divisor;
        let cr = window[(record + COLOUR_CELL_CR_SUM) as usize] / divisor;
        let model = colour_table_value(table, plane * COLOUR_CELLS, luma, cb, cr);

        ceilings[index as usize] = select(count > 0.0f32, model + LN_2, f32::MAX);
    }
}

/// Writes each plane's log gain table from its log grain table.
///
/// Cube `p` takes plane `p`. With `luma`, the gain is the model's grain over the pipeline's
/// brightness curve times one calibration factor. The factor is the median over the window's
/// neutral blocks of grain over the curve, taken from the calibration records after the cells. The
/// curve is `curve` per brightness bin, or flat when `derived` says it is not valid. Otherwise the
/// gain is the model's grain over the median grain of all the plane's blocks, taken from the plane
/// record, clamped between 0.4 and 3. Each log gain then scales by `n / (n + 256)` of the plane's
/// kept blocks, and luma's also by the same share of its neutral blocks, so thin windows stay near
/// 1. A plane with no populated cell, or with no blocks to normalise by, gets gains of 1.
#[cube(launch_unchecked)]
pub fn noise_colour_fit_gains(
    window: &[f32],
    table: &[f32],
    fit_planes: &[f32],
    curve: &[f32],
    derived: &[f32],
    gains: &mut [f32],
    #[comptime] planes: u32,
    #[comptime] luma: bool,
) {
    let plane = CUBE_POS_X;
    let plane_cells = plane * COLOUR_CELLS;
    let header = plane * COLOUR_FIT_PLANE_WORDS;

    let mut histograms = Shared::<[f32]>::new_slice(NORMALISER_VALUES as usize);
    let mut cumulative = Shared::<[f32]>::new_slice(NORMALISER_POINTS as usize);
    let mut shifts = Shared::<[f32]>::new_slice(NORMALISER_RECORDS as usize);
    let mut below_points = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut above_points = Shared::<[f32]>::new_slice(COLOUR_FIT_THREADS as usize);
    let mut normaliser = Shared::<[f32]>::new_slice(2usize);

    let curve_valid = derived[DERIVED_CURVE_VALID as usize] != 0.0f32;

    if UNIT_POS_X < NORMALISER_RECORDS {
        let record = UNIT_POS_X;
        let mut shift = 0.0f32.runtime();
        if comptime!(luma) {
            let ratio = curve[record as usize];
            let usable = curve_valid && ratio > 0.0f32;
            let applied = select(usable, ratio, 1.0f32);
            shift = -f32::ln(applied);
        }
        shifts[record as usize] = shift;
    }

    // Luma reads one calibration record per brightness bin. Chroma reads its plane record first.
    let extra_base = comptime!(planes * COLOUR_CELLS);
    let mut value = UNIT_POS_X;
    while value < NORMALISER_VALUES {
        let record = value / COLOUR_HISTOGRAM_BINS;
        let bin = value % COLOUR_HISTOGRAM_BINS;
        let mut count = 0.0f32.runtime();

        if comptime!(luma) {
            let calibration = extra_base + record;
            count = window[(calibration * COLOUR_CELL_WORDS + COLOUR_CELL_HISTOGRAM + bin) as usize];
        } else if record == 0u32 {
            let plane_record = extra_base + plane;
            count = window[(plane_record * COLOUR_CELL_WORDS + COLOUR_CELL_HISTOGRAM + bin) as usize];
        }

        histograms[value as usize] = count;
        value += COLOUR_FIT_THREADS;
    }

    sync_cube();

    if UNIT_POS_X < NORMALISER_RECORDS {
        let record = UNIT_POS_X;
        let first_edge = record * HISTOGRAM_EDGES;
        let mut running = 0.0f32;
        cumulative[first_edge as usize] = running;

        for bin in 0..COLOUR_HISTOGRAM_BINS {
            running += histograms[(record * COLOUR_HISTOGRAM_BINS + bin) as usize];
            cumulative[(first_edge + bin + 1u32) as usize] = running;
        }
    }

    sync_cube();

    // The median of the shifted histograms lies between the last breakpoint below half the blocks
    // and the first at or above it, where the share below grows linearly.
    let total = histogram_total(&cumulative);
    let target = total * 0.5f32;
    let mut best_below = f32::MIN.runtime();
    let mut best_above = f32::MAX.runtime();
    let mut point = UNIT_POS_X;
    while point < NORMALISER_POINTS {
        let record = point / HISTOGRAM_EDGES;
        let edge = point % HISTOGRAM_EDGES;
        let position = COLOUR_HISTOGRAM_LOG_MIN
            + f32::cast_from(edge) * COLOUR_HISTOGRAM_LOG_STEP
            + shifts[record as usize];
        let share = shifted_share(&histograms, &cumulative, &shifts, position);

        if share < target {
            best_below = f32::max(best_below, position);
        } else {
            best_above = f32::min(best_above, position);
        }

        point += COLOUR_FIT_THREADS;
    }

    let thread = UNIT_POS_X as usize;
    below_points[thread] = best_below;
    above_points[thread] = best_above;

    sync_cube();

    let half_threads = COLOUR_FIT_THREADS / 2u32;
    let mut half = half_threads.runtime();
    while half > 0u32 {
        if UNIT_POS_X < half {
            let partner = thread + half as usize;
            let below_pair = f32::max(below_points[thread], below_points[partner]);
            let above_pair = f32::min(above_points[thread], above_points[partner]);
            below_points[thread] = below_pair;
            above_points[thread] = above_pair;
        }

        sync_cube();
        half /= 2u32;
    }

    if UNIT_POS_X == 0u32 {
        let below = below_points[0];
        let above = above_points[0];
        let below_share = shifted_share(&histograms, &cumulative, &shifts, below);
        let above_share = shifted_share(&histograms, &cumulative, &shifts, above);
        let rise = f32::max(above_share - below_share, 1.0e-30f32);
        let median = below + (target - below_share) / rise * (above - below);
        let has_median = total > 0.0f32 && below > f32::MIN && above < f32::MAX;

        normaliser[0] = select(has_median, median, 0.0f32);
        normaliser[1] = select(has_median, 1.0f32, 0.0f32);
    }

    sync_cube();

    let populated = fit_planes[(header + COLOUR_FIT_PLANE_POPULATED) as usize] > 0.0f32;
    let applies = populated && normaliser[1] > 0.0f32;
    let log_normaliser = normaliser[0];
    let log_min = f32::ln(COLOUR_CHROMA_GAIN_MIN);
    let log_max = f32::ln(COLOUR_CHROMA_GAIN_MAX);

    // The gains ramp in with the evidence behind them, luma's also with its neutral blocks.
    let blocks = fit_planes[(header + COLOUR_FIT_PLANE_BLOCKS) as usize];
    let block_ramp = blocks / (blocks + COLOUR_GAIN_BLOCK_RAMP);
    let neutral_ramp = total / (total + COLOUR_GAIN_NEUTRAL_RAMP);
    let mut ramp = block_ramp;
    if comptime!(luma) {
        ramp = block_ramp * neutral_ramp;
    }

    let mut cell = UNIT_POS_X;
    while cell < COLOUR_CELLS {
        let log_grain = table[(plane_cells + cell) as usize];
        let brightness = cell / COLOUR_CELLS_PER_BRIGHTNESS;
        let mut log_gain = log_grain - log_normaliser;

        if comptime!(luma) {
            log_gain += shifts[brightness as usize];
        } else {
            log_gain = f32::clamp(log_gain, log_min, log_max);
        }

        let ramped = ramp * log_gain;
        gains[(plane_cells + cell) as usize] = select(applies, ramped, 0.0f32);
        cell += COLOUR_FIT_THREADS;
    }
}

/// The blocks of the shifted histograms whose log grain lies below `position`.
///
/// Histogram `r` holds bins of log grain moved by `shifts[r]`, and `cumulative` its blocks below
/// each bin edge. A bin's blocks spread evenly across it. The histograms add in order, so the share
/// is the same on every run.
#[cube]
fn shifted_share(
    histograms: &Shared<[f32]>,
    cumulative: &Shared<[f32]>,
    shifts: &Shared<[f32]>,
    position: f32,
) -> f32 {
    let mut share = 0.0f32;
    for record in 0..NORMALISER_RECORDS {
        let offset = position - shifts[record as usize] - COLOUR_HISTOGRAM_LOG_MIN;
        let bin_position = offset / COLOUR_HISTOGRAM_LOG_STEP;
        let clamped = f32::clamp(bin_position, 0.0f32, COLOUR_HISTOGRAM_BINS as f32);
        let floor = f32::floor(clamped);
        let bin = u32::min(u32::cast_from(floor), COLOUR_HISTOGRAM_BINS - 1u32);
        let fraction = clamped - f32::cast_from(bin);
        let below_bin = cumulative[(record * HISTOGRAM_EDGES + bin) as usize];
        let bin_count = histograms[(record * COLOUR_HISTOGRAM_BINS + bin) as usize];

        share += below_bin + fraction * bin_count;
    }

    share
}

/// The total of the shifted histograms, from their blocks below each bin edge.
#[cube]
fn histogram_total(cumulative: &Shared<[f32]>) -> f32 {
    let mut total = 0.0f32;
    for record in 0..NORMALISER_RECORDS {
        total += cumulative[(record * HISTOGRAM_EDGES + COLOUR_HISTOGRAM_BINS) as usize];
    }

    total
}

/// Picks `fraction` for the upper of two corners and its complement for the lower.
#[cube]
fn corner_weight(fraction: f32, #[comptime] upper: bool) -> f32 {
    if comptime!(upper) {
        fraction
    } else {
        1.0f32 - fraction
    }
}

const _: () = assert!(COLOUR_CALIBRATION_RECORDS == NORMALISER_RECORDS);
