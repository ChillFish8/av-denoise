use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, CELL, TEMPLATE_SEEDS};
use crate::nl4d::grain::gaussian::GAUSSIAN_SEQUENCE;

const TEMPLATE_ROWS: usize = 73;
const TEMPLATE_COLS: usize = 82;
/// The interior every statistic reads, past the rows and columns the AR filter never reaches.
const INTERIOR_START: usize = 9;
const INTERIOR_SIZE: usize = 64;

/// AV1's 16-bit linear feedback shift register for grain samples.
pub(crate) struct Av1Random {
    register: u16,
}

impl Av1Random {
    pub(crate) fn new(seed: u16) -> Self {
        Self { register: seed }
    }

    /// The next `bits`-bit random number.
    pub(crate) fn next(&mut self, bits: u32) -> u32 {
        let register = self.register as u32;
        let bit = (register ^ (register >> 1) ^ (register >> 3) ^ (register >> 12)) & 1;
        let shifted = (register >> 1) | (bit << 15);
        self.register = shifted as u16;
        (shifted >> (16 - bits)) & ((1 << bits) - 1)
    }
}

fn round2(value: i32, shift: u32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

/// Builds the 73x82 luma grain template an 8-bit AV1 decoder makes for these weights and seed.
pub(crate) fn luma_template(coeffs: &[i32; AR_COEFFS], ar_shift: u32, seed: u16) -> Vec<i32> {
    let mut rng = Av1Random::new(seed);
    let mut grain = vec![0i32; TEMPLATE_ROWS * TEMPLATE_COLS];
    for sample in grain.iter_mut() {
        let index = rng.next(11) as usize;
        let gaussian = GAUSSIAN_SEQUENCE[index] as i32;
        *sample = round2(gaussian, 4);
    }

    for y in 3..TEMPLATE_ROWS {
        for x in 3..TEMPLATE_COLS - 3 {
            let mut total = 0i32;
            for (&(dy, dx), &coeff) in AR_OFFSETS.iter().zip(coeffs.iter()) {
                let neighbour_y = (y as i32 + dy) as usize;
                let neighbour_x = (x as i32 + dx) as usize;
                total += coeff * grain[neighbour_y * TEMPLATE_COLS + neighbour_x];
            }

            let filtered = grain[y * TEMPLATE_COLS + x] + round2(total, ar_shift);
            grain[y * TEMPLATE_COLS + x] = filtered.clamp(-128, 127);
        }
    }

    grain
}

/// The seed for calibration template `index`.
pub(crate) fn template_seed(index: u32) -> u16 {
    ((1000 + 7919 * index) & 0xFFFF) as u16
}

/// The grain std of these weights' templates, and the median std of their 8x8 blocks.
///
/// Both read the 64x64 interior of [TEMPLATE_SEEDS](crate::nl4d::grain::consts::TEMPLATE_SEEDS)
/// templates. A block's std removes its own mean.
pub(crate) fn template_stats(coeffs: &[i32; AR_COEFFS], ar_shift: u32) -> (f64, f64) {
    let capacity = TEMPLATE_SEEDS as usize * INTERIOR_SIZE * INTERIOR_SIZE;
    let mut samples = Vec::with_capacity(capacity);
    let mut block_stds = Vec::new();
    for index in 0..TEMPLATE_SEEDS {
        let seed = template_seed(index);
        let template = luma_template(coeffs, ar_shift, seed);
        let interior = interior_of(&template);
        block_stds.extend(block_stds_of(&interior));
        samples.extend(interior);
    }

    let sigma = population_std(&samples);
    let median = median_of(&mut block_stds);
    (sigma, median)
}

fn interior_of(template: &[i32]) -> Vec<f64> {
    let mut interior = Vec::with_capacity(INTERIOR_SIZE * INTERIOR_SIZE);
    for y in INTERIOR_START..INTERIOR_START + INTERIOR_SIZE {
        let row_start = y * TEMPLATE_COLS + INTERIOR_START;
        let row = &template[row_start..row_start + INTERIOR_SIZE];
        interior.extend(row.iter().map(|&value| value as f64));
    }
    interior
}

fn block_stds_of(interior: &[f64]) -> Vec<f64> {
    let cell = CELL as usize;
    let blocks_per_side = INTERIOR_SIZE / cell;
    let mut stds = Vec::with_capacity(blocks_per_side * blocks_per_side);
    for block_y in 0..blocks_per_side {
        for block_x in 0..blocks_per_side {
            let mut block = Vec::with_capacity(cell * cell);
            for y in 0..cell {
                let start = (block_y * cell + y) * INTERIOR_SIZE + block_x * cell;
                block.extend_from_slice(&interior[start..start + cell]);
            }

            stds.push(sample_std(&block));
        }
    }
    stds
}

pub(crate) fn sample_std(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let squares: f64 = values.iter().map(|value| (value - mean) * (value - mean)).sum();
    (squares / (count - 1.0)).sqrt()
}

fn population_std(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let squares: f64 = values.iter().map(|value| (value - mean) * (value - mean)).sum();
    (squares / count).sqrt()
}

/// The median, averaging the two middle values of an even count.
pub(crate) fn median_of(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    values[middle]
}
