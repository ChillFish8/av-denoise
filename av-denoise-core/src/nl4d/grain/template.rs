use super::consts::{AR_COEFFS, AR_OFFSETS, CELL, TEMPLATE_SEEDS};
use super::gaussian::GAUSSIAN_SEQUENCE;

/// The size of a grain template and the interior its statistics read.
#[derive(Copy, Clone)]
struct TemplateLayout {
    rows: usize,
    cols: usize,
    /// The first row and column of the interior, past the samples the AR filter never reaches.
    interior_start: usize,
    /// The side length of the square interior.
    interior_size: usize,
}

const LUMA_LAYOUT: TemplateLayout = TemplateLayout {
    rows: 73,
    cols: 82,
    interior_start: 9,
    interior_size: 64,
};

#[cfg_attr(not(test), expect(dead_code, reason = "only the chroma grain tests read it"))]
const CHROMA_LAYOUT: TemplateLayout = TemplateLayout {
    rows: 38,
    cols: 44,
    interior_start: 6,
    interior_size: 32,
};

/// A chroma plane of an AV1 film grain template.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(not(test), expect(dead_code, reason = "only the chroma grain tests read it"))]
pub(crate) enum ChromaPlane {
    Cb,
    Cr,
}

#[cfg_attr(not(test), expect(dead_code, reason = "only the chroma grain tests read it"))]
impl ChromaPlane {
    /// The mask AV1 applies to the grain seed before drawing this plane's samples.
    pub(crate) fn seed_mask(self) -> u16 {
        match self {
            ChromaPlane::Cb => 0xb524,
            ChromaPlane::Cr => 0x49d8,
        }
    }
}

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
    filtered_template(coeffs, ar_shift, seed, LUMA_LAYOUT)
}

/// Builds the 38x44 4:2:0 chroma grain template an 8-bit AV1 decoder makes for these weights.
///
/// The luma weight is taken as zero, so the template depends on the chroma samples alone.
#[cfg_attr(not(test), expect(dead_code, reason = "only the chroma grain tests read it"))]
pub(crate) fn chroma_template(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    seed: u16,
    plane: ChromaPlane,
) -> Vec<i32> {
    let plane_seed = seed ^ plane.seed_mask();
    filtered_template(coeffs, ar_shift, plane_seed, CHROMA_LAYOUT)
}

/// Draws a template of rounded Gaussian samples and runs the AR filter over it in raster order.
fn filtered_template(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    seed: u16,
    layout: TemplateLayout,
) -> Vec<i32> {
    let cols = layout.cols;
    let mut random = Av1Random::new(seed);
    let mut grain = vec![0i32; layout.rows * cols];
    for sample in grain.iter_mut() {
        let index = random.next(11) as usize;
        let gaussian = GAUSSIAN_SEQUENCE[index] as i32;
        *sample = round2(gaussian, 4);
    }

    for y in 3..layout.rows {
        for x in 3..cols - 3 {
            let mut total = 0i32;
            for (&(dy, dx), &coeff) in AR_OFFSETS.iter().zip(coeffs.iter()) {
                let neighbour_y = (y as i32 + dy) as usize;
                let neighbour_x = (x as i32 + dx) as usize;
                total += coeff * grain[neighbour_y * cols + neighbour_x];
            }

            let filtered = grain[y * cols + x] + round2(total, ar_shift);
            grain[y * cols + x] = filtered.clamp(-128, 127);
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
/// Both read the 64x64 interior of [TEMPLATE_SEEDS] templates. A block's std removes its own mean.
pub(crate) fn template_stats(coeffs: &[i32; AR_COEFFS], ar_shift: u32) -> (f64, f64) {
    let build = |seed: u16| luma_template(coeffs, ar_shift, seed);
    stats_over_seeds(LUMA_LAYOUT, build)
}

/// The grain std of these weights' chroma templates, and the median std of their 8x8 blocks.
///
/// Both read the 32x32 interior of [TEMPLATE_SEEDS] templates. A block's std removes its own mean.
#[cfg_attr(not(test), expect(dead_code, reason = "only the chroma grain tests read it"))]
pub(crate) fn chroma_template_stats(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    plane: ChromaPlane,
) -> (f64, f64) {
    let build = |seed: u16| chroma_template(coeffs, ar_shift, seed, plane);
    stats_over_seeds(CHROMA_LAYOUT, build)
}

fn stats_over_seeds(layout: TemplateLayout, build: impl Fn(u16) -> Vec<i32>) -> (f64, f64) {
    let interior_size = layout.interior_size;
    let capacity = TEMPLATE_SEEDS as usize * interior_size * interior_size;
    let mut samples = Vec::with_capacity(capacity);
    let mut block_stds = Vec::new();
    for index in 0..TEMPLATE_SEEDS {
        let seed = template_seed(index);
        let template = build(seed);
        let interior = interior_of(&template, layout);
        let interior_block_stds = block_stds_of(&interior, interior_size);
        block_stds.extend(interior_block_stds);
        samples.extend(interior);
    }

    let sigma = population_std(&samples);
    let median = median_of(&mut block_stds);
    (sigma, median)
}

fn interior_of(template: &[i32], layout: TemplateLayout) -> Vec<f64> {
    let start = layout.interior_start;
    let size = layout.interior_size;
    let mut interior = Vec::with_capacity(size * size);
    for y in start..start + size {
        let row_start = y * layout.cols + start;
        let row = &template[row_start..row_start + size];
        let row_values = row.iter().map(|&value| value as f64);
        interior.extend(row_values);
    }

    interior
}

fn block_stds_of(interior: &[f64], interior_size: usize) -> Vec<f64> {
    let cell = CELL as usize;
    let blocks_per_side = interior_size / cell;
    let mut stds = Vec::with_capacity(blocks_per_side * blocks_per_side);
    for block_y in 0..blocks_per_side {
        for block_x in 0..blocks_per_side {
            let mut block = Vec::with_capacity(cell * cell);
            for y in 0..cell {
                let start = (block_y * cell + y) * interior_size + block_x * cell;
                block.extend_from_slice(&interior[start..start + cell]);
            }

            let block_std = sample_std(&block);
            stds.push(block_std);
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
