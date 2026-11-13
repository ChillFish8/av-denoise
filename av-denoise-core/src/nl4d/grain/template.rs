use std::collections::HashMap;

use super::consts::{AR_COEFFS, AR_OFFSETS, CELL, STD_MAX, TEMPLATE_SEEDS};
use super::gaussian::GAUSSIAN_SEQUENCE;

/// Pixels in one 8x8 cell.
pub(crate) const CELL_PIXELS: usize = (CELL * CELL) as usize;
/// The summed squared neighbour differences of a cell over this give its high-pass variance.
///
/// Each axis has 56 pairs, and half of each squared difference reads one sample's variance.
const HIGH_PASS_DIVISOR: f64 = 224.0;
/// The field std, in codes, up to which a rounded curve steps evenly.
const EVEN_GRID_CODES: f64 = 3.0;
/// The even step of a rounded curve, in codes.
const EVEN_GRID_STEP: f64 = 0.05;
/// The ratio between neighbouring field stds past the even part of a curve.
const GEOMETRIC_GRID_RATIO: f64 = 1.1;
/// How far past [STD_MAX] the median at the top of a curve reaches.
const CURVE_HEADROOM: f64 = 1.5;
/// The largest field std a curve reaches, the whole signal range.
const MAX_FIELD_STD: f64 = 1.0;
/// The fewest cells a curve reads, as many as the luma templates hold.
///
/// The smaller chroma templates repeat their cells, each copy on its own level offsets.
const CURVE_CELLS: usize = TEMPLATE_SEEDS as usize * LUMA_CELLS_PER_TEMPLATE;
/// The 8x8 cells in one luma template's interior.
const LUMA_CELLS_PER_TEMPLATE: usize = (LUMA_LAYOUT.interior_size / CELL as usize).pow(2);

/// The size of a grain template and the interior its statistics read.
#[derive(Copy, Clone)]
struct TemplateLayout {
    rows: usize,
    cols: usize,
    /// The first row and column of the interior, past the samples the AR filter never reaches.
    interior_start: usize,
    /// The side length of the square interior, a whole number of 8x8 cells.
    interior_size: usize,
}

/// The number of columns in a luma grain template.
pub(crate) const LUMA_TEMPLATE_COLS: usize = 82;

const LUMA_LAYOUT: TemplateLayout = TemplateLayout {
    rows: 73,
    cols: LUMA_TEMPLATE_COLS,
    interior_start: 9,
    interior_size: 64,
};

/// The 4:2:0 chroma template, whose 8x8 cells match the one-frame chroma cells.
const CHROMA_LAYOUT: TemplateLayout = TemplateLayout {
    rows: 38,
    cols: 44,
    interior_start: 6,
    interior_size: 32,
};

/// A chroma plane of an AV1 film grain template.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum ChromaPlane {
    Cb,
    Cr,
}

/// A plane of an AV1 film grain template.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum TemplatePlane {
    Luma,
    Chroma(ChromaPlane),
}

impl TemplatePlane {
    fn layout(self) -> TemplateLayout {
        match self {
            TemplatePlane::Luma => LUMA_LAYOUT,
            TemplatePlane::Chroma(_) => CHROMA_LAYOUT,
        }
    }

    fn template(self, coeffs: &[i32; AR_COEFFS], ar_shift: u32, seed: u16) -> Vec<i32> {
        match self {
            TemplatePlane::Luma => luma_template(coeffs, ar_shift, seed),
            TemplatePlane::Chroma(plane) => chroma_template(coeffs, ar_shift, seed, plane),
        }
    }
}

impl ChromaPlane {
    /// The mask AV1 applies to the grain seed before drawing this plane's samples.
    pub(crate) fn seed_mask(self) -> u16 {
        match self {
            ChromaPlane::Cb => 0xb524,
            ChromaPlane::Cr => 0x49d8,
        }
    }

    /// The plane's position in a chroma record, 0 for Cb and 1 for Cr.
    pub(crate) fn index(self) -> usize {
        match self {
            ChromaPlane::Cb => 0,
            ChromaPlane::Cr => 1,
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

/// The grain std of these weights' templates on `plane`.
///
/// It reads the interior of [TEMPLATE_SEEDS] templates.
pub(crate) fn template_sigma(coeffs: &[i32; AR_COEFFS], ar_shift: u32, plane: TemplatePlane) -> f64 {
    let interiors = template_interiors(coeffs, ar_shift, plane);
    let samples = interiors.concat();

    population_std(&samples)
}

/// The grain std of the one template these weights and `seed` make on `plane`.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "only tests read a single seed's strength")
)]
pub(crate) fn seed_sigma(coeffs: &[i32; AR_COEFFS], ar_shift: u32, plane: TemplatePlane, seed: u16) -> f64 {
    let layout = plane.layout();
    let template = plane.template(coeffs, ar_shift, seed);
    let interior = interior_of(&template, layout);

    population_std(&interior)
}

/// Field stds and the median one-frame high-pass std that template grain reads at each.
#[derive(Debug, Clone)]
pub(crate) struct HighPassCurve {
    /// The field stds, rising from 0, in normalised units.
    stds: Vec<f64>,
    /// The median high-pass std at each field std, never falling, in normalised units.
    medians: Vec<f64>,
}

impl HighPassCurve {
    /// The field std whose grain reads `median`, interpolated linearly between grid points.
    ///
    /// A median of 0 gives 0, and a median past the top gives the top field std.
    pub(crate) fn field_std(&self, median: f64) -> f64 {
        if median <= 0.0 {
            return 0.0;
        }

        let index = self.medians.partition_point(|&value| value < median);
        if index == 0 {
            return self.stds[0];
        }

        let last = self.stds.len() - 1;
        if index > last {
            return self.stds[last];
        }

        let low_median = self.medians[index - 1];
        let high_median = self.medians[index];
        let fraction = (median - low_median) / (high_median - low_median);
        let low_std = self.stds[index - 1];
        let high_std = self.stds[index];

        low_std + fraction * (high_std - low_std)
    }
}

/// The high-pass curve of these weights' template grain on `plane`.
///
/// At each grid field std, the [curve_cells] are scaled to it and cell `i` of `n` sits
/// `(i + 0.5) / n` codes above a whole code. With `code_step` set the cells round to whole codes.
/// The curve reads the median high-pass std over every cell, kept as a running maximum.
pub(crate) fn high_pass_curve(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    plane: TemplatePlane,
    code_step: Option<f64>,
) -> HighPassCurve {
    let cells = curve_cells(coeffs, ar_shift, plane);
    let offsets = even_offsets(cells.len());
    let unit_median = median_high_pass(&cells, 1.0, &offsets, None);
    let stds = curve_grid(unit_median, code_step);

    let mut medians = Vec::with_capacity(stds.len());
    let mut highest = 0.0f64;
    for &std in &stds {
        let median = median_high_pass(&cells, std, &offsets, code_step);
        highest = highest.max(median);
        medians.push(highest);
    }

    HighPassCurve { stds, medians }
}

/// The curve's field stds, from 0 to where the median reads [CURVE_HEADROOM] times [STD_MAX].
///
/// With a code step, they rise by [EVEN_GRID_STEP] codes up to [EVEN_GRID_CODES] codes and then
/// by [GEOMETRIC_GRID_RATIO] each. Without rounding the curve is a straight line, so its two ends
/// are enough. No field std passes [MAX_FIELD_STD].
fn curve_grid(unit_median: f64, code_step: Option<f64>) -> Vec<f64> {
    let reach = CURVE_HEADROOM * STD_MAX as f64 / unit_median;
    let top = reach.min(MAX_FIELD_STD);
    let Some(step) = code_step else {
        return vec![0.0, top];
    };

    let even_points = (EVEN_GRID_CODES / EVEN_GRID_STEP).round() as usize;
    let mut stds = Vec::with_capacity(even_points + 1);
    for point in 0..=even_points {
        let codes = point as f64 * EVEN_GRID_STEP;
        stds.push(codes * step);
    }

    let mut std = EVEN_GRID_CODES * step;
    while std < top {
        std *= GEOMETRIC_GRID_RATIO;
        stds.push(std.min(top));
    }

    stds
}

/// Level offsets spread evenly over one code, one per cell.
pub(crate) fn even_offsets(count: usize) -> Vec<f64> {
    let spacing = 1.0 / count as f64;

    (0..count).map(|index| (index as f64 + 0.5) * spacing).collect()
}

/// The median high-pass std of `cells` scaled to field std `std`.
///
/// Each cell sits its offset, in codes, above a whole code and rounds to whole codes `code_step`
/// apart. Without a code step nothing rounds and the offsets cancel.
pub(crate) fn median_high_pass(
    cells: &[[f64; CELL_PIXELS]],
    std: f64,
    offsets: &[f64],
    code_step: Option<f64>,
) -> f64 {
    let unit = code_step.unwrap_or(1.0);
    let gain = std / unit;

    let mut readings = Vec::with_capacity(cells.len());
    let mut levels = [0.0f64; CELL_PIXELS];
    for (cell, &offset) in cells.iter().zip(offsets) {
        for (level, &sample) in levels.iter_mut().zip(cell) {
            let scaled = gain * sample;
            *level = match code_step {
                Some(_) => (offset + scaled).round(),
                None => scaled,
            };
        }

        let reading = high_pass_std(&levels);
        readings.push(reading);
    }

    let median = median_of(&mut readings);

    median * unit
}

/// The high-pass std of one row-major 8x8 cell, from its neighbours' squared differences.
pub(crate) fn high_pass_std(cell: &[f64]) -> f64 {
    let side = CELL as usize;
    let mut sum_sq = 0.0f64;
    for y in 0..side {
        for x in 0..side {
            let here = cell[y * side + x];
            if x + 1 < side {
                let right = cell[y * side + x + 1] - here;
                sum_sq += right * right;
            }

            if y + 1 < side {
                let down = cell[(y + 1) * side + x] - here;
                sum_sq += down * down;
            }
        }
    }

    let variance = sum_sq / HIGH_PASS_DIVISOR;
    variance.sqrt()
}

/// The 8x8 cells of these weights' template interiors on `plane`, scaled to a field std of 1.
pub(crate) fn unit_cells(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    plane: TemplatePlane,
) -> Vec<[f64; CELL_PIXELS]> {
    let layout = plane.layout();
    let interiors = template_interiors(coeffs, ar_shift, plane);
    let samples = interiors.concat();
    let sigma = population_std(&samples);

    let mut cells = Vec::new();
    for interior in &interiors {
        let interior_cells = cells_of(interior, layout, sigma);
        cells.extend(interior_cells);
    }

    cells
}

/// The [unit_cells] repeated until there are at least [CURVE_CELLS].
pub(crate) fn curve_cells(
    coeffs: &[i32; AR_COEFFS],
    ar_shift: u32,
    plane: TemplatePlane,
) -> Vec<[f64; CELL_PIXELS]> {
    let cells = unit_cells(coeffs, ar_shift, plane);
    let copies = CURVE_CELLS.div_ceil(cells.len());

    cells.repeat(copies)
}

/// High-pass curves built for one fit, one per texture and plane.
pub(crate) struct CurveCache {
    /// The code step every curve rounds to, `None` for no rounding.
    code_step: Option<f64>,
    curves: HashMap<([i32; AR_COEFFS], u32, TemplatePlane), HighPassCurve>,
}

impl CurveCache {
    pub(crate) fn new(code_step: Option<f64>) -> Self {
        Self {
            code_step,
            curves: HashMap::new(),
        }
    }

    /// The curve of these weights on `plane`, built on its first use.
    pub(crate) fn curve(
        &mut self,
        coeffs: &[i32; AR_COEFFS],
        ar_shift: u32,
        plane: TemplatePlane,
    ) -> &HighPassCurve {
        let code_step = self.code_step;
        let key = (*coeffs, ar_shift, plane);

        self.curves
            .entry(key)
            .or_insert_with(|| high_pass_curve(coeffs, ar_shift, plane, code_step))
    }
}

/// The interiors of the [TEMPLATE_SEEDS] templates of these weights on `plane`.
fn template_interiors(coeffs: &[i32; AR_COEFFS], ar_shift: u32, plane: TemplatePlane) -> Vec<Vec<f64>> {
    let layout = plane.layout();
    let mut interiors = Vec::with_capacity(TEMPLATE_SEEDS as usize);
    for index in 0..TEMPLATE_SEEDS {
        let seed = template_seed(index);
        let template = plane.template(coeffs, ar_shift, seed);
        let interior = interior_of(&template, layout);
        interiors.push(interior);
    }

    interiors
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

/// The interior's 8x8 cells in raster order, each sample divided by `scale`.
fn cells_of(interior: &[f64], layout: TemplateLayout, scale: f64) -> Vec<[f64; CELL_PIXELS]> {
    let size = layout.interior_size;
    let side = CELL as usize;
    let per_side = size / side;
    let mut cells = Vec::with_capacity(per_side * per_side);
    for cell_y in 0..per_side {
        for cell_x in 0..per_side {
            let mut cell = [0.0f64; CELL_PIXELS];
            for y in 0..side {
                let start = (cell_y * side + y) * size + cell_x * side;
                let row = &interior[start..start + side];
                let slots = &mut cell[y * side..(y + 1) * side];
                for (slot, &value) in slots.iter_mut().zip(row) {
                    *slot = value / scale;
                }
            }

            cells.push(cell);
        }
    }

    cells
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
