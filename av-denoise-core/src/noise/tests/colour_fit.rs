use std::f64::consts::{LN_2, PI, TAU};

use cubecl::prelude::*;

use super::records::{BlockSpec, frame_records, hashed_unit};
use crate::noise::NOISE_CURVE_BINS;
use crate::noise::colour::{
    COLOUR_BRIGHTNESS_BINS,
    COLOUR_CELL_CB_SUM,
    COLOUR_CELL_COUNT,
    COLOUR_CELL_CR_SUM,
    COLOUR_CELL_HISTOGRAM,
    COLOUR_CELL_LOG_SUM,
    COLOUR_CELL_LUMA_SUM,
    COLOUR_CELL_WORDS,
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    COLOUR_CHROMA_GAIN_MAX,
    COLOUR_CHROMA_GAIN_MIN,
    COLOUR_FIT_BANDWIDTH,
    COLOUR_FIT_KNOT_PENALTY,
    COLOUR_FIT_KNOTS,
    COLOUR_FIT_LUMA_DEAD_ZONE,
    COLOUR_FIT_MIN_BLOCKS,
    COLOUR_FIT_PRIOR_SHARE,
    COLOUR_FIT_PULL,
    COLOUR_FIT_RIDGE,
    COLOUR_GAIN_BLOCK_RAMP,
    COLOUR_GAIN_NEUTRAL_RAMP,
    COLOUR_HISTOGRAM_BINS,
    COLOUR_HISTOGRAM_LOG_MIN,
    COLOUR_HISTOGRAM_LOG_STEP,
    COLOUR_HUE_BINS,
    COLOUR_KERNEL_CHROMA_STEP,
    COLOUR_KERNEL_LUMA_STEP,
    COLOUR_SATURATION_CENTRES,
    COLOUR_SATURATION_EDGES,
    CellFrame,
    ColourCells,
    ColourFit,
};
use crate::noise::derived::DerivedValues;
use crate::noise::history::{WindowBounds, history_capacity};
use crate::noise::kernels::colour_gain;
use crate::noise::temporal::temporal_stats_blocks;
use crate::tune::gpu::{bind, read_f32};
use crate::tune::tests::runtime::test_client;

/// The factor from a normalised 8-bit residual to codes.
const EIGHT_BIT: f32 = 255.0;
/// The width of every pooled test frame, 64 blocks across.
const FRAME_WIDTH: u32 = 1024;
/// The blocks one pooled test frame holds.
const FRAME_BLOCKS: usize = 64 * 64;
/// The most frames one test window pools.
const WINDOW_FRAMES: usize = 17;
const CELLS: usize = COLOUR_CELLS as usize;
const PER_BRIGHTNESS: usize = COLOUR_CELLS_PER_BRIGHTNESS as usize;
const HUES: usize = COLOUR_HUE_BINS as usize;
const KNOTS: usize = COLOUR_FIT_KNOTS as usize;
const LEVELS: usize = COLOUR_SATURATION_CENTRES.as_slice().len();

/// One block's mean colour on the 8-bit scale, chroma centred on 0, and each channel's grain in
/// codes.
#[derive(Debug, Clone, Copy)]
struct FitBlock {
    luma: f32,
    cb: f32,
    cr: f32,
    grain: [f32; 2],
}

impl FitBlock {
    /// A block near the centre of brightness bin `brightness`, saturation level `level` and hue
    /// sector `sector`, nudged by `jitter` between -1 and 1.
    fn in_cell(brightness: usize, level: usize, sector: usize, jitter: f32, grain: [f32; 2]) -> Self {
        let luma = (brightness as f32 + 0.5) * 16.0 + 4.0 * jitter;
        let saturation = match level {
            0 => 2.0 + jitter,
            _ => COLOUR_SATURATION_CENTRES[level] + jitter,
        };
        let sector_width = std::f32::consts::TAU / HUES as f32;
        let angle = -std::f32::consts::PI + (sector as f32 + 0.5 + 0.2 * jitter) * sector_width;

        Self {
            luma,
            cb: saturation * angle.cos(),
            cr: saturation * angle.sin(),
            grain,
        }
    }

    /// A block that never joins a cell, which pads a frame.
    fn padding() -> Self {
        Self {
            luma: 0.0,
            cb: 0.0,
            cr: 0.0,
            grain: [0.0; 2],
        }
    }
}

/// A seeded stream of values between 0 and 1, from SplitMix64.
struct SceneRng(u64);

impl SceneRng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;

        (mixed >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A log spread between -0.6 and 0.6, roughly normal with a deviation of 0.2.
    fn log_noise(&mut self) -> f32 {
        let total = self.unit() + self.unit() + self.unit();

        (total - 1.5) * 0.4
    }

    /// A value between -1 and 1.
    fn jitter(&mut self) -> f32 {
        let unit = self.unit();

        2.0 * unit - 1.0
    }
}

/// The fit's inputs and outputs for one pass, with the window it pooled.
struct FitRun {
    client: Client,
    planes: u32,
    cells: ColourCells,
    fit: ColourFit,
}

impl FitRun {
    /// Pools `blocks` into a fresh window, frame by frame, and fits it against the brightness curve
    /// `ratios`, which counts with `curve_valid`.
    fn new(
        planes: u32,
        luma: bool,
        blocks: &[FitBlock],
        ratios: [f32; NOISE_CURVE_BINS],
        curve_valid: bool,
    ) -> Self {
        let client = test_client();
        let cells = ColourCells::new(&client, planes, luma, history_capacity(1));
        let fit = ColourFit::new(&client, planes, luma);

        let frames: Vec<&[FitBlock]> = blocks.chunks(FRAME_BLOCKS).collect();
        assert!(frames.len() <= WINDOW_FRAMES, "the blocks fit one window");

        for (frame, frame_blocks) in frames.iter().enumerate() {
            pool_frame(&client, &cells, planes, frame_blocks, frame as u32);
        }

        let newest = frames.len() as u32 - 1;
        let centre = newest.min(8);
        let bounds = WindowBounds::around(centre, newest);
        cells.queue_window(&client, bounds);

        let ratio_bytes = f32::as_bytes(&ratios);
        let curve = client.create_from_slice(ratio_bytes);
        let derived_values = DerivedValues {
            curve_valid,
            ..DerivedValues::default()
        };
        let derived_words = derived_values.to_words();
        let derived_bytes = f32::as_bytes(&derived_words);
        let derived = client.create_from_slice(derived_bytes);
        let window = cells.window();
        fit.queue(&client, window, &curve, &derived);

        Self {
            client,
            planes,
            cells,
            fit,
        }
    }

    fn window(&self) -> Vec<f32> {
        let words = self.cells.words();
        read_f32(&self.client, self.cells.window(), words).expect("the window reads back")
    }

    fn gains(&self) -> Vec<f32> {
        let len = (self.planes * COLOUR_CELLS) as usize;
        read_f32(&self.client, self.fit.gains(), len).expect("the gains read back")
    }

    fn table(&self) -> Vec<f32> {
        let len = (self.planes * COLOUR_CELLS) as usize;
        read_f32(&self.client, self.fit.table(), len).expect("the table reads back")
    }

    /// The GPU lookup of plane `plane`'s gains at every query colour.
    fn looked_up(&self, plane: u32, queries: &[[f32; 3]]) -> Vec<f32> {
        let flat: Vec<f32> = queries.iter().flatten().copied().collect();
        let query_bytes = f32::as_bytes(&flat);
        let query_handle = self.client.create_from_slice(query_bytes);
        let count = queries.len() as u32;
        let output = self.client.empty(queries.len() * size_of::<f32>());
        let table_len = (self.planes * COLOUR_CELLS) as usize;

        unsafe {
            colour_gain_probe::launch_unchecked(
                &self.client,
                CubeCount::new_1d(count.div_ceil(64)),
                CubeDim::new_1d(64),
                bind(self.fit.gains(), table_len),
                bind(&query_handle, flat.len()),
                bind(&output, queries.len()),
                plane * COLOUR_CELLS,
                count,
            );
        }

        read_f32(&self.client, &output, queries.len()).expect("the looked up gains read back")
    }
}

/// Reads `table`'s gain at each query colour through [colour_gain].
#[cube(launch_unchecked)]
fn colour_gain_probe(table: &[f32], queries: &[f32], gains: &mut [f32], base: u32, #[comptime] count: u32) {
    let index = ABSOLUTE_POS_X;

    if index < count {
        let query = index * 3u32;
        let luma = queries[query as usize];
        let cb = queries[(query + 1u32) as usize];
        let cr = queries[(query + 2u32) as usize];

        gains[index as usize] = colour_gain(table, base, luma, cb, cr);
    }
}

/// Pools one frame of `blocks` as stream frame `frame`, padded to a full frame.
fn pool_frame(client: &Client, cells: &ColourCells, planes: u32, blocks: &[FitBlock], frame: u32) {
    let height = (FRAME_BLOCKS as u32 / 64) * 16;
    let (blocks_x, blocks_y) = temporal_stats_blocks(FRAME_WIDTH, height);
    let frame_len = (blocks_x * blocks_y) as usize;

    let mut padded = blocks.to_vec();
    padded.resize(frame_len, FitBlock::padding());

    let specs: Vec<BlockSpec> = padded
        .iter()
        .map(|block| {
            let mut spec = BlockSpec::uniform(0.0, block.grain[0] / EIGHT_BIT, 0.0);
            spec.sigma[1] = block.grain[1] / EIGHT_BIT;
            spec
        })
        .collect();
    let records = frame_records(FRAME_WIDTH, height, planes, &specs);
    let means: Vec<f32> = padded
        .iter()
        .flat_map(|block| [block.luma, block.cb, block.cr])
        .collect();

    let records_bytes = f32::as_bytes(&records);
    let means_bytes = f32::as_bytes(&means);
    let records_handle = client.create_from_slice(records_bytes);
    let means_handle = client.create_from_slice(means_bytes);
    let input = CellFrame {
        records: &records_handle,
        records_len: records.len(),
        means: &means_handle,
        width: FRAME_WIDTH,
        height,
        grain_scale: EIGHT_BIT,
    };

    cells.queue_frame(client, &input, frame);
}

// ---------------------------------------------------------------- host reference

/// One cell's kept blocks and level, judged against a ceiling.
#[derive(Debug, Clone, Copy)]
struct HostLevel {
    kept: f64,
    level: f64,
}

/// The levels of `cells` cells of `window`, leaving out the blocks above `ceilings`.
fn host_levels(window: &[f32], ceilings: &[f64], cells: usize, mean_level: bool) -> Vec<HostLevel> {
    let step = f64::from(COLOUR_HISTOGRAM_LOG_STEP);
    let log_min = f64::from(COLOUR_HISTOGRAM_LOG_MIN);
    let bins = COLOUR_HISTOGRAM_BINS as usize;

    (0..cells)
        .map(|cell| {
            let record = &window[cell * COLOUR_CELL_WORDS as usize..(cell + 1) * COLOUR_CELL_WORDS as usize];
            let histogram = &record[COLOUR_CELL_HISTOGRAM as usize..COLOUR_CELL_HISTOGRAM as usize + bins];
            let count = f64::from(record[COLOUR_CELL_COUNT as usize]);
            let log_sum = f64::from(record[COLOUR_CELL_LOG_SUM as usize]);
            let ceiling = ceilings[cell];

            let mut dropped = 0.0;
            let mut dropped_log = 0.0;
            let mut kept_counts = Vec::with_capacity(bins);
            for (bin, &bin_count) in histogram.iter().enumerate() {
                let low = log_min + bin as f64 * step;
                let high = low + step;
                let above = ((high - ceiling) / step).clamp(0.0, 1.0);
                let cut = low.max(ceiling);
                let bin_count = f64::from(bin_count);

                dropped += bin_count * above;
                dropped_log += bin_count * above * (cut + high) * 0.5;
                kept_counts.push((low, bin_count * (1.0 - above), 1.0 - above));
            }

            let kept = (count - dropped).max(0.0);
            let level = if kept <= 0.0 {
                0.0
            } else if mean_level {
                (log_sum - dropped_log) / kept.max(1.0)
            } else {
                let target = kept * 0.5;
                let mut below = 0.0;
                let mut level = 0.0;
                for (low, kept_count, kept_share) in kept_counts {
                    if kept_count > 0.0 && below + kept_count >= target {
                        level = low + (target - below) / kept_count * kept_share * step;
                        break;
                    }
                    below += kept_count;
                }
                level
            };

            HostLevel { kept, level }
        })
        .collect()
}

/// The colour at the centre of cell `cell`.
fn host_centre(cell: usize) -> [f64; 3] {
    let brightness = cell / PER_BRIGHTNESS;
    let ring = cell % PER_BRIGHTNESS;
    let luma = (brightness as f64 + 0.5) * 16.0;
    if ring == 0 {
        return [luma, 0.0, 0.0];
    }

    let level = (ring - 1) / HUES + 1;
    let sector = (ring - 1) % HUES;
    let saturation = f64::from(COLOUR_SATURATION_CENTRES[level]);
    let angle = -PI + (sector as f64 + 0.5) * TAU / HUES as f64;

    [luma, saturation * angle.cos(), saturation * angle.sin()]
}

fn host_index(brightness: usize, level: usize, sector: usize) -> usize {
    let ring = if level == 0 {
        0
    } else {
        1 + (level - 1) * HUES + sector
    };
    brightness * PER_BRIGHTNESS + ring
}

/// The table's value at a colour, interpolated between the 8 nearest cell centres.
fn host_lookup(table: &[f64], luma: f64, cb: f64, cr: f64) -> f64 {
    let last_brightness = f64::from(COLOUR_BRIGHTNESS_BINS - 1);
    let brightness_position = (luma / 16.0 - 0.5).clamp(0.0, last_brightness);
    let brightness_low = (brightness_position.floor() as usize).min(COLOUR_BRIGHTNESS_BINS as usize - 2);
    let brightness_fraction = brightness_position - brightness_low as f64;

    let centres: Vec<f64> = COLOUR_SATURATION_CENTRES
        .iter()
        .map(|&centre| f64::from(centre))
        .collect();
    let saturation = cb.hypot(cr);
    let level_position = if saturation <= centres[0] {
        0.0
    } else if saturation >= centres[LEVELS - 1] {
        (LEVELS - 1) as f64
    } else {
        let segment = centres.windows(2).position(|pair| saturation < pair[1]).unwrap();
        segment as f64 + (saturation - centres[segment]) / (centres[segment + 1] - centres[segment])
    };
    let level_low = (level_position.floor() as usize).min(LEVELS - 2);
    let level_fraction = level_position - level_low as f64;

    let hue_position = (cr.atan2(cb) + PI) / TAU * HUES as f64 - 0.5;
    let hue_floor = hue_position.floor();
    let hue_fraction = hue_position - hue_floor;
    let sector_low = (hue_floor as i64).rem_euclid(HUES as i64) as usize;

    let mut value = 0.0;
    for corner in 0..8 {
        let brightness_step = corner / 4;
        let level_step = (corner / 2) % 2;
        let sector_step = corner % 2;
        let pick = |fraction: f64, upper: usize| if upper == 1 { fraction } else { 1.0 - fraction };
        let weight = pick(brightness_fraction, brightness_step)
            * pick(level_fraction, level_step)
            * pick(hue_fraction, sector_step);
        let sector = (sector_low + sector_step) % HUES;
        let index = host_index(brightness_low + brightness_step, level_low + level_step, sector);

        value += weight * table[index];
    }

    value
}

/// The tent weights of mean luma `luma` on the brightness base's knots.
fn host_tents(luma: f64) -> (usize, f64) {
    let last = (KNOTS - 1) as f64;
    let position = (luma * last / 255.0).clamp(0.0, last);
    let low = (position.floor() as usize).min(KNOTS - 2);

    (low, position - low as f64)
}

fn host_base_curve(knots: &[f64], luma: f64) -> f64 {
    let (low, fraction) = host_tents(luma);
    knots[low] * (1.0 - fraction) + knots[low + 1] * fraction
}

/// Solves `matrix · x = right` by Gaussian elimination with partial pivoting.
fn host_solve(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Vec<f64> {
    let size = right.len();
    for column in 0..size {
        let pivot = (column..size)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot);
        right.swap(column, pivot);

        let pivot_row = matrix[column].clone();
        for row in column + 1..size {
            let factor = matrix[row][column] / pivot_row[column];
            let updates = matrix[row].iter_mut().zip(&pivot_row).skip(column);
            for (value, &pivot_value) in updates {
                *value -= factor * pivot_value;
            }
            right[row] -= factor * right[column];
        }
    }

    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let tail: f64 = (row + 1..size)
            .map(|inner| matrix[row][inner] * solution[inner])
            .sum();
        solution[row] = (right[row] - tail) / matrix[row][row];
    }

    solution
}

/// One plane's fitted model.
struct HostModel {
    /// The log grain at every cell centre.
    table: Vec<f64>,
    populated: bool,
    /// The blocks every cell keeps.
    blocks: f64,
}

fn host_model(window: &[f32], levels: &[HostLevel], plane: usize, luma: bool) -> HostModel {
    let words = COLOUR_CELL_WORDS as usize;
    let mut means = Vec::with_capacity(CELLS);
    let mut weights = Vec::with_capacity(CELLS);
    let mut cell_levels = Vec::with_capacity(CELLS);
    let mut blocks = 0.0;

    for cell in 0..CELLS {
        let record = &window[(plane * CELLS + cell) * words..(plane * CELLS + cell + 1) * words];
        let count = f64::from(record[COLOUR_CELL_COUNT as usize]).max(1.0);
        let mean = [
            f64::from(record[COLOUR_CELL_LUMA_SUM as usize]) / count,
            f64::from(record[COLOUR_CELL_CB_SUM as usize]) / count,
            f64::from(record[COLOUR_CELL_CR_SUM as usize]) / count,
        ];
        let level = levels[plane * CELLS + cell];
        let populated = level.kept >= f64::from(COLOUR_FIT_MIN_BLOCKS);

        means.push(mean);
        weights.push(if populated { level.kept } else { 0.0 });
        cell_levels.push(level.level);
        blocks += level.kept;
    }

    let populated: Vec<usize> = (0..CELLS).filter(|&cell| weights[cell] > 0.0).collect();
    if populated.is_empty() {
        return HostModel {
            table: vec![0.0; CELLS],
            populated: false,
            blocks,
        };
    }

    // The brightness base holds flat past the populated cells' luma span.
    let span_low = populated
        .iter()
        .map(|&cell| means[cell][0])
        .fold(f64::MAX, f64::min);
    let span_high = populated
        .iter()
        .map(|&cell| means[cell][0])
        .fold(f64::MIN, f64::max);
    let held_curve = |knots: &[f64], luma: f64| {
        let held = luma.clamp(span_low, span_high);
        host_base_curve(knots, held)
    };

    // Least squares weighted by the root count, with the second-difference penalty.
    let mut matrix = vec![vec![0.0; KNOTS]; KNOTS];
    let mut right = vec![0.0; KNOTS];
    for &cell in &populated {
        let (low, fraction) = host_tents(means[cell][0]);
        let tents = [(low, 1.0 - fraction), (low + 1, fraction)];
        for &(row, row_weight) in &tents {
            right[row] += weights[cell] * cell_levels[cell] * row_weight;
            for &(column, column_weight) in &tents {
                matrix[row][column] += weights[cell] * row_weight * column_weight;
            }
        }
    }

    let mean_root: f64 =
        populated.iter().map(|&cell| weights[cell].sqrt()).sum::<f64>() / populated.len() as f64;
    let penalty_scale = f64::from(COLOUR_FIT_KNOT_PENALTY).powi(2) * mean_root;
    for row in 0..KNOTS - 2 {
        let coefficients = [(row, 1.0), (row + 1, -2.0), (row + 2, 1.0)];
        for &(a, weight_a) in &coefficients {
            for &(b, weight_b) in &coefficients {
                matrix[a][b] += penalty_scale * weight_a * weight_b;
            }
        }
    }

    for (knot, row) in matrix.iter_mut().enumerate() {
        row[knot] *= 1.0 + f64::from(COLOUR_FIT_RIDGE);
    }

    let knots = host_solve(matrix, right);
    let residuals: Vec<f64> = (0..CELLS)
        .map(|cell| cell_levels[cell] - held_curve(&knots, means[cell][0]))
        .collect();
    let prior = f64::from(COLOUR_FIT_PRIOR_SHARE) * blocks;
    let bandwidth = f64::from(COLOUR_FIT_BANDWIDTH);
    let luma_step = f64::from(COLOUR_KERNEL_LUMA_STEP);
    let chroma_step = f64::from(COLOUR_KERNEL_CHROMA_STEP);

    let base = |query: [f64; 3]| {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for &cell in &populated {
            let luma_gap = (query[0] - means[cell][0]) / luma_step;
            let cb_gap = (query[1] - means[cell][1]) / chroma_step;
            let cr_gap = (query[2] - means[cell][2]) / chroma_step;
            let distance = luma_gap * luma_gap + cb_gap * cb_gap + cr_gap * cr_gap;
            let kernel = (-distance / (2.0 * bandwidth * bandwidth)).exp();
            numerator += kernel * weights[cell] * residuals[cell];
            denominator += kernel * weights[cell];
        }

        let lift = if denominator + prior > 0.0 {
            numerator / (denominator + prior)
        } else {
            0.0
        };
        held_curve(&knots, query[0]) + lift
    };

    let dead_zone = if luma {
        f64::from(COLOUR_FIT_LUMA_DEAD_ZONE)
    } else {
        0.0
    };
    let pull = f64::from(COLOUR_FIT_PULL);
    let mut moves = vec![0.0; CELLS];
    for &cell in &populated {
        let gap = cell_levels[cell] - base(means[cell]);
        let shrunk = gap.signum() * (gap.abs() - dead_zone).max(0.0);
        moves[cell] = weights[cell] / (weights[cell] + pull) * shrunk;
    }

    let has = |cell: usize| weights[cell] > 0.0;
    let table = (0..CELLS)
        .map(|cell| {
            let filled = if has(cell) {
                moves[cell]
            } else {
                let brightness = cell / PER_BRIGHTNESS;
                let ring = cell % PER_BRIGHTNESS;
                let mut neighbours: Vec<f64> = Vec::new();
                if brightness > 0 && has(cell - PER_BRIGHTNESS) {
                    neighbours.push(moves[cell - PER_BRIGHTNESS]);
                }
                if brightness + 1 < COLOUR_BRIGHTNESS_BINS as usize && has(cell + PER_BRIGHTNESS) {
                    neighbours.push(moves[cell + PER_BRIGHTNESS]);
                }

                let row = brightness * PER_BRIGHTNESS;
                if ring == 0 {
                    let ring_moves: Vec<f64> = (0..HUES)
                        .map(|sector| row + 1 + sector)
                        .filter(|&neighbour| has(neighbour))
                        .map(|neighbour| moves[neighbour])
                        .collect();
                    if !ring_moves.is_empty() {
                        neighbours.push(ring_moves.iter().sum::<f64>() / ring_moves.len() as f64);
                    }
                } else {
                    let level = (ring - 1) / HUES;
                    let sector = (ring - 1) % HUES;
                    let mut candidates = vec![if level == 0 { row } else { cell - HUES }];
                    if level + 2 < LEVELS {
                        candidates.push(cell + HUES);
                    }
                    let level_start = row + 1 + level * HUES;
                    candidates.push(level_start + (sector + 1) % HUES);
                    candidates.push(level_start + (sector + HUES - 1) % HUES);
                    for neighbour in candidates {
                        if has(neighbour) {
                            neighbours.push(moves[neighbour]);
                        }
                    }
                }

                if neighbours.is_empty() {
                    0.0
                } else {
                    neighbours.iter().sum::<f64>() / neighbours.len() as f64
                }
            };

            base(host_centre(cell)) + filled
        })
        .collect();

    HostModel {
        table,
        populated: true,
        blocks,
    }
}

/// The median log grain of histograms whose bins are moved by their shift, by bisection.
fn host_median(histograms: &[(Vec<f64>, f64)]) -> Option<f64> {
    let step = f64::from(COLOUR_HISTOGRAM_LOG_STEP);
    let log_min = f64::from(COLOUR_HISTOGRAM_LOG_MIN);
    let share = |position: f64| -> f64 {
        histograms
            .iter()
            .map(|(bins, shift)| {
                bins.iter()
                    .enumerate()
                    .map(|(bin, &count)| {
                        let low = log_min + bin as f64 * step + shift;
                        count * ((position - low) / step).clamp(0.0, 1.0)
                    })
                    .sum::<f64>()
            })
            .sum()
    };

    let total: f64 = histograms.iter().map(|(bins, _)| bins.iter().sum::<f64>()).sum();
    if total <= 0.0 {
        return None;
    }

    let mut low = -100.0;
    let mut high = 100.0;
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if share(middle) < total * 0.5 {
            low = middle;
        } else {
            high = middle;
        }
    }

    Some(0.5 * (low + high))
}

fn histogram_of(window: &[f32], record: usize) -> Vec<f64> {
    let start = record * COLOUR_CELL_WORDS as usize + COLOUR_CELL_HISTOGRAM as usize;
    let bins = &window[start..start + COLOUR_HISTOGRAM_BINS as usize];
    bins.iter().map(|&count| f64::from(count)).collect()
}

/// Every plane's log grain table and log gain table, from the window's sums.
fn host_reference(
    window: &[f32],
    planes: usize,
    luma: bool,
    ratios: [f32; NOISE_CURVE_BINS],
    curve_valid: bool,
) -> (Vec<f64>, Vec<f64>) {
    let cells = planes * CELLS;
    let open = vec![f64::from(f32::MAX); cells];
    let first_levels = host_levels(window, &open, cells, luma);

    let words = COLOUR_CELL_WORDS as usize;
    let mut ceilings = Vec::with_capacity(cells);
    for plane in 0..planes {
        let first_model = host_model(window, &first_levels, plane, luma);
        let table = first_model.table;
        for cell in 0..CELLS {
            let record = &window[(plane * CELLS + cell) * words..];
            let count = f64::from(record[COLOUR_CELL_COUNT as usize]);
            let divisor = count.max(1.0);
            let colour = [
                f64::from(record[COLOUR_CELL_LUMA_SUM as usize]) / divisor,
                f64::from(record[COLOUR_CELL_CB_SUM as usize]) / divisor,
                f64::from(record[COLOUR_CELL_CR_SUM as usize]) / divisor,
            ];
            let ceiling = if count > 0.0 {
                host_lookup(&table, colour[0], colour[1], colour[2]) + LN_2
            } else {
                f64::from(f32::MAX)
            };
            ceilings.push(ceiling);
        }
    }

    let second_levels = host_levels(window, &ceilings, cells, luma);
    let mut tables = Vec::with_capacity(cells);
    let mut gains = Vec::with_capacity(cells);
    for plane in 0..planes {
        let model = host_model(window, &second_levels, plane, luma);
        let table = model.table;
        let populated = model.populated;
        let shifts: Vec<f64> = ratios
            .iter()
            .map(|&ratio| {
                let usable = luma && curve_valid && ratio > 0.0;
                if usable { -f64::from(ratio).ln() } else { 0.0 }
            })
            .collect();

        let histograms: Vec<(Vec<f64>, f64)> = if luma {
            (0..COLOUR_BRIGHTNESS_BINS as usize)
                .map(|bin| (histogram_of(window, planes * CELLS + bin), shifts[bin]))
                .collect()
        } else {
            let mut merged = vec![0.0; COLOUR_HISTOGRAM_BINS as usize];
            for cell in 0..CELLS {
                let bins = histogram_of(window, plane * CELLS + cell);
                for (total, count) in merged.iter_mut().zip(bins) {
                    *total += count;
                }
            }
            vec![(merged, 0.0)]
        };
        let normaliser = host_median(&histograms);

        // The gains ramp in with the plane's kept blocks, and luma's also with its neutral blocks.
        let block_ramp = f64::from(COLOUR_GAIN_BLOCK_RAMP);
        let neutral_ramp = f64::from(COLOUR_GAIN_NEUTRAL_RAMP);
        let mut ramp = model.blocks / (model.blocks + block_ramp);
        if luma {
            let neutral: f64 = histograms.iter().map(|(bins, _)| bins.iter().sum::<f64>()).sum();
            ramp *= neutral / (neutral + neutral_ramp);
        }

        for (cell, &log_grain) in table.iter().enumerate() {
            let log_gain = match (populated, normaliser) {
                (true, Some(normaliser)) if luma => {
                    let brightness = cell / PER_BRIGHTNESS;
                    log_grain - normaliser + shifts[brightness]
                },
                (true, Some(normaliser)) => {
                    let min = f64::from(COLOUR_CHROMA_GAIN_MIN).ln();
                    let max = f64::from(COLOUR_CHROMA_GAIN_MAX).ln();
                    (log_grain - normaliser).clamp(min, max)
                },
                _ => 0.0,
            };
            gains.push(ramp * log_gain);
        }

        tables.extend(table);
    }

    (tables, gains)
}

// ---------------------------------------------------------------- scenes

/// Blocks at random colours across the whole table, with grain from `grain_of` and a seeded spread.
///
/// A share of the blocks carries 6 times the grain, like motion residual.
fn random_scene(count: usize, seed: u64, grain_of: impl Fn(f32, f32, f32) -> [f32; 2]) -> Vec<FitBlock> {
    let mut rng = SceneRng(seed);

    (0..count)
        .map(|_| {
            let luma = 22.0 + 216.0 * rng.unit();
            let saturation_unit = rng.unit();
            let saturation = 60.0 * saturation_unit * saturation_unit;
            let angle = TAU as f32 * rng.unit();
            let cb = saturation * angle.cos();
            let cr = saturation * angle.sin();
            let grain = grain_of(luma, cb, cr);
            let residual = if rng.unit() < 0.03 { 6.0 } else { 1.0 };
            let spread = rng.log_noise().exp() * residual;
            let second_spread = rng.log_noise().exp() * residual;

            FitBlock {
                luma,
                cb,
                cr,
                grain: [grain[0] * spread, grain[1] * second_spread],
            }
        })
        .collect()
}

/// Whether a colour sits in the planted cells, saturated colours in the first quarter of hues.
fn planted(cb: f32, cr: f32) -> bool {
    let saturation = cb.hypot(cr);
    let turn = (cr.atan2(cb) + std::f32::consts::PI) / std::f32::consts::TAU;
    let quarter = (turn * 4.0).floor() as u32 % 4;

    saturation >= COLOUR_SATURATION_EDGES[2] && quarter == 0
}

/// `per_cell` blocks in each of the neutral cell and the cells of saturation level 3 of each of
/// the brightness bins `bins`.
///
/// Each channel's grain is 2 and 1.2 codes times `factor_of` at the block's brightness bin and
/// chroma, with a spread. The neutral cells also hold motion residual at 8 times the grain.
fn cell_scene(
    per_cell: usize,
    seed: u64,
    bins: std::ops::Range<usize>,
    factor_of: impl Fn(usize, f32, f32) -> f32,
) -> Vec<FitBlock> {
    let mut rng = SceneRng(seed);
    let mut blocks = Vec::new();
    for brightness in bins {
        for ring in 0..1 + HUES {
            let (level, sector) = if ring == 0 { (0, 0) } else { (3, ring - 1) };
            for _ in 0..per_cell {
                let nudge = rng.jitter();
                let probe = FitBlock::in_cell(brightness, level, sector, nudge, [1.0; 2]);
                let factor = factor_of(brightness, probe.cb, probe.cr);
                let spread = rng.log_noise().exp();
                let second_spread = rng.log_noise().exp();
                let block = FitBlock {
                    grain: [2.0 * factor * spread, 1.2 * factor * second_spread],
                    ..probe
                };
                blocks.push(block);
            }
        }

        for _ in 0..per_cell / 20 {
            let nudge = rng.jitter();
            let block = FitBlock::in_cell(brightness, 0, 0, nudge, [16.0, 9.6]);
            blocks.push(block);
        }
    }

    blocks
}

/// Luma grain 3 times stronger in the planted cells.
fn planted_factor(_brightness: usize, cb: f32, cr: f32) -> f32 {
    if planted(cb, cr) { 3.0 } else { 1.0 }
}

fn flat_curve() -> [f32; NOISE_CURVE_BINS] {
    [1.0; NOISE_CURVE_BINS]
}

/// A curve that rises from 0.7 to 1.4 with brightness.
fn rising_curve() -> [f32; NOISE_CURVE_BINS] {
    std::array::from_fn(|bin| 0.7 + 0.7 * bin as f32 / (NOISE_CURVE_BINS - 1) as f32)
}

fn populated_cells(window: &[f32], plane: usize) -> Vec<usize> {
    let words = COLOUR_CELL_WORDS as usize;
    (0..CELLS)
        .filter(|&cell| window[(plane * CELLS + cell) * words + COLOUR_CELL_COUNT as usize] >= 40.0)
        .collect()
}

// ---------------------------------------------------------------- tests

#[test]
fn gpu_fit_matches_the_host_reference() {
    let luma_scene = random_scene(60_000, 11, |luma, cb, cr| {
        let warm = (cr - cb).max(0.0) / 40.0;
        let grain = (1.2 + luma / 200.0) * (1.0 + warm);
        [grain, grain]
    });
    let chroma_scene = random_scene(60_000, 23, |luma, cb, cr| {
        let cb_grain = 0.8 + 0.6 * (cb.max(0.0) / 30.0) + luma / 400.0;
        let cr_grain = 0.6 + 0.9 * (cr.abs() / 40.0);
        [cb_grain, cr_grain]
    });

    let cases = [
        (1u32, true, &luma_scene, rising_curve(), true),
        (1u32, true, &luma_scene, rising_curve(), false),
        (2u32, false, &chroma_scene, flat_curve(), false),
    ];
    for (planes, luma, scene, ratios, curve_valid) in cases {
        let run = FitRun::new(planes, luma, scene, ratios, curve_valid);
        let window = run.window();
        let (tables, gains) = host_reference(&window, planes as usize, luma, ratios, curve_valid);
        let gpu_tables = run.table();
        let gpu_gains = run.gains();

        for (cell, (&gpu, &host)) in gpu_tables.iter().zip(&tables).enumerate() {
            let gap = (f64::from(gpu) - host).abs();
            assert!(
                gap <= 1.0e-4,
                "luma {luma}: table cell {cell} is {gpu} against {host}"
            );
        }
        for (cell, (&gpu, &host)) in gpu_gains.iter().zip(&gains).enumerate() {
            let gap = (f64::from(gpu) - host).abs();
            assert!(
                gap <= 1.0e-4,
                "luma {luma}: log gain {cell} is {gpu} against {host}"
            );
        }

        let queries: Vec<[f32; 3]> = (0..512u32)
            .map(|index| {
                let luma_code = 255.0 * hashed_unit(91, index);
                let cb = 140.0 * hashed_unit(92, index) - 70.0;
                let cr = 140.0 * hashed_unit(93, index) - 70.0;
                [luma_code, cb, cr]
            })
            .collect();
        for plane in 0..planes as usize {
            let looked_up = run.looked_up(plane as u32, &queries);
            let plane_gains = &gains[plane * CELLS..(plane + 1) * CELLS];
            for (query, &gpu) in queries.iter().zip(&looked_up) {
                let host_log = host_lookup(plane_gains, query[0].into(), query[1].into(), query[2].into());
                let host = host_log.exp();
                let gap = (f64::from(gpu) - host).abs();
                assert!(
                    gap <= 1.0e-4 * host.max(1.0),
                    "lookup at {query:?} is {gpu} against {host}"
                );
            }
        }
    }
}

#[test]
fn the_fit_recovers_a_planted_colour_field() {
    let scene = cell_scene(220, 5, 2..14, planted_factor);
    let run = FitRun::new(1, true, &scene, flat_curve(), true);
    let window = run.window();
    let gains = run.gains();

    let cells = populated_cells(&window, 0);
    assert!(cells.len() >= 100, "the scene populates {} cells", cells.len());

    // The scene is small next to a real window, so the ramp keeps a visible share of each gain.
    let words = COLOUR_CELL_WORDS as usize;
    let count_of = |record: usize| f64::from(window[record * words + COLOUR_CELL_COUNT as usize]);
    let blocks: f64 = (0..CELLS).map(count_of).sum();
    let neutral: f64 = (CELLS..CELLS + COLOUR_BRIGHTNESS_BINS as usize)
        .map(count_of)
        .sum();
    let block_share = blocks / (blocks + f64::from(COLOUR_GAIN_BLOCK_RAMP));
    let neutral_share = neutral / (neutral + f64::from(COLOUR_GAIN_NEUTRAL_RAMP));
    let ramp = block_share * neutral_share;
    // The dead zone holds every coloured cell that much short of its planted level.
    let tolerance = 0.07 + f64::from(COLOUR_FIT_LUMA_DEAD_ZONE);

    for cell in cells {
        let centre = host_centre(cell);
        let planted_gain: f64 = if planted(centre[1] as f32, centre[2] as f32) {
            3.0
        } else {
            1.0
        };
        let expected = (ramp * planted_gain.ln()).exp();
        let gain = f64::from(gains[cell]).exp();
        let gap = (gain.ln() - expected.ln()).abs();
        assert!(
            gap <= tolerance,
            "cell {cell} at {centre:?} has gain {gain}, planted {expected}"
        );
    }
}

#[test]
fn a_field_without_colour_effect_gives_gains_near_one() {
    let scene = cell_scene(300, 37, 2..14, |_, _, _| 1.0);
    let luma_run = FitRun::new(1, true, &scene, flat_curve(), true);
    let chroma_run = FitRun::new(2, false, &scene, flat_curve(), false);

    for (run, planes) in [(&luma_run, 1usize), (&chroma_run, 2)] {
        let window = run.window();
        let gains = run.gains();

        for plane in 0..planes {
            let cells = populated_cells(&window, plane);
            assert!(cells.len() >= 100, "the scene populates {} cells", cells.len());

            let mut total_gap = 0.0;
            for &cell in &cells {
                let gain = f64::from(gains[plane * CELLS + cell]).exp();
                let gap = (gain - 1.0).abs();
                assert!(gap <= 0.1, "plane {plane} cell {cell} has gain {gain}");
                total_gap += gap;
            }

            let mean_gap = total_gap / cells.len() as f64;
            assert!(
                mean_gap <= 0.055,
                "plane {plane} gains stray {mean_gap} from 1 on average"
            );
        }
    }
}

#[test]
fn a_thin_scene_gives_gains_near_one() {
    let scene = cell_scene(5, 9, 2..14, planted_factor);
    let luma_run = FitRun::new(1, true, &scene, flat_curve(), true);
    let chroma_run = FitRun::new(2, false, &scene, flat_curve(), false);

    for run in [&luma_run, &chroma_run] {
        let gains = run.gains();
        for (cell, &log_gain) in gains.iter().enumerate() {
            assert_eq!(
                log_gain, 0.0,
                "cell {cell} of a thin scene has log gain {log_gain}"
            );
        }
    }
}

#[test]
fn a_scene_with_one_or_two_brightness_bins_keeps_gains_near_one_elsewhere() {
    // Grain rises between bins 7 and 8, and the curve follows it and holds flat past them.
    let scene = cell_scene(
        300,
        61,
        7..9,
        |brightness, _, _| {
            if brightness == 7 { 0.9 } else { 1.1 }
        },
    );
    let ratios: [f32; NOISE_CURVE_BINS] = std::array::from_fn(|bin| if bin < 8 { 0.9 } else { 1.1 });
    let run = FitRun::new(1, true, &scene, ratios, true);
    let gains = run.gains();

    let far_bins = (0..5).chain(11..COLOUR_BRIGHTNESS_BINS as usize);
    for brightness in far_bins {
        for ring in 0..PER_BRIGHTNESS {
            let cell = brightness * PER_BRIGHTNESS + ring;
            let log_gain = gains[cell];
            let within = log_gain.abs() <= COLOUR_FIT_LUMA_DEAD_ZONE;
            assert!(within, "far cell {cell} has log gain {log_gain}");
        }
    }
}

#[test]
fn a_single_neutral_block_barely_moves_the_luma_gains() {
    let mut scene = cell_scene(220, 71, 2..14, planted_factor);
    scene.retain(|block| block.cb.hypot(block.cr) >= 8.0);

    let neutral = FitBlock::in_cell(8, 0, 0, 0.0, [2.0, 1.2]);
    scene.push(neutral);

    let run = FitRun::new(1, true, &scene, flat_curve(), true);
    let gains = run.gains();

    // One neutral block keeps 1/257 of each log gain, and the planted gains are about log 3.
    for (cell, &log_gain) in gains.iter().enumerate() {
        assert!(log_gain.abs() <= 0.01, "cell {cell} has log gain {log_gain}");
    }

    let moved = gains.iter().any(|&log_gain| log_gain != 0.0);
    assert!(moved, "the luma gains ramp in rather than switch off");
}

#[test]
fn chroma_gains_preserve_the_plane_average() {
    let scene = random_scene(60_000, 53, |luma, cb, cr| {
        let cb_grain = 1.0 + 1.5 * (cb.max(0.0) / 40.0) + luma / 300.0;
        let cr_grain = 0.7 + 1.2 * ((-cr).max(0.0) / 40.0);
        [cb_grain, cr_grain]
    });
    let run = FitRun::new(2, false, &scene, flat_curve(), false);
    let gains = run.gains();

    // Dividing every block's grain by its gain keeps the plane's median grain.
    let measured: Vec<&FitBlock> = scene
        .iter()
        .filter(|block| block.luma > 20.0 && block.luma < 240.0)
        .collect();
    let queries: Vec<[f32; 3]> = measured
        .iter()
        .map(|block| [block.luma, block.cb, block.cr])
        .collect();
    for plane in 0..2u32 {
        let block_gains = run.looked_up(plane, &queries);
        let mut grains: Vec<f64> = measured
            .iter()
            .map(|block| f64::from(block.grain[plane as usize]))
            .collect();
        let mut evened: Vec<f64> = grains
            .iter()
            .zip(&block_gains)
            .map(|(&grain, &gain)| grain / f64::from(gain))
            .collect();
        grains.sort_by(f64::total_cmp);
        evened.sort_by(f64::total_cmp);

        let median = grains[grains.len() / 2];
        let evened_median = evened[evened.len() / 2];
        let shift = (evened_median / median).ln();
        assert!(
            shift.abs() <= 0.05,
            "plane {plane}'s median grain moves from {median} to {evened_median}"
        );
    }

    // Each plane keeps its own gains. Cb grain rises on blue, while Cr grain rises on green.
    let blue = [128.0f64, 40.0, 0.0];
    let cb_table: Vec<f64> = gains[..CELLS].iter().map(|&gain| f64::from(gain)).collect();
    let cr_table: Vec<f64> = gains[CELLS..].iter().map(|&gain| f64::from(gain)).collect();
    let cb_on_blue = host_lookup(&cb_table, blue[0], blue[1], blue[2]);
    let cr_on_blue = host_lookup(&cr_table, blue[0], blue[1], blue[2]);
    assert!(
        cb_on_blue - cr_on_blue >= 0.3,
        "blue reads Cb {cb_on_blue} against Cr {cr_on_blue}"
    );
}
