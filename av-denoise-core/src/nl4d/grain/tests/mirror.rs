#![cfg(any(feature = "vulkan", feature = "metal"))]

use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    CELL,
    CLIP_HIGH,
    CLIP_LOW,
    CONF_MIN,
    FLAT_RANGE,
    HIST_LEN,
    LAG_COUNT,
    LAGS,
    LUMA_BINS,
    LUMA_HIGH,
    LUMA_LOW,
    STD_BUCKETS,
    STD_MIN,
};
use crate::nl4d::grain::fit::bucket_of;

/// One completed frame's inputs, luma only, with `stored_ch = 1`.
pub(super) struct MirrorFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub source_t: &'a [f32],
    pub source_next: &'a [f32],
    pub out_t: &'a [f32],
    pub out_prev: &'a [f32],
    /// `(dx, dy)` per motion block for `t` to `t + 1`.
    pub source_mv: &'a [(i32, i32)],
    pub source_conf: &'a [f32],
    /// `(dx, dy)` per motion block for `t - 1` to `t`.
    pub kept_mv: &'a [(i32, i32)],
    pub kept_conf: &'a [f32],
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step: u32,
    pub has_source: bool,
    pub has_kept: bool,
}

pub(super) struct MirrorRecord {
    pub hist: Vec<u32>,
    pub autocov: Vec<f64>,
}

struct BlockStats {
    mean: f32,
    std: f32,
}

fn block_of(frame: &MirrorFrame, cell_x: u32, cell_y: u32) -> usize {
    let block_x = (cell_x * CELL / frame.step).min(frame.blocks_x - 1);
    let block_y = (cell_y * CELL / frame.step).min(frame.blocks_y - 1);
    (block_y * frame.blocks_x + block_x) as usize
}

fn clamp_coord(value: i32, limit: u32) -> u32 {
    value.clamp(0, limit as i32 - 1) as u32
}

fn warped_index(frame: &MirrorFrame, x: u32, y: u32, shift: (i32, i32)) -> usize {
    let warped_x = clamp_coord(x as i32 + shift.0, frame.width);
    let warped_y = clamp_coord(y as i32 + shift.1, frame.height);
    (warped_y * frame.width + warped_x) as usize
}

fn source_grain(frame: &MirrorFrame, x: u32, y: u32) -> f32 {
    let block = block_of(frame, x / CELL, y / CELL);
    let next_index = warped_index(frame, x, y, frame.source_mv[block]);
    let next = frame.source_next[next_index];
    let current = frame.source_t[(y * frame.width + x) as usize];
    (next - current) * std::f32::consts::FRAC_1_SQRT_2
}

fn stats_of(values: &[f32]) -> BlockStats {
    let sum: f32 = values.iter().sum();
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    let count = values.len() as f32;
    let variance = (sum_sq - sum * sum / count) / (count - 1.0);

    BlockStats {
        mean: sum / count,
        std: variance.max(0.0).sqrt(),
    }
}

fn range_of(values: &[f32]) -> (f32, f32) {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (low, high)
}

fn hist_slot(mean: f32, std: f32, edges: &[f32]) -> usize {
    let bin = ((mean * LUMA_BINS as f32) as usize).min(LUMA_BINS - 1);
    let bucket = bucket_of(std, edges);
    bin * STD_BUCKETS + bucket
}

/// Adds the cell's lag products to `autocov`, each pixel against its neighbour at every lag.
fn add_lag_sums(frame: &MirrorFrame, x0: u32, y0: u32, autocov: &mut [f64]) {
    for y in y0..y0 + CELL {
        for x in x0..x0 + CELL {
            let centre = source_grain(frame, x, y) as f64;

            for (lane, &(dy, dx)) in LAGS.iter().enumerate() {
                let neighbour_y = (y as i32 + dy) as u32;
                let neighbour_x = (x as i32 + dx) as u32;
                let neighbour = source_grain(frame, neighbour_x, neighbour_y) as f64;
                autocov[lane] += centre * neighbour;
            }
        }
    }

    autocov[LAG_COUNT] += (CELL * CELL) as f64;
}

/// Runs the measure kernel's maths on the host.
pub(super) fn mirror_measure(frame: &MirrorFrame, edges: &[f32]) -> MirrorRecord {
    let cells_x = frame.width / CELL;
    let cells_y = frame.height / CELL;
    let mut hist = vec![0u32; 2 * HIST_LEN];
    let mut autocov = vec![0.0f64; AUTOCOV_LEN];

    for cell_y in 0..cells_y {
        for cell_x in 0..cells_x {
            let x0 = cell_x * CELL;
            let y0 = cell_y * CELL;
            let block = block_of(frame, cell_x, cell_y);
            let pixels = (CELL * CELL) as usize;
            let mut grain = Vec::with_capacity(pixels);
            let mut clean = Vec::with_capacity(pixels);
            let mut noisy = Vec::with_capacity(2 * pixels);
            let mut kept = Vec::with_capacity(pixels);
            let mut prev = Vec::with_capacity(pixels);

            for y in y0..y0 + CELL {
                for x in x0..x0 + CELL {
                    let index = (y * frame.width + x) as usize;
                    grain.push(source_grain(frame, x, y));
                    clean.push(frame.out_t[index]);
                    noisy.push(frame.source_t[index]);

                    let next_index = warped_index(frame, x, y, frame.source_mv[block]);
                    noisy.push(frame.source_next[next_index]);

                    let kept_index = warped_index(frame, x, y, frame.kept_mv[block]);
                    let warped = frame.out_t[kept_index];
                    let previous = frame.out_prev[index];
                    kept.push((warped - previous) * std::f32::consts::FRAC_1_SQRT_2);
                    prev.push(previous);
                }
            }

            let interior = cell_x >= 1 && cell_x + 2 <= cells_x && cell_y + 2 <= cells_y;
            let grain_stats = stats_of(&grain);
            let clean_stats = stats_of(&clean);
            let (clean_low, clean_high) = range_of(&clean);
            let (noisy_low, noisy_high) = range_of(&noisy);
            let source_ok = frame.has_source
                && interior
                && frame.source_conf[block] >= CONF_MIN
                && clean_high - clean_low < FLAT_RANGE
                && clean_stats.mean > LUMA_LOW
                && clean_stats.mean < LUMA_HIGH
                && noisy_low >= CLIP_LOW
                && noisy_high <= CLIP_HIGH
                && grain_stats.std > STD_MIN;
            if source_ok {
                let slot = hist_slot(clean_stats.mean, grain_stats.std, edges);
                hist[slot] += 1;
                add_lag_sums(frame, x0, y0, &mut autocov);
            }

            let kept_stats = stats_of(&kept);
            let prev_stats = stats_of(&prev);
            let (prev_low, prev_high) = range_of(&prev);
            let kept_ok = frame.has_kept
                && frame.kept_conf[block] >= CONF_MIN
                && prev_high - prev_low < FLAT_RANGE
                && prev_stats.mean > LUMA_LOW
                && prev_stats.mean < LUMA_HIGH
                && prev_low >= CLIP_LOW
                && prev_high <= CLIP_HIGH
                && kept_stats.std > STD_MIN;
            if kept_ok {
                let slot = hist_slot(prev_stats.mean, kept_stats.std, edges);
                hist[HIST_LEN + slot] += 1;
            }
        }
    }

    MirrorRecord { hist, autocov }
}
