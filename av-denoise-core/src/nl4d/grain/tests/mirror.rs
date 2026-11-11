#![cfg(any(feature = "vulkan", feature = "metal"))]

use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    BUCKETS_PER_GROUP,
    CELL,
    CHUNK_FRAMES,
    CLIP_HIGH,
    CLIP_LOW,
    CONF_MIN,
    FLAT_RANGE,
    GROUPED_AUTOCOV_LEN,
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
use crate::noise::TEMPORAL_NOISE_BLOCK;
use crate::noise::colour::{COLOUR_BRIGHTNESS_STEP, COLOUR_MEAN_FIELDS};

/// One completed frame's inputs, each pixel holding `channels` interleaved samples.
pub(super) struct MirrorFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// The channel every sample is read from.
    pub channel: u32,
    /// Whether a cell's mean must sit inside the luma band to count.
    pub luma_band: bool,
    pub source_t: &'a [f32],
    pub source_next: &'a [f32],
    pub out_t: &'a [f32],
    /// `(dx, dy)` per motion block for `t` to `t + 1`.
    pub source_mv: &'a [(i32, i32)],
    pub source_conf: &'a [f32],
    /// Each colour block's luma, Cb and Cr means on the 8-bit scale, `None` when not held.
    pub luma_means: Option<&'a [f32]>,
    /// The colour blocks in one row, 0 for a frame that bins one-frame cells by its own mean.
    pub means_blocks_x: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step: u32,
    pub has_source: bool,
}

pub(super) struct MirrorRecord {
    pub hist: Vec<u32>,
    /// One autocovariance record per strength group.
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

/// The index of the measured channel's sample at pixel `(x, y)`.
fn sample_index(frame: &MirrorFrame, x: u32, y: u32) -> usize {
    let pixel = y * frame.width + x;
    (pixel * frame.channels + frame.channel) as usize
}

fn warped_index(frame: &MirrorFrame, x: u32, y: u32, shift: (i32, i32)) -> usize {
    let warped_x = clamp_coord(x as i32 + shift.0, frame.width);
    let warped_y = clamp_coord(y as i32 + shift.1, frame.height);
    sample_index(frame, warped_x, warped_y)
}

fn in_luma_band(frame: &MirrorFrame, mean: f32) -> bool {
    !frame.luma_band || (mean > LUMA_LOW && mean < LUMA_HIGH)
}

fn source_grain(frame: &MirrorFrame, x: u32, y: u32) -> f32 {
    let block = block_of(frame, x / CELL, y / CELL);
    let next_index = warped_index(frame, x, y, frame.source_mv[block]);
    let next = frame.source_next[next_index];
    let current_index = sample_index(frame, x, y);
    let current = frame.source_t[current_index];
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

fn luma_bin(mean: f32) -> usize {
    let bin = (mean * LUMA_BINS as f32) as usize;
    bin.min(LUMA_BINS - 1)
}

fn hist_slot(bin: usize, std: f32, edges: &[f32]) -> usize {
    let bucket = bucket_of(std, edges);
    bin * STD_BUCKETS + bucket
}

/// The brightness bin of a one-frame cell, `None` when its colour block's luma is not held.
///
/// A frame without a colour grid bins by the cell's denoised `mean`. Otherwise the cell reads the
/// colour block its origin falls in.
fn frame_bin(frame: &MirrorFrame, cell_x: u32, cell_y: u32, mean: f32) -> Option<usize> {
    if frame.means_blocks_x == 0 {
        let bin = luma_bin(mean);
        return Some(bin);
    }

    let means = frame.luma_means?;
    let block_x = cell_x * CELL / TEMPORAL_NOISE_BLOCK;
    let block_y = cell_y * CELL / TEMPORAL_NOISE_BLOCK;
    let block = block_y * frame.means_blocks_x + block_x;
    let block_luma = means[(block * COLOUR_MEAN_FIELDS) as usize];
    let bin = (block_luma / COLOUR_BRIGHTNESS_STEP) as usize;

    Some(bin.min(LUMA_BINS - 1))
}

/// Adds the cell's lag products to `record`, each pixel against its neighbour at every lag.
///
/// `mean` is the cell's mean grain, taken off every pixel and neighbour first.
fn add_lag_sums(frame: &MirrorFrame, cell_left: u32, cell_top: u32, mean: f32, record: &mut [f64]) {
    for y in cell_top..cell_top + CELL {
        for x in cell_left..cell_left + CELL {
            let centre = (source_grain(frame, x, y) - mean) as f64;

            for (lane, &(dy, dx)) in LAGS.iter().enumerate() {
                let neighbour_y = (y as i32 + dy) as u32;
                let neighbour_x = (x as i32 + dx) as u32;
                let neighbour = (source_grain(frame, neighbour_x, neighbour_y) - mean) as f64;
                record[lane] += centre * neighbour;
            }
        }
    }

    record[LAG_COUNT] += (CELL * CELL) as f64;
}

/// Runs the measure kernel's maths on the host.
pub(super) fn mirror_measure(frame: &MirrorFrame, edges: &[f32]) -> MirrorRecord {
    let cells_x = frame.width / CELL;
    let cells_y = frame.height / CELL;
    let mut hist = vec![0u32; 2 * HIST_LEN];
    let mut autocov = vec![0.0f64; GROUPED_AUTOCOV_LEN];

    for cell_y in 0..cells_y {
        for cell_x in 0..cells_x {
            let cell_left = cell_x * CELL;
            let cell_top = cell_y * CELL;
            let block = block_of(frame, cell_x, cell_y);
            let pixels = (CELL * CELL) as usize;
            let origin_index = sample_index(frame, cell_left, cell_top);
            let origin = frame.source_t[origin_index];
            let mut grain = Vec::with_capacity(pixels);
            let mut clean = Vec::with_capacity(pixels);
            let mut noisy = Vec::with_capacity(2 * pixels);
            let mut centred = Vec::with_capacity(pixels);

            for y in cell_top..cell_top + CELL {
                for x in cell_left..cell_left + CELL {
                    let index = sample_index(frame, x, y);
                    let pixel_grain = source_grain(frame, x, y);
                    grain.push(pixel_grain);
                    clean.push(frame.out_t[index]);
                    noisy.push(frame.source_t[index]);
                    centred.push(frame.source_t[index] - origin);

                    let next_index = warped_index(frame, x, y, frame.source_mv[block]);
                    noisy.push(frame.source_next[next_index]);
                }
            }

            let interior = cell_x >= 1 && cell_x + 2 <= cells_x && cell_y + 2 <= cells_y;
            let grain_stats = stats_of(&grain);
            let clean_stats = stats_of(&clean);
            let (clean_low, clean_high) = range_of(&clean);
            let (noisy_low, noisy_high) = range_of(&noisy);
            let shared_ok = frame.has_source
                && clean_high - clean_low < FLAT_RANGE
                && in_luma_band(frame, clean_stats.mean)
                && noisy_low >= CLIP_LOW
                && noisy_high <= CLIP_HIGH;
            let source_ok =
                shared_ok && interior && frame.source_conf[block] >= CONF_MIN && grain_stats.std > STD_MIN;
            if source_ok {
                let bin = luma_bin(clean_stats.mean);
                let slot = hist_slot(bin, grain_stats.std, edges);
                hist[slot] += 1;

                let group = bucket_of(grain_stats.std, edges) / BUCKETS_PER_GROUP;
                let record = &mut autocov[group * AUTOCOV_LEN..(group + 1) * AUTOCOV_LEN];
                add_lag_sums(frame, cell_left, cell_top, grain_stats.mean, record);
            }

            let frame_stats = stats_of(&centred);
            let bin = frame_bin(frame, cell_x, cell_y, clean_stats.mean);
            let frame_ok = shared_ok && frame_stats.std > STD_MIN;
            if frame_ok && let Some(bin) = bin {
                let slot = hist_slot(bin, frame_stats.std, edges);
                hist[HIST_LEN + slot] += 1;
            }
        }
    }

    MirrorRecord { hist, autocov }
}

/// A luma stream with no motion, each frame with its denoised output.
pub(super) struct StillStream {
    pub width: u32,
    pub height: u32,
    /// The motion block size of the stream's vectors.
    pub step: u32,
    pub sources: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

impl StillStream {
    pub fn blocks_x(&self) -> u32 {
        self.width / self.step
    }

    pub fn blocks_y(&self) -> u32 {
        self.height / self.step
    }

    fn blocks(&self) -> usize {
        (self.blocks_x() * self.blocks_y()) as usize
    }
}

/// Runs [mirror_measure] over every frame of `stream`, each block with a zero vector at full
/// confidence, and sums each chunk of `CHUNK_FRAMES` frames.
pub(super) fn mirror_still_chunks(stream: &StillStream, edges: &[f32]) -> Vec<MirrorRecord> {
    let frame_count = stream.sources.len();
    let still_mv = vec![(0, 0); stream.blocks()];
    let full_conf = vec![1.0f32; stream.blocks()];
    let mut chunks: Vec<MirrorRecord> = Vec::new();

    for index in 0..frame_count {
        let has_source = index + 1 < frame_count;
        let next = if has_source { index + 1 } else { index };
        let frame = MirrorFrame {
            width: stream.width,
            height: stream.height,
            channels: 1,
            channel: 0,
            luma_band: true,
            source_t: &stream.sources[index],
            source_next: &stream.sources[next],
            out_t: &stream.outputs[index],
            source_mv: &still_mv,
            source_conf: &full_conf,
            luma_means: None,
            means_blocks_x: 0,
            blocks_x: stream.blocks_x(),
            blocks_y: stream.blocks_y(),
            step: stream.step,
            has_source,
        };
        let record = mirror_measure(&frame, edges);

        if index % CHUNK_FRAMES as usize == 0 {
            let empty = MirrorRecord {
                hist: vec![0; 2 * HIST_LEN],
                autocov: vec![0.0; GROUPED_AUTOCOV_LEN],
            };
            chunks.push(empty);
        }

        let chunk = chunks.last_mut().expect("a chunk is open");
        for (total, count) in chunk.hist.iter_mut().zip(record.hist) {
            *total += count;
        }

        for (total, sum) in chunk.autocov.iter_mut().zip(record.autocov) {
            *total += sum;
        }
    }

    chunks
}
