use std::f32::consts::{PI, TAU};

use cubecl::prelude::*;

use super::sample::residual_sigma;
use crate::noise::NOISE_CHANNELS;
use crate::noise::colour::{
    COLOUR_BRIGHTNESS_BINS,
    COLOUR_BRIGHTNESS_STEP,
    COLOUR_CELL_CB_SUM,
    COLOUR_CELL_COUNT,
    COLOUR_CELL_CR_SUM,
    COLOUR_CELL_HISTOGRAM,
    COLOUR_CELL_LOG_SUM,
    COLOUR_CELL_LUMA_SUM,
    COLOUR_CELL_WORDS,
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    COLOUR_CHROMA_CENTRE,
    COLOUR_CHROMA_GRAIN_FLOOR,
    COLOUR_HISTOGRAM_BINS,
    COLOUR_HISTOGRAM_LOG_MAX,
    COLOUR_HISTOGRAM_LOG_MIN,
    COLOUR_HISTOGRAM_LOG_STEP,
    COLOUR_HUE_BINS,
    COLOUR_LEVEL_FIELDS,
    COLOUR_LOG_FIXED,
    COLOUR_LUMA_GRAIN_FLOOR,
    COLOUR_LUMA_MAX,
    COLOUR_LUMA_MIN,
    COLOUR_MEAN_FIELDS,
    COLOUR_MEAN_FIXED,
    COLOUR_MEAN_THREADS,
    COLOUR_NEUTRAL_SATURATION,
    COLOUR_SATURATION_EDGES,
};
use crate::noise::temporal::{STATIC_GATE, TEMPORAL_NOISE_BLOCK, temporal_stats_record_len};

const LAST_BRIGHTNESS: f32 = (COLOUR_BRIGHTNESS_BINS - 1) as f32;
const HUE_SECTORS: f32 = COLOUR_HUE_BINS as f32;
const LAST_HISTOGRAM_BIN: f32 = (COLOUR_HISTOGRAM_BINS - 1) as f32;
const SATURATION_EDGE_1: f32 = COLOUR_SATURATION_EDGES[0];
const SATURATION_EDGE_2: f32 = COLOUR_SATURATION_EDGES[1];
const SATURATION_EDGE_3: f32 = COLOUR_SATURATION_EDGES[2];
const SATURATION_EDGE_4: f32 = COLOUR_SATURATION_EDGES[3];
const SATURATION_EDGE_5: f32 = COLOUR_SATURATION_EDGES[4];
/// The columns of a block-mean cube's thread grid, one block row wide.
const MEAN_COLUMNS: u32 = TEMPORAL_NOISE_BLOCK;
/// The rows of a block-mean cube's thread grid.
const MEAN_ROWS: u32 = COLOUR_MEAN_THREADS / MEAN_COLUMNS;
/// The factors that decode a frame's fixed-point log grain and colour sums.
const LOG_DECODE: f32 = 1.0 / COLOUR_LOG_FIXED;
const MEAN_DECODE: f32 = 1.0 / COLOUR_MEAN_FIXED;

/// The colour cell of a block with mean luma `luma` and chroma `cb` and `cr`, centred on 0.
///
/// Every value is on the 8-bit scale. Brightness takes one of 16 bins of 16 codes. A block whose
/// chroma magnitude sits below the first saturation edge is neutral and has no hue. Every other
/// block takes its saturation level and one of 12 equal hue sectors. Values past either end clamp
/// into the table.
#[cube]
fn colour_cell(luma: f32, cb: f32, cr: f32) -> u32 {
    let brightness = brightness_bin(luma);

    let saturation = f32::sqrt(cb * cb + cr * cr);
    let level_1 = select(saturation >= SATURATION_EDGE_1, 1u32, 0u32);
    let level_2 = select(saturation >= SATURATION_EDGE_2, 1u32, 0u32);
    let level_3 = select(saturation >= SATURATION_EDGE_3, 1u32, 0u32);
    let level_4 = select(saturation >= SATURATION_EDGE_4, 1u32, 0u32);
    let level_5 = select(saturation >= SATURATION_EDGE_5, 1u32, 0u32);
    let level = level_1 + level_2 + level_3 + level_4 + level_5;

    let hue = f32::atan2(cr, cb);
    let turn = (hue + PI) / TAU;
    let raw_sector = f32::floor(turn * HUE_SECTORS);
    let sector_position = f32::max(raw_sector, 0.0f32);
    let sector = u32::cast_from(sector_position) % COLOUR_HUE_BINS;
    let coloured = 1u32 + (u32::max(level, 1u32) - 1u32) * COLOUR_HUE_BINS + sector;
    let ring = select(level == 0u32, 0u32, coloured);

    brightness * COLOUR_CELLS_PER_BRIGHTNESS + ring
}

/// The brightness bin of a block with mean luma `luma` on the 8-bit scale, clamped into the table.
#[cube]
fn brightness_bin(luma: f32) -> u32 {
    let scaled = f32::floor(luma / COLOUR_BRIGHTNESS_STEP);
    let clamped = f32::clamp(scaled, 0.0f32, LAST_BRIGHTNESS);

    u32::cast_from(clamped)
}

/// The histogram bin of a block's log grain, clamped into the histogram.
#[cube]
fn histogram_bin(log_grain: f32) -> u32 {
    let position = f32::floor((log_grain - COLOUR_HISTOGRAM_LOG_MIN) / COLOUR_HISTOGRAM_LOG_STEP);
    let clamped = f32::clamp(position, 0.0f32, LAST_HISTOGRAM_BIN);

    u32::cast_from(clamped)
}

/// Adds one block to the cell record that starts at word `base` of `cells`, in fixed point.
///
/// The log grain clamps to the histogram's range, so the sum and the histogram agree on every
/// block.
#[cube]
fn add_block(cells: &mut [Atomic<i32>], base: u32, log_grain: f32, luma: f32, cb: f32, cr: f32) {
    let clamped_log = f32::clamp(log_grain, COLOUR_HISTOGRAM_LOG_MIN, COLOUR_HISTOGRAM_LOG_MAX);
    let log_fixed = f32::round(clamped_log * COLOUR_LOG_FIXED) as i32;
    let luma_fixed = f32::round(luma * COLOUR_MEAN_FIXED) as i32;
    let cb_fixed = f32::round(cb * COLOUR_MEAN_FIXED) as i32;
    let cr_fixed = f32::round(cr * COLOUR_MEAN_FIXED) as i32;
    let bin = histogram_bin(log_grain);

    Atomic::fetch_add(&cells[(base + COLOUR_CELL_COUNT) as usize], 1i32);
    Atomic::fetch_add(&cells[(base + COLOUR_CELL_LOG_SUM) as usize], log_fixed);
    Atomic::fetch_add(&cells[(base + COLOUR_CELL_LUMA_SUM) as usize], luma_fixed);
    Atomic::fetch_add(&cells[(base + COLOUR_CELL_CB_SUM) as usize], cb_fixed);
    Atomic::fetch_add(&cells[(base + COLOUR_CELL_CR_SUM) as usize], cr_fixed);
    Atomic::fetch_add(&cells[(base + COLOUR_CELL_HISTOGRAM + bin) as usize], 1i32);
}

/// Reads sample `index` of a plane as a code on its own scale.
///
/// Word-packed planes hold `samples_per_word` codes per word. A float plane holds one normalised
/// sample per word.
#[cube]
fn read_sample(plane: &[u32], index: u32, #[comptime] samples_per_word: u32, #[comptime] float: bool) -> f32 {
    let bits = comptime!(32u32 / samples_per_word);
    let mask = comptime!(
        if samples_per_word == 1 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        }
    );
    let word = plane[(index / samples_per_word) as usize];

    if comptime!(float) {
        f32::reinterpret(word)
    } else {
        let shift = (index % samples_per_word) * bits;
        let code = (word >> shift) & mask;
        f32::cast_from(code)
    }
}

/// The sum of the samples of `plane` inside a rectangle, taken by every thread of the cube.
///
/// The threads hold a grid of [MEAN_COLUMNS] columns by [MEAN_ROWS] rows, which steps over the
/// rectangle. The sums are combined in the cube's shared memory by the caller.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a rectangle bound or a comptime shape"
)]
fn rectangle_sum(
    plane: &[u32],
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    #[comptime] stride: u32,
    #[comptime] samples_per_word: u32,
    #[comptime] float: bool,
) -> f32 {
    let mut sum = 0.0f32;
    let mut y = top + UNIT_POS_X / MEAN_COLUMNS;

    while y < bottom {
        let mut x = left + UNIT_POS_X % MEAN_COLUMNS;

        while x < right {
            let index = y * stride + x;
            sum += read_sample(plane, index, samples_per_word, float);
            x += MEAN_COLUMNS;
        }

        y += MEAN_ROWS;
    }

    sum
}

/// Writes the mean luma, Cb and Cr of every block of the luma and the chroma noise grids.
///
/// Both grids hold 16x16 blocks of their own plane, with ragged edges truncated. Cubes in the first
/// `luma_blocks_y` rows take the luma grid and the rest take the chroma grid. A luma block averages
/// the chroma pixels its area covers, and a chroma block the luma pixels its area covers. Means are
/// on the 8-bit scale, codes times `eight_bit_scale`, with chroma centred on 0. Each block writes
/// [COLOUR_MEAN_FIELDS](crate::noise::colour::COLOUR_MEAN_FIELDS) values. The sums reduce in a fixed
/// order, so the means are the same on every run.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_colour_means(
    luma: &[u32],
    cb: &[u32],
    cr: &[u32],
    luma_means: &mut [f32],
    chroma_means: &mut [f32],
    eight_bit_scale: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] shift_x: u32,
    #[comptime] shift_y: u32,
    #[comptime] samples_per_word: u32,
    #[comptime] float: bool,
) {
    let chroma_width = comptime!(width.div_ceil(1u32 << shift_x));
    let chroma_height = comptime!(height.div_ceil(1u32 << shift_y));
    let luma_blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let luma_blocks_y = comptime!(height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let chroma_blocks_x = comptime!(chroma_width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let chroma_blocks_y = comptime!(chroma_height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let step_x = comptime!(1u32 << shift_x);
    let step_y = comptime!(1u32 << shift_y);

    let mut luma_sums = Shared::<[f32]>::new_slice(COLOUR_MEAN_THREADS as usize);
    let mut cb_sums = Shared::<[f32]>::new_slice(COLOUR_MEAN_THREADS as usize);
    let mut cr_sums = Shared::<[f32]>::new_slice(COLOUR_MEAN_THREADS as usize);

    let in_luma_grid = CUBE_POS_Y < luma_blocks_y;
    let block_x = CUBE_POS_X;
    let luma_rows = u32::min(CUBE_POS_Y, luma_blocks_y);
    let chroma_row = CUBE_POS_Y - luma_rows;
    let block_y = select(in_luma_grid, CUBE_POS_Y, chroma_row);
    let blocks_x = select(in_luma_grid, luma_blocks_x, chroma_blocks_x);
    let blocks_y = select(in_luma_grid, luma_blocks_y, chroma_blocks_y);
    let active = block_x < blocks_x && block_y < blocks_y;

    // A luma block's chroma area rounds outwards, and a chroma block's luma area clips to the plane.
    let own_left = block_x * TEMPORAL_NOISE_BLOCK;
    let own_top = block_y * TEMPORAL_NOISE_BLOCK;
    let own_width = select(in_luma_grid, width, chroma_width);
    let own_height = select(in_luma_grid, height, chroma_height);
    let own_right = u32::min(own_left + TEMPORAL_NOISE_BLOCK, own_width);
    let own_bottom = u32::min(own_top + TEMPORAL_NOISE_BLOCK, own_height);

    let luma_left = select(in_luma_grid, own_left, own_left * step_x);
    let luma_top = select(in_luma_grid, own_top, own_top * step_y);
    let covered_right = u32::min(own_right * step_x, width);
    let covered_bottom = u32::min(own_bottom * step_y, height);
    let luma_right = select(in_luma_grid, own_right, covered_right);
    let luma_bottom = select(in_luma_grid, own_bottom, covered_bottom);
    let chroma_left = select(in_luma_grid, own_left >> shift_x, own_left);
    let chroma_top = select(in_luma_grid, own_top >> shift_y, own_top);
    let chroma_right = select(in_luma_grid, (own_right + step_x - 1u32) >> shift_x, own_right);
    let chroma_bottom = select(in_luma_grid, (own_bottom + step_y - 1u32) >> shift_y, own_bottom);

    let thread = UNIT_POS_X as usize;
    luma_sums[thread] = 0.0f32;
    cb_sums[thread] = 0.0f32;
    cr_sums[thread] = 0.0f32;

    if active {
        luma_sums[thread] = rectangle_sum(
            luma,
            luma_left,
            luma_top,
            luma_right,
            luma_bottom,
            width,
            samples_per_word,
            float,
        );
        cb_sums[thread] = rectangle_sum(
            cb,
            chroma_left,
            chroma_top,
            chroma_right,
            chroma_bottom,
            chroma_width,
            samples_per_word,
            float,
        );
        cr_sums[thread] = rectangle_sum(
            cr,
            chroma_left,
            chroma_top,
            chroma_right,
            chroma_bottom,
            chroma_width,
            samples_per_word,
            float,
        );
    }

    sync_cube();

    let half_threads = COLOUR_MEAN_THREADS / 2u32;
    let mut half = half_threads.runtime();
    while half > 0u32 {
        if UNIT_POS_X < half {
            let partner = thread + half as usize;
            let luma_pair = luma_sums[thread] + luma_sums[partner];
            let cb_pair = cb_sums[thread] + cb_sums[partner];
            let cr_pair = cr_sums[thread] + cr_sums[partner];
            luma_sums[thread] = luma_pair;
            cb_sums[thread] = cb_pair;
            cr_sums[thread] = cr_pair;
        }

        sync_cube();
        half /= 2u32;
    }

    if active && UNIT_POS_X == 0u32 {
        let luma_pixels = f32::cast_from((luma_right - luma_left) * (luma_bottom - luma_top));
        let chroma_pixels = f32::cast_from((chroma_right - chroma_left) * (chroma_bottom - chroma_top));
        let mean_luma = luma_sums[0] / luma_pixels * eight_bit_scale;
        let mean_cb = cb_sums[0] / chroma_pixels * eight_bit_scale - COLOUR_CHROMA_CENTRE;
        let mean_cr = cr_sums[0] / chroma_pixels * eight_bit_scale - COLOUR_CHROMA_CENTRE;
        let base = (block_y * blocks_x + block_x) * COLOUR_MEAN_FIELDS;

        if in_luma_grid {
            luma_means[base as usize] = mean_luma;
            luma_means[(base + 1u32) as usize] = mean_cb;
            luma_means[(base + 2u32) as usize] = mean_cr;
        } else {
            chroma_means[base as usize] = mean_luma;
            chroma_means[(base + 1u32) as usize] = mean_cb;
            chroma_means[(base + 2u32) as usize] = mean_cr;
        }
    }
}

/// Adds every measurable block of one slot's stats to one frame's cell sums.
///
/// The frame's sums start at word `slot_base` of `cells`, which [noise_colour_clear] zeroed.
/// Thread `i` takes block `i`. Each channel judges the block on its own residual. A block joins a
/// channel when its mean residual passes the static gate, its mean luma sits between 20 and 240
/// codes and its grain is above the channel's floor. Its grain is the residual's sigma times
/// `grain_scale`, on the 8-bit scale. It then adds its count, log grain, mean colour and a
/// histogram count to channel `c`'s cell of its colour. Past every channel's cells, with `luma` a
/// neutral block also adds to the calibration record of its brightness. Without it the block adds to
/// channel `c`'s plane record.
///
/// Every block that joins is counted, however heavy. The colour-relative outlier rule runs on the
/// pooled cells, through [noise_colour_levels], so it judges each block against the whole window.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_colour_accumulate(
    records: &[f32],
    means: &[f32],
    cells: &mut [Atomic<i32>],
    grain_scale: f32,
    slot_base: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] luma: bool,
) {
    comptime! {
        assert!(channels <= NOISE_CHANNELS, "noise_colour_accumulate needs a slot per channel");
    }

    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let blocks = comptime!(blocks_x * height.div_ceil(TEMPORAL_NOISE_BLOCK));
    let record_len = comptime!(temporal_stats_record_len(channels));
    let grain_floor = comptime!(
        if luma {
            COLOUR_LUMA_GRAIN_FLOOR
        } else {
            COLOUR_CHROMA_GRAIN_FLOOR
        }
    );
    let extra_base = comptime!(channels * COLOUR_CELLS);

    let index = ABSOLUTE_POS_X;

    if index < blocks {
        let block_x = index % blocks_x;
        let block_y = index / blocks_x;
        let block_width = u32::min(TEMPORAL_NOISE_BLOCK, width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = u32::min(TEMPORAL_NOISE_BLOCK, height - block_y * TEMPORAL_NOISE_BLOCK);
        let pixels = f32::cast_from(block_width * block_height);
        let base = index * record_len;

        let mean_base = index * COLOUR_MEAN_FIELDS;
        let mean_luma = means[mean_base as usize];
        let mean_cb = means[(mean_base + 1u32) as usize];
        let mean_cr = means[(mean_base + 2u32) as usize];
        let cell = colour_cell(mean_luma, mean_cb, mean_cr);
        let saturation = f32::sqrt(mean_cb * mean_cb + mean_cr * mean_cr);
        let in_range = mean_luma > COLOUR_LUMA_MIN && mean_luma < COLOUR_LUMA_MAX;

        let brightness = brightness_bin(mean_luma);
        let neutral = saturation < COLOUR_NEUTRAL_SATURATION;

        #[unroll]
        for channel in 0..NOISE_CHANNELS {
            if comptime!(channel < channels) {
                let sum_d = records[(base + channel) as usize];
                let sum_d2 = records[(base + channels + channel) as usize];
                let (mean, _, sigma) = residual_sigma(sum_d, sum_d2, pixels);
                let grain = sigma * grain_scale;
                let joins = in_range && f32::abs(mean) < STATIC_GATE && grain > grain_floor;

                if joins {
                    let log_grain = f32::ln(grain);
                    let record = comptime!(channel * COLOUR_CELLS) + cell;
                    let record_base = slot_base + record * COLOUR_CELL_WORDS;
                    add_block(cells, record_base, log_grain, mean_luma, mean_cb, mean_cr);

                    if comptime!(luma) {
                        if neutral {
                            let calibration = extra_base + brightness;
                            let calibration_word = slot_base + calibration * COLOUR_CELL_WORDS;
                            add_block(cells, calibration_word, log_grain, mean_luma, mean_cb, mean_cr);
                        }
                    } else {
                        let plane_record = comptime!(extra_base + channel);
                        let plane_word = slot_base + plane_record * COLOUR_CELL_WORDS;
                        add_block(cells, plane_word, log_grain, mean_luma, mean_cb, mean_cr);
                    }
                }
            }
        }
    }
}

/// Zeroes the `words` words of one frame's cell sums, which start at word `slot_base` of `cells`.
#[cube(launch_unchecked)]
pub fn noise_colour_clear(cells: &mut [i32], slot_base: u32, #[comptime] words: u32) {
    let index = ABSOLUTE_POS_X;

    if index < words {
        cells[(slot_base + index) as usize] = 0i32;
    }
}

/// Sums the cell sums of the frames `first..=last` into the window's sums.
///
/// Stream frame `frame` holds its sums in ring slot `frame % capacity`, `words` words each. Thread
/// `i` takes word `i`. Fixed-point words decode back to their own units, so the window holds counts,
/// log grain and colour on the 8-bit scale. Frames add from the oldest to the newest, so the same
/// frames give the same sums whatever the stream read before them.
#[cube(launch_unchecked)]
pub fn noise_colour_window(
    cells: &[i32],
    window: &mut [f32],
    first: u32,
    last: u32,
    capacity: u32,
    #[comptime] words: u32,
) {
    let index = ABSOLUTE_POS_X;

    if index < words {
        let word = index % COLOUR_CELL_WORDS;
        let is_log = word == COLOUR_CELL_LOG_SUM;
        let past_log = word >= COLOUR_CELL_LUMA_SUM;
        let before_histogram = word < COLOUR_CELL_HISTOGRAM;
        let is_mean = past_log && before_histogram;
        let mean_scale = select(is_mean, MEAN_DECODE, 1.0f32);
        let scale = select(is_log, LOG_DECODE, mean_scale);

        let mut sum = 0.0f32;
        let mut frame = first;
        while frame <= last {
            let slot = frame % capacity;
            let fixed = cells[(slot * words + index) as usize];
            sum += f32::cast_from(fixed) * scale;
            frame += 1u32;
        }

        window[index as usize] = sum;
    }
}

/// Writes every colour cell's level from the window's sums, leaving out the blocks above the cell's
/// ceiling.
///
/// This is the colour-relative outlier rule. Thread `c` takes colour cell `c` of `cells` cells, and
/// `ceilings[c]` is the highest log grain the cell keeps. A ceiling past the histogram keeps every
/// block. The kept blocks come from the cell's histogram, with a bin the ceiling splits kept in
/// proportion. With `mean_level`, the level is the mean log grain of the kept blocks, taken from
/// the exact sum less the dropped blocks. Otherwise it is the median of the kept blocks' log grain,
/// interpolated within its bin. Each cell writes its kept count, then its level, which is 0 when
/// it keeps nothing. A ceiling inside the top bin splits it as if its blocks spread evenly, since
/// heavier log grain clamps into that bin.
///
/// Judging the pooled cells against a first fit's ceilings matches refitting on the blocks, since
/// each cell holds every block of the window. The ceilings come from that first fit.
#[cube(launch_unchecked)]
pub fn noise_colour_levels(
    window: &[f32],
    ceilings: &[f32],
    levels: &mut [f32],
    #[comptime] cells: u32,
    #[comptime] mean_level: bool,
) {
    let cell = ABSOLUTE_POS_X;

    if cell < cells {
        let base = cell * COLOUR_CELL_WORDS;
        let ceiling = ceilings[cell as usize];
        let count = window[(base + COLOUR_CELL_COUNT) as usize];
        let log_sum = window[(base + COLOUR_CELL_LOG_SUM) as usize];

        let mut dropped = 0.0f32;
        let mut dropped_log = 0.0f32;
        for bin in 0..COLOUR_HISTOGRAM_BINS {
            let bin_count = window[(base + COLOUR_CELL_HISTOGRAM + bin) as usize];
            let low = COLOUR_HISTOGRAM_LOG_MIN + f32::cast_from(bin) * COLOUR_HISTOGRAM_LOG_STEP;
            let high = low + COLOUR_HISTOGRAM_LOG_STEP;
            let above = f32::clamp((high - ceiling) / COLOUR_HISTOGRAM_LOG_STEP, 0.0f32, 1.0f32);
            let cut = f32::max(low, ceiling);

            dropped += bin_count * above;
            dropped_log += bin_count * above * (cut + high) * 0.5f32;
        }

        let kept = f32::max(count - dropped, 0.0f32);
        let mut level = 0.0f32;

        if comptime!(mean_level) {
            let divisor = f32::max(kept, 1.0f32);
            level = (log_sum - dropped_log) / divisor;
        } else {
            let target = kept * 0.5f32;
            let mut below = 0.0f32;
            let mut found = false;

            for bin in 0..COLOUR_HISTOGRAM_BINS {
                let bin_count = window[(base + COLOUR_CELL_HISTOGRAM + bin) as usize];
                let low = COLOUR_HISTOGRAM_LOG_MIN + f32::cast_from(bin) * COLOUR_HISTOGRAM_LOG_STEP;
                let high = low + COLOUR_HISTOGRAM_LOG_STEP;
                let above = f32::clamp((high - ceiling) / COLOUR_HISTOGRAM_LOG_STEP, 0.0f32, 1.0f32);
                let kept_share = 1.0f32 - above;
                let kept_count = bin_count * kept_share;
                let reaches = below + kept_count >= target && kept_count > 0.0f32;

                if !found && reaches {
                    let fraction = (target - below) / kept_count;
                    level = low + fraction * kept_share * COLOUR_HISTOGRAM_LOG_STEP;
                    found = true;
                }

                below += kept_count;
            }
        }

        let has_blocks = kept > 0.0f32;
        let output = cell * COLOUR_LEVEL_FIELDS;
        levels[output as usize] = kept;
        levels[(output + 1u32) as usize] = select(has_blocks, level, 0.0f32);
    }
}
