use std::f32::consts::{LN_2, PI, TAU};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::records::{BlockSpec, frame_records, hashed_unit};
use crate::engine::align::StorageAlign;
use crate::engine::{ChromaLayout, DevicePlane, EngineSpec, Geometry, SampleFormat};
use crate::nl4d::Nl4dOptions;
use crate::nlmeans::{DenoisingMode, NlmeansAlgorithm, NlmeansHqOptions, NlmeansOptions};
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
    COLOUR_HISTOGRAM_LOG_MIN,
    COLOUR_HISTOGRAM_LOG_STEP,
    COLOUR_HUE_BINS,
    COLOUR_LEVEL_FIELDS,
    COLOUR_MEAN_FIELDS,
    COLOUR_SATURATION_EDGES,
    CellFrame,
    ColourBlocks,
    ColourCells,
    ColourGrid,
    queue_colour_levels,
};
use crate::noise::derived::DerivedValues;
use crate::noise::fold::NoiseSettings;
use crate::noise::history::{WindowBounds, history_capacity};
use crate::noise::state::{NoiseShape, NoiseState};
use crate::noise::temporal::temporal_stats_blocks;
use crate::tune::gpu::read_f32;
use crate::tune::tests::runtime::test_client;

/// The factor from a normalised 8-bit residual to codes.
const EIGHT_BIT: f32 = 255.0;
/// A ceiling past every histogram bin, which keeps every block.
const NO_CEILING: f32 = f32::MAX;

/// One block's mean colour on the 8-bit scale, chroma centred on 0, and its grain in codes.
#[derive(Debug, Clone, Copy)]
struct ColourBlock {
    luma: f32,
    cb: f32,
    cr: f32,
    grain: f32,
    /// The block's mean residual in codes.
    mean: f32,
    /// Channel 1's mean residual in codes, when it differs from channel 0's.
    second_mean: Option<f32>,
}

impl ColourBlock {
    fn new(luma: f32, cb: f32, cr: f32, grain: f32) -> Self {
        Self {
            luma,
            cb,
            cr,
            grain,
            mean: 0.0,
            second_mean: None,
        }
    }

    /// A block at the centre of hue sector `sector`, `saturation` codes from neutral.
    fn coloured(luma: f32, saturation: f32, sector: u32, grain: f32) -> Self {
        let sector_width = TAU / COLOUR_HUE_BINS as f32;
        let angle = -PI + (sector as f32 + 0.5) * sector_width;
        let cb = saturation * angle.cos();
        let cr = saturation * angle.sin();

        Self::new(luma, cb, cr, grain)
    }
}

/// The colour cell a block of mean `luma`, `cb` and `cr` belongs to.
fn reference_cell(luma: f32, cb: f32, cr: f32) -> usize {
    let per_brightness = COLOUR_CELLS_PER_BRIGHTNESS as usize;
    let last_brightness = (COLOUR_BRIGHTNESS_BINS - 1) as f32;
    let brightness = (luma / 16.0).floor().clamp(0.0, last_brightness) as usize;
    let saturation = cb.hypot(cr);
    let level = COLOUR_SATURATION_EDGES
        .iter()
        .filter(|&&edge| saturation >= edge)
        .count();
    if level == 0 {
        return brightness * per_brightness;
    }

    let hues = COLOUR_HUE_BINS as usize;
    let turn = (cr.atan2(cb) + PI) / TAU;
    let sector = ((turn * hues as f32).floor().max(0.0) as usize) % hues;

    brightness * per_brightness + 1 + (level - 1) * hues + sector
}

fn histogram_bin(log_grain: f32) -> usize {
    let position = ((log_grain - COLOUR_HISTOGRAM_LOG_MIN) / COLOUR_HISTOGRAM_LOG_STEP).floor();
    position.max(0.0) as usize
}

/// One slot's records and block means for `blocks` of a `width` by `height` plane.
fn frame_inputs(width: u32, height: u32, channels: u32, blocks: &[ColourBlock]) -> (Vec<f32>, Vec<f32>) {
    let specs: Vec<BlockSpec> = blocks
        .iter()
        .map(|block| {
            let mean = block.mean / EIGHT_BIT;
            let sigma = block.grain / EIGHT_BIT;
            let mut spec = BlockSpec::uniform(mean, sigma, 0.0);
            if let Some(second_mean) = block.second_mean {
                spec.mean[1] = second_mean / EIGHT_BIT;
            }

            spec
        })
        .collect();
    let records = frame_records(width, height, channels, &specs);
    let means = blocks
        .iter()
        .flat_map(|block| [block.luma, block.cb, block.cr])
        .collect();

    (records, means)
}

/// Pools frames of blocks into fresh cells as stream frames 0 onwards and reads their window's
/// sums back.
struct CellRun {
    client: Client,
    width: u32,
    height: u32,
    channels: u32,
    cells: ColourCells,
    pooled: std::cell::Cell<u32>,
}

impl CellRun {
    fn new(width: u32, height: u32, channels: u32, luma: bool) -> Self {
        let client = test_client();
        let capacity = history_capacity(1);
        let cells = ColourCells::new(&client, channels, luma, capacity);

        Self {
            client,
            width,
            height,
            channels,
            cells,
            pooled: std::cell::Cell::new(0),
        }
    }

    fn pool(&self, blocks: &[ColourBlock]) {
        let frame = self.pooled.get();
        self.pooled.set(frame + 1);

        let (records, means) = frame_inputs(self.width, self.height, self.channels, blocks);
        let records_bytes = f32::as_bytes(&records);
        let means_bytes = f32::as_bytes(&means);
        let records_handle = self.client.create_from_slice(records_bytes);
        let means_handle = self.client.create_from_slice(means_bytes);

        let input = CellFrame {
            records: &records_handle,
            records_len: records.len(),
            means: &means_handle,
            width: self.width,
            height: self.height,
            grain_scale: EIGHT_BIT,
        };
        self.cells.queue_frame(&self.client, &input, frame);
    }

    /// Sums the window around stream frame 0, which holds every pooled frame, and reads it back.
    fn scene(&self) -> Vec<f32> {
        let newest = self.pooled.get() - 1;
        let bounds = WindowBounds::around(0, newest);
        self.cells.queue_window(&self.client, bounds);

        let words = self.cells.words();
        let window = self.cells.window();
        read_f32(&self.client, window, words).expect("the window's sums read back")
    }

    /// Every colour cell's kept count and level in the last summed window, judged against
    /// `ceilings`.
    fn levels(&self, ceilings: &[f32], mean_level: bool) -> Vec<f32> {
        let cells = self.channels * COLOUR_CELLS;
        let ceilings_bytes = f32::as_bytes(ceilings);
        let ceilings_handle = self.client.create_from_slice(ceilings_bytes);
        let levels_len = (cells * COLOUR_LEVEL_FIELDS) as usize;
        let levels = self.client.empty(levels_len * size_of::<f32>());

        queue_colour_levels(
            &self.client,
            self.cells.window(),
            &ceilings_handle,
            &levels,
            cells,
            mean_level,
        );

        read_f32(&self.client, &levels, levels_len).expect("the levels read back")
    }
}

fn record(scene: &[f32], record: usize) -> &[f32] {
    let words = COLOUR_CELL_WORDS as usize;
    &scene[record * words..(record + 1) * words]
}

fn total_count(scene: &[f32], records: std::ops::Range<usize>) -> f32 {
    records
        .map(|index| record(scene, index)[COLOUR_CELL_COUNT as usize])
        .sum()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, what: &str) {
    let gap = (actual - expected).abs();
    assert!(gap <= tolerance, "{what}: {actual} against {expected}");
}

#[test]
fn blocks_bin_into_the_expected_cells() {
    let width = 64;
    let height = 48;
    let blocks = [
        ColourBlock::new(100.0, 0.0, 0.0, 2.0),
        ColourBlock::coloured(100.0, 9.0, 0, 2.0),
        ColourBlock::coloured(100.0, 9.0, 11, 1.5),
        ColourBlock::coloured(130.0, 16.0, 3, 3.0),
        ColourBlock::coloured(130.0, 25.0, 6, 2.5),
        ColourBlock::coloured(60.0, 37.0, 8, 1.2),
        ColourBlock::coloured(60.0, 80.0, 5, 4.0),
        ColourBlock::new(200.0, 3.0, -4.0, 0.8),
        ColourBlock::coloured(200.0, 13.0, 9, 1.0),
        ColourBlock::coloured(25.0, 50.0, 2, 2.0),
        ColourBlock::coloured(235.0, 7.0, 7, 2.2),
        ColourBlock::coloured(150.0, 44.5, 10, 0.6),
    ];
    let run = CellRun::new(width, height, 2, false);
    run.pool(&blocks);
    let scene = run.scene();

    // Hand-worked cells: brightness 6 is cell 366 onwards. Sector 0 at level 1 is the next cell
    // and sector 11 the last of level 1.
    assert_eq!(reference_cell(100.0, 0.0, 0.0), 6 * 61);
    assert_eq!(
        reference_cell(blocks[1].luma, blocks[1].cb, blocks[1].cr),
        6 * 61 + 1
    );
    assert_eq!(
        reference_cell(blocks[2].luma, blocks[2].cb, blocks[2].cr),
        6 * 61 + 12
    );

    for channel in 0..2usize {
        let channel_base = channel * COLOUR_CELLS as usize;
        let pooled = total_count(&scene, channel_base..channel_base + COLOUR_CELLS as usize);
        assert_eq!(pooled, blocks.len() as f32, "channel {channel} pools every block");

        for block in &blocks {
            let cell = channel_base + reference_cell(block.luma, block.cb, block.cr);
            let words = record(&scene, cell);
            let log_grain = block.grain.ln();
            let bin = histogram_bin(log_grain);

            assert_eq!(words[COLOUR_CELL_COUNT as usize], 1.0, "{block:?}");
            assert_close(
                words[COLOUR_CELL_LOG_SUM as usize],
                log_grain,
                1.0e-3,
                "log grain",
            );
            assert_close(words[COLOUR_CELL_LUMA_SUM as usize], block.luma, 0.04, "luma");
            assert_close(words[COLOUR_CELL_CB_SUM as usize], block.cb, 0.04, "cb");
            assert_close(words[COLOUR_CELL_CR_SUM as usize], block.cr, 0.04, "cr");
            assert_eq!(words[COLOUR_CELL_HISTOGRAM as usize + bin], 1.0, "{block:?}");
        }
    }
}

/// The blocks in brightness bin `brightness` at saturation level `level` of channel `channel`,
/// summed over every hue.
fn level_count(scene: &[f32], channel: usize, brightness: usize, level: usize) -> f32 {
    let per_brightness = COLOUR_CELLS_PER_BRIGHTNESS as usize;
    let hues = COLOUR_HUE_BINS as usize;
    let first = channel * COLOUR_CELLS as usize + brightness * per_brightness;
    let cells = if level == 0 {
        first..first + 1
    } else {
        let start = first + 1 + (level - 1) * hues;
        start..start + hues
    };

    total_count(scene, cells)
}

#[test]
fn blocks_at_the_gate_and_level_edges() {
    let width = 64;
    let height = 64;
    let moving = ColourBlock {
        mean: 3.0,
        ..ColourBlock::new(100.0, 0.0, 0.0, 2.0)
    };
    let blocks = [
        ColourBlock::new(20.0, 0.0, 0.0, 2.0),
        ColourBlock::new(240.0, 0.0, 0.0, 2.0),
        ColourBlock::new(20.01, 0.0, 0.0, 2.0),
        ColourBlock::new(239.9, 0.0, 0.0, 2.0),
        ColourBlock::new(100.0, 6.0, 0.0, 2.0),
        ColourBlock::new(100.0, 5.99, 0.0, 2.0),
        ColourBlock::new(100.0, 0.0, 44.0, 2.0),
        ColourBlock::new(100.0, 0.0, 43.99, 2.0),
        ColourBlock::new(100.0, 7.5, 0.0, 2.0),
        ColourBlock::new(100.0, 8.0, 0.0, 2.0),
        ColourBlock::new(100.0, 0.0, 0.0, 0.04),
        ColourBlock::new(100.0, 0.0, 0.0, 0.06),
        moving,
        moving,
        moving,
        moving,
    ];

    let luma_run = CellRun::new(width, height, 1, true);
    luma_run.pool(&blocks);
    let luma = luma_run.scene();

    assert_eq!(
        total_count(&luma, 0..COLOUR_CELLS as usize),
        9.0,
        "luma pools 9 blocks"
    );
    assert_eq!(level_count(&luma, 0, 1, 0), 1.0, "luma just above 20 joins");
    assert_eq!(level_count(&luma, 0, 14, 0), 1.0, "luma just below 240 joins");
    assert_eq!(
        level_count(&luma, 0, 6, 0),
        2.0,
        "below 6 is neutral, 0.04 is under the luma floor"
    );
    assert_eq!(
        level_count(&luma, 0, 6, 1),
        3.0,
        "saturation 6, 7.5 and 8 open level 1"
    );
    assert_eq!(level_count(&luma, 0, 6, 4), 1.0, "just below 44 is level 4");
    assert_eq!(level_count(&luma, 0, 6, 5), 1.0, "44 opens level 5");

    let calibration = |brightness: usize| {
        let index = COLOUR_CELLS as usize + brightness;
        record(&luma, index)[COLOUR_CELL_COUNT as usize]
    };
    assert_eq!(
        calibration(6),
        4.0,
        "saturation below 8 calibrates, coloured or not"
    );
    assert_eq!(calibration(1), 1.0);
    assert_eq!(calibration(14), 1.0);

    let chroma_run = CellRun::new(width, height, 2, false);
    chroma_run.pool(&blocks);
    let chroma = chroma_run.scene();

    for channel in 0..2 {
        let neutral = level_count(&chroma, channel, 6, 0);
        assert_eq!(neutral, 3.0, "channel {channel}: 0.04 clears the chroma floor");
    }
}

#[test]
fn each_chroma_channel_gates_on_its_own_residual() {
    let width = 64;
    let height = 48;
    let moving = ColourBlock {
        mean: 3.0,
        ..ColourBlock::new(100.0, 0.0, 0.0, 2.0)
    };
    let cb_static = ColourBlock {
        second_mean: Some(3.0),
        ..ColourBlock::coloured(100.0, 25.0, 3, 2.0)
    };
    let cr_static = ColourBlock {
        mean: 3.0,
        second_mean: Some(0.0),
        ..ColourBlock::coloured(180.0, 37.0, 9, 1.5)
    };
    let mut blocks = vec![cb_static, cr_static];
    blocks.extend([moving; 10]);

    let run = CellRun::new(width, height, 2, false);
    run.pool(&blocks);
    let scene = run.scene();

    let cells = COLOUR_CELLS as usize;
    let cb_cell = reference_cell(cb_static.luma, cb_static.cb, cb_static.cr);
    let cr_cell = reference_cell(cr_static.luma, cr_static.cb, cr_static.cr);
    assert_eq!(total_count(&scene, 0..cells), 1.0, "Cb keeps one block");
    assert_eq!(total_count(&scene, cells..2 * cells), 1.0, "Cr keeps one block");
    assert_eq!(record(&scene, cb_cell)[COLOUR_CELL_COUNT as usize], 1.0);
    assert_eq!(record(&scene, cells + cr_cell)[COLOUR_CELL_COUNT as usize], 1.0);
}

#[test]
fn heavy_colour_grain_is_kept_and_motion_residual_dropped() {
    let width = 128;
    let height = 64;
    let neutral = ColourBlock::new(120.0, 0.0, 0.0, 1.0);
    let residual = ColourBlock::new(120.0, 0.0, 0.0, 5.0);
    let moving = ColourBlock {
        mean: 3.0,
        ..ColourBlock::new(120.0, 0.0, 0.0, 1.0)
    };
    let red = ColourBlock::coloured(120.0, 40.0, 7, 4.0);
    let mut blocks = vec![neutral; 16];
    blocks.extend([residual; 4]);
    blocks.extend([red; 10]);
    blocks.extend([moving; 2]);

    let neutral_cell = reference_cell(neutral.luma, neutral.cb, neutral.cr);
    let red_cell = reference_cell(red.luma, red.cb, red.cr);

    for mean_level in [true, false] {
        let run = CellRun::new(width, height, 1, mean_level);
        run.pool(&blocks);
        run.scene();

        let unbounded = vec![NO_CEILING; COLOUR_CELLS as usize];
        let first = run.levels(&unbounded, mean_level);
        let ceilings: Vec<f32> = first
            .chunks(COLOUR_LEVEL_FIELDS as usize)
            .map(|level| level[1] + LN_2)
            .collect();
        let second = run.levels(&ceilings, mean_level);

        let fields = COLOUR_LEVEL_FIELDS as usize;
        let first_neutral = &first[neutral_cell * fields..(neutral_cell + 1) * fields];
        let neutral_level = &second[neutral_cell * fields..(neutral_cell + 1) * fields];
        let red_level = &second[red_cell * fields..(red_cell + 1) * fields];

        assert_eq!(first_neutral[0], 20.0, "the moving blocks fail the static gate");
        assert_close(neutral_level[0], 16.0, 1.0e-3, "the residual blocks drop out");
        assert_close(neutral_level[1], 0.0, COLOUR_HISTOGRAM_LOG_STEP, "neutral level");
        assert_close(red_level[0], 10.0, 1.0e-3, "every heavy red block stays");
        assert_close(red_level[1], 4.0f32.ln(), COLOUR_HISTOGRAM_LOG_STEP, "red level");
    }
}

#[test]
fn extreme_chroma_values_index_inside_the_table() {
    let width = 64;
    let height = 64;
    let extremes = [-128.0, 127.0, -300.0, 300.0];
    let mut blocks = Vec::new();
    for luma in [20.5, 239.5] {
        for cb in extremes {
            for cr in [extremes[0], extremes[1]] {
                blocks.push(ColourBlock::new(luma, cb, cr, 2.0));
            }
        }
    }

    let run = CellRun::new(width, height, 1, true);
    run.pool(&blocks);
    let scene = run.scene();

    let pooled = total_count(&scene, 0..COLOUR_CELLS as usize);
    assert_eq!(pooled, blocks.len() as f32, "every block lands in a cell");

    for block in &blocks {
        let cell = reference_cell(block.luma, block.cb, block.cr);
        assert!(cell < COLOUR_CELLS as usize, "{block:?}");
        assert!(
            record(&scene, cell)[COLOUR_CELL_COUNT as usize] >= 1.0,
            "{block:?}"
        );
    }

    let calibration = COLOUR_CELLS as usize..(COLOUR_CELLS + COLOUR_BRIGHTNESS_BINS) as usize;
    let neutral_count = total_count(&scene, calibration);
    assert_eq!(neutral_count, 0.0, "no saturated block calibrates");
}

/// Packs `codes` into the word layout of `format`.
fn pack(codes: &[u32], format: SampleFormat) -> Vec<u8> {
    match format {
        SampleFormat::U8 => codes.iter().map(|&code| code as u8).collect(),
        SampleFormat::U16 { .. } => codes
            .iter()
            .flat_map(|&code| (code as u16).to_le_bytes())
            .collect(),
        SampleFormat::F32 => codes
            .iter()
            .flat_map(|&code| (code as f32 / 255.0).to_le_bytes())
            .collect(),
    }
}

/// The 8-bit codes of a test frame's planes, luma then Cb then Cr.
fn colour_planes(width: u32, height: u32, chroma_width: u32, chroma_height: u32) -> [Vec<u32>; 3] {
    let luma = (0..width * height)
        .map(|index| 30 + (hashed_unit(1, index) * 170.0) as u32)
        .collect();
    let cb = (0..chroma_width * chroma_height)
        .map(|index| 16 + (hashed_unit(2, index) * 224.0) as u32)
        .collect();
    let cr = (0..chroma_width * chroma_height)
        .map(|index| 16 + (hashed_unit(3, index) * 224.0) as u32)
        .collect();

    [luma, cb, cr]
}

/// The mean of `plane` over the rectangle between `left`, `top` and `right`, `bottom`.
fn rectangle_mean(plane: &[u32], stride: u32, left: u32, top: u32, right: u32, bottom: u32) -> f32 {
    let mut sum = 0u64;
    for y in top..bottom {
        for x in left..right {
            sum += u64::from(plane[(y * stride + x) as usize]);
        }
    }

    let pixels = (right - left) * (bottom - top);
    sum as f32 / pixels as f32
}

/// Reads both grids' block means of a frame of `codes` uploaded in `format`.
fn measured_means(client: &Client, geometry: Geometry, codes: &[Vec<u32>; 3]) -> [Vec<f32>; 2] {
    let blocks = ColourBlocks::new(client, geometry).expect("a colour frame has block means");
    let (chroma_width, chroma_height) = geometry.plane_dims(1);
    let handles: Vec<Handle> = codes
        .iter()
        .map(|plane| {
            let bytes = pack(plane, geometry.input);
            client.create_from_slice(&bytes)
        })
        .collect();
    let planes = [
        DevicePlane::new(&handles[0], geometry.width, geometry.height),
        DevicePlane::new(&handles[1], chroma_width, chroma_height),
        DevicePlane::new(&handles[2], chroma_width, chroma_height),
    ];
    blocks.queue(&planes, 0);

    [ColourGrid::Luma, ColourGrid::Chroma].map(|grid| {
        let (width, height) = match grid {
            ColourGrid::Luma => (geometry.width, geometry.height),
            ColourGrid::Chroma => (chroma_width, chroma_height),
        };
        let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
        let len = (blocks_x * blocks_y * COLOUR_MEAN_FIELDS) as usize;
        let means = blocks.read(grid, 0).expect("frame 0's means are held");
        read_f32(client, &means, len).expect("the means read back")
    })
}

#[test]
fn high_bit_depth_maps_to_the_same_cells() {
    let client = test_client();
    let width = 72;
    let height = 40;
    let base = Geometry {
        width,
        height,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let (chroma_width, chroma_height) = base.plane_dims(1);
    let codes = colour_planes(width, height, chroma_width, chroma_height);
    let eight_bit = measured_means(&client, base, &codes);

    // The luma grid's first block covers chroma pixels 0..8, and the chroma grid's last column
    // covers luma pixels 64..72.
    let luma_block = [
        rectangle_mean(&codes[0], width, 0, 0, 16, 16),
        rectangle_mean(&codes[1], chroma_width, 0, 0, 8, 8) - 128.0,
        rectangle_mean(&codes[2], chroma_width, 0, 0, 8, 8) - 128.0,
    ];
    let chroma_block = [
        rectangle_mean(&codes[0], width, 64, 0, 72, 32),
        rectangle_mean(&codes[1], chroma_width, 32, 0, 36, 16) - 128.0,
        rectangle_mean(&codes[2], chroma_width, 32, 0, 36, 16) - 128.0,
    ];
    let (chroma_blocks_x, _) = temporal_stats_blocks(chroma_width, chroma_height);
    let last_column = ((chroma_blocks_x - 1) * COLOUR_MEAN_FIELDS) as usize;
    for field in 0..3 {
        assert_close(eight_bit[0][field], luma_block[field], 1.0e-3, "luma grid mean");
        assert_close(
            eight_bit[1][last_column + field],
            chroma_block[field],
            1.0e-3,
            "chroma grid mean",
        );
    }

    for depth in [10u8, 12] {
        let shift = u32::from(depth) - 8;
        let deep_codes = codes
            .clone()
            .map(|plane| plane.iter().map(|&code| code << shift).collect());
        let format = SampleFormat::U16 { depth };
        let geometry = Geometry {
            input: format,
            ..base
        };
        let deep = measured_means(&client, geometry, &deep_codes);

        assert_eq!(deep, eight_bit, "{depth}-bit means");

        let blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
        let source = blocks.source(ColourGrid::Luma);
        let max = ((1u32 << depth) - 1) as f32;
        let grain = 2.5 * (1u32 << shift) as f32 / max * source.grain_scale();
        assert_close(grain, 2.5, 1.0e-4, "the grain scale");
    }

    let float = Geometry {
        input: SampleFormat::F32,
        ..base
    };
    let float_means = measured_means(&client, float, &codes);
    for (grid, (measured, expected)) in float_means.iter().zip(&eight_bit).enumerate() {
        for (index, (value, reference)) in measured.iter().zip(expected).enumerate() {
            assert_close(
                *value,
                *reference,
                1.0e-3,
                &format!("float grid {grid} value {index}"),
            );
        }

        let cells: Vec<usize> = measured
            .chunks(3)
            .map(|mean| reference_cell(mean[0], mean[1], mean[2]))
            .collect();
        let expected_cells: Vec<usize> = expected
            .chunks(3)
            .map(|mean| reference_cell(mean[0], mean[1], mean[2]))
            .collect();
        assert_eq!(cells, expected_cells, "float grid {grid} cells");
    }
}

#[test]
fn letterbox_bars_do_not_shift_the_neutral_calibration() {
    let width = 64;
    let height = 64;
    let picture = ColourBlock::new(120.0, 2.0, -3.0, 1.5);
    let black_bar = ColourBlock::new(16.0, 0.0, 0.0, 0.0);
    let coded_bar = ColourBlock::new(16.0, 0.0, 0.0, 0.4);
    let flat_grey = ColourBlock::new(60.0, 0.0, 0.0, 0.0);

    let mut boxed = vec![black_bar; 4];
    boxed.extend([picture; 8]);
    boxed.extend([coded_bar, coded_bar, flat_grey, flat_grey]);

    let mut excluded = vec![ColourBlock { mean: 3.0, ..picture }; 4];
    excluded.extend([picture; 8]);
    excluded.extend([ColourBlock { mean: 3.0, ..picture }; 4]);

    let boxed_run = CellRun::new(width, height, 1, true);
    boxed_run.pool(&boxed);
    let picture_run = CellRun::new(width, height, 1, true);
    picture_run.pool(&excluded);

    let boxed_scene = boxed_run.scene();
    let picture_scene = picture_run.scene();
    assert_eq!(boxed_scene, picture_scene, "the bars add nothing");

    let brightness = (picture.luma / 16.0) as usize;
    let calibration = record(&boxed_scene, COLOUR_CELLS as usize + brightness);
    assert_eq!(calibration[COLOUR_CELL_COUNT as usize], 8.0);
    assert_close(
        calibration[COLOUR_CELL_LUMA_SUM as usize],
        8.0 * picture.luma,
        0.1,
        "luma",
    );
}

/// The shape of a two-frame luma noise state over a `width` by `height` plane.
fn noise_shape(client: &Client, width: u32, height: u32) -> NoiseShape {
    NoiseShape {
        width,
        height,
        channels: 1,
        frames: 2,
        ring_radius: 1,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

/// Two normalised frames of flat `luma` codes, each with its own hashed noise of about 2 codes.
fn noisy_ring(client: &Client, width: u32, height: u32, luma: f32, seed: u32) -> Handle {
    let pixels = width * height;
    let mut ring = Vec::with_capacity(2 * pixels as usize);
    for frame in 0..2 {
        for index in 0..pixels {
            let unit = hashed_unit(seed + frame, index);
            let code = luma + 7.0 * (unit - 0.5);
            ring.push(code / 255.0);
        }
    }

    let bytes = f32::as_bytes(&ring);
    client.create_from_slice(bytes)
}

/// Uploads flat 4:4:4 planes of `codes` and queues their block means as stream frame `frame`.
fn queue_flat_means(client: &Client, blocks: &ColourBlocks, geometry: Geometry, codes: [u8; 3], frame: u32) {
    let pixels = (geometry.width * geometry.height) as usize;
    let handles: Vec<Handle> = codes
        .iter()
        .map(|&code| {
            let plane = vec![code; pixels];
            client.create_from_slice(&plane)
        })
        .collect();
    let planes: Vec<DevicePlane<'_>> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, geometry.width, geometry.height))
        .collect();

    blocks.queue(&planes, frame);
}

/// A luma noise state pooling colour cells from `blocks`.
fn colour_state(client: &Client, blocks: &ColourBlocks, width: u32, height: u32) -> NoiseState {
    let shape = noise_shape(client, width, height);
    let mut state = NoiseState::new(client, shape, NoiseSettings::default(), DerivedValues::default());
    let source = blocks.source(ColourGrid::Luma);
    state.measure_colour(source);

    state
}

/// Pools one frame of flat `codes` with noise from `seed` into `state` as stream frame `frame`.
fn pool_frame(
    client: &Client,
    state: &NoiseState,
    blocks: &ColourBlocks,
    codes: [u8; 3],
    seed: u32,
    frame: u32,
) {
    let shape = state.shape();
    let geometry = Geometry {
        width: shape.width,
        height: shape.height,
        chroma: ChromaLayout::Yuv444,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let luma = f32::from(codes[0]);
    let ring = noisy_ring(client, shape.width, shape.height, luma, seed);

    queue_flat_means(client, blocks, geometry, codes, frame);
    state.queue_stats(&ring, 1, Some(0)).expect("the stats queue");
    state
        .queue_colour_cells(1, frame)
        .expect("the frame's means are held");
    state.record_reading(1, frame);
}

/// Folds the window around stream frame 0 of the frames `state` recorded, and reads its colour
/// sums back.
fn window_of(client: &Client, state: &NoiseState) -> Vec<f32> {
    let recorded = state.stream_frame_count();
    let bounds = WindowBounds::around(0, recorded - 1);
    state.queue_window_fold(bounds).expect("the window folds");

    let cells = state.colour_cells().expect("the state pools colour cells");
    read_f32(client, cells.window(), cells.words()).expect("the window's sums read back")
}

#[test]
fn a_frame_read_again_replaces_its_cells() {
    let client = test_client();
    let geometry = Geometry {
        width: 64,
        height: 48,
        chroma: ChromaLayout::Yuv444,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let first_read = [120, 128, 128];
    let second_read = [70, 160, 100];

    let blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
    let state = colour_state(&client, &blocks, geometry.width, geometry.height);
    pool_frame(&client, &state, &blocks, first_read, 11, 0);
    pool_frame(&client, &state, &blocks, second_read, 12, 0);
    let read_twice = window_of(&client, &state);

    let fresh_blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
    let fresh = colour_state(&client, &fresh_blocks, geometry.width, geometry.height);
    pool_frame(&client, &fresh, &fresh_blocks, second_read, 12, 0);
    let expected = window_of(&client, &fresh);

    assert_eq!(read_twice, expected, "the second read replaces the first");
}

#[test]
fn a_stream_reset_windows_only_the_new_stream() {
    let client = test_client();
    let width = 64;
    let height = 48;
    let geometry = Geometry {
        width,
        height,
        chroma: ChromaLayout::Yuv444,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let scene_a = [120, 128, 128];
    let scene_b = [70, 160, 100];

    let blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
    let state = colour_state(&client, &blocks, width, height);
    pool_frame(&client, &state, &blocks, scene_a, 11, 0);
    pool_frame(&client, &state, &blocks, scene_a, 12, 1);
    let first_scene = window_of(&client, &state);
    state.reset_stream();
    pool_frame(&client, &state, &blocks, scene_b, 13, 0);
    let after_reset = window_of(&client, &state);

    let fresh_blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
    let fresh = colour_state(&client, &fresh_blocks, width, height);
    pool_frame(&client, &fresh, &fresh_blocks, scene_b, 13, 0);
    let expected = window_of(&client, &fresh);

    let blocks_per_frame = {
        let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
        (blocks_x * blocks_y) as f32
    };
    let pooled = total_count(&first_scene, 0..COLOUR_CELLS as usize);
    assert_eq!(
        pooled,
        2.0 * blocks_per_frame,
        "both frames of the first scene pool"
    );
    assert_eq!(
        after_reset, expected,
        "the new stream's window holds only its own frames"
    );
}

#[test]
fn a_frame_without_held_means_is_an_error() {
    let client = test_client();
    let geometry = Geometry {
        width: 64,
        height: 48,
        chroma: ChromaLayout::Yuv444,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let blocks = ColourBlocks::new(&client, geometry).expect("a colour frame has block means");
    let state = colour_state(&client, &blocks, geometry.width, geometry.height);
    queue_flat_means(&client, &blocks, geometry, [120, 128, 128], 0);

    let stale = state.queue_colour_cells(1, 2);

    assert!(stale.is_err(), "frame 2's means were never measured");
}

fn hq_nlmeans() -> NlmeansAlgorithm {
    let nlm = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        ..NlmeansOptions::default()
    };
    let hq = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };

    NlmeansAlgorithm::Hq(hq)
}

#[test]
fn greyscale_builds_no_colour_cells() {
    let client = test_client();
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let pinned = Nl4dOptions {
        sigma: Some(2.0 / 255.0),
        ..options
    };
    let geometry = |chroma| Geometry {
        width: 64,
        height: 48,
        chroma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let gray = EngineSpec::Nl4d {
        luma: Some(options),
        chroma: None,
        geometry: geometry(ChromaLayout::Gray),
    };
    let colour = EngineSpec::Nl4d {
        luma: Some(options),
        chroma: None,
        geometry: geometry(ChromaLayout::Yuv420),
    };
    let fixed_sigma = EngineSpec::Nl4d {
        luma: Some(pinned),
        chroma: Some(pinned),
        geometry: geometry(ChromaLayout::Yuv420),
    };
    let nlmeans = EngineSpec::Nlmeans {
        luma: Some(hq_nlmeans()),
        chroma: Some(hq_nlmeans()),
        geometry: geometry(ChromaLayout::Yuv420),
    };

    assert!(ColourBlocks::new(&client, geometry(ChromaLayout::Gray)).is_none());
    assert!(gray.colour_blocks(&client).is_none());
    assert!(fixed_sigma.colour_blocks(&client).is_none());
    assert!(nlmeans.colour_blocks(&client).is_none());
    assert!(colour.colour_blocks(&client).is_some());
}
