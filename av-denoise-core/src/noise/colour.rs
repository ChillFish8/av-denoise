use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::history::{WindowBounds, history_capacity};
use super::kernels::{
    noise_colour_accumulate,
    noise_colour_clear,
    noise_colour_levels,
    noise_colour_means,
    noise_colour_window,
};
use super::state::NoiseState;
use super::temporal::temporal_stats_blocks;
use crate::engine::{ChromaLayout, DevicePlane, Geometry, SampleFormat};
use crate::tune::gpu::{bind, zeroed_words};

/// The threads of one cube of the per-block, per-word and per-cell colour launches.
pub(crate) const COLOUR_BLOCK_THREADS: u32 = 256;
/// The threads of one block-mean cube, each summing a stride of the block's pixels.
pub(crate) const COLOUR_MEAN_THREADS: u32 = 64;
/// The brightness bins of the cell table.
pub(crate) const COLOUR_BRIGHTNESS_BINS: u32 = 16;
/// The codes one brightness bin spans, on the 8-bit scale.
pub(crate) const COLOUR_BRIGHTNESS_STEP: f32 = 16.0;
/// The chroma magnitudes that open each saturation level past neutral, on the 8-bit scale.
pub(crate) const COLOUR_SATURATION_EDGES: [f32; 5] = [6.0, 12.0, 20.0, 30.0, 44.0];
/// The equal hue sectors of every saturation level past neutral.
pub(crate) const COLOUR_HUE_BINS: u32 = 12;
/// The cells of one brightness bin, one neutral cell then a hue sector at each saturation level.
pub(crate) const COLOUR_CELLS_PER_BRIGHTNESS: u32 =
    1 + COLOUR_SATURATION_EDGES.len() as u32 * COLOUR_HUE_BINS;
/// The colour cells of one plane.
pub(crate) const COLOUR_CELLS: u32 = COLOUR_BRIGHTNESS_BINS * COLOUR_CELLS_PER_BRIGHTNESS;
/// The calibration records of the luma plane, one per brightness bin.
///
/// They pool the neutral blocks, whose chroma magnitude sits below
/// [COLOUR_NEUTRAL_SATURATION], which the luma gain is calibrated against.
pub(crate) const COLOUR_CALIBRATION_RECORDS: u32 = COLOUR_BRIGHTNESS_BINS;
/// The chroma magnitude below which a block counts towards the luma calibration.
pub(crate) const COLOUR_NEUTRAL_SATURATION: f32 = 8.0;

/// The cell record word of the block count.
pub(crate) const COLOUR_CELL_COUNT: u32 = 0;
/// The cell record word of the summed log grain.
pub(crate) const COLOUR_CELL_LOG_SUM: u32 = 1;
/// The cell record word of the summed block luma.
pub(crate) const COLOUR_CELL_LUMA_SUM: u32 = 2;
/// The cell record word of the summed block Cb, centred on 0.
pub(crate) const COLOUR_CELL_CB_SUM: u32 = 3;
/// The cell record word of the summed block Cr, centred on 0.
pub(crate) const COLOUR_CELL_CR_SUM: u32 = 4;
/// The cell record word of the first histogram bin.
pub(crate) const COLOUR_CELL_HISTOGRAM: u32 = 5;
/// The bins of each cell's histogram of log grain.
pub(crate) const COLOUR_HISTOGRAM_BINS: u32 = 64;
/// The log grain at the low edge of the first histogram bin.
///
/// Log grain below it counts in the first bin and above the last edge in the last bin.
pub(crate) const COLOUR_HISTOGRAM_LOG_MIN: f32 = -3.5;
/// The log grain one histogram bin spans.
pub(crate) const COLOUR_HISTOGRAM_LOG_STEP: f32 = 0.125;
/// The log grain at the high edge of the last histogram bin.
pub(crate) const COLOUR_HISTOGRAM_LOG_MAX: f32 =
    COLOUR_HISTOGRAM_LOG_MIN + COLOUR_HISTOGRAM_BINS as f32 * COLOUR_HISTOGRAM_LOG_STEP;
/// The words of one cell record.
pub(crate) const COLOUR_CELL_WORDS: u32 = COLOUR_CELL_HISTOGRAM + COLOUR_HISTOGRAM_BINS;

/// The means each block of a grid holds, luma then Cb then Cr.
pub(crate) const COLOUR_MEAN_FIELDS: u32 = 3;
/// The values each cell's level holds, its kept block count then its level.
pub(crate) const COLOUR_LEVEL_FIELDS: u32 = 2;
/// The chroma code of no colour, on the 8-bit scale.
pub(crate) const COLOUR_CHROMA_CENTRE: f32 = 128.0;
/// The block luma at or below which a block is left out, on the 8-bit scale.
pub(crate) const COLOUR_LUMA_MIN: f32 = 20.0;
/// The block luma at or above which a block is left out, on the 8-bit scale.
pub(crate) const COLOUR_LUMA_MAX: f32 = 240.0;
/// The luma grain at or below which a block carries no measurable grain, on the 8-bit scale.
///
/// Letterbox bars and other perfectly flat areas read a grain of 0.
pub(crate) const COLOUR_LUMA_GRAIN_FLOOR: f32 = 0.05;
/// The chroma grain at or below which a block carries no measurable grain, on the 8-bit scale.
pub(crate) const COLOUR_CHROMA_GRAIN_FLOOR: f32 = 0.03;

/// The fixed-point scale of a frame's log grain sums.
///
/// Every block of an 8K frame, all in one cell at the histogram's edge, still fits an `i32`.
pub(crate) const COLOUR_LOG_FIXED: f32 = 2048.0;
/// The fixed-point scale of a frame's colour sums.
pub(crate) const COLOUR_MEAN_FIXED: f32 = 16.0;

/// The frames the block means are held for.
///
/// A motion-compensated stream reads its first frame again on its second push.
const COLOUR_SLOTS: usize = 2;

/// Which noise grid a pass reads its block means on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColourGrid {
    /// The 16x16 blocks of the luma plane.
    Luma,
    /// The 16x16 blocks of the chroma planes.
    Chroma,
}

/// The mean luma, Cb and Cr of every noise block of the newest frames, on both noise grids.
///
/// The engine measures each frame's means from all three planes before its passes see the frame,
/// so the luma pass reads the chroma of its blocks and the chroma pass the luma of its blocks. Each
/// slot is tagged with its frame, so a read never returns another frame's means.
#[derive(Clone)]
pub(crate) struct ColourBlocks {
    client: Client,
    geometry: Geometry,
    /// Each slot's luma grid means, then its chroma grid means.
    slots: Arc<[[Handle; 2]]>,
    tags: Arc<Mutex<[Option<u32>; COLOUR_SLOTS]>>,
}

impl ColourBlocks {
    /// The block means of a stream of `geometry` frames, `None` for gray frames.
    pub(crate) fn new(client: &Client, geometry: Geometry) -> Option<Self> {
        if geometry.chroma == ChromaLayout::Gray {
            return None;
        }

        let luma_bytes =
            grid_blocks(geometry, ColourGrid::Luma) * COLOUR_MEAN_FIELDS as usize * size_of::<f32>();
        let chroma_bytes =
            grid_blocks(geometry, ColourGrid::Chroma) * COLOUR_MEAN_FIELDS as usize * size_of::<f32>();
        let slots: Arc<[[Handle; 2]]> = (0..COLOUR_SLOTS)
            .map(|_| [client.empty(luma_bytes), client.empty(chroma_bytes)])
            .collect();

        let blocks = Self {
            client: client.clone(),
            geometry,
            slots,
            tags: Arc::new(Mutex::new([None; COLOUR_SLOTS])),
        };

        Some(blocks)
    }

    /// Queues the measure of stream frame `frame`'s block means from its `planes`, luma first.
    pub(crate) fn queue(&self, planes: &[DevicePlane<'_>], frame: u32) {
        let index = frame as usize % COLOUR_SLOTS;
        let [luma_means, chroma_means] = &self.slots[index];
        let geometry = self.geometry;
        let format = geometry.input;
        let subsampling = geometry.chroma.subsampling();
        let (luma_blocks_x, luma_blocks_y) = temporal_stats_blocks(geometry.width, geometry.height);
        let (chroma_width, chroma_height) = geometry.plane_dims(1);
        let (_, chroma_blocks_y) = temporal_stats_blocks(chroma_width, chroma_height);
        let luma_words = plane_words(format, geometry.width, geometry.height);
        let chroma_words = plane_words(format, chroma_width, chroma_height);
        let luma_len = grid_blocks(geometry, ColourGrid::Luma) * COLOUR_MEAN_FIELDS as usize;
        let chroma_len = grid_blocks(geometry, ColourGrid::Chroma) * COLOUR_MEAN_FIELDS as usize;
        let float = format == SampleFormat::F32;

        unsafe {
            noise_colour_means::launch_unchecked(
                &self.client,
                CubeCount::new_2d(luma_blocks_x, luma_blocks_y + chroma_blocks_y),
                CubeDim::new_1d(COLOUR_MEAN_THREADS),
                bind(planes[0].handle(), luma_words),
                bind(planes[1].handle(), chroma_words),
                bind(planes[2].handle(), chroma_words),
                bind(luma_means, luma_len),
                bind(chroma_means, chroma_len),
                eight_bit_scale(format),
                geometry.width,
                geometry.height,
                subsampling.shift_x,
                subsampling.shift_y,
                format.samples_per_word(),
                float,
            );
        }

        let mut tags = self.lock_tags();
        tags[index] = Some(frame);
    }

    /// Stream frame `frame`'s means on `grid`, or `None` if its slot holds another frame.
    pub(crate) fn read(&self, grid: ColourGrid, frame: u32) -> Option<Handle> {
        let index = frame as usize % COLOUR_SLOTS;
        let tags = self.lock_tags();
        let holds_frame = tags[index] == Some(frame);
        let means = &self.slots[index][grid_index(grid)];

        holds_frame.then(|| means.clone())
    }

    /// Empties every slot.
    pub(crate) fn clear(&self) {
        let mut tags = self.lock_tags();
        tags.fill(None);
    }

    /// The source a pass on `grid` reads its block means from.
    pub(crate) fn source(&self, grid: ColourGrid) -> ColourSource {
        ColourSource {
            blocks: self.clone(),
            grid,
        }
    }

    fn lock_tags(&self) -> MutexGuard<'_, [Option<u32>; COLOUR_SLOTS]> {
        self.tags.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The block means one pass reads, on its own grid.
#[derive(Clone)]
pub(crate) struct ColourSource {
    blocks: ColourBlocks,
    grid: ColourGrid,
}

impl ColourSource {
    /// The width and height of the plane the grid covers.
    fn plane_dims(&self) -> (u32, u32) {
        let geometry = self.blocks.geometry;
        match self.grid {
            ColourGrid::Luma => (geometry.width, geometry.height),
            ColourGrid::Chroma => geometry.plane_dims(1),
        }
    }

    /// The factor from a normalised residual to the 8-bit scale.
    pub(crate) fn grain_scale(&self) -> f32 {
        let format = self.blocks.geometry.input;
        format.max_value() * eight_bit_scale(format)
    }
}

/// The factor from a sample code in `format` to the 8-bit scale.
///
/// A high bit depth code divides by its extra bits, so the same content lands on the same value. A
/// float sample is normalised, so it scales by 255.
fn eight_bit_scale(format: SampleFormat) -> f32 {
    match format {
        SampleFormat::U8 => 1.0,
        SampleFormat::U16 { depth } => {
            let extra_bits = u32::from(depth) - 8;
            let divisor = (1u32 << extra_bits) as f32;
            1.0 / divisor
        },
        SampleFormat::F32 => 255.0,
    }
}

fn plane_words(format: SampleFormat, width: u32, height: u32) -> usize {
    let pixels = u64::from(width) * u64::from(height);
    let bytes = format.plane_bytes(pixels);

    bytes as usize / size_of::<u32>()
}

fn grid_index(grid: ColourGrid) -> usize {
    match grid {
        ColourGrid::Luma => 0,
        ColourGrid::Chroma => 1,
    }
}

/// The noise blocks of `grid` in a frame of `geometry`.
fn grid_blocks(geometry: Geometry, grid: ColourGrid) -> usize {
    let (width, height) = match grid {
        ColourGrid::Luma => (geometry.width, geometry.height),
        ColourGrid::Chroma => geometry.plane_dims(1),
    };
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);

    (blocks_x * blocks_y) as usize
}

/// One frame's inputs to the cell sums.
pub(crate) struct CellFrame<'a> {
    /// One slot's temporal-stats records.
    pub records: &'a Handle,
    pub records_len: usize,
    /// The block means of the same frame, on the records' grid.
    pub means: &'a Handle,
    /// The width and height of the plane the records cover.
    pub width: u32,
    pub height: u32,
    /// The factor from a normalised residual to the 8-bit scale.
    pub grain_scale: f32,
}

/// The grain of one pass's planes per colour cell, one ring slot per stream frame, and the sums of
/// the last noise window.
///
/// Each plane holds [COLOUR_CELLS] cell records of [COLOUR_CELL_WORDS] words. The luma plane also
/// holds [COLOUR_CALIBRATION_RECORDS] calibration records after its cells. Stream frame `frame`
/// keeps its fixed-point sums in slot `frame % capacity`, and a frame read again replaces them. The
/// window's sums are summed from the slots afresh for every window, so they depend only on the
/// window's own frames.
pub(crate) struct ColourCells {
    planes: u32,
    luma: bool,
    capacity: u32,
    /// Every slot's sums in fixed point.
    frames: Handle,
    /// The last window's sums, decoded.
    window: Handle,
}

impl ColourCells {
    /// Cells for `planes` planes, which are luma with `luma` and chroma otherwise, with a ring of
    /// `capacity` frames.
    pub(crate) fn new(client: &Client, planes: u32, luma: bool, capacity: u32) -> Self {
        let records = record_count(planes, luma);
        let words = (records * COLOUR_CELL_WORDS) as usize;
        let ring_words = words * capacity as usize;

        Self {
            planes,
            luma,
            capacity,
            frames: zeroed_words(client, ring_words),
            window: zeroed_words(client, words),
        }
    }

    /// The cell and calibration records each frame holds.
    pub(crate) fn records(&self) -> u32 {
        record_count(self.planes, self.luma)
    }

    /// The words one frame's sums and the window's sums hold.
    pub(crate) fn words(&self) -> usize {
        (self.records() * COLOUR_CELL_WORDS) as usize
    }

    /// The last window's decoded sums.
    pub(crate) fn window(&self) -> &Handle {
        &self.window
    }

    /// The fixed-point sums of every ring slot, slot after slot.
    #[cfg(test)]
    pub(crate) fn frames(&self) -> &Handle {
        &self.frames
    }

    /// The ring slot of stream frame `frame`.
    pub(crate) fn slot(&self, frame: u32) -> u32 {
        frame % self.capacity
    }

    /// Queues the sums of stream frame `frame`'s blocks into its slot, in two launches.
    pub(crate) fn queue_frame(&self, client: &Client, input: &CellFrame<'_>, frame: u32) {
        let (blocks_x, blocks_y) = temporal_stats_blocks(input.width, input.height);
        let blocks = blocks_x * blocks_y;
        let means_len = (blocks * COLOUR_MEAN_FIELDS) as usize;
        let words = self.words() as u32;
        let ring_len = (words * self.capacity) as usize;
        let slot = self.slot(frame);
        let slot_base = slot * words;
        let word_cubes = words.div_ceil(COLOUR_BLOCK_THREADS);
        let block_cubes = blocks.div_ceil(COLOUR_BLOCK_THREADS);

        unsafe {
            noise_colour_clear::launch_unchecked(
                client,
                CubeCount::new_1d(word_cubes),
                CubeDim::new_1d(COLOUR_BLOCK_THREADS),
                bind(&self.frames, ring_len),
                slot_base,
                words,
            );

            noise_colour_accumulate::launch_unchecked(
                client,
                CubeCount::new_1d(block_cubes),
                CubeDim::new_1d(COLOUR_BLOCK_THREADS),
                bind(input.records, input.records_len),
                bind(input.means, means_len),
                bind(&self.frames, ring_len),
                input.grain_scale,
                slot_base,
                input.width,
                input.height,
                self.planes,
                self.luma,
            );
        }
    }

    /// Queues the sums of the stream frames `bounds` names into the window's sums, in one launch.
    pub(crate) fn queue_window(&self, client: &Client, bounds: WindowBounds) {
        let words = self.words() as u32;
        let ring_len = (words * self.capacity) as usize;
        let word_cubes = words.div_ceil(COLOUR_BLOCK_THREADS);

        unsafe {
            noise_colour_window::launch_unchecked(
                client,
                CubeCount::new_1d(word_cubes),
                CubeDim::new_1d(COLOUR_BLOCK_THREADS),
                bind(&self.frames, ring_len),
                bind(&self.window, words as usize),
                bounds.first(),
                bounds.last(),
                self.capacity,
                words,
            );
        }
    }
}

fn record_count(planes: u32, luma: bool) -> u32 {
    let calibration = if luma { COLOUR_CALIBRATION_RECORDS } else { 0 };
    planes * COLOUR_CELLS + calibration
}

/// Queues every colour cell's kept count and level from the window's sums `window`, leaving out
/// the blocks above `ceilings`, into `levels`.
///
/// `cells` covers every plane's colour cells. The levels are mean log grain with `mean_level` and
/// median log grain otherwise.
pub(crate) fn queue_colour_levels(
    client: &Client,
    window: &Handle,
    ceilings: &Handle,
    levels: &Handle,
    cells: u32,
    mean_level: bool,
) {
    let window_len = (cells * COLOUR_CELL_WORDS) as usize;
    let levels_len = (cells * COLOUR_LEVEL_FIELDS) as usize;

    unsafe {
        noise_colour_levels::launch_unchecked(
            client,
            CubeCount::new_1d(cells.div_ceil(COLOUR_BLOCK_THREADS)),
            CubeDim::new_1d(COLOUR_BLOCK_THREADS),
            bind(window, window_len),
            bind(ceilings, cells as usize),
            bind(levels, levels_len),
            cells,
            mean_level,
        );
    }
}

/// A pass's block means and the cells it pools them into.
pub(crate) struct ColourMeasure {
    source: ColourSource,
    cells: ColourCells,
}

impl NoiseState {
    /// Pools each reading's blocks into colour cells from the next reading on, with the block means
    /// `source` holds.
    ///
    /// A state that holds fixed values or keeps no temporal stats pools nothing.
    pub(crate) fn measure_colour(&mut self, source: ColourSource) {
        let measurement = self.measurement.as_ref();
        let has_stats = measurement.is_some_and(|measurement| measurement.stats.is_some());
        if !has_stats {
            return;
        }

        let plane_dims = source.plane_dims();
        let shape_dims = (self.shape.width, self.shape.height);
        debug_assert_eq!(plane_dims, shape_dims, "the block means cover the measured plane");

        let luma = source.grid == ColourGrid::Luma;
        let capacity = history_capacity(self.shape.ring_radius);
        let cells = ColourCells::new(&self.client, self.shape.channels, luma, capacity);
        let measure = ColourMeasure { source, cells };

        self.colour = Some(measure);
    }

    /// Queues the pooling of the stats in `slot`, read as stream frame `frame`, into that frame's
    /// colour cells.
    ///
    /// A state that pools no colour cells queues nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when the block means of `frame` are not held.
    pub(crate) fn queue_colour_cells(&self, slot: u32, frame: u32) -> Result<(), anyhow::Error> {
        let Some(colour) = self.colour.as_ref() else {
            return Ok(());
        };

        let source = &colour.source;
        let means = source.blocks.read(source.grid, frame);
        let means =
            means.ok_or_else(|| anyhow::anyhow!("the block colour of stream frame {frame} is not held"))?;
        let records = self.records_view(slot);
        let (records, records_len) = records.expect("colour cells imply temporal stats");

        let input = CellFrame {
            records: &records,
            records_len,
            means: &means,
            width: self.shape.width,
            height: self.shape.height,
            grain_scale: source.grain_scale(),
        };
        colour.cells.queue_frame(&self.client, &input, frame);

        Ok(())
    }

    /// Queues the sums of the colour cells of the stream frames `bounds` names.
    ///
    /// A state that pools no colour cells queues nothing.
    pub(super) fn queue_colour_window(&self, bounds: WindowBounds) {
        if let Some(colour) = self.colour.as_ref() {
            colour.cells.queue_window(&self.client, bounds);
        }
    }

    /// The colour cells, `None` for a state that pools none.
    #[cfg(test)]
    pub(crate) fn colour_cells(&self) -> Option<&ColourCells> {
        let colour = self.colour.as_ref();
        colour.map(|colour| &colour.cells)
    }
}
