use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{ChromaLayout, DevicePlane, Geometry, SampleFormat};
use crate::noise::colour::{
    COLOUR_CELLS,
    COLOUR_LEVEL_FIELDS,
    CellFrame,
    ColourBlocks,
    ColourCells,
    ColourGrid,
    queue_colour_levels,
};
use crate::noise::{
    NOISE_WINDOW_LEN,
    NOISE_WINDOW_RADIUS,
    WindowBounds,
    history_capacity,
    temporal_stats_blocks,
    temporal_stats_record_len,
};

/// The temporal radius of nl4d's default window, which sizes the bench's frame ring.
const BENCH_RING_RADIUS: u32 = 2;

/// The launches of the colour cell measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourStage {
    /// The block means of both grids, from all three planes.
    Means,
    /// One frame's cell sums, cleared and added into its ring slot.
    Pool,
    /// The sum of a full noise window's frames.
    Window,
    /// Every cell's level against a ceiling.
    Levels,
}

/// The colour cells of one pass over a 4:2:0 frame of 8-bit planes with hashed colour and grain.
pub struct ColourMeasurement {
    client: Client,
    planes: Vec<Handle>,
    geometry: Geometry,
    blocks: ColourBlocks,
    cells: ColourCells,
    grid: ColourGrid,
    channels: u32,
    records: Handle,
    records_len: usize,
    ceilings: Handle,
    levels: Handle,
}

impl ColourMeasurement {
    /// The measurement of the luma pass, or with `chroma` the chroma pass, of a `width` by `height`
    /// frame, with frame 0's block means already queued.
    pub fn new(client: &Client, width: u32, height: u32, chroma: bool) -> Self {
        let geometry = Geometry {
            width,
            height,
            chroma: ChromaLayout::Yuv420,
            input: SampleFormat::U8,
            output: SampleFormat::U8,
        };
        let (chroma_width, chroma_height) = geometry.plane_dims(1);
        let luma_plane = hashed_plane(width, height, 1);
        let cb_plane = hashed_plane(chroma_width, chroma_height, 2);
        let cr_plane = hashed_plane(chroma_width, chroma_height, 3);
        let planes = vec![
            client.create_from_slice(&luma_plane),
            client.create_from_slice(&cb_plane),
            client.create_from_slice(&cr_plane),
        ];

        let blocks = ColourBlocks::new(client, geometry).expect("a 4:2:0 frame has block means");
        let (grid, channels, plane_width, plane_height) = if chroma {
            (ColourGrid::Chroma, 2, chroma_width, chroma_height)
        } else {
            (ColourGrid::Luma, 1, width, height)
        };
        let capacity = history_capacity(BENCH_RING_RADIUS);
        let cells = ColourCells::new(client, channels, !chroma, capacity);

        let records = hashed_records(plane_width, plane_height, channels);
        let records_len = records.len();
        let records_bytes = f32::as_bytes(&records);
        let records = client.create_from_slice(records_bytes);

        let cell_count = (channels * COLOUR_CELLS) as usize;
        let ceilings = vec![1.0f32; cell_count];
        let ceilings_bytes = f32::as_bytes(&ceilings);
        let ceilings = client.create_from_slice(ceilings_bytes);
        let levels_bytes = cell_count * COLOUR_LEVEL_FIELDS as usize * size_of::<f32>();
        let levels = client.empty(levels_bytes);

        let measurement = Self {
            client: client.clone(),
            planes,
            geometry,
            blocks,
            cells,
            grid,
            channels,
            records,
            records_len,
            ceilings,
            levels,
        };
        measurement.queue_means();
        for frame in 0..NOISE_WINDOW_LEN {
            measurement.queue_pool_frame(frame);
        }
        measurement.queue_window();

        measurement
    }

    /// Queues the launches of `stage`.
    pub fn queue_stage(&self, stage: ColourStage) {
        match stage {
            ColourStage::Means => self.queue_means(),
            ColourStage::Pool => self.queue_pool_frame(0),
            ColourStage::Window => self.queue_window(),
            ColourStage::Levels => self.queue_levels(),
        }
    }

    fn queue_means(&self) {
        let geometry = self.geometry;
        let (chroma_width, chroma_height) = geometry.plane_dims(1);
        let planes = [
            DevicePlane::new(&self.planes[0], geometry.width, geometry.height),
            DevicePlane::new(&self.planes[1], chroma_width, chroma_height),
            DevicePlane::new(&self.planes[2], chroma_width, chroma_height),
        ];

        self.blocks.queue(&planes, 0);
    }

    /// Queues the pooling of the held frame 0's means and the stats as stream frame `frame`.
    fn queue_pool_frame(&self, frame: u32) {
        let means = self.blocks.read(self.grid, 0).expect("frame 0's means are held");
        let (width, height) = match self.grid {
            ColourGrid::Luma => (self.geometry.width, self.geometry.height),
            ColourGrid::Chroma => self.geometry.plane_dims(1),
        };
        let input = CellFrame {
            records: &self.records,
            records_len: self.records_len,
            means: &means,
            width,
            height,
            grain_scale: 255.0,
        };

        self.cells.queue_frame(&self.client, &input, frame);
    }

    /// Queues the sum of a full window centred on stream frame `NOISE_WINDOW_RADIUS`.
    fn queue_window(&self) {
        let newest = NOISE_WINDOW_LEN - 1;
        let bounds = WindowBounds::around(NOISE_WINDOW_RADIUS, newest);

        self.cells.queue_window(&self.client, bounds);
    }

    fn queue_levels(&self) {
        let cells = self.channels * COLOUR_CELLS;
        let mean_level = self.grid == ColourGrid::Luma;

        queue_colour_levels(
            &self.client,
            self.cells.window(),
            &self.ceilings,
            &self.levels,
            cells,
            mean_level,
        );
    }
}

/// A value between 0 and 1 from a hash of `seed` and `index`.
fn hashed_unit(seed: u32, index: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// An 8-bit plane whose codes drift smoothly between 16 and 240, plus a little hashed noise.
fn hashed_plane(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut plane = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let phase = (x as f32 * 0.01 + y as f32 * 0.013 + seed as f32).sin();
            let noise = hashed_unit(seed, y * width + x) - 0.5;
            let code = 128.0 + 100.0 * phase + 6.0 * noise;
            plane.push(code.clamp(16.0, 240.0) as u8);
        }
    }

    plane
}

/// One slot's records of a `width` by `height` plane, each channel of each block static with a
/// hashed grain between 0.5 and 4.5 codes.
fn hashed_records(width: u32, height: u32, channels: u32) -> Vec<f32> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let record_len = temporal_stats_record_len(channels) as usize;
    let blocks = (blocks_x * blocks_y) as usize;
    let pixels = 256.0f32;
    let mut records = vec![0.0f32; blocks * record_len];

    for block in 0..blocks {
        for channel in 0..channels as usize {
            let unit = hashed_unit(7 + channel as u32, block as u32);
            let sigma = (0.5 + 4.0 * unit) / 255.0;
            let variance = 2.0 * sigma * sigma;
            records[block * record_len + channels as usize + channel] = variance * pixels;
        }
    }

    records
}
