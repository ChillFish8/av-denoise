use av_denoise_core::bench_api::nl4d_kernels::{
    grain_measure,
    grain_persist,
    grain_persist_reduce,
    grain_project,
    grain_reduce_partials,
    grain_save_vectors,
};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame};

const STEP: u32 = 8;
const NEIGHBOURS: u32 = 4;
const RING: u32 = 5;
const SAVE_THREADS: u32 = 256;
const CELL: u32 = 8;
/// Two histograms of 16 luma bins by 64 std buckets.
const HIST_TOTAL: usize = 2 * 16 * 64;
/// 46 lag sums and a pixel count, the lanes of one autocovariance record.
const RECORD_LANES: usize = 47;
/// One record and a strength group per cell.
const PARTIAL_LANES: usize = RECORD_LANES + 1;
const STRENGTH_GROUPS: usize = 16;
/// One record per strength group.
const GROUPED_RECORD_LANES: usize = STRENGTH_GROUPS * RECORD_LANES;
/// Frames in one grain chunk, each with its own histogram and record slot.
const CHUNK_FRAMES: usize = 24;
/// The chunk slot the benched frame writes, the last one.
const MEASURED_FRAME: usize = CHUNK_FRAMES - 1;
const REDUCE_THREADS: u32 = 128;
const EDGE_COUNT: usize = 65;
const PROJECTIONS: usize = 8;
/// Two words per projection pattern.
const PATTERN_WORDS: usize = 2 * PROJECTIONS;
/// The projections and the flat flag of one cell in one history frame.
const HISTORY_ENTRY_LEN: usize = PROJECTIONS + 1;
const HISTORY_FRAMES: u32 = 9;
const PERSISTENCE_LAGS: u32 = 8;
/// A cell count, auto sum and cross sum per lag.
const FRAME_RECORD_LEN: usize = 3 * PERSISTENCE_LAGS as usize;
const PERSIST_THREADS: u32 = 256;
/// How many luma pixels span one chroma pixel each way in 4:2:0.
const CHROMA_SUBSAMPLING: u32 = 2;
/// The Cb and Cr channels of a chroma frame.
const CHROMA_CHANNELS: u32 = 2;
/// The channel a luma frame is read on.
const LUMA_CHANNEL: u32 = 0;
/// The Cr channel of a chroma frame, after Cb.
const CR_CHANNEL: u32 = 1;
/// Cells one persisting cube adds, four per thread.
const PERSIST_BLOCK_CELLS: u32 = 4 * PERSIST_THREADS;

/// A frame size the grain benches run at.
#[derive(Clone, Copy)]
pub struct GrainSize {
    pub width: u32,
    pub height: u32,
    pub label: &'static str,
}

impl GrainSize {
    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn blocks(&self) -> u32 {
        self.width.div_ceil(STEP) * self.height.div_ceil(STEP)
    }

    fn cells(&self) -> usize {
        ((self.width / CELL) * (self.height / CELL)) as usize
    }

    fn persist_blocks(&self) -> u32 {
        let cells = self.cells() as u32;
        cells.div_ceil(PERSIST_BLOCK_CELLS)
    }

    fn persist_partials_len(&self) -> usize {
        self.persist_blocks() as usize * FRAME_RECORD_LEN
    }
}

pub const GRAIN_SIZES: &[GrainSize] = &[
    GrainSize {
        width: WIDTH,
        height: HEIGHT,
        label: "1080p",
    },
    GrainSize {
        width: 3840,
        height: 2160,
        label: "4k",
    },
];

/// Saves one neighbour's motion field into the grain export's vector ring.
pub struct GrainSaveVectorsBench {
    pub client: Client,
    pub size: GrainSize,
}

#[derive(Clone)]
pub struct GrainSaveVectorsInput {
    pub mv: Handle,
    pub conf: Handle,
    pub saved_mv: Handle,
    pub saved_conf: Handle,
}

impl Benchmark for GrainSaveVectorsBench {
    type Input = GrainSaveVectorsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks = self.size.blocks() as usize;
        let mv_host = vec![1i32; NEIGHBOURS as usize * blocks * 2];
        let conf_host = vec![0.9f32; NEIGHBOURS as usize * blocks];
        let mv_bytes = i32::as_bytes(&mv_host);
        let mv = self.client.create_from_slice(mv_bytes);
        let conf_bytes = f32::as_bytes(&conf_host);
        let conf = self.client.create_from_slice(conf_bytes);
        let saved_mv = self.client.empty(RING as usize * blocks * 2 * size_of::<i32>());
        let saved_conf = self.client.empty(RING as usize * blocks * size_of::<f32>());

        GrainSaveVectorsInput {
            mv,
            conf,
            saved_mv,
            saved_conf,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let blocks = self.size.blocks();
        let grid = blocks.div_ceil(SAVE_THREADS);

        unsafe {
            grain_save_vectors::launch_unchecked(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(SAVE_THREADS),
                BufferArg::from_raw_parts(args.mv, (NEIGHBOURS * blocks * 2) as usize),
                BufferArg::from_raw_parts(args.conf, (NEIGHBOURS * blocks) as usize),
                BufferArg::from_raw_parts(args.saved_mv, (RING * blocks * 2) as usize),
                BufferArg::from_raw_parts(args.saved_conf, (RING * blocks) as usize),
                2 * blocks * 2,
                2 * blocks,
                1u32,
                blocks,
                grid * SAVE_THREADS,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_save_vectors_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Measures the source and kept grain of one luma frame.
pub struct GrainMeasureBench {
    pub client: Client,
    pub size: GrainSize,
}

#[derive(Clone)]
pub struct GrainMeasureInput {
    pub input: Handle,
    pub out_t: Handle,
    pub out_prev: Handle,
    pub saved_mv: Handle,
    pub saved_conf: Handle,
    pub edges: Handle,
    pub hist: Handle,
    pub partials: Handle,
}

impl Benchmark for GrainMeasureBench {
    type Input = GrainMeasureInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = self.size.pixels();
        let blocks = self.size.blocks() as usize;
        let cells = self.size.cells();

        // Two ring slots, a frame and a copy moved by one pixel, so every cell has per-pixel grain.
        let frame = make_synthetic_frame(self.size.width, self.size.height, 1);
        let mut ring = frame.clone();
        ring.extend_from_slice(&frame[1..]);
        ring.push(frame[0]);

        // Flat outputs with a small per-pixel ripple, so the kept grain is not zero either.
        let out_prev_host = vec![0.5f32; pixels];
        let out_t_host: Vec<f32> = (0..pixels)
            .map(|index| 0.5 + 0.001 * ((index % 7) as f32 - 3.0))
            .collect();
        let saved_mv_host = vec![0i32; 2 * blocks * 2];
        let saved_conf_host = vec![1.0f32; 2 * blocks];
        let edges_host: Vec<f32> = (0..EDGE_COUNT).map(|index| index as f32 * 0.002).collect();
        let hist_host = vec![0i32; CHUNK_FRAMES * HIST_TOTAL];

        let ring_bytes = f32::as_bytes(&ring);
        let input = self.client.create_from_slice(ring_bytes);
        let out_t_bytes = f32::as_bytes(&out_t_host);
        let out_t = self.client.create_from_slice(out_t_bytes);
        let out_prev_bytes = f32::as_bytes(&out_prev_host);
        let out_prev = self.client.create_from_slice(out_prev_bytes);
        let saved_mv_bytes = i32::as_bytes(&saved_mv_host);
        let saved_mv = self.client.create_from_slice(saved_mv_bytes);
        let saved_conf_bytes = f32::as_bytes(&saved_conf_host);
        let saved_conf = self.client.create_from_slice(saved_conf_bytes);
        let edges_bytes = f32::as_bytes(&edges_host);
        let edges = self.client.create_from_slice(edges_bytes);
        let hist_bytes = i32::as_bytes(&hist_host);
        let hist = self.client.create_from_slice(hist_bytes);
        let partials = self.client.empty(cells * PARTIAL_LANES * size_of::<f32>());

        GrainMeasureInput {
            input,
            out_t,
            out_prev,
            saved_mv,
            saved_conf,
            edges,
            hist,
            partials,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let width = self.size.width;
        let height = self.size.height;
        let pixels = self.size.pixels();
        let blocks = self.size.blocks() as usize;
        let cells = self.size.cells();
        let blocks_x = width.div_ceil(STEP);
        let blocks_y = height.div_ceil(STEP);
        let hist_offset = (MEASURED_FRAME * HIST_TOTAL) as u32;

        unsafe {
            grain_measure::launch_unchecked(
                &self.client,
                CubeCount::new_2d(width / CELL, height / CELL),
                CubeDim::new_2d(CELL, CELL),
                1usize,
                BufferArg::from_raw_parts(args.input, 2 * pixels),
                BufferArg::from_raw_parts(args.out_t, pixels),
                BufferArg::from_raw_parts(args.out_prev, pixels),
                BufferArg::from_raw_parts(args.saved_mv, 2 * blocks * 2),
                BufferArg::from_raw_parts(args.saved_conf, 2 * blocks),
                BufferArg::from_raw_parts(args.edges, EDGE_COUNT),
                BufferArg::from_raw_parts(args.hist, CHUNK_FRAMES * HIST_TOTAL),
                BufferArg::from_raw_parts(args.partials, cells * PARTIAL_LANES),
                0u32,
                1u32,
                0u32,
                1u32,
                1u32,
                1u32,
                hist_offset,
                width,
                height,
                1u32,
                blocks_x,
                blocks_y,
                STEP,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_measure_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Adds every cell's partials of one frame into its strength group's chunk record.
pub struct GrainReducePartialsBench {
    pub client: Client,
    pub size: GrainSize,
}

#[derive(Clone)]
pub struct GrainReducePartialsInput {
    pub partials: Handle,
    pub chunk: Handle,
}

impl Benchmark for GrainReducePartialsBench {
    type Input = GrainReducePartialsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Cells spread over every strength group.
        let cells = self.size.cells();
        let mut partials_host = Vec::with_capacity(cells * PARTIAL_LANES);
        for cell in 0..cells {
            partials_host.extend_from_slice(&[0.5f32; RECORD_LANES]);
            partials_host.push((cell % STRENGTH_GROUPS) as f32);
        }

        let chunk_host = vec![0.0f32; CHUNK_FRAMES * GROUPED_RECORD_LANES];

        let partials_bytes = f32::as_bytes(&partials_host);
        let partials = self.client.create_from_slice(partials_bytes);
        let chunk_bytes = f32::as_bytes(&chunk_host);
        let chunk = self.client.create_from_slice(chunk_bytes);

        GrainReducePartialsInput { partials, chunk }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let cells = self.size.cells();
        let record_offset = (MEASURED_FRAME * GROUPED_RECORD_LANES) as u32;

        unsafe {
            grain_reduce_partials::launch_unchecked(
                &self.client,
                CubeCount::new_1d(RECORD_LANES as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(args.partials, cells * PARTIAL_LANES),
                BufferArg::from_raw_parts(args.chunk, CHUNK_FRAMES * GROUPED_RECORD_LANES),
                record_offset,
                cells as u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_reduce_partials_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Projects every cell of one frame into a persistence history slot.
pub struct GrainProjectBench {
    pub client: Client,
    /// The size of the luma plane.
    pub size: GrainSize,
    /// Whether the frame is a half-resolution chroma frame of two channels, read on Cr without
    /// the luma band.
    pub chroma: bool,
}

impl GrainProjectBench {
    /// The size of the projected plane.
    fn plane(&self) -> GrainSize {
        if !self.chroma {
            return self.size;
        }

        GrainSize {
            width: self.size.width / CHROMA_SUBSAMPLING,
            height: self.size.height / CHROMA_SUBSAMPLING,
            label: self.size.label,
        }
    }

    fn channels(&self) -> u32 {
        if self.chroma { CHROMA_CHANNELS } else { 1 }
    }
}

#[derive(Clone)]
pub struct GrainProjectInput {
    pub input: Handle,
    pub output: Handle,
    pub patterns: Handle,
    pub history: Handle,
}

impl Benchmark for GrainProjectBench {
    type Input = GrainProjectInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let plane = self.plane();
        let channels = self.channels();
        let samples = plane.pixels() * channels as usize;
        let history_len = HISTORY_FRAMES as usize * plane.cells() * HISTORY_ENTRY_LEN;
        let frame = make_synthetic_frame(plane.width, plane.height, channels);
        let output_host = vec![0.5f32; samples];
        let patterns_host: Vec<u32> = (0..PATTERN_WORDS as u32)
            .map(|index| 0x5A5A_5A5Au32.rotate_left(index))
            .collect();

        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let output_bytes = f32::as_bytes(&output_host);
        let output = self.client.create_from_slice(output_bytes);
        let patterns_bytes = u32::as_bytes(&patterns_host);
        let patterns = self.client.create_from_slice(patterns_bytes);
        let history = self.client.empty(history_len * size_of::<f32>());

        GrainProjectInput {
            input,
            output,
            patterns,
            history,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let plane = self.plane();
        let channels = self.channels();
        let samples = plane.pixels() * channels as usize;
        let history_len = HISTORY_FRAMES as usize * plane.cells() * HISTORY_ENTRY_LEN;
        let channel = if self.chroma { CR_CHANNEL } else { LUMA_CHANNEL };
        let luma_band = !self.chroma;

        unsafe {
            grain_project::launch_unchecked(
                &self.client,
                CubeCount::new_2d(plane.width / CELL, plane.height / CELL),
                CubeDim::new_2d(CELL, CELL),
                channels as usize,
                BufferArg::from_raw_parts(args.input, samples),
                BufferArg::from_raw_parts(args.output, samples),
                BufferArg::from_raw_parts(args.patterns, PATTERN_WORDS),
                BufferArg::from_raw_parts(args.history, history_len),
                0u32,
                1u32,
                plane.width,
                plane.height,
                channels,
                channel,
                luma_band,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        if self.chroma {
            return format!("grain_project_chroma_{}", self.size.label);
        }

        format!("grain_project_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Adds every lag's persistence sums of one frame from a full history, block partials then the
/// final reduce.
pub struct GrainPersistBench {
    pub client: Client,
    pub size: GrainSize,
}

#[derive(Clone)]
pub struct GrainPersistInput {
    pub history: Handle,
    pub partials: Handle,
    pub record: Handle,
}

impl Benchmark for GrainPersistBench {
    type Input = GrainPersistInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Every cell flagged flat, so each lag sums every cell.
        let entries = HISTORY_FRAMES as usize * self.size.cells();
        let mut history_host = Vec::with_capacity(entries * HISTORY_ENTRY_LEN);
        for entry in 0..entries {
            let projections = (0..PROJECTIONS).map(|projection| ((entry + projection) % 5) as f32 * 0.01);
            history_host.extend(projections);
            history_host.push(1.0);
        }

        let record_host = vec![0.0f32; FRAME_RECORD_LEN];

        let history_bytes = f32::as_bytes(&history_host);
        let history = self.client.create_from_slice(history_bytes);
        let record_bytes = f32::as_bytes(&record_host);
        let record = self.client.create_from_slice(record_bytes);

        let partials_len = self.size.persist_partials_len();
        let partials = self.client.empty(partials_len * size_of::<f32>());

        GrainPersistInput {
            history,
            partials,
            record,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let cells = self.size.cells();
        let history_len = HISTORY_FRAMES as usize * cells * HISTORY_ENTRY_LEN;
        let blocks = self.size.persist_blocks();
        let partials_len = self.size.persist_partials_len();

        unsafe {
            grain_persist::launch_unchecked(
                &self.client,
                CubeCount::new_2d(blocks, PERSISTENCE_LAGS),
                CubeDim::new_1d(PERSIST_THREADS),
                BufferArg::from_raw_parts(args.history, history_len),
                BufferArg::from_raw_parts(args.partials.clone(), partials_len),
                HISTORY_FRAMES - 1,
                cells as u32,
                blocks,
            );

            grain_persist_reduce::launch_unchecked(
                &self.client,
                CubeCount::new_1d(PERSISTENCE_LAGS),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(args.partials, partials_len),
                BufferArg::from_raw_parts(args.record, FRAME_RECORD_LEN),
                0u32,
                blocks,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_persist_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}
