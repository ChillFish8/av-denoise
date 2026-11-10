use cubecl::prelude::*;
use cubecl::server::Handle;

use super::chunk::GrainChunk;
use super::consts::{
    AUTOCOV_LEN,
    CELL,
    CHUNK_FRAMES,
    FRAME_RECORD_LEN,
    GROUPED_AUTOCOV_LEN,
    HIST_LEN,
    HISTORY_ENTRY_LEN,
    HISTORY_FRAMES,
    LAG_COUNT,
    LAG_RECORD_LEN,
    PARTIAL_LEN,
    PATTERNS,
    PERSIST_BLOCK_CELLS,
    PERSIST_THREADS,
    PERSISTENCE_LAGS,
    PROJECTIONS,
    REDUCE_THREADS,
    STD_BUCKETS,
};
use super::fit::bucket_edges;
use crate::nl4d::kernels::{
    grain_measure,
    grain_persist,
    grain_persist_reduce,
    grain_project,
    grain_reduce_partials,
    grain_save_vectors,
};
use crate::nl4d::snapshot::LastFields;

const SAVE_THREADS: u32 = 256;
/// The std bucket edges, one more than the bucket count.
const EDGES_LEN: usize = STD_BUCKETS + 1;
/// Values in one chunk's frame records.
const CHUNK_RECORD_LEN: usize = CHUNK_FRAMES as usize * FRAME_RECORD_LEN;
/// The pattern words the project kernel reads, two per projection.
const PATTERN_WORDS_LEN: usize = 2 * PROJECTIONS;

/// The projection patterns split into words, the low 32 bits of each pattern first.
pub(crate) fn pattern_words() -> Vec<u32> {
    let mut words = Vec::with_capacity(PATTERN_WORDS_LEN);
    for pattern in PATTERNS {
        let low_word = pattern as u32;
        let high_word = (pattern >> 32) as u32;
        words.push(low_word);
        words.push(high_word);
    }

    words
}

fn cell_count(width: u32, height: u32) -> u64 {
    u64::from(width / CELL) * u64::from(height / CELL)
}

fn persist_blocks(width: u32, height: u32) -> u64 {
    let cells = cell_count(width, height);
    cells.div_ceil(u64::from(PERSIST_BLOCK_CELLS))
}

/// Values in the persistence history of a frame size, an entry per cell for each history frame.
pub(crate) fn persistence_history_len(width: u32, height: u32) -> u64 {
    let cells = cell_count(width, height);
    cells * u64::from(HISTORY_FRAMES) * HISTORY_ENTRY_LEN as u64
}

/// Values in the persistence partials of a frame size, one lag record part per block and lag.
pub(crate) fn persistence_partials_len(width: u32, height: u32) -> u64 {
    let blocks = persist_blocks(width, height);
    blocks * (PERSISTENCE_LAGS * LAG_RECORD_LEN) as u64
}

/// Which measurements a [GrainExport] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GrainParts {
    /// The source grain table measurement, histograms and autocovariance.
    pub table: bool,
    /// The per-frame grain persistence measurement.
    pub persistence: bool,
}

/// The geometry a [GrainExport] is built for.
pub(crate) struct GrainGeometry {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step: u32,
    pub ring_frames: u32,
}

impl GrainGeometry {
    fn blocks(&self) -> u32 {
        self.blocks_x * self.blocks_y
    }

    fn cells_x(&self) -> u32 {
        self.width / CELL
    }

    fn cells_y(&self) -> u32 {
        self.height / CELL
    }

    fn cells(&self) -> u32 {
        self.cells_x() * self.cells_y()
    }

    fn frame_len(&self) -> usize {
        (self.width * self.height * self.channels) as usize
    }

    fn ring_len(&self) -> usize {
        self.frame_len() * self.ring_frames as usize
    }

    fn saved_mv_len(&self) -> usize {
        (self.ring_frames * self.blocks() * 2) as usize
    }

    fn saved_conf_len(&self) -> usize {
        (self.ring_frames * self.blocks()) as usize
    }

    fn partials_len(&self) -> usize {
        self.cells() as usize * PARTIAL_LEN
    }

    fn history_len(&self) -> usize {
        persistence_history_len(self.width, self.height) as usize
    }

    fn persist_blocks(&self) -> u32 {
        persist_blocks(self.width, self.height) as u32
    }

    fn persist_partials_len(&self) -> usize {
        persistence_partials_len(self.width, self.height) as usize
    }
}

/// One completed frame's buffers and ring positions.
struct CompletedFrame<'a> {
    input: &'a Handle,
    outputs: &'a [Handle; 2],
    frame_slot: u32,
    /// The ring slot of the next real frame, `None` for a stream's last frame.
    next_slot: Option<u32>,
    output_slot: usize,
}

/// One chunk's table histograms and autocovariance.
#[derive(Clone)]
struct TableChunk {
    hist: Handle,
    autocov: Handle,
}

#[derive(Clone)]
struct ChunkBuffers {
    /// Present when the table is measured.
    table: Option<TableChunk>,
    /// Each frame's persistence sums, present when persistence is measured.
    frame_record: Option<Handle>,
    frames: u32,
}

#[derive(Clone, Copy)]
struct Completed {
    ring_slot: u32,
    output_slot: usize,
}

/// The grain table measurement.
struct TableMeasure {
    /// Each frame's vectors to the next frame, in a ring entry keyed by its physical ring slot.
    ///
    /// A frame's vectors must be saved before the frame is measured.
    saved_mv: Handle,
    /// The confidence of each saved vector, in the same entries.
    saved_conf: Handle,
    /// Whether each ring entry holds vectors to a real next frame.
    saved_valid: Vec<bool>,
    edges: Handle,
    partials: Handle,
    last_completed: Option<Completed>,
    /// How many measured frames had a saved entry to their next frame.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    measured_with_entry: u32,
}

impl TableMeasure {
    fn new(client: &Client, geometry: &GrainGeometry) -> Self {
        let edges_host = bucket_edges();
        let saved_mv_host = vec![0i32; geometry.saved_mv_len()];
        let saved_conf_host = vec![0.0f32; geometry.saved_conf_len()];

        let saved_mv_bytes = i32::as_bytes(&saved_mv_host);
        let saved_mv = client.create_from_slice(saved_mv_bytes);
        let saved_conf_bytes = f32::as_bytes(&saved_conf_host);
        let saved_conf = client.create_from_slice(saved_conf_bytes);
        let edges_bytes = f32::as_bytes(&edges_host);
        let edges = client.create_from_slice(edges_bytes);
        let partials_bytes = geometry.partials_len().max(1) * size_of::<f32>();
        let partials = client.empty(partials_bytes);

        Self {
            saved_mv,
            saved_conf,
            saved_valid: vec![false; geometry.ring_frames as usize],
            edges,
            partials,
            last_completed: None,
            #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
            measured_with_entry: 0,
        }
    }

    fn save_vectors(
        &mut self,
        client: &Client,
        geometry: &GrainGeometry,
        fields: &LastFields,
        centre_slot: u32,
        next_neighbour: Option<u32>,
    ) {
        let Some(neighbour) = next_neighbour else {
            self.saved_valid[centre_slot as usize] = false;
            return;
        };

        let blocks = geometry.blocks();
        let grid = blocks.div_ceil(SAVE_THREADS);
        let mv_len = (fields.neighbours * fields.mv_stride) as usize;
        let conf_len = (fields.neighbours * fields.conf_stride) as usize;

        unsafe {
            grain_save_vectors::launch_unchecked(
                client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(SAVE_THREADS),
                BufferArg::from_raw_parts(fields.mv_field.clone(), mv_len),
                BufferArg::from_raw_parts(fields.confidence.clone(), conf_len),
                BufferArg::from_raw_parts(self.saved_mv.clone(), geometry.saved_mv_len()),
                BufferArg::from_raw_parts(self.saved_conf.clone(), geometry.saved_conf_len()),
                neighbour * fields.mv_stride,
                neighbour * fields.conf_stride,
                centre_slot,
                blocks,
                grid * SAVE_THREADS,
            );
        }

        self.saved_valid[centre_slot as usize] = true;
    }

    fn measure(
        &mut self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        chunk: &TableChunk,
    ) {
        let frame_slot = frame.frame_slot;
        let output_slot = frame.output_slot;
        let has_source = frame.next_slot.is_some() && self.saved_valid[frame_slot as usize];

        #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
        if self.saved_valid[frame_slot as usize] {
            self.measured_with_entry += 1;
        }

        let kept_from = self
            .last_completed
            .filter(|previous| self.saved_valid[previous.ring_slot as usize]);
        let has_kept = kept_from.is_some();
        let next_or_own_slot = frame.next_slot.unwrap_or(frame_slot);
        let kept_entry = kept_from.map_or(frame_slot, |previous| previous.ring_slot);
        let prev_output = kept_from.map_or(output_slot, |previous| previous.output_slot);
        let frame_len = geometry.frame_len();

        unsafe {
            grain_measure::launch_unchecked(
                client,
                CubeCount::new_2d(geometry.cells_x(), geometry.cells_y()),
                CubeDim::new_2d(CELL, CELL),
                geometry.channels as usize,
                BufferArg::from_raw_parts(frame.input.clone(), geometry.ring_len()),
                BufferArg::from_raw_parts(frame.outputs[output_slot].clone(), frame_len),
                BufferArg::from_raw_parts(frame.outputs[prev_output].clone(), frame_len),
                BufferArg::from_raw_parts(self.saved_mv.clone(), geometry.saved_mv_len()),
                BufferArg::from_raw_parts(self.saved_conf.clone(), geometry.saved_conf_len()),
                BufferArg::from_raw_parts(self.edges.clone(), EDGES_LEN),
                BufferArg::from_raw_parts(chunk.hist.clone(), 2 * HIST_LEN),
                BufferArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                frame_slot,
                next_or_own_slot,
                frame_slot,
                kept_entry,
                has_source as u32,
                has_kept as u32,
                geometry.width,
                geometry.height,
                geometry.channels,
                geometry.blocks_x,
                geometry.blocks_y,
                geometry.step,
            );

            grain_reduce_partials::launch_unchecked(
                client,
                CubeCount::new_1d(AUTOCOV_LEN as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                BufferArg::from_raw_parts(chunk.autocov.clone(), GROUPED_AUTOCOV_LEN),
                geometry.cells(),
            );
        }

        self.last_completed = Some(Completed {
            ring_slot: frame_slot,
            output_slot,
        });
    }

    fn reset_stream(&mut self) {
        self.saved_valid.fill(false);
        self.last_completed = None;
    }

    fn open_chunk(client: &Client) -> TableChunk {
        let hist_host = vec![0i32; 2 * HIST_LEN];
        let autocov_host = vec![0.0f32; GROUPED_AUTOCOV_LEN];
        let hist_bytes = i32::as_bytes(&hist_host);
        let hist = client.create_from_slice(hist_bytes);
        let autocov_bytes = f32::as_bytes(&autocov_host);
        let autocov = client.create_from_slice(autocov_bytes);

        TableChunk { hist, autocov }
    }
}

/// The grain persistence measurement.
struct PersistenceMeasure {
    /// The projection patterns as words, two per projection.
    patterns: Handle,
    /// Each cell's projections and flat flag for the last [HISTORY_FRAMES] completed frames.
    history: Handle,
    /// Each lag's persistence sums per block of cells, before they add into a frame record.
    partials: Handle,
    /// Completed frames measured since the stream started.
    frames_in_stream: u32,
}

impl PersistenceMeasure {
    fn new(client: &Client, geometry: &GrainGeometry) -> Self {
        let words = pattern_words();
        let patterns_bytes = u32::as_bytes(&words);
        let patterns = client.create_from_slice(patterns_bytes);
        let history_bytes = geometry.history_len().max(1) * size_of::<f32>();
        let history = client.empty(history_bytes);
        let partials_bytes = geometry.persist_partials_len().max(1) * size_of::<f32>();
        let partials = client.empty(partials_bytes);

        Self {
            patterns,
            history,
            partials,
            frames_in_stream: 0,
        }
    }

    /// Projects the completed frame into the history and adds its pairs into the frame's record.
    ///
    /// `chunk_frames` is how many frames the chunk held before this one.
    fn measure(
        &mut self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        frame_record: &Handle,
        chunk_frames: u32,
    ) {
        self.launch(client, geometry, frame, frame_record, chunk_frames);
        self.frames_in_stream += 1;
    }

    /// Launches the frame's projection, then each lag whose earlier frame is in this stream.
    fn launch(
        &self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        frame_record: &Handle,
        chunk_frames: u32,
    ) {
        let history_len = geometry.history_len();
        let history_slot = self.frames_in_stream % HISTORY_FRAMES;
        let earlier_frames = self.frames_in_stream.min(PERSISTENCE_LAGS as u32);
        let record_offset = chunk_frames * FRAME_RECORD_LEN as u32;
        let output = &frame.outputs[frame.output_slot];

        unsafe {
            grain_project::launch_unchecked(
                client,
                CubeCount::new_2d(geometry.cells_x(), geometry.cells_y()),
                CubeDim::new_2d(CELL, CELL),
                geometry.channels as usize,
                BufferArg::from_raw_parts(frame.input.clone(), geometry.ring_len()),
                BufferArg::from_raw_parts(output.clone(), geometry.frame_len()),
                BufferArg::from_raw_parts(self.patterns.clone(), PATTERN_WORDS_LEN),
                BufferArg::from_raw_parts(self.history.clone(), history_len),
                frame.frame_slot,
                history_slot,
                geometry.width,
                geometry.height,
                geometry.channels,
            );
        }

        if earlier_frames == 0 {
            return;
        }

        let blocks = geometry.persist_blocks();
        let partials_len = geometry.persist_partials_len();

        unsafe {
            grain_persist::launch_unchecked(
                client,
                CubeCount::new_2d(blocks, earlier_frames),
                CubeDim::new_1d(PERSIST_THREADS),
                BufferArg::from_raw_parts(self.history.clone(), history_len),
                BufferArg::from_raw_parts(self.partials.clone(), partials_len),
                history_slot,
                geometry.cells(),
                blocks,
            );

            grain_persist_reduce::launch_unchecked(
                client,
                CubeCount::new_1d(earlier_frames),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(self.partials.clone(), partials_len),
                BufferArg::from_raw_parts(frame_record.clone(), CHUNK_RECORD_LEN),
                record_offset,
                blocks,
            );
        }
    }

    fn reset_stream(&mut self) {
        self.frames_in_stream = 0;
    }

    fn open_chunk(client: &Client) -> Handle {
        let record_host = vec![0.0f32; CHUNK_RECORD_LEN];
        let record_bytes = f32::as_bytes(&record_host);

        client.create_from_slice(record_bytes)
    }
}

/// The grain measurement state of one nl4d denoiser.
pub(crate) struct GrainExport {
    geometry: GrainGeometry,
    /// Present when the table is measured.
    table: Option<TableMeasure>,
    /// Present when persistence is measured.
    persistence: Option<PersistenceMeasure>,
    /// Each completed frame adds into the newest chunk, which stays on the GPU until
    /// [Self::drain] reads every chunk back.
    chunks: Vec<ChunkBuffers>,
    chunk_open: bool,
}

impl GrainExport {
    pub(crate) fn new(client: &Client, geometry: GrainGeometry, parts: GrainParts) -> Self {
        let table = parts.table.then(|| TableMeasure::new(client, &geometry));
        let persistence = parts
            .persistence
            .then(|| PersistenceMeasure::new(client, &geometry));

        Self {
            geometry,
            table,
            persistence,
            chunks: Vec::new(),
            chunk_open: false,
        }
    }

    /// Saves the centre frame's vectors to the next frame, or marks the entry empty.
    ///
    /// `next_neighbour` is the motion field's neighbour index of the frame after the centre,
    /// `None` when that frame is not a real frame of the same stream. Nothing is saved without the
    /// table measurement.
    pub(in crate::nl4d) fn save_vectors(
        &mut self,
        client: &Client,
        fields: &LastFields,
        centre_slot: u32,
        next_neighbour: Option<u32>,
    ) {
        let Some(table) = self.table.as_mut() else {
            return;
        };

        table.save_vectors(client, &self.geometry, fields, centre_slot, next_neighbour);
    }

    /// Measures the frame that just completed into the newest chunk.
    ///
    /// `next_slot` is the ring slot of the next real frame, `None` for a stream's last frame.
    pub(crate) fn measure(
        &mut self,
        client: &Client,
        input: &Handle,
        outputs: &[Handle; 2],
        frame_slot: u32,
        next_slot: Option<u32>,
        output_slot: usize,
    ) {
        let frame = CompletedFrame {
            input,
            outputs,
            frame_slot,
            next_slot,
            output_slot,
        };
        let chunk = self.open_chunk(client);

        let table = self.table.as_mut();
        if let (Some(table), Some(table_chunk)) = (table, &chunk.table) {
            table.measure(client, &self.geometry, &frame, table_chunk);
        }

        let persistence = self.persistence.as_mut();
        if let (Some(persistence), Some(frame_record)) = (persistence, &chunk.frame_record) {
            persistence.measure(client, &self.geometry, &frame, frame_record, chunk.frames);
        }

        let newest = self.chunks.last_mut().expect("open_chunk pushed a chunk");
        newest.frames += 1;
        if newest.frames == CHUNK_FRAMES {
            self.chunk_open = false;
        }
    }

    /// A copy of the newest chunk's buffers, opening a chunk when none is open.
    ///
    /// The copy's frame count is taken before the frame being measured.
    fn open_chunk(&mut self, client: &Client) -> ChunkBuffers {
        if !self.chunk_open {
            let table = self.table.as_ref().map(|_| TableMeasure::open_chunk(client));
            let frame_record = self
                .persistence
                .as_ref()
                .map(|_| PersistenceMeasure::open_chunk(client));
            self.chunks.push(ChunkBuffers {
                table,
                frame_record,
                frames: 0,
            });
            self.chunk_open = true;
        }

        let chunk = self.chunks.last().expect("a chunk is open");

        chunk.clone()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn measured_with_entry(&self) -> u32 {
        self.table.as_ref().map_or(0, |table| table.measured_with_entry)
    }

    /// Forgets the stream's saved vectors and history, and closes the open chunk.
    pub(crate) fn reset_stream(&mut self) {
        if let Some(table) = self.table.as_mut() {
            table.reset_stream();
        }

        if let Some(persistence) = self.persistence.as_mut() {
            persistence.reset_stream();
        }

        self.chunk_open = false;
    }

    /// Reads every chunk back, in order, and forgets them.
    pub(crate) fn drain(&mut self, client: &Client) -> Result<Vec<GrainChunk>, anyhow::Error> {
        let buffers = std::mem::take(&mut self.chunks);
        self.chunk_open = false;

        let mut chunks = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let chunk = read_chunk(client, buffer)?;
            chunks.push(chunk);
        }

        Ok(chunks)
    }
}

fn read_chunk(client: &Client, buffer: ChunkBuffers) -> Result<GrainChunk, anyhow::Error> {
    let mut chunk = GrainChunk::empty();
    chunk.frames = buffer.frames;

    if let Some(table) = buffer.table {
        read_table(client, table, &mut chunk)?;
    }

    if let Some(frame_record) = buffer.frame_record {
        let record_bytes = client
            .read_one(frame_record)
            .map_err(|error| anyhow::anyhow!("grain frame record readback failed: {error}"))?;
        let values = f32::from_bytes(&record_bytes);
        let recorded_len = buffer.frames as usize * FRAME_RECORD_LEN;
        let recorded = values[..recorded_len].iter().map(|&value| value as f64);
        chunk.frame_records = recorded.collect();
    }

    Ok(chunk)
}

fn read_table(client: &Client, table: TableChunk, chunk: &mut GrainChunk) -> Result<(), anyhow::Error> {
    let hist_bytes = client
        .read_one(table.hist)
        .map_err(|error| anyhow::anyhow!("grain histogram readback failed: {error}"))?;
    let autocov_bytes = client
        .read_one(table.autocov)
        .map_err(|error| anyhow::anyhow!("grain autocovariance readback failed: {error}"))?;
    let hist = i32::from_bytes(&hist_bytes);
    let autocov = f32::from_bytes(&autocov_bytes);
    let (source_counts, kept_counts) = hist.split_at(HIST_LEN);

    for (target, &count) in chunk.source_hist.iter_mut().zip(source_counts) {
        *target = count as u32;
    }

    for (target, &count) in chunk.kept_hist.iter_mut().zip(kept_counts) {
        *target = count as u32;
    }

    let (records, _) = autocov.as_chunks::<AUTOCOV_LEN>();
    let (targets, _) = chunk.autocov.as_chunks_mut::<LAG_COUNT>();
    let groups = targets.iter_mut().zip(chunk.pixels.iter_mut());
    for ((target, pixels), record) in groups.zip(records) {
        for (sum, &value) in target.iter_mut().zip(&record[..LAG_COUNT]) {
            *sum = value as f64;
        }

        *pixels = record[LAG_COUNT] as f64;
    }

    Ok(())
}
