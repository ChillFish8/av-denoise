use cubecl::prelude::*;
use cubecl::server::Handle;

use super::chunk::GrainChunk;
use super::consts::{
    AUTOCOV_LEN,
    CELL,
    CHUNK_FRAMES,
    FRAME_TABLE_LEN,
    GROUPED_AUTOCOV_LEN,
    HIST_LEN,
    PARTIAL_LEN,
    PERSIST_BLOCK_CELLS,
    REDUCE_THREADS,
    STD_BUCKETS,
};
use super::fit::bucket_edges;
use super::persistence::{
    MeasuredPlanes,
    PersistenceMeasure,
    persistence_history_len,
    persistence_partials_len,
};
use crate::nl4d::kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use crate::nl4d::snapshot::LastFields;

const SAVE_THREADS: u32 = 256;
/// The std bucket edges, one more than the bucket count.
const EDGES_LEN: usize = STD_BUCKETS + 1;
/// Counts in one frame's histograms, source grain and then kept grain.
const FRAME_HIST_LEN: usize = 2 * HIST_LEN;
/// Counts in one chunk's histograms, one slot per frame.
const CHUNK_HIST_LEN: usize = CHUNK_FRAMES as usize * FRAME_HIST_LEN;
/// Values in one chunk's grouped records, one slot per frame.
const CHUNK_AUTOCOV_LEN: usize = CHUNK_FRAMES as usize * GROUPED_AUTOCOV_LEN;

/// Which measurements a [GrainExport] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GrainParts {
    /// The source grain table measurement, histograms and autocovariance, on luma only.
    pub table: bool,
    /// The planes the per-frame grain persistence measurement reads, `None` when it is off.
    pub persistence: Option<MeasuredPlanes>,
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

    pub(super) fn cells_x(&self) -> u32 {
        self.width / CELL
    }

    pub(super) fn cells_y(&self) -> u32 {
        self.height / CELL
    }

    pub(super) fn cells(&self) -> u32 {
        self.cells_x() * self.cells_y()
    }

    pub(super) fn frame_len(&self) -> usize {
        (self.width * self.height * self.channels) as usize
    }

    pub(super) fn ring_len(&self) -> usize {
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

    /// Values in one channel's persistence history.
    pub(super) fn history_len(&self) -> usize {
        persistence_history_len(self.width, self.height) as usize
    }

    /// Blocks of cells the persist kernel covers.
    pub(super) fn persist_blocks(&self) -> u32 {
        self.cells().div_ceil(PERSIST_BLOCK_CELLS)
    }

    /// Values in one channel's persistence partials over `lags` lags.
    pub(super) fn persist_partials_len(&self, lags: usize) -> usize {
        persistence_partials_len(self.width, self.height, lags) as usize
    }
}

/// One completed frame's buffers and ring positions.
pub(super) struct CompletedFrame<'a> {
    pub input: &'a Handle,
    pub outputs: &'a [Handle; 2],
    pub frame_slot: u32,
    /// The ring slot of the next real frame, `None` for a stream's last frame.
    pub next_slot: Option<u32>,
    pub output_slot: usize,
}

/// One chunk's table histograms and autocovariance, one slot per frame.
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

    /// Measures one frame into its slot of `chunk`.
    ///
    /// `chunk_frames` is how many frames the chunk held before this one.
    fn measure(
        &mut self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        chunk: &TableChunk,
        chunk_frames: u32,
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
        let hist_offset = chunk_frames * FRAME_HIST_LEN as u32;
        let record_offset = chunk_frames * GROUPED_AUTOCOV_LEN as u32;

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
                BufferArg::from_raw_parts(chunk.hist.clone(), CHUNK_HIST_LEN),
                BufferArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                frame_slot,
                next_or_own_slot,
                frame_slot,
                kept_entry,
                has_source as u32,
                has_kept as u32,
                hist_offset,
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
                BufferArg::from_raw_parts(chunk.autocov.clone(), CHUNK_AUTOCOV_LEN),
                record_offset,
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

    /// A chunk with every frame slot zeroed.
    fn open_chunk(client: &Client) -> TableChunk {
        let hist_host = vec![0i32; CHUNK_HIST_LEN];
        let autocov_host = vec![0.0f32; CHUNK_AUTOCOV_LEN];
        let hist_bytes = i32::as_bytes(&hist_host);
        let hist = client.create_from_slice(hist_bytes);
        let autocov_bytes = f32::as_bytes(&autocov_host);
        let autocov = client.create_from_slice(autocov_bytes);

        TableChunk { hist, autocov }
    }
}

/// The grain measurement state of one nl4d denoiser.
pub(crate) struct GrainExport {
    geometry: GrainGeometry,
    /// Present when the table is measured.
    table: Option<TableMeasure>,
    /// Present when persistence is measured.
    persistence: Option<PersistenceMeasure>,
    /// Each completed frame fills its own slot in the newest chunk, which stays on the GPU until
    /// [Self::drain] reads every chunk back.
    chunks: Vec<ChunkBuffers>,
    chunk_open: bool,
}

impl GrainExport {
    pub(crate) fn new(client: &Client, geometry: GrainGeometry, parts: GrainParts) -> Self {
        let table = parts.table.then(|| TableMeasure::new(client, &geometry));
        let persistence = parts
            .persistence
            .map(|planes| PersistenceMeasure::new(client, &geometry, planes));

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
            table.measure(client, &self.geometry, &frame, table_chunk, chunk.frames);
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
                .map(|persistence| persistence.open_chunk(client));
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

        let frame_record_len = self
            .persistence
            .as_ref()
            .map_or(0, PersistenceMeasure::frame_record_len);

        let mut chunks = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let chunk = read_chunk(client, buffer, frame_record_len)?;
            chunks.push(chunk);
        }

        Ok(chunks)
    }
}

/// Reads one chunk back, with `frame_record_len` values in each frame's record.
fn read_chunk(
    client: &Client,
    buffer: ChunkBuffers,
    frame_record_len: usize,
) -> Result<GrainChunk, anyhow::Error> {
    let mut chunk = GrainChunk::empty();
    chunk.frames = buffer.frames;

    if let Some(table) = buffer.table {
        chunk.frame_tables = read_table(client, table, buffer.frames)?;
    }

    if let Some(frame_record) = buffer.frame_record {
        let record_bytes = client
            .read_one(frame_record)
            .map_err(|error| anyhow::anyhow!("grain frame record readback failed: {error}"))?;
        let values = f32::from_bytes(&record_bytes);
        let recorded_len = buffer.frames as usize * frame_record_len;
        let recorded = values[..recorded_len].iter().map(|&value| value as f64);
        chunk.frame_records = recorded.collect();
    }

    Ok(chunk)
}

/// Reads the table slots of a chunk's first `frames` frames, `FRAME_TABLE_LEN` values per frame.
fn read_table(client: &Client, table: TableChunk, frames: u32) -> Result<Vec<f32>, anyhow::Error> {
    let hist_bytes = client
        .read_one(table.hist)
        .map_err(|error| anyhow::anyhow!("grain histogram readback failed: {error}"))?;
    let autocov_bytes = client
        .read_one(table.autocov)
        .map_err(|error| anyhow::anyhow!("grain autocovariance readback failed: {error}"))?;
    let hist = i32::from_bytes(&hist_bytes);
    let autocov = f32::from_bytes(&autocov_bytes);
    let (frame_hists, _) = hist.as_chunks::<FRAME_HIST_LEN>();
    let (frame_records, _) = autocov.as_chunks::<GROUPED_AUTOCOV_LEN>();
    let frame_count = frames as usize;

    let mut frame_tables = Vec::with_capacity(frame_count * FRAME_TABLE_LEN);
    let slots = frame_hists.iter().zip(frame_records).take(frame_count);
    for (frame_hist, frame_record) in slots {
        let counts = frame_hist.iter().map(|&count| count as f32);
        frame_tables.extend(counts);
        frame_tables.extend_from_slice(frame_record);
    }

    Ok(frame_tables)
}
