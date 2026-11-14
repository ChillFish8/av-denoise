use cubecl::prelude::*;
use cubecl::server::Handle;

use super::chunk::GrainChunk;
use super::consts::{
    AUTOCOV_LEN,
    CELL,
    CHROMA_PLANES,
    CHUNK_FRAMES,
    FRAME_TABLE_LEN,
    GROUPED_AUTOCOV_LEN,
    HIST_LEN,
    PARTIAL_LEN,
    REDUCE_THREADS,
    STD_BUCKETS,
};
use super::fit::bucket_edges;
use super::persistence::{
    MeasuredPlanes,
    PersistenceMeasure,
    cell_count,
    persistence_history_len,
    persistence_partials_len,
};
use super::snapshot::{PassSources, SnapshotChunk, SnapshotMeasure};
use crate::nl4d::kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use crate::nl4d::snapshot::LastFields;
use crate::noise::TEMPORAL_NOISE_BLOCK;
use crate::noise::colour::{COLOUR_MEAN_FIELDS, ColourSource};

const SAVE_THREADS: u32 = 256;
/// The std bucket edges, one more than the bucket count.
const EDGES_LEN: usize = STD_BUCKETS + 1;
/// Counts in one plane's histograms of one frame, source grain and then one-frame grain.
const FRAME_HIST_LEN: usize = 2 * HIST_LEN;
/// Counts in one plane's histograms of one chunk, one slot per frame.
const CHUNK_HIST_LEN: usize = CHUNK_FRAMES as usize * FRAME_HIST_LEN;
/// Values in one plane's grouped records of one chunk, one slot per frame.
const CHUNK_AUTOCOV_LEN: usize = CHUNK_FRAMES as usize * GROUPED_AUTOCOV_LEN;

/// Which measurements a [GrainExport] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GrainParts {
    /// The planes the pass measures.
    pub planes: MeasuredPlanes,
    /// Whether each frame keeps the noise reading it was filtered with.
    pub noise: bool,
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
        let cells = cell_count(self.width, self.height);
        cells as u32
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

    /// The colour blocks in one row of the measured plane.
    fn means_blocks_x(&self) -> u32 {
        self.width.div_ceil(TEMPORAL_NOISE_BLOCK)
    }

    /// Values in the measured plane's colour block means.
    fn means_len(&self) -> usize {
        let blocks_y = self.height.div_ceil(TEMPORAL_NOISE_BLOCK);
        let blocks = self.means_blocks_x() * blocks_y;

        (blocks * COLOUR_MEAN_FIELDS) as usize
    }

    /// Values in one channel's persistence history.
    pub(super) fn history_len(&self) -> usize {
        persistence_history_len(self.width, self.height) as usize
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
    /// The frame's index in its stream.
    pub stream_frame: u32,
}

/// One chunk's table histograms and autocovariance, one slot per plane of each frame.
///
/// Frame `f`'s plane `p` sits in slot `f * planes + p`.
#[derive(Clone)]
struct TableChunk {
    hist: Handle,
    autocov: Handle,
    planes: usize,
}

#[derive(Clone)]
struct ChunkBuffers {
    table: TableChunk,
    /// Each frame's persistence sums.
    frame_record: Handle,
    /// Each frame's noise reading and colour fit, each present when it is kept.
    snapshots: SnapshotChunk,
    frames: u32,
}

/// How one plane's table is measured.
#[derive(Clone, Copy)]
struct TablePlan {
    /// The channel read from each pixel.
    channel: u32,
    /// Whether a cell's mean must sit inside the luma band to count.
    luma_band: bool,
    /// The colour blocks in one row, 0 when one-frame cells bin by their own mean.
    means_blocks_x: u32,
}

impl TablePlan {
    fn for_planes(planes: MeasuredPlanes, geometry: &GrainGeometry) -> Vec<TablePlan> {
        match planes {
            MeasuredPlanes::Luma => {
                let luma = TablePlan {
                    channel: 0,
                    luma_band: true,
                    means_blocks_x: 0,
                };
                vec![luma]
            },
            MeasuredPlanes::Chroma => {
                let means_blocks_x = geometry.means_blocks_x();
                let plans = (0..CHROMA_PLANES).map(|plane| TablePlan {
                    channel: plane as u32,
                    luma_band: false,
                    means_blocks_x,
                });
                plans.collect()
            },
        }
    }
}

/// The ring positions, pair and block means one completed frame is measured with.
struct FramePairs {
    frame_slot: u32,
    /// The next frame's ring slot, or the frame's own without one.
    next_or_own_slot: u32,
    has_source: bool,
    /// The frame's colour block means, present while they are held.
    luma_means: Option<Handle>,
}

/// The grain table measurement.
struct TableMeasure {
    /// One plan per measured plane, in table order.
    plans: Vec<TablePlan>,
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
    /// The colour block means chroma cells bin by, present once the colour model measures them.
    luma_means: Option<ColourSource>,
    /// One value bound in place of block means that are not held.
    means_placeholder: Handle,
    /// How many measured frames had a saved entry to their next frame.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    measured_with_entry: u32,
}

impl TableMeasure {
    fn new(client: &Client, geometry: &GrainGeometry, planes: MeasuredPlanes) -> Self {
        let plans = TablePlan::for_planes(planes, geometry);
        let edges_host = bucket_edges();
        let saved_mv_host = vec![0i32; geometry.saved_mv_len()];
        let saved_conf_host = vec![0.0f32; geometry.saved_conf_len()];
        let placeholder_host = [0.0f32];

        let saved_mv_bytes = i32::as_bytes(&saved_mv_host);
        let saved_mv = client.create_from_slice(saved_mv_bytes);
        let saved_conf_bytes = f32::as_bytes(&saved_conf_host);
        let saved_conf = client.create_from_slice(saved_conf_bytes);
        let edges_bytes = f32::as_bytes(&edges_host);
        let edges = client.create_from_slice(edges_bytes);
        let partials_bytes = geometry.partials_len().max(1) * size_of::<f32>();
        let partials = client.empty(partials_bytes);
        let placeholder_bytes = f32::as_bytes(&placeholder_host);
        let means_placeholder = client.create_from_slice(placeholder_bytes);

        Self {
            plans,
            saved_mv,
            saved_conf,
            saved_valid: vec![false; geometry.ring_frames as usize],
            edges,
            partials,
            luma_means: None,
            means_placeholder,
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

    /// Measures each plane of one frame into its slot of `chunk`.
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
        let has_source = frame.next_slot.is_some() && self.saved_valid[frame_slot as usize];

        #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
        if self.saved_valid[frame_slot as usize] {
            self.measured_with_entry += 1;
        }

        let next_or_own_slot = frame.next_slot.unwrap_or(frame_slot);
        let source = self.luma_means.as_ref();
        let luma_means = source.and_then(|source| source.held(frame.stream_frame));
        let pairs = FramePairs {
            frame_slot,
            next_or_own_slot,
            has_source,
            luma_means,
        };

        let planes = self.plans.len() as u32;
        for (plane, plan) in self.plans.iter().enumerate() {
            let frame_plane = chunk_frames * planes + plane as u32;
            self.launch_plan(client, geometry, frame, chunk, &pairs, *plan, frame_plane);
        }
    }

    /// Measures one plane of a frame into slot `frame_plane` of `chunk`.
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a buffer, ring position or slot the plane's launches bind"
    )]
    fn launch_plan(
        &self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        chunk: &TableChunk,
        pairs: &FramePairs,
        plan: TablePlan,
        frame_plane: u32,
    ) {
        let frame_len = geometry.frame_len();
        let hist_len = chunk.planes * CHUNK_HIST_LEN;
        let autocov_len = chunk.planes * CHUNK_AUTOCOV_LEN;
        let hist_offset = frame_plane * FRAME_HIST_LEN as u32;
        let record_offset = frame_plane * GROUPED_AUTOCOV_LEN as u32;
        let output = &frame.outputs[frame.output_slot];
        let held_means = pairs.luma_means.as_ref().filter(|_| plan.means_blocks_x > 0);
        let has_luma_means = held_means.is_some();
        let (means, means_len) = match held_means {
            Some(means) => (means, geometry.means_len()),
            None => (&self.means_placeholder, 1),
        };

        unsafe {
            grain_measure::launch_unchecked(
                client,
                CubeCount::new_2d(geometry.cells_x(), geometry.cells_y()),
                CubeDim::new_2d(CELL, CELL),
                geometry.channels as usize,
                BufferArg::from_raw_parts(frame.input.clone(), geometry.ring_len()),
                BufferArg::from_raw_parts(output.clone(), frame_len),
                BufferArg::from_raw_parts(self.saved_mv.clone(), geometry.saved_mv_len()),
                BufferArg::from_raw_parts(self.saved_conf.clone(), geometry.saved_conf_len()),
                BufferArg::from_raw_parts(self.edges.clone(), EDGES_LEN),
                BufferArg::from_raw_parts(means.clone(), means_len),
                BufferArg::from_raw_parts(chunk.hist.clone(), hist_len),
                BufferArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                pairs.frame_slot,
                pairs.next_or_own_slot,
                pairs.frame_slot,
                pairs.has_source as u32,
                has_luma_means as u32,
                hist_offset,
                geometry.width,
                geometry.height,
                geometry.channels,
                geometry.blocks_x,
                geometry.blocks_y,
                geometry.step,
                plan.channel,
                plan.luma_band,
                plan.means_blocks_x,
            );

            grain_reduce_partials::launch_unchecked(
                client,
                CubeCount::new_1d(AUTOCOV_LEN as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                BufferArg::from_raw_parts(chunk.autocov.clone(), autocov_len),
                record_offset,
                geometry.cells(),
            );
        }
    }

    fn reset_stream(&mut self) {
        self.saved_valid.fill(false);
    }

    /// A chunk with every frame slot of every plane zeroed.
    fn open_chunk(&self, client: &Client) -> TableChunk {
        let planes = self.plans.len();
        let hist_host = vec![0i32; planes * CHUNK_HIST_LEN];
        let autocov_host = vec![0.0f32; planes * CHUNK_AUTOCOV_LEN];
        let hist_bytes = i32::as_bytes(&hist_host);
        let hist = client.create_from_slice(hist_bytes);
        let autocov_bytes = f32::as_bytes(&autocov_host);
        let autocov = client.create_from_slice(autocov_bytes);

        TableChunk {
            hist,
            autocov,
            planes,
        }
    }
}

/// The grain measurement state of one nl4d denoiser.
pub(crate) struct GrainExport {
    geometry: GrainGeometry,
    /// The source and one-frame grain table of each measured plane.
    table: TableMeasure,
    /// The persistence sums each frame adds to its chunk.
    persistence: PersistenceMeasure,
    /// The noise reading and colour fit each frame was filtered with.
    snapshots: SnapshotMeasure,
    /// Each completed frame fills its own slot in the newest chunk, which stays on the GPU until
    /// [Self::drain] reads every chunk back.
    chunks: Vec<ChunkBuffers>,
    chunk_open: bool,
}

/// Values in one frame of each of a chunk's per-frame buffers.
struct FrameLens {
    record: usize,
    noise: usize,
    colour: usize,
}

impl GrainExport {
    pub(crate) fn new(client: &Client, geometry: GrainGeometry, parts: GrainParts) -> Self {
        let table = TableMeasure::new(client, &geometry, parts.planes);
        let persistence = PersistenceMeasure::new(client, &geometry, parts.planes);
        let snapshots = SnapshotMeasure::new(client, geometry.ring_frames, parts.noise);

        Self {
            geometry,
            table,
            persistence,
            snapshots,
            chunks: Vec::new(),
            chunk_open: false,
        }
    }

    /// Keeps each frame's colour fit of `planes` planes from the next pass on.
    pub(crate) fn keep_colour(&mut self, client: &Client, planes: u32) {
        self.snapshots.keep_colour(client, planes);
    }

    /// Bins each chroma frame's one-frame cells by the luma of its colour blocks in `source`.
    pub(crate) fn keep_luma_means(&mut self, source: ColourSource) {
        self.table.luma_means = Some(source);
    }

    /// Copies what a pass filtered its centre frame with into the frame's ring slot `centre_slot`.
    pub(crate) fn keep_snapshots(&self, client: &Client, sources: &PassSources, centre_slot: u32) {
        self.snapshots.keep(client, sources, centre_slot);
    }

    /// Saves the centre frame's vectors to the next frame, or marks the entry empty.
    ///
    /// `next_neighbour` is the motion field's neighbour index of the frame after the centre,
    /// `None` when that frame is not a real frame of the same stream. `fields` must sit on the
    /// measured plane's motion block grid.
    pub(in crate::nl4d) fn save_vectors(
        &mut self,
        client: &Client,
        fields: &LastFields,
        centre_slot: u32,
        next_neighbour: Option<u32>,
    ) {
        self.table
            .save_vectors(client, &self.geometry, fields, centre_slot, next_neighbour);
    }

    /// Measures the frame that just completed into the newest chunk.
    ///
    /// `next_slot` is the ring slot of the next real frame, `None` for a stream's last frame.
    /// `stream_frame` is the completed frame's index in its stream.
    #[expect(
        clippy::too_many_arguments,
        reason = "every argument is a buffer or ring position of the completed frame"
    )]
    pub(crate) fn measure(
        &mut self,
        client: &Client,
        input: &Handle,
        outputs: &[Handle; 2],
        frame_slot: u32,
        next_slot: Option<u32>,
        output_slot: usize,
        stream_frame: u32,
    ) {
        let frame = CompletedFrame {
            input,
            outputs,
            frame_slot,
            next_slot,
            output_slot,
            stream_frame,
        };
        let chunk = self.open_chunk(client);

        self.table
            .measure(client, &self.geometry, &frame, &chunk.table, chunk.frames);

        self.persistence
            .measure(client, &self.geometry, &frame, &chunk.frame_record, chunk.frames);

        self.snapshots
            .measure(client, frame_slot, &chunk.snapshots, chunk.frames);

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
            let table = self.table.open_chunk(client);
            let frame_record = self.persistence.open_chunk(client);
            let snapshots = self.snapshots.open_chunk(client);
            self.chunks.push(ChunkBuffers {
                table,
                frame_record,
                snapshots,
                frames: 0,
            });
            self.chunk_open = true;
        }

        let chunk = self.chunks.last().expect("a chunk is open");

        chunk.clone()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn measured_with_entry(&self) -> u32 {
        self.table.measured_with_entry
    }

    /// Forgets the stream's saved vectors and history, and closes the open chunk.
    pub(crate) fn reset_stream(&mut self) {
        self.table.reset_stream();
        self.persistence.reset_stream();
        self.chunk_open = false;
    }

    /// Reads every chunk back, in order, and forgets them.
    pub(crate) fn drain(&mut self, client: &Client) -> Result<Vec<GrainChunk>, anyhow::Error> {
        let buffers = std::mem::take(&mut self.chunks);
        self.chunk_open = false;

        let lens = FrameLens {
            record: self.persistence.frame_record_len(),
            noise: self.snapshots.noise_len(),
            colour: self.snapshots.colour_len(),
        };

        let mut chunks = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let chunk = read_chunk(client, buffer, &lens)?;
            chunks.push(chunk);
        }

        Ok(chunks)
    }
}

/// Reads one chunk back, with `lens` values in each frame of each buffer.
fn read_chunk(client: &Client, buffer: ChunkBuffers, lens: &FrameLens) -> Result<GrainChunk, anyhow::Error> {
    let mut chunk = GrainChunk::empty();
    let frames = buffer.frames;
    chunk.frames = frames;

    chunk.frame_tables = read_table(client, buffer.table, frames)?;

    let records = read_frames(client, buffer.frame_record, frames, lens.record, "frame record")?;
    let recorded = records.iter().map(|&value| value as f64);
    chunk.frame_records = recorded.collect();

    if let Some(noise) = buffer.snapshots.noise {
        chunk.frame_noise = read_frames(client, noise, frames, lens.noise, "noise snapshot")?;
    }

    if let Some(colour) = buffer.snapshots.colour {
        chunk.frame_colour = read_frames(client, colour, frames, lens.colour, "colour snapshot")?;
    }

    Ok(chunk)
}

/// Reads the first `frames` frames of `handle`, `frame_len` values per frame.
fn read_frames(
    client: &Client,
    handle: Handle,
    frames: u32,
    frame_len: usize,
    what: &str,
) -> Result<Vec<f32>, anyhow::Error> {
    let bytes = client
        .read_one(handle)
        .map_err(|error| anyhow::anyhow!("grain {what} readback failed: {error}"))?;
    let values = f32::from_bytes(&bytes);
    let kept_len = frames as usize * frame_len;

    Ok(values[..kept_len].to_vec())
}

/// Reads the table slots of a chunk's first `frames` frames.
///
/// Each frame holds `FRAME_TABLE_LEN` values per plane, its planes in turn.
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
    let slot_count = frames as usize * table.planes;

    let mut frame_tables = Vec::with_capacity(slot_count * FRAME_TABLE_LEN);
    let slots = frame_hists.iter().zip(frame_records).take(slot_count);
    for (frame_hist, frame_record) in slots {
        let counts = frame_hist.iter().map(|&count| count as f32);
        frame_tables.extend(counts);
        frame_tables.extend_from_slice(frame_record);
    }

    Ok(frame_tables)
}
