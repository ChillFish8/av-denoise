use cubecl::prelude::*;
use cubecl::server::Handle;

use super::chunk::GrainChunk;
use super::consts::{
    AUTOCOV_LEN,
    CELL,
    CHUNK_FRAMES,
    GROUPED_AUTOCOV_LEN,
    HIST_LEN,
    LAG_COUNT,
    PARTIAL_LEN,
    REDUCE_THREADS,
    STD_BUCKETS,
};
use super::fit::bucket_edges;
use crate::nl4d::kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use crate::nl4d::snapshot::LastFields;

const SAVE_THREADS: u32 = 256;
/// The std bucket edges, one more than the bucket count.
const EDGES_LEN: usize = STD_BUCKETS + 1;

/// The geometry a [GrainExport] is built for.
pub(crate) struct GrainGeometry {
    pub width: u32,
    pub height: u32,
    pub stored_ch: u32,
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

    fn saved_mv_len(&self) -> usize {
        (self.ring_frames * self.blocks() * 2) as usize
    }

    fn saved_conf_len(&self) -> usize {
        (self.ring_frames * self.blocks()) as usize
    }

    fn partials_len(&self) -> usize {
        self.cells() as usize * PARTIAL_LEN
    }
}

struct ChunkBuffers {
    hist: Handle,
    autocov: Handle,
    frames: u32,
}

#[derive(Clone, Copy)]
struct Completed {
    ring_slot: u32,
    output_slot: usize,
}

/// The grain measurement state of one nl4d denoiser.
///
/// Each frame's vectors to the next frame are saved in a ring entry keyed by its physical ring
/// slot. Each completed frame adds into the newest chunk record, which stays on the GPU until
/// [Self::drain] reads every chunk back.
pub(crate) struct GrainExport {
    geometry: GrainGeometry,
    saved_mv: Handle,
    saved_conf: Handle,
    saved_valid: Vec<bool>,
    edges: Handle,
    partials: Handle,
    chunks: Vec<ChunkBuffers>,
    chunk_open: bool,
    last_completed: Option<Completed>,
    /// How many measured frames had a saved entry to their next frame.
    #[cfg(test)]
    measured_with_entry: u32,
}

impl GrainExport {
    pub(crate) fn new<R: Runtime>(client: &ComputeClient<R>, geometry: GrainGeometry) -> Self {
        let edges_host = bucket_edges();
        let saved_mv_host = vec![0i32; geometry.saved_mv_len()];
        let saved_conf_host = vec![0.0f32; geometry.saved_conf_len()];

        let saved_mv = client.create_from_slice(i32::as_bytes(&saved_mv_host));
        let saved_conf = client.create_from_slice(f32::as_bytes(&saved_conf_host));
        let edges = client.create_from_slice(f32::as_bytes(&edges_host));
        let partials = client.empty(geometry.partials_len().max(1) * size_of::<f32>());

        Self {
            saved_valid: vec![false; geometry.ring_frames as usize],
            geometry,
            saved_mv,
            saved_conf,
            edges,
            partials,
            chunks: Vec::new(),
            chunk_open: false,
            last_completed: None,
            #[cfg(test)]
            measured_with_entry: 0,
        }
    }

    /// Saves the centre frame's vectors to the next frame, or marks the entry empty.
    ///
    /// `next_neighbour` is the motion field's neighbour index of the frame after the centre,
    /// `None` when that frame is not a real frame of the same stream.
    pub(in crate::nl4d) fn save_vectors<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        fields: &LastFields,
        centre_slot: u32,
        next_neighbour: Option<u32>,
    ) {
        let Some(neighbour) = next_neighbour else {
            self.saved_valid[centre_slot as usize] = false;
            return;
        };

        let blocks = self.geometry.blocks();
        let grid = blocks.div_ceil(SAVE_THREADS);
        let mv_len = (fields.neighbours * fields.mv_stride) as usize;
        let conf_len = (fields.neighbours * fields.conf_stride) as usize;

        unsafe {
            grain_save_vectors::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(SAVE_THREADS),
                ArrayArg::from_raw_parts(fields.mv_field.clone(), mv_len),
                ArrayArg::from_raw_parts(fields.confidence.clone(), conf_len),
                ArrayArg::from_raw_parts(self.saved_mv.clone(), self.geometry.saved_mv_len()),
                ArrayArg::from_raw_parts(self.saved_conf.clone(), self.geometry.saved_conf_len()),
                neighbour * fields.mv_stride,
                neighbour * fields.conf_stride,
                centre_slot,
                blocks,
                grid * SAVE_THREADS,
            );
        }

        self.saved_valid[centre_slot as usize] = true;
    }

    /// Measures the frame that just completed into the newest chunk.
    ///
    /// `slot_next` is the ring slot of the next real frame, `None` for a stream's last frame.
    pub(crate) fn measure<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        input: &Handle,
        outputs: &[Handle; 2],
        slot_t: u32,
        slot_next: Option<u32>,
        output_slot: usize,
    ) {
        let has_source = slot_next.is_some() && self.saved_valid[slot_t as usize];

        #[cfg(test)]
        if self.saved_valid[slot_t as usize] {
            self.measured_with_entry += 1;
        }

        let kept_from = self
            .last_completed
            .filter(|previous| self.saved_valid[previous.ring_slot as usize]);
        let has_kept = kept_from.is_some();
        let next = slot_next.unwrap_or(slot_t);
        let kept_entry = kept_from.map_or(slot_t, |previous| previous.ring_slot);
        let prev_output = kept_from.map_or(output_slot, |previous| previous.output_slot);

        let (chunk_hist, chunk_autocov) = self.open_chunk(client);
        let geometry = &self.geometry;
        let frame_len = (geometry.width * geometry.height * geometry.stored_ch) as usize;
        let ring_len = frame_len * geometry.ring_frames as usize;
        let cells_x = geometry.cells_x();
        let cells_y = geometry.cells_y();

        unsafe {
            grain_measure::launch_unchecked::<R>(
                client,
                CubeCount::new_2d(cells_x, cells_y),
                CubeDim::new_2d(CELL, CELL),
                geometry.stored_ch as usize,
                ArrayArg::from_raw_parts(input.clone(), ring_len),
                ArrayArg::from_raw_parts(outputs[output_slot].clone(), frame_len),
                ArrayArg::from_raw_parts(outputs[prev_output].clone(), frame_len),
                ArrayArg::from_raw_parts(self.saved_mv.clone(), geometry.saved_mv_len()),
                ArrayArg::from_raw_parts(self.saved_conf.clone(), geometry.saved_conf_len()),
                ArrayArg::from_raw_parts(self.edges.clone(), EDGES_LEN),
                ArrayArg::from_raw_parts(chunk_hist, 2 * HIST_LEN),
                ArrayArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                slot_t,
                next,
                slot_t,
                kept_entry,
                has_source as u32,
                has_kept as u32,
                geometry.width,
                geometry.height,
                geometry.stored_ch,
                geometry.blocks_x,
                geometry.blocks_y,
                geometry.step,
            );

            grain_reduce_partials::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(AUTOCOV_LEN as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                ArrayArg::from_raw_parts(self.partials.clone(), geometry.partials_len()),
                ArrayArg::from_raw_parts(chunk_autocov, GROUPED_AUTOCOV_LEN),
                geometry.cells(),
            );
        }

        let newest = self.chunks.last_mut().expect("open_chunk pushed a chunk");
        newest.frames += 1;
        if newest.frames == CHUNK_FRAMES {
            self.chunk_open = false;
        }

        self.last_completed = Some(Completed {
            ring_slot: slot_t,
            output_slot,
        });
    }

    /// The newest chunk's histogram and autocovariance handles, opening a chunk when none is open.
    fn open_chunk<R: Runtime>(&mut self, client: &ComputeClient<R>) -> (Handle, Handle) {
        if !self.chunk_open {
            let hist_host = vec![0i32; 2 * HIST_LEN];
            let autocov_host = vec![0.0f32; GROUPED_AUTOCOV_LEN];
            let hist = client.create_from_slice(i32::as_bytes(&hist_host));
            let autocov = client.create_from_slice(f32::as_bytes(&autocov_host));
            self.chunks.push(ChunkBuffers {
                hist,
                autocov,
                frames: 0,
            });
            self.chunk_open = true;
        }

        let chunk = self.chunks.last().expect("a chunk is open");
        (chunk.hist.clone(), chunk.autocov.clone())
    }

    #[cfg(test)]
    pub(crate) fn measured_with_entry(&self) -> u32 {
        self.measured_with_entry
    }

    /// Forgets the stream's saved vectors and closes the open chunk.
    pub(crate) fn reset_stream(&mut self) {
        self.saved_valid.fill(false);
        self.last_completed = None;
        self.chunk_open = false;
    }

    /// Reads every chunk back, in order, and forgets them.
    pub(crate) fn drain<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
    ) -> Result<Vec<GrainChunk>, anyhow::Error> {
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

fn read_chunk<R: Runtime>(
    client: &ComputeClient<R>,
    buffer: ChunkBuffers,
) -> Result<GrainChunk, anyhow::Error> {
    let hist_bytes = client
        .read_one(buffer.hist)
        .map_err(|error| anyhow::anyhow!("grain histogram readback failed: {error}"))?;
    let autocov_bytes = client
        .read_one(buffer.autocov)
        .map_err(|error| anyhow::anyhow!("grain autocovariance readback failed: {error}"))?;
    let hist = i32::from_bytes(&hist_bytes);
    let autocov = f32::from_bytes(&autocov_bytes);
    let (source_counts, kept_counts) = hist.split_at(HIST_LEN);

    let mut chunk = GrainChunk::empty();
    chunk.frames = buffer.frames;
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

    Ok(chunk)
}
