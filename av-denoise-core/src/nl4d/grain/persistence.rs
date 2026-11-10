use cubecl::prelude::*;
use cubecl::server::Handle;

use super::consts::{
    CELL,
    CHROMA_RECORD_LEN,
    CHUNK_FRAMES,
    FRAME_RECORD_LEN,
    HISTORY_ENTRY_LEN,
    HISTORY_FRAMES,
    LAG_RECORD_LEN,
    PATTERNS,
    PERSIST_BLOCK_CELLS,
    PERSIST_THREADS,
    PERSISTENCE_LAGS,
    PROJECTIONS,
    REDUCE_THREADS,
};
use super::export::{CompletedFrame, GrainGeometry};
use crate::nl4d::kernels::{grain_persist, grain_persist_reduce, grain_project};

/// The pattern words the project kernel reads, two per projection.
const PATTERN_WORDS_LEN: usize = 2 * PROJECTIONS;
/// The lags each chroma plane measures, lag 1 only.
const CHROMA_LAGS: usize = 1;
/// The chroma planes measured, Cb then Cr.
const CHROMA_CHANNELS: u32 = 2;

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

/// Values in the persistence history of one channel, an entry per cell for each history frame.
pub(crate) fn persistence_history_len(width: u32, height: u32) -> u64 {
    let cells = cell_count(width, height);
    cells * u64::from(HISTORY_FRAMES) * HISTORY_ENTRY_LEN as u64
}

/// Values in the persistence partials of one channel, one lag record part per block and lag.
pub(crate) fn persistence_partials_len(width: u32, height: u32, lags: usize) -> u64 {
    let blocks = persist_blocks(width, height);
    blocks * (lags * LAG_RECORD_LEN) as u64
}

/// The planes a persistence measurement reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeasuredPlanes {
    /// The luma plane, every lag from 1 to [PERSISTENCE_LAGS].
    Luma,
    /// Both chroma planes, lag 1 only.
    Chroma,
}

impl MeasuredPlanes {
    /// Values in one frame's record.
    pub(crate) fn frame_record_len(self) -> usize {
        match self {
            MeasuredPlanes::Luma => FRAME_RECORD_LEN,
            MeasuredPlanes::Chroma => CHROMA_RECORD_LEN,
        }
    }

    fn channel_plans(self) -> Vec<ChannelPlan> {
        match self {
            MeasuredPlanes::Luma => {
                let luma = ChannelPlan {
                    channel: 0,
                    lags: PERSISTENCE_LAGS,
                    luma_band: true,
                    record_offset: 0,
                };
                vec![luma]
            },
            MeasuredPlanes::Chroma => {
                let plans = (0..CHROMA_CHANNELS).map(|channel| ChannelPlan {
                    channel,
                    lags: CHROMA_LAGS,
                    luma_band: false,
                    record_offset: channel as usize * CHROMA_LAGS * LAG_RECORD_LEN,
                });
                plans.collect()
            },
        }
    }

    /// The history and partials values the measurement holds over every channel it reads.
    pub(crate) fn buffer_lens(self, width: u32, height: u32) -> (u64, u64) {
        let plans = self.channel_plans();
        let history_len = persistence_history_len(width, height);

        let mut history = 0;
        let mut partials = 0;
        for plan in plans {
            history += history_len;
            partials += persistence_partials_len(width, height, plan.lags);
        }

        (history, partials)
    }
}

/// How one channel is measured.
#[derive(Clone, Copy)]
struct ChannelPlan {
    /// The channel read from each pixel.
    channel: u32,
    /// How many lags the channel measures, from lag 1.
    lags: usize,
    /// Whether a cell must sit inside the luma band to count.
    luma_band: bool,
    /// Where the channel's sums start in a frame record.
    record_offset: usize,
}

/// One channel's persistence buffers.
struct ChannelMeasure {
    plan: ChannelPlan,
    /// Each cell's projections and flat flag for the last [HISTORY_FRAMES] completed frames.
    history: Handle,
    /// Each lag's persistence sums per block of cells, before they add into a frame record.
    partials: Handle,
}

impl ChannelMeasure {
    fn new(client: &Client, geometry: &GrainGeometry, plan: ChannelPlan) -> Self {
        let history_len = geometry.history_len();
        let partials_len = geometry.persist_partials_len(plan.lags);
        let history_bytes = history_len.max(1) * size_of::<f32>();
        let history = client.empty(history_bytes);
        let partials_bytes = partials_len.max(1) * size_of::<f32>();
        let partials = client.empty(partials_bytes);

        Self {
            plan,
            history,
            partials,
        }
    }
}

/// Where one frame's projections and sums land.
struct FrameTarget<'a> {
    frame_record: &'a Handle,
    /// The record's first value for this frame.
    record_offset: u32,
    history_slot: u32,
    /// Completed frames measured in this stream before this one.
    frames_in_stream: u32,
}

/// The grain persistence measurement.
pub(super) struct PersistenceMeasure {
    /// The projection patterns as words, two per projection.
    patterns: Handle,
    channels: Vec<ChannelMeasure>,
    /// Values in one frame's record.
    frame_record_len: usize,
    /// Completed frames measured since the stream started.
    frames_in_stream: u32,
}

impl PersistenceMeasure {
    pub(super) fn new(client: &Client, geometry: &GrainGeometry, planes: MeasuredPlanes) -> Self {
        let words = pattern_words();
        let patterns_bytes = u32::as_bytes(&words);
        let patterns = client.create_from_slice(patterns_bytes);
        let plans = planes.channel_plans();
        let channels = plans
            .into_iter()
            .map(|plan| ChannelMeasure::new(client, geometry, plan))
            .collect();

        Self {
            patterns,
            channels,
            frame_record_len: planes.frame_record_len(),
            frames_in_stream: 0,
        }
    }

    pub(super) fn frame_record_len(&self) -> usize {
        self.frame_record_len
    }

    fn chunk_record_len(&self) -> usize {
        CHUNK_FRAMES as usize * self.frame_record_len
    }

    /// Projects the completed frame into each channel's history and adds its pairs into the
    /// frame's record.
    ///
    /// `chunk_frames` is how many frames the chunk held before this one.
    pub(super) fn measure(
        &mut self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        frame_record: &Handle,
        chunk_frames: u32,
    ) {
        let record_offset = chunk_frames * self.frame_record_len as u32;
        let target = FrameTarget {
            frame_record,
            record_offset,
            history_slot: self.frames_in_stream % HISTORY_FRAMES,
            frames_in_stream: self.frames_in_stream,
        };

        for channel in &self.channels {
            self.launch(client, geometry, frame, channel, &target);
        }

        self.frames_in_stream += 1;
    }

    /// Launches the channel's projection, then each of its lags whose earlier frame is in this
    /// stream.
    fn launch(
        &self,
        client: &Client,
        geometry: &GrainGeometry,
        frame: &CompletedFrame,
        channel: &ChannelMeasure,
        target: &FrameTarget,
    ) {
        let plan = channel.plan;
        let history_len = geometry.history_len();
        let earlier_frames = target.frames_in_stream.min(plan.lags as u32);
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
                BufferArg::from_raw_parts(channel.history.clone(), history_len),
                frame.frame_slot,
                target.history_slot,
                geometry.width,
                geometry.height,
                geometry.channels,
                plan.channel,
                plan.luma_band,
            );
        }

        if earlier_frames == 0 {
            return;
        }

        let blocks = geometry.persist_blocks();
        let partials_len = geometry.persist_partials_len(plan.lags);
        let record_offset = target.record_offset + plan.record_offset as u32;
        let chunk_record_len = self.chunk_record_len();

        unsafe {
            grain_persist::launch_unchecked(
                client,
                CubeCount::new_2d(blocks, earlier_frames),
                CubeDim::new_1d(PERSIST_THREADS),
                BufferArg::from_raw_parts(channel.history.clone(), history_len),
                BufferArg::from_raw_parts(channel.partials.clone(), partials_len),
                target.history_slot,
                geometry.cells(),
                blocks,
            );

            grain_persist_reduce::launch_unchecked(
                client,
                CubeCount::new_1d(earlier_frames),
                CubeDim::new_1d(REDUCE_THREADS),
                BufferArg::from_raw_parts(channel.partials.clone(), partials_len),
                BufferArg::from_raw_parts(target.frame_record.clone(), chunk_record_len),
                record_offset,
                blocks,
            );
        }
    }

    pub(super) fn reset_stream(&mut self) {
        self.frames_in_stream = 0;
    }

    pub(super) fn open_chunk(&self, client: &Client) -> Handle {
        let record_host = vec![0.0f32; self.chunk_record_len()];
        let record_bytes = f32::as_bytes(&record_host);

        client.create_from_slice(record_bytes)
    }
}
