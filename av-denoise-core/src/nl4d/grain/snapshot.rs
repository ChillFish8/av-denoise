use cubecl::prelude::*;
use cubecl::server::Handle;

use super::consts::CHUNK_FRAMES;
use crate::engine::{BLOCK_1D, MAX_GRID_1D};
use crate::nlmeans::kernels::gpu_copy;
use crate::noise::colour::{COLOUR_CELLS, COLOUR_FIT_PLANE_WORDS, ColourFit};
use crate::noise::{READING_WORDS, ReadingHistory};
use crate::tune::gpu::zeroed_words;

/// Values in one frame's colour fit for `planes` planes, each plane's log grain table, log gain
/// table and fit header.
///
/// The planes follow in turn, each with its table, then its gains, then its header.
pub(crate) fn colour_frame_len(planes: u32) -> usize {
    let plane_len = 2 * COLOUR_CELLS + COLOUR_FIT_PLANE_WORDS;
    let frame_len = planes * plane_len;

    frame_len as usize
}

/// What one pass filtered its centre frame with.
pub(crate) struct PassSources<'a> {
    /// The readings of the latest stream frames, `None` when the noise is not measured.
    pub history: Option<&'a ReadingHistory>,
    /// The stream frame the pass is centred on.
    pub centre_frame: u32,
    /// The grain model the pass scaled its thresholds by, `None` without the colour model.
    pub colour: Option<&'a ColourFit>,
}

/// One chunk's snapshot buffers, one slot per frame.
#[derive(Clone)]
pub(super) struct SnapshotChunk {
    /// Each frame's noise reading, present when the reading is kept.
    pub noise: Option<Handle>,
    /// Each frame's colour fit, present when the colour fit is kept.
    pub colour: Option<Handle>,
}

/// The noise reading and colour fit each frame was filtered with.
pub(super) struct SnapshotMeasure {
    /// Frames in the ring, one staging slot each.
    ring_frames: u32,
    /// [READING_WORDS] values per ring slot, present when the reading is kept.
    noise: Option<Handle>,
    /// One colour fit per ring slot, present once the colour fit is kept.
    colour: Option<ColourSlots>,
}

/// The staged colour fit of each ring slot.
struct ColourSlots {
    planes: u32,
    /// [colour_frame_len] values per ring slot.
    slots: Handle,
}

/// A buffer of `buffer_len` values, read or written from `offset`.
struct Span<'a> {
    buffer: &'a Handle,
    buffer_len: usize,
    offset: usize,
}

impl SnapshotMeasure {
    /// Slots for a ring of `ring_frames` frames, keeping the noise reading with `keeps_noise`.
    pub(super) fn new(client: &Client, ring_frames: u32, keeps_noise: bool) -> Self {
        let noise_len = ring_frames as usize * READING_WORDS as usize;
        let noise = keeps_noise.then(|| zeroed_words(client, noise_len));

        Self {
            ring_frames,
            noise,
            colour: None,
        }
    }

    /// Keeps the colour fit of `planes` planes from the next pass on.
    pub(super) fn keep_colour(&mut self, client: &Client, planes: u32) {
        let slots_len = self.ring_frames as usize * colour_frame_len(planes);
        let slots = zeroed_words(client, slots_len);

        self.colour = Some(ColourSlots { planes, slots });
    }

    /// Values in one frame's noise reading, 0 when the reading is not kept.
    pub(super) fn noise_len(&self) -> usize {
        if self.noise.is_some() {
            READING_WORDS as usize
        } else {
            0
        }
    }

    /// Values in one frame's colour fit, 0 when the colour fit is not kept.
    pub(super) fn colour_len(&self) -> usize {
        self.colour
            .as_ref()
            .map_or(0, |colour| colour_frame_len(colour.planes))
    }

    /// Copies what the pass filtered with into the ring slot `ring_slot` of its centre frame.
    ///
    /// The slot is copied into the open chunk once its frame completes, see [Self::measure].
    pub(super) fn keep(&self, client: &Client, sources: &PassSources, ring_slot: u32) {
        let ring_slot = ring_slot as usize;

        if let (Some(noise), Some(history)) = (self.noise.as_ref(), sources.history) {
            let words = READING_WORDS as usize;
            let capacity = history.capacity();
            let history_slot = (sources.centre_frame % capacity) as usize;
            let source = Span {
                buffer: history.records(),
                buffer_len: capacity as usize * words,
                offset: history_slot * words,
            };
            let target = Span {
                buffer: noise,
                buffer_len: self.ring_frames as usize * words,
                offset: ring_slot * words,
            };
            copy_values(client, &source, &target, words);
        }

        if let (Some(colour), Some(fit)) = (self.colour.as_ref(), sources.colour) {
            self.keep_colour_fit(client, colour, fit, ring_slot);
        }
    }

    /// Copies each plane's table, gains and header into ring slot `ring_slot`.
    fn keep_colour_fit(&self, client: &Client, colour: &ColourSlots, fit: &ColourFit, ring_slot: usize) {
        debug_assert_eq!(colour.planes, fit.planes(), "the slots match the fit's planes");

        let cells = COLOUR_CELLS as usize;
        let header_len = COLOUR_FIT_PLANE_WORDS as usize;
        let planes = colour.planes as usize;
        let frame_len = colour_frame_len(colour.planes);
        let plane_len = 2 * cells + header_len;
        let tables_len = planes * cells;
        let headers_len = planes * header_len;
        let slots_len = self.ring_frames as usize * frame_len;

        for plane in 0..planes {
            let plane_start = ring_slot * frame_len + plane * plane_len;
            let parts = [
                (fit.table(), tables_len, plane * cells, cells),
                (fit.gains(), tables_len, plane * cells, cells),
                (fit.fit_planes(), headers_len, plane * header_len, header_len),
            ];

            let mut target_offset = plane_start;
            for (buffer, buffer_len, offset, len) in parts {
                let source = Span {
                    buffer,
                    buffer_len,
                    offset,
                };
                let target = Span {
                    buffer: &colour.slots,
                    buffer_len: slots_len,
                    offset: target_offset,
                };
                copy_values(client, &source, &target, len);
                target_offset += len;
            }
        }
    }

    /// A chunk with every frame slot zeroed.
    pub(super) fn open_chunk(&self, client: &Client) -> SnapshotChunk {
        let frames = CHUNK_FRAMES as usize;
        let noise_len = frames * self.noise_len();
        let colour_len = frames * self.colour_len();
        let noise = self.noise.as_ref().map(|_| zeroed_words(client, noise_len));
        let colour = self.colour.as_ref().map(|_| zeroed_words(client, colour_len));

        SnapshotChunk { noise, colour }
    }

    /// Copies ring slot `ring_slot` into frame slot `chunk_frames` of `chunk`.
    pub(super) fn measure(&self, client: &Client, ring_slot: u32, chunk: &SnapshotChunk, chunk_frames: u32) {
        debug_assert!(chunk_frames < CHUNK_FRAMES, "the chunk has a slot for the frame");

        let ring_slot = ring_slot as usize;
        let frame_slot = chunk_frames as usize;

        if let (Some(noise), Some(chunk_noise)) = (self.noise.as_ref(), chunk.noise.as_ref()) {
            let len = self.noise_len();
            copy_slot(
                client,
                noise,
                chunk_noise,
                ring_slot,
                frame_slot,
                self.ring_frames,
                len,
            );
        }

        let colour = self.colour.as_ref();
        if let (Some(colour), Some(chunk_colour)) = (colour, chunk.colour.as_ref()) {
            let len = self.colour_len();
            copy_slot(
                client,
                &colour.slots,
                chunk_colour,
                ring_slot,
                frame_slot,
                self.ring_frames,
                len,
            );
        }
    }
}

/// Copies slot `ring_slot` of `ring`, `len` values per slot, into slot `frame_slot` of `chunk`.
fn copy_slot(
    client: &Client,
    ring: &Handle,
    chunk: &Handle,
    ring_slot: usize,
    frame_slot: usize,
    ring_frames: u32,
    len: usize,
) {
    let source = Span {
        buffer: ring,
        buffer_len: ring_frames as usize * len,
        offset: ring_slot * len,
    };
    let target = Span {
        buffer: chunk,
        buffer_len: CHUNK_FRAMES as usize * len,
        offset: frame_slot * len,
    };

    copy_values(client, &source, &target, len);
}

/// Copies `len` values from `source` into `target` on the device.
fn copy_values(client: &Client, source: &Span, target: &Span, len: usize) {
    let length = len as u32;
    let grid = length.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        gpu_copy::launch_unchecked(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            BufferArg::from_raw_parts(source.buffer.clone(), source.buffer_len),
            BufferArg::from_raw_parts(target.buffer.clone(), target.buffer_len),
            source.offset as u32,
            target.offset as u32,
            length,
            total_threads,
        );
    }
}
