use std::sync::atomic::Ordering;

use anyhow::ensure;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NOISE_BLOCK_1D;
use super::kernels::noise_record;
use super::reading::READING_WORDS;
use super::spatial::RESULTS_PER_SLOT;
use super::state::NoiseState;
use crate::engine::MAX_GRID_1D;
use crate::tune::gpu::{bind, zeroed_words};

/// The frames a noise window reaches on each side of its centre.
pub(crate) const NOISE_WINDOW_RADIUS: u32 = 8;
/// The most frames one noise window holds.
pub(crate) const NOISE_WINDOW_LEN: u32 = 2 * NOISE_WINDOW_RADIUS + 1;

/// The history slots a stream with `temporal_radius` keeps.
pub(crate) fn history_capacity(temporal_radius: u32) -> u32 {
    NOISE_WINDOW_LEN + temporal_radius
}

/// The stream frames one window fold reads, inclusive at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowBounds {
    first: u32,
    last: u32,
}

impl WindowBounds {
    /// The window centred on frame `centre` of a stream whose newest recorded frame is `newest`.
    ///
    /// The window shortens where it would reach before frame 0 or past `newest`.
    pub(crate) fn around(centre: u32, newest: u32) -> Self {
        assert!(
            centre <= newest,
            "the centre {centre} is past the newest frame {newest}"
        );

        let first = centre.saturating_sub(NOISE_WINDOW_RADIUS);
        let reach = centre + NOISE_WINDOW_RADIUS;
        let last = reach.min(newest);

        Self { first, last }
    }

    pub(crate) fn first(&self) -> u32 {
        self.first
    }

    pub(crate) fn last(&self) -> u32 {
        self.last
    }

    pub(crate) fn len(&self) -> u32 {
        self.last - self.first + 1
    }
}

/// The readings of the latest stream frames, one slot per frame.
///
/// Stream frame `frame` lives in slot `frame % capacity`. Each slot holds the frame's reading
/// record, its Immerkær abs sums and, with a strength map, its map.
pub(crate) struct ReadingHistory {
    capacity: u32,
    /// The map quarters per slot, 0 without a strength map.
    map_len: u32,
    /// [READING_WORDS](crate::noise::reading::READING_WORDS) values per slot.
    records: Handle,
    /// [RESULTS_PER_SLOT](crate::noise::spatial::RESULTS_PER_SLOT) values per slot.
    results: Handle,
    /// `map_len` threshold multipliers per slot, `None` without a strength map.
    maps: Option<Handle>,
}

impl ReadingHistory {
    /// A zeroed history for a stream with `temporal_radius`, with `map_len` map quarters per slot.
    pub(crate) fn new(client: &Client, temporal_radius: u32, map_len: u32) -> Self {
        let capacity = history_capacity(temporal_radius);
        let slots = capacity as usize;
        let records = zeroed_words(client, slots * READING_WORDS as usize);
        let results = zeroed_words(client, slots * RESULTS_PER_SLOT);
        let maps = (map_len > 0).then(|| zeroed_words(client, slots * map_len as usize));

        Self {
            capacity,
            map_len,
            records,
            results,
            maps,
        }
    }

    /// A history whose slots hold `records`, `results` and `maps`, laid out slot after slot.
    ///
    /// # Panics
    ///
    /// Panics when a buffer does not hold exactly one entry per slot.
    #[cfg(test)]
    pub(crate) fn from_words(
        client: &Client,
        temporal_radius: u32,
        map_len: u32,
        records: &[f32],
        results: &[f32],
        maps: &[f32],
    ) -> Self {
        let capacity = history_capacity(temporal_radius);
        let slots = capacity as usize;
        let records_len = records.len();
        let results_len = results.len();
        let maps_len = maps.len();
        assert_eq!(records_len, slots * READING_WORDS as usize, "one record per slot");
        assert_eq!(results_len, slots * RESULTS_PER_SLOT, "one results row per slot");
        assert_eq!(maps_len, slots * map_len as usize, "one map per slot");

        let records_bytes = f32::as_bytes(records);
        let results_bytes = f32::as_bytes(results);
        let records = client.create_from_slice(records_bytes);
        let results = client.create_from_slice(results_bytes);
        let maps = (map_len > 0).then(|| {
            let maps_bytes = f32::as_bytes(maps);
            client.create_from_slice(maps_bytes)
        });

        Self {
            capacity,
            map_len,
            records,
            results,
            maps,
        }
    }

    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    pub(crate) fn map_len(&self) -> u32 {
        self.map_len
    }

    pub(crate) fn records(&self) -> &Handle {
        &self.records
    }

    pub(crate) fn results(&self) -> &Handle {
        &self.results
    }

    pub(crate) fn maps(&self) -> Option<&Handle> {
        self.maps.as_ref()
    }
}

impl NoiseState {
    /// Queues the copy of ring slot `slot`'s reading, Immerkær abs sums and map into the history
    /// slot of stream frame `frame`, in one launch.
    ///
    /// The stream then counts every frame up to `frame` as recorded. A state that holds fixed
    /// values queues nothing.
    pub(crate) fn record_reading(&self, slot: u32, frame: u32) {
        self.record(slot, frame, true);
    }

    /// Records ring slot `slot` as stream frame `frame` like [Self::record_reading], without its
    /// temporal sample.
    ///
    /// The frame then feeds the window its Immerkær values alone, as a frame without a sample does.
    pub(crate) fn record_spatial_reading(&self, slot: u32, frame: u32) {
        self.record(slot, frame, false);
    }

    fn record(&self, slot: u32, frame: u32, keeps_sample: bool) {
        let Some(measurement) = self.measurement.as_ref() else {
            return;
        };

        let history = &measurement.history;
        let frames = self.shape.frames;
        let capacity = history.capacity;
        let map_len = history.map_len;
        let history_slot = frame % capacity;
        let words = READING_WORDS.max(map_len);
        let cubes = words.div_ceil(NOISE_BLOCK_1D);
        let grid = cubes.clamp(1, MAX_GRID_1D);

        let readings_len = (frames * READING_WORDS) as usize;
        let results_len = frames as usize * RESULTS_PER_SLOT;
        let records_len = (capacity * READING_WORDS) as usize;
        let history_results_len = capacity as usize * RESULTS_PER_SLOT;
        let maps = self.record_map_args(history);

        unsafe {
            noise_record::launch_unchecked(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(NOISE_BLOCK_1D),
                bind(&measurement.readings, readings_len),
                bind(&measurement.results, results_len),
                maps.slot_maps,
                bind(&history.records, records_len),
                bind(&history.results, history_results_len),
                maps.history_maps,
                slot,
                history_slot,
                u32::from(keeps_sample),
                map_len,
            );
        }

        let count = frame + 1;
        measurement.recorded.fetch_max(count, Ordering::Relaxed);
    }

    /// The stream frames recorded since the last reset, 0 for a state that holds fixed values.
    pub(crate) fn stream_frame_count(&self) -> u32 {
        let Some(measurement) = self.measurement.as_ref() else {
            return 0;
        };

        measurement.recorded.load(Ordering::Relaxed)
    }

    /// Checks that `bounds` lies within the frames recorded since the last reset.
    pub(super) fn check_recorded(&self, bounds: WindowBounds) -> Result<(), anyhow::Error> {
        let recorded = self.stream_frame_count();
        let last = bounds.last();
        ensure!(
            last < recorded,
            "the window reaches frame {last} of the {recorded} recorded since the reset"
        );

        Ok(())
    }

    /// The slot maps and the history's maps, bound over every slot.
    ///
    /// Without a map the kernel never touches them, so the history's records stand in.
    fn record_map_args(&self, history: &ReadingHistory) -> RecordMapArgs {
        let records_len = (history.capacity * READING_WORDS) as usize;
        let (Some(slot_maps), Some(history_maps)) = (self.slot_maps(), history.maps.as_ref()) else {
            return RecordMapArgs {
                slot_maps: bind(&history.records, records_len),
                history_maps: bind(&history.records, records_len),
            };
        };

        let map_len = history.map_len as usize;
        let slot_maps_len = map_len * self.shape.frames as usize;
        let history_maps_len = map_len * history.capacity as usize;

        RecordMapArgs {
            slot_maps: bind(slot_maps, slot_maps_len),
            history_maps: bind(history_maps, history_maps_len),
        }
    }
}

/// The record launch's map bindings.
struct RecordMapArgs {
    slot_maps: BufferArg,
    history_maps: BufferArg,
}
