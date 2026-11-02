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
/// record and its Immerkær abs sums.
pub(crate) struct ReadingHistory {
    capacity: u32,
    /// [READING_WORDS](crate::noise::reading::READING_WORDS) values per slot.
    records: Handle,
    /// [RESULTS_PER_SLOT](crate::noise::spatial::RESULTS_PER_SLOT) values per slot.
    results: Handle,
}

impl ReadingHistory {
    /// A zeroed history for a stream with `temporal_radius`.
    pub(crate) fn new(client: &Client, temporal_radius: u32) -> Self {
        let capacity = history_capacity(temporal_radius);
        let slots = capacity as usize;
        let records = zeroed_words(client, slots * READING_WORDS as usize);
        let results = zeroed_words(client, slots * RESULTS_PER_SLOT);

        Self {
            capacity,
            records,
            results,
        }
    }

    /// A history whose slots hold `records` and `results`, laid out slot after slot.
    ///
    /// # Panics
    ///
    /// Panics when a buffer does not hold exactly one entry per slot.
    #[cfg(test)]
    pub(crate) fn from_words(
        client: &Client,
        temporal_radius: u32,
        records: &[f32],
        results: &[f32],
    ) -> Self {
        let capacity = history_capacity(temporal_radius);
        let slots = capacity as usize;
        let records_len = records.len();
        let results_len = results.len();
        assert_eq!(records_len, slots * READING_WORDS as usize, "one record per slot");
        assert_eq!(results_len, slots * RESULTS_PER_SLOT, "one results row per slot");

        let records_bytes = f32::as_bytes(records);
        let results_bytes = f32::as_bytes(results);
        let records = client.create_from_slice(records_bytes);
        let results = client.create_from_slice(results_bytes);

        Self {
            capacity,
            records,
            results,
        }
    }

    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    pub(crate) fn records(&self) -> &Handle {
        &self.records
    }

    pub(crate) fn results(&self) -> &Handle {
        &self.results
    }
}

impl NoiseState {
    /// Queues the copy of ring slot `slot`'s reading and Immerkær abs sums into the history slot of
    /// stream frame `frame`, in one launch.
    ///
    /// The stream then counts every frame up to `frame` as recorded. A state that holds fixed
    /// values queues nothing.
    pub(crate) fn record_reading(&self, slot: u32, frame: u32) {
        let Some(measurement) = self.measurement.as_ref() else {
            return;
        };

        let history = &measurement.history;
        let frames = self.shape.frames;
        let capacity = history.capacity;
        let history_slot = frame % capacity;
        let cubes = READING_WORDS.div_ceil(NOISE_BLOCK_1D);
        let grid = cubes.clamp(1, MAX_GRID_1D);

        let readings_len = (frames * READING_WORDS) as usize;
        let results_len = frames as usize * RESULTS_PER_SLOT;
        let records_len = (capacity * READING_WORDS) as usize;
        let history_results_len = capacity as usize * RESULTS_PER_SLOT;

        unsafe {
            noise_record::launch_unchecked(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(NOISE_BLOCK_1D),
                bind(&measurement.readings, readings_len),
                bind(&measurement.results, results_len),
                bind(&history.records, records_len),
                bind(&history.results, history_results_len),
                slot,
                history_slot,
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
}
