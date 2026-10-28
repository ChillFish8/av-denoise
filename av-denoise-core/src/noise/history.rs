use anyhow::{anyhow, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::reading::READING_WORDS;
use super::spatial::RESULTS_PER_SLOT;
use super::state::{NoiseState, quarter_count};

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
    /// A history whose slots hold `records`, `results` and `maps`, laid out slot after slot.
    ///
    /// # Panics
    ///
    /// Panics when a buffer does not hold exactly one entry per slot.
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
    /// A history whose every slot holds the reading, results and map of ring slot `slot`.
    ///
    /// It reads the slot back, so it waits for every queued launch.
    pub(crate) fn history_of_slot(
        &self,
        slot: u32,
        temporal_radius: u32,
    ) -> Result<ReadingHistory, anyhow::Error> {
        let frames = self.shape.frames;
        ensure!(slot < frames, "slot {slot} is past the {frames} slots");

        let measurement = self.measurement();
        let capacity = history_capacity(temporal_radius) as usize;
        let record = self.read_slot_words(&measurement.readings, slot, READING_WORDS as usize)?;
        let results = self.read_slot_words(&measurement.results, slot, RESULTS_PER_SLOT)?;

        let (map_len, map) = match self.slot_maps() {
            Some(slot_maps) => {
                let quarters = quarter_count(self.shape);
                let map = self.read_slot_words(slot_maps, slot, quarters as usize)?;
                (quarters, map)
            },
            None => (0, Vec::new()),
        };

        let records = record.repeat(capacity);
        let results = results.repeat(capacity);
        let maps = map.repeat(capacity);
        let history =
            ReadingHistory::from_words(&self.client, temporal_radius, map_len, &records, &results, &maps);

        Ok(history)
    }

    /// The `len` words of `slot` in a buffer that holds `len` words per slot.
    fn read_slot_words(&self, buffer: &Handle, slot: u32, len: usize) -> Result<Vec<f32>, anyhow::Error> {
        let bytes = self
            .client
            .read_one(buffer.clone())
            .map_err(|error| anyhow!("the noise slot read back failed: {error:?}"))?;
        let words = f32::from_bytes(&bytes);
        let start = slot as usize * len;
        let slot_words = words[start..start + len].to_vec();

        Ok(slot_words)
    }
}
