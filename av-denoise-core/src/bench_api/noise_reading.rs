use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::noise::{NoiseShape, NoiseState, ReadingSettings};

/// One launch of a slot's reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadingStage {
    BlockDerive,
    BlockAccept,
    SampleFinish,
}

/// A two-slot noise state whose slot 1 holds the stats of `ring`'s second frame against its first.
pub struct NoiseReading<R: Runtime> {
    state: NoiseState<R>,
}

impl<R: Runtime> NoiseReading<R> {
    /// Measures the stats of a two-frame `ring` of `width` by `height` frames.
    ///
    /// A single-channel plane measures the luma quarter fields, as nl4d's luma plane does.
    pub fn new(
        client: &ComputeClient<R>,
        width: u32,
        height: u32,
        channels: u32,
        stored_ch: u32,
        ring: &Handle,
    ) -> Result<Self, anyhow::Error> {
        let shape = NoiseShape {
            width,
            height,
            channels,
            stored_ch,
            frames: 2,
            temporal: true,
            align: StorageAlign::from_client(client),
        };
        let settings = ReadingSettings {
            luma_fields: channels == 1,
            ..ReadingSettings::default()
        };
        let state = NoiseState::new(client, shape, settings);
        state.queue_stats(ring, 1, Some(0))?;

        Ok(Self { state })
    }

    /// Queues every launch of the reading that runs before `stage`.
    pub fn queue_before(&self, stage: ReadingStage) {
        if stage == ReadingStage::BlockDerive {
            return;
        }

        self.state.queue_block_derive(1);
        self.state.queue_immerkaer_batch();

        if stage == ReadingStage::BlockAccept {
            return;
        }

        self.state.queue_block_accept(1);
        self.state.queue_sample_batch();
    }

    /// Queues the launch of `stage`.
    pub fn queue_stage(&self, stage: ReadingStage) {
        match stage {
            ReadingStage::BlockDerive => self.state.queue_block_derive(1),
            ReadingStage::BlockAccept => self.state.queue_block_accept(1),
            ReadingStage::SampleFinish => self.state.queue_sample_finish(1),
        }
    }

    /// The histogram and state of each select batch with their lengths in words, which the finish
    /// clears.
    pub fn batch_buffers(&self) -> Vec<(Handle, u32)> {
        self.state.reading_batch_buffers()
    }
}
