use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::noise::{
    DerivedValues,
    FoldSettings,
    NOISE_WINDOW_LEN,
    NOISE_WINDOW_RADIUS,
    NoiseSettings,
    NoiseShape,
    NoiseState,
    ReadingSettings,
    WindowBounds,
};

/// One launch of a slot's reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadingStage {
    BlockDerive,
    BlockAccept,
    CurveBins,
    /// The curve batch's three histogram passes.
    CurveBatch,
    SampleFinish,
}

/// What a reading builds and what its folds write.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReadingOptions {
    /// Whether the stats carry the luma quarter fields and the reading builds a curve, as nl4d's
    /// luma plane does.
    pub luma_fields: bool,
    /// The search radius of the NLM offset tables every fold writes.
    pub offset_tables: Option<u32>,
}

/// A two-slot noise state whose slot 1 holds the stats of `ring`'s second frame against its first.
pub struct NoiseReading {
    state: NoiseState,
}

impl NoiseReading {
    /// Measures the stats of a two-frame `ring` of `width` by `height` frames, building what
    /// `options` asks for.
    pub fn new(
        client: &Client,
        width: u32,
        height: u32,
        channels: u32,
        ring: &Handle,
        options: ReadingOptions,
    ) -> Result<Self, anyhow::Error> {
        let luma_fields = options.luma_fields;
        let shape = NoiseShape {
            width,
            height,
            channels,
            frames: 2,
            ring_radius: 1,
            temporal: true,
            align: StorageAlign::from_client(client),
        };
        let settings = ReadingSettings { luma_fields };
        let fold = FoldSettings {
            offset_tables: options.offset_tables,
            ..FoldSettings::default()
        };
        let noise_settings = NoiseSettings {
            reading: settings,
            fold,
        };
        let state = NoiseState::new(client, shape, noise_settings, DerivedValues::default());
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

        if stage == ReadingStage::CurveBins {
            return;
        }

        self.state.queue_curve_bins(1);

        if stage == ReadingStage::CurveBatch {
            return;
        }

        self.state.queue_curve_batch();
    }

    /// Queues the launch of `stage`.
    pub fn queue_stage(&self, stage: ReadingStage) {
        match stage {
            ReadingStage::BlockDerive => self.state.queue_block_derive(1),
            ReadingStage::BlockAccept => self.state.queue_block_accept(1),
            ReadingStage::CurveBins => self.state.queue_curve_bins(1),
            ReadingStage::CurveBatch => self.state.queue_curve_batch(),
            ReadingStage::SampleFinish => self.state.queue_sample_finish(1),
        };
    }

    /// Queues every launch of the reading.
    pub fn queue_reading(&self) {
        self.queue_before(ReadingStage::SampleFinish);
        self.queue_stage(ReadingStage::SampleFinish);
    }

    /// Queues the copy of slot 1's reading into the history as stream frame 0.
    pub fn queue_record(&self) {
        self.state.record_reading(1, 0);
    }

    /// Queues the copies of slot 1's reading into the history as every frame of a full window.
    pub fn record_window(&self) {
        for frame in 0..NOISE_WINDOW_LEN {
            self.state.record_reading(1, frame);
        }
    }

    /// Queues a window fold over a full window of the recorded history.
    ///
    /// # Errors
    ///
    /// Returns an error before [Self::record_window] has run.
    pub fn queue_window_fold(&self) -> Result<(), anyhow::Error> {
        let newest = NOISE_WINDOW_LEN - 1;
        let bounds = WindowBounds::around(NOISE_WINDOW_RADIUS, newest);

        self.state.queue_window_fold(bounds)
    }

    /// The curve batch's histogram and state with their lengths in words, which its passes add to.
    pub fn curve_batch_buffers(&self) -> Vec<(Handle, u32)> {
        self.state.curve_batch_buffers()
    }
}
