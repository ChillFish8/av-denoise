use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::noise::{
    NoiseShape,
    NoiseState,
    QuarterSettings,
    ReadingSettings,
    StrengthMapKind,
    StrengthMapParams,
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
    Classify,
    StrengthMap,
}

/// The luma strength map a reading builds after its curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadingMap {
    pub flat_boost: f32,
    pub shadow_soften: f32,
    /// The texture veto's coherence cut, where 1.0 or more turns the veto off.
    pub texture_cut: f32,
    /// The line ring's radius in quarters, where 0 turns the ring off.
    pub line_ring: usize,
}

/// A two-slot noise state whose slot 1 holds the stats of `ring`'s second frame against its first.
pub struct NoiseReading<R: Runtime> {
    state: NoiseState<R>,
}

impl<R: Runtime> NoiseReading<R> {
    /// Measures the stats of a two-frame `ring` of `width` by `height` frames.
    ///
    /// With `luma_fields` the stats carry the luma quarter fields and the reading builds a curve, as
    /// nl4d's luma plane does. With `map` it then builds a luma strength map.
    #[expect(
        clippy::too_many_arguments,
        reason = "the plane's geometry, its frames and what the reading builds"
    )]
    pub fn new(
        client: &ComputeClient<R>,
        width: u32,
        height: u32,
        channels: u32,
        stored_ch: u32,
        ring: &Handle,
        luma_fields: bool,
        map: Option<ReadingMap>,
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
        let settings = match map {
            Some(map) => map_settings(luma_fields, map),
            None => ReadingSettings {
                luma_fields,
                ..ReadingSettings::default()
            },
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

        if stage == ReadingStage::CurveBins {
            return;
        }

        self.state.queue_curve_bins(1);

        if stage == ReadingStage::CurveBatch {
            return;
        }

        self.state.queue_curve_batch();

        if stage == ReadingStage::SampleFinish {
            return;
        }

        self.state.queue_sample_finish(1);

        if stage == ReadingStage::Classify {
            return;
        }

        self.state.queue_classify(1);
    }

    /// Queues the launch of `stage`.
    pub fn queue_stage(&self, stage: ReadingStage) {
        match stage {
            ReadingStage::BlockDerive => self.state.queue_block_derive(1),
            ReadingStage::BlockAccept => self.state.queue_block_accept(1),
            ReadingStage::CurveBins => self.state.queue_curve_bins(1),
            ReadingStage::CurveBatch => self.state.queue_curve_batch(),
            ReadingStage::SampleFinish => self.state.queue_sample_finish(1),
            ReadingStage::Classify => self.state.queue_classify(1),
            ReadingStage::StrengthMap => self.state.queue_strength_map(1),
        };
    }

    /// The curve batch's histogram and state with their lengths in words, which its passes add to.
    pub fn curve_batch_buffers(&self) -> Vec<(Handle, u32)> {
        self.state.curve_batch_buffers()
    }
}

fn map_settings(luma_fields: bool, map: ReadingMap) -> ReadingSettings {
    let params = StrengthMapParams {
        flat_boost: map.flat_boost,
        shadow_soften: map.shadow_soften,
    };
    let quarter = QuarterSettings {
        texture_cut: Some(map.texture_cut),
        line_ring: Some(map.line_ring).filter(|&radius| radius > 0),
    };

    ReadingSettings {
        luma_fields,
        quarter,
        map: StrengthMapKind::Luma(params),
    }
}
