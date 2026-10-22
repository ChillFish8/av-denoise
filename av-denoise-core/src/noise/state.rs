use cubecl::prelude::*;
use cubecl::server::Handle;

use super::spatial::{
    RESULTS_PER_SLOT,
    noise_partials_slot_stride_bytes,
    noise_partials_slot_view,
    noise_results_slot_view,
};
use super::strength_map::{QuarterSettings, StrengthMapParams};
use super::temporal::{temporal_stats_buf_bytes, temporal_stats_slot_view};
use crate::engine::align::StorageAlign;

/// The plane geometry one noise state measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NoiseShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    /// Ring slots, one stats region and one reading each.
    pub frames: u32,
    /// Whether temporal statistics are measured, which needs a temporal radius of at least 1.
    pub temporal: bool,
    pub align: StorageAlign,
}

/// Which strength map a reading builds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no production caller builds a strength map")
)]
pub(crate) enum StrengthMapKind {
    #[default]
    Off,
    Luma(StrengthMapParams),
    Chroma {
        flat_boost: f32,
    },
}

/// What one slot's reading measures.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ReadingSettings {
    /// Whether the stats carry the luma quarter fields the curve and the classes read.
    pub luma_fields: bool,
    pub quarter: QuarterSettings,
    pub map: StrengthMapKind,
}

/// The noise measurement buffers of one ring of frames.
pub(crate) struct NoiseState<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) shape: NoiseShape,
    pub(super) settings: ReadingSettings,
    /// The first stage of each slot's Immerkær estimate, one padded region per slot.
    pub(super) partials: Handle,
    /// The per-channel Immerkær totals, one row per slot.
    pub(super) results: Handle,
    /// The temporal residual statistics, one padded region per slot, when they are measured.
    pub(super) stats: Option<Handle>,
}

impl<R: Runtime> NoiseState<R> {
    pub(crate) fn new(client: &ComputeClient<R>, shape: NoiseShape, settings: ReadingSettings) -> Self {
        let frames = u64::from(shape.frames);
        let partials_stride = noise_partials_slot_stride_bytes(shape.width, shape.height, shape.align);
        let partials_bytes = partials_stride * frames;
        let partials = client.empty(partials_bytes as usize);

        let results_len = shape.frames as usize * RESULTS_PER_SLOT;
        let results = client.empty(results_len * size_of::<f32>());

        let stats = if shape.temporal {
            let stats_bytes = temporal_stats_buf_bytes(
                shape.width,
                shape.height,
                shape.stored_ch,
                shape.frames,
                shape.align,
            );
            let stats = client.empty(stats_bytes);
            Some(stats)
        } else {
            None
        };

        Self {
            client: client.clone(),
            shape,
            settings,
            partials,
            results,
            stats,
        }
    }

    pub(crate) fn shape(&self) -> NoiseShape {
        self.shape
    }

    pub(crate) fn settings(&self) -> ReadingSettings {
        self.settings
    }

    /// Replaces the settings before the first queued slot.
    pub(crate) fn set_settings(&mut self, settings: ReadingSettings) {
        self.settings = settings;
    }

    /// The results row, the partials and, with temporal stats, the records of `slot`.
    pub(crate) fn reading_handles(&self, slot: u32) -> Vec<Handle> {
        let shape = self.shape;
        let row = noise_results_slot_view(&self.results, shape.frames, slot);
        let partials = noise_partials_slot_view(
            &self.partials,
            shape.width,
            shape.height,
            shape.frames,
            slot,
            shape.align,
        );
        let mut handles = vec![row, partials];

        if let Some(stats_buf) = self.stats.as_ref() {
            let stats = temporal_stats_slot_view(
                stats_buf,
                shape.width,
                shape.height,
                shape.stored_ch,
                shape.frames,
                slot,
                shape.align,
            );
            handles.push(stats);
        }

        handles
    }

    /// The partials ring, offset to the start of `slot`.
    pub(super) fn partials_slot(&self, slot: u32) -> Handle {
        let stride = noise_partials_slot_stride_bytes(self.shape.width, self.shape.height, self.shape.align);
        let start = u64::from(slot) * stride;

        self.partials.clone().offset_start(start)
    }

    #[cfg(test)]
    pub(crate) fn partials(&self) -> &Handle {
        &self.partials
    }

    #[cfg(test)]
    pub(crate) fn results(&self) -> &Handle {
        &self.results
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> Option<&Handle> {
        self.stats.as_ref()
    }
}
