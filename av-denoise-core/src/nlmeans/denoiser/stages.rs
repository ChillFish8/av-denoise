use anyhow::Context;

use super::NlmDenoiser;
use crate::engine::timing::timed;
use crate::motion::{self, MotionEstimation, run_pyramid_build};
use crate::nlmeans::prefilter::{PrefilterCtx, PrefilterMode, run_prefilter};

impl NlmDenoiser {
    /// Runs every per-frame stage on a freshly pushed `slot`, through to the ring advance.
    ///
    /// The GPU stages run in device spans when the frame is sampled. The first frame's noise seed
    /// is queued between two spans.
    pub(super) fn run_post_upload_stages(&mut self, slot: usize) -> Result<(), anyhow::Error> {
        let client = self.client.clone();
        let frame_slot = slot as u32;

        timed(&client, self, move |front| front.queue_noise_stats(frame_slot))?;
        self.seed_noise_estimate_if_first_frame(frame_slot)?;

        timed(&client, self, move |front| front.run_reference_stages(slot))?;

        self.advance_ring();
        self.prime_leading_edge_if_first()
    }

    /// Queues the prefilter, the motion pyramids and the pair analyse for `slot`.
    fn run_reference_stages(&mut self, slot: usize) -> Result<(), anyhow::Error> {
        if let PrefilterMode::NlmSpatial { .. } = self.params.prefilter {
            self.run_nlm_spatial_pilot(slot as u32)
                .context("nlm spatial pilot dispatch failed")?;
        } else if self.params.prefilter.is_gpu_internal() {
            self.run_prefilter_for_slot(slot)?;
        }

        self.build_pyramids_for_slot(slot as u32)?;
        self.build_confidence_pyramid_for_slot(slot as u32)?;
        self.run_pair_analyse_for_slot(slot as u32)
    }

    fn run_prefilter_for_slot(&self, slot: usize) -> Result<(), anyhow::Error> {
        let reference_buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist for GPU prefilter");

        let prefilter_ctx = PrefilterCtx {
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            frame_count: self.params.total_frames(),
            frame: slot as u32,
            input_buf: &self.input_buf,
            reference_buf,
        };

        run_prefilter(self.params.prefilter, &self.client, &prefilter_ctx)
            .context("prefilter dispatch failed")
    }

    /// Builds the motion pyramids for `slot` on the input ring and, when present, the reference ring.
    pub(super) fn build_pyramids_for_slot(&self, slot: u32) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        let stored_ch = self.params.channels.storage_count();
        let frame_count = self.params.total_frames();

        if let Some(input_pyramid) = self.pyramid_input.as_ref() {
            run_pyramid_build(
                &self.client,
                motion_ctx,
                self.width,
                self.height,
                frame_count,
                slot,
                &self.input_buf,
                input_pyramid,
                stored_ch,
            )
            .context("input pyramid build dispatch failed")?;
        }

        if let (Some(reference_pyramid), Some(reference_buf)) =
            (self.pyramid_reference.as_ref(), self.reference_buf.as_ref())
        {
            run_pyramid_build(
                &self.client,
                motion_ctx,
                self.width,
                self.height,
                frame_count,
                slot,
                reference_buf,
                reference_pyramid,
                stored_ch,
            )
            .context("reference pyramid build dispatch failed")?;
        }

        Ok(())
    }

    /// Builds the luma pyramid for `slot` that the confidence-only pass reads.
    ///
    /// It always reads `input_buf`, even with a prefilter, to avoid a second reference pyramid.
    pub(super) fn build_confidence_pyramid_for_slot(&self, slot: u32) -> Result<(), anyhow::Error> {
        let (Some(confidence_ctx), Some(pyramid)) =
            (self.confidence_ctx.as_ref(), self.confidence_pyramid.as_ref())
        else {
            return Ok(());
        };

        run_pyramid_build(
            &self.client,
            confidence_ctx,
            self.width,
            self.height,
            self.params.total_frames(),
            slot,
            &self.input_buf,
            pyramid,
            self.params.channels.storage_count(),
        )
        .context("confidence pyramid build dispatch failed")
    }

    /// Whether `Chained` motion estimation is in use, either asked for or resolved from `Auto`.
    ///
    /// This ignores whether motion compensation is active, which also needs a temporal radius
    /// above 0.
    pub(in crate::nlmeans) fn is_chained(&self) -> bool {
        let estimation = self
            .params
            .motion_compensation
            .resolved_estimation(self.params.temporal_radius);
        matches!(estimation, Some(MotionEstimation::Chained { .. }))
    }

    /// Measures motion both ways between the slot just written and the one before it.
    ///
    /// The result goes into the pair ring. A stream's first frame has no older partner and is
    /// skipped, so composition reads the priming duplicate's zeroed pair instead.
    fn run_pair_analyse_for_slot(&self, newer_slot: u32) -> Result<(), anyhow::Error> {
        if self.ring_head == 0 {
            return Ok(());
        }

        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        if !self.is_chained() {
            return Ok(());
        }

        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");

        // Match against the cleaner of the two pyramids.
        let pyramid = self.pyramid_reference.as_ref().unwrap_or_else(|| {
            self.pyramid_input
                .as_ref()
                .expect("pyramid_input allocated when mc_ctx is Some")
        });

        let total_frames = self.params.total_frames();
        let older_slot = (newer_slot + total_frames - 1) % total_frames;
        let pair_slot = self.pair_slot(0);
        let noise = self.derived_handle();

        motion::run_pair_analyse(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            total_frames,
            older_slot,
            newer_slot,
            pair_slot,
            pyramid,
            &self.ring_slot_tables,
            pair_ring,
            &self.confidence_dummy,
            &noise,
        )
        .context("pair analyse dispatch failed")
    }

    /// Zeroes the pair-ring slot of a duplicated frame.
    pub(super) fn zero_pair_slot_for_duplicate(&self) {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return;
        };

        if !self.is_chained() {
            return;
        }

        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");
        let pair_slot = self.pair_slot(0);
        motion::zero_pair_slot(&self.client, motion_ctx, pair_ring, pair_slot);
    }
}
