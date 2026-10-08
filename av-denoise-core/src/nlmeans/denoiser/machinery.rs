use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use crate::nlmeans::dispatch::mc_sad_noise_floor_sigma;
use crate::nlmeans::motion::{self, MotionCtx};
use crate::nlmeans::params::ChannelMode;

/// Handles and geometry a collaborative stage needs to read the ring.
///
/// The handles view the denoiser's own buffers and stay valid until the next push or machinery
/// step reuses their slots.
pub(crate) struct RingView {
    /// The whole input ring, indexable by physical frame slot.
    pub input: Handle,
    /// The f16 search ring, when the denoiser has one.
    pub search_input: Option<Handle>,
    /// Chained motion fields, one per neighbour.
    pub mv_field: Handle,
    /// Per-block confidence, one plane per neighbour.
    pub confidence: Handle,
    /// Physical ring slot of the centre frame.
    pub centre_slot: u32,
    /// Physical ring slot of each neighbour in logical order, skipping the centre.
    pub neighbour_slots: Vec<u32>,
    /// `i32` element stride between neighbours in `mv_field`.
    pub mv_stride: u32,
    /// `f32` element stride between neighbours in `confidence`.
    pub conf_stride: u32,
    /// The luma pyramid the motion estimator analysed.
    ///
    /// It is built from the reference ring when a prefilter is active. Level 0 of slot `s` starts
    /// at `pyramid_slot_byte_offset(width, height, frame_count, 0, s, align)`.
    pub pyramid: Handle,
    pub frame_count: u32,
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Runs the per-submit noise and motion estimates without launching any NLM kernel.
    ///
    /// The view is centred on logical ring position `center_t` and reads the unshifted input
    /// ring, so a caller searches around the predicted motion itself. Returns `Ok(None)` while
    /// the window is still filling.
    ///
    /// # Errors
    ///
    /// Returns an error unless motion compensation and temporal confidence are both active.
    pub(crate) fn submit_machinery(&mut self, center_t: u32) -> Result<Option<RingView>, anyhow::Error> {
        debug_assert!(
            center_t < self.params.total_frames(),
            "center_t must be a logical ring position"
        );

        let total_frames = self.params.total_frames() as usize;
        if self.frames_loaded < total_frames {
            return Ok(None);
        }

        if self.noise_results.is_some() {
            self.update_noise_estimate(center_t)?;
        }

        let neighbour_slots = self.run_motion_machinery(center_t)?;

        let motion_ctx = self
            .mc_ctx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("submit_machinery requires motion compensation to be active"))?;
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some")
            .clone();
        let confidence = self
            .confidence_buf
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("submit_machinery requires HQ temporal confidence to be active"))?
            .clone();
        let pyramid = self
            .pyramid_reference
            .as_ref()
            .or(self.pyramid_input.as_ref())
            .expect("pyramid allocated when mc_ctx is Some")
            .clone();

        let centre_slot = self.phys_frame(center_t as i32);
        let mv_stride = (motion_ctx.mv_field_bytes_per_neighbour() / size_of::<i32>() as u64) as u32;
        let conf_stride = (motion_ctx.confidence_bytes_per_neighbour() / size_of::<f32>() as u64) as u32;

        let view = RingView {
            input: self.input_buf.clone(),
            search_input: self.search_buf.clone(),
            mv_field,
            confidence,
            centre_slot,
            neighbour_slots,
            mv_stride,
            conf_stride,
            pyramid,
            frame_count: self.params.total_frames(),
        };
        Ok(Some(view))
    }

    /// The motion-compensation geometry.
    ///
    /// # Panics
    ///
    /// Panics unless motion compensation is active.
    pub(crate) fn motion_ctx(&self) -> &MotionCtx {
        self.mc_ctx
            .as_ref()
            .expect("motion_ctx called without motion compensation active")
    }

    /// The SAD two noisy copies of one block show by chance, which confidence is scored against.
    ///
    /// # Panics
    ///
    /// Panics unless motion compensation is active.
    pub(crate) fn sad_noise_floor_value(&self) -> f32 {
        let blksize = self.motion_ctx().blksize;
        let sigma = mc_sad_noise_floor_sigma(self.params.prefilter, self.sigma_y);
        motion::sad_noise_floor(blksize, sigma)
    }

    /// The SAD threshold that confidence is scored against.
    ///
    /// # Panics
    ///
    /// Panics unless motion compensation is active.
    pub(crate) fn thsad_value(&self) -> f32 {
        let blksize = self.motion_ctx().blksize;
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        motion::thsad(blksize, thsad_scale)
    }

    pub(crate) fn compute_client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub(crate) fn input_ring(&self) -> &Handle {
        &self.input_buf
    }

    #[cfg(test)]
    pub(crate) fn search_ring(&self) -> Option<&Handle> {
        self.search_buf.as_ref()
    }

    pub(crate) fn frame_shape(&self) -> (u32, u32, ChannelMode) {
        (self.width, self.height, self.params.channels)
    }
}
