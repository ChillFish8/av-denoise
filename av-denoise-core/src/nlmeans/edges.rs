use std::sync::Arc;

use cubecl::prelude::*;

use super::NlmDenoiser;
use super::denoiser::readings::SlotNoiseReading;

impl<R: Runtime> NlmDenoiser<R> {
    /// Whether every ring slot holds a frame, so a pass can run at any centre.
    pub(crate) fn window_ready(&self) -> bool {
        self.frames_loaded >= self.params.total_frames() as usize
    }

    /// The physical slot holding logical ring position `logical`, where 0 is the oldest frame.
    pub(crate) fn ring_slot(&self, logical: u32) -> u32 {
        self.phys_frame(logical as i32)
    }

    /// Turns the shifted-edge stream layout on or off.
    pub(crate) fn set_shifted_edges(&mut self, on: bool) {
        self.shifted_edges = on;
    }

    /// Copies the last pushed frame forward until every ring slot holds a frame.
    ///
    /// A stream shorter than the ring uses this at flush so its real frames can run as centres.
    pub(crate) fn fill_ring_with_last_frame(&mut self) -> Result<(), anyhow::Error> {
        while !self.window_ready() {
            self.duplicate_last_frame()?;
            self.frames_loaded += 1;
        }

        Ok(())
    }

    /// The reading whose temporal side a pass centred on `center_t` folds.
    ///
    /// It is the centre's own reading, unless shifted edges are on and the centre has no usable
    /// sample. Then it is the first reading with a usable sample at a later ring position, or the
    /// centre's own when there is none.
    pub(super) fn borrow_reading_ahead(
        &mut self,
        center_t: u32,
        centre: &Arc<SlotNoiseReading>,
    ) -> Result<Arc<SlotNoiseReading>, anyhow::Error> {
        let has_sample = centre.temporal.sample.is_some();
        if has_sample || !self.shifted_edges {
            return Ok(Arc::clone(centre));
        }

        let total_frames = self.params.total_frames();
        for logical in center_t + 1..total_frames {
            let candidate = self.noise_reading_at(logical)?;
            if candidate.temporal.sample.is_some() {
                return Ok(candidate);
            }
        }

        Ok(Arc::clone(centre))
    }

    #[cfg(test)]
    pub(crate) fn frames_loaded_for_test(&self) -> usize {
        self.frames_loaded
    }
}
