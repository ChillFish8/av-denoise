use cubecl::prelude::*;

use super::NlmDenoiser;
use super::noise::TemporalNoiseReading;

impl<R: Runtime> NlmDenoiser<R> {
    /// Whether every ring slot holds a frame, so a pass can run at any centre.
    pub(crate) fn window_ready(&self) -> bool {
        self.frames_loaded >= self.params.total_frames() as usize
    }

    /// The physical slot holding logical ring position `logical`, where 0 is the oldest frame.
    pub(crate) fn ring_slot(&self, logical: u32) -> u32 {
        self.phys_frame(logical as i32)
    }

    /// Turns the shifted-edge stream layout on or off. See the `shifted_edges` field.
    pub(crate) fn set_shifted_edges(&mut self, on: bool) {
        self.shifted_edges = on;
    }

    /// Copies the last pushed frame forward until every ring slot holds a frame.
    ///
    /// A stream shorter than the ring uses this at flush so its real frames can run as centres.
    pub(crate) fn fill_ring_with_last_frame(&mut self) {
        while !self.window_ready() {
            self.duplicate_last_frame();
            self.frames_loaded += 1;
        }
    }

    /// The temporal reading a pass centred on `center_t` folds.
    ///
    /// This is the centre's own reading. With shifted edges on and no
    /// usable sample at the centre, it is the first usable reading at a
    /// later ring position, or the centre's empty one when there is none.
    pub(super) fn borrow_reading_ahead(&self, center_t: u32) -> Result<TemporalNoiseReading, anyhow::Error> {
        let centre_slot = self.ring_slot(center_t);
        let own = self.read_temporal_noise(centre_slot)?;
        if own.sample.is_some() || !self.shifted_edges {
            return Ok(own);
        }

        let total_frames = self.params.total_frames();
        for logical in center_t + 1..total_frames {
            let slot = self.ring_slot(logical);
            let candidate = self.read_temporal_noise(slot)?;
            if candidate.sample.is_some() {
                return Ok(candidate);
            }
        }

        Ok(own)
    }

    #[cfg(test)]
    pub(crate) fn frames_loaded_for_test(&self) -> usize {
        self.frames_loaded
    }
}
