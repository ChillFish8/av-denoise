use super::NlmDenoiser;

impl NlmDenoiser {
    /// Whether every ring slot holds a frame, so a pass can run at any centre.
    pub(crate) fn window_ready(&self) -> bool {
        self.frames_loaded >= self.params.total_frames() as usize
    }

    /// The physical slot holding logical ring position `logical`, where 0 is the oldest frame.
    pub(crate) fn ring_slot(&self, logical: u32) -> u32 {
        self.phys_frame(logical as i32)
    }

    /// Turns the shifted-edge stream layout on or off, and with it borrowing a sample ahead.
    pub(crate) fn set_shifted_edges(&mut self, on: bool) {
        self.shifted_edges = on;
        self.set_borrow_ahead(on);
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

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn frames_loaded_for_test(&self) -> usize {
        self.frames_loaded
    }
}
