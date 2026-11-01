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

    /// The stream frame ring slot `slot` holds, where a duplicate holds the frame it copies.
    pub(crate) fn slot_frame(&self, slot: u32) -> u32 {
        self.slot_frames[slot as usize]
    }

    /// The stream frame at the window centre.
    pub(crate) fn centre_frame(&self) -> u32 {
        let radius = self.params.temporal_radius;
        let slot = self.ring_slot(radius);

        self.slot_frame(slot)
    }

    /// The frames the stream has taken, counting those still in the look-ahead.
    pub(crate) fn stream_frames(&self) -> u32 {
        let taken = self.real_pushes + self.held_frames();
        taken as u32
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

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn frames_loaded_for_test(&self) -> usize {
        self.frames_loaded
    }
}
