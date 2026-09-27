use cubecl::prelude::*;

use super::NlmDenoiser;

impl<R: Runtime> NlmDenoiser<R> {
    /// Whether every ring slot holds a frame, so a pass can run at any centre.
    pub(crate) fn window_ready(&self) -> bool {
        self.frames_loaded >= self.params.total_frames() as usize
    }

    /// The physical slot holding logical ring position `logical`, where 0 is the oldest frame.
    pub(crate) fn ring_slot(&self, logical: u32) -> u32 {
        self.phys_frame(logical as i32)
    }
}
