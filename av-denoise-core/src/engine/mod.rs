mod io;
pub(crate) mod kernels;
mod plane;

#[cfg(test)]
mod tests;

use crate::error::Error;
use crate::nl4d::grain::GrainChunk;

#[cfg(test)]
pub(crate) use self::io::{EgressSource, IngestTarget, egress, ingest};
pub use self::plane::{DevicePlane, Geometry, SampleFormat};

/// How many frames before and after a target frame are needed to denoise it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSpan {
    pub behind: usize,
    pub ahead: usize,
    pub edges: EdgePadding,
}

/// How a window is filled where it runs past a clip's ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgePadding {
    /// The boundary frame is repeated, so every window has the same length.
    Repeat,
    /// The window stops at the clip's ends and the engine runs off-centre passes there.
    Shifted,
}

impl WindowSpan {
    /// The full window length, target frame included.
    pub fn frame_count(&self) -> usize {
        self.behind + 1 + self.ahead
    }
}

/// A stateful denoiser that reads and writes GPU planes.
///
/// Push frames in order with [Engine::push]. It returns how many frames are ready, and every ready frame
/// must be written out with [Engine::emit_into] before the next push. At the end of a stream call
/// [Engine::finish] and emit the frames it reports. The next push then starts a new stream.
///
/// Any GPU error leaves the engine refusing calls with [Error::NeedsReset] until [Engine::reset].
pub trait Engine: Send {
    /// Ingests one frame and returns how many frames are ready to emit.
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error>;

    /// Ingests one frame as context only, producing no output.
    ///
    /// A stream that starts with this picks up mid-clip instead of at a scene start.
    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error>;

    /// Writes the oldest ready frame into `planes`.
    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error>;

    /// Ends the stream and returns how many tail frames are ready to emit.
    fn finish(&mut self) -> Result<usize, Error>;

    /// Abandons the current stream, keeping every allocation.
    fn reset(&mut self);

    fn window_span(&self) -> WindowSpan;

    /// The most frames the engine holds before it emits one.
    fn max_held_frames(&self) -> usize;

    /// Reads back the film grain measured since the last call, in frame order.
    fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error> {
        Ok(Vec::new())
    }
}
