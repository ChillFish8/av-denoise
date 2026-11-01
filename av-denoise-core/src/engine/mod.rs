pub(crate) mod align;
mod frame;
mod io;
pub(crate) mod kernels;
mod passthrough;
mod planar;
mod plane;
mod spec;
pub(crate) mod timing;

#[cfg(test)]
mod tests;

use std::ops::Range;

use cubecl::server::Handle;

pub use self::frame::{ChromaLayout, Geometry};
pub(crate) use self::io::{EgressSource, IngestTarget, copy_words_into, egress, ingest};
pub(crate) use self::plane::PassGeometry;
pub use self::plane::{DevicePlane, SampleFormat};
pub use self::spec::EngineSpec;
pub(crate) use self::spec::PassSpec;
pub use self::timing::EngineTiming;
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;

/// Largest 1D grid a dispatch may ask for, set by the WebGPU and Vulkan limits.
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for the 1D copy and zero kernels.
pub(crate) const BLOCK_1D: u32 = 256;

/// A buffer's name and its element count, `None` when the count overflows `u64`.
pub(crate) type BufferSize = (&'static str, Option<u64>);

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
/// Push frames in order with [PassEngine::push]. It returns how many frames are ready, and every ready frame
/// must be written out with [PassEngine::emit_into] before the next push. At the end of a stream call
/// [PassEngine::finish] and emit the frames it reports. The next push then starts a new stream.
///
/// Any GPU error leaves the engine refusing calls with [Error::NeedsReset] until [PassEngine::reset].
pub(crate) trait PassEngine: Send {
    /// Ingests one frame and returns how many frames are ready to emit.
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error>;

    /// Ingests one frame as context only, producing no output from this call.
    ///
    /// A stream that starts with this picks up mid-clip instead of at a scene start. Context frames
    /// still fill the window, so a later push or [PassEngine::finish] may emit them, and `finish` alone
    /// reports at most [PassEngine::max_held_frames] of them. It returns [Error::ContextAfterPush] once the
    /// stream has had a real push.
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

    /// Device time spent so far.
    ///
    /// Device time is only gathered after [PassEngine::sample_device_time] turns it on.
    fn timing(&mut self) -> EngineTiming {
        EngineTiming::default()
    }

    /// Samples the device time of one frame in every `every`, or of none at 0.
    fn sample_device_time(&mut self, _every: u32) {}
}

/// The device time of each pass of an [Engine], `None` for a plane group it does not clean.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PassTimings {
    pub luma: Option<EngineTiming>,
    pub chroma: Option<EngineTiming>,
}

/// A [PassEngine] that names the stream frame behind each frame it emits.
pub(crate) trait FramePass: PassEngine {
    /// The stream frame the next push or context push takes.
    fn next_frame(&self) -> u32;

    /// The stream frame the last [PassEngine::emit_into] wrote out.
    fn emitted_frame(&self) -> Option<u32>;
}

/// Uploads a frame's planes when an [Engine] asks for them.
pub trait PlaneSource {
    /// Uploads the planes at indices `planes`, luma first, and returns their handles in order.
    fn upload(&mut self, planes: Range<usize>) -> Vec<Handle>;
}

/// A stateful denoiser that reads and writes every plane of a frame on the GPU.
///
/// Push frames in order with [Engine::push]. It returns how many frames are ready, and every ready frame
/// must be written out with [Engine::emit_into] before the next push. At the end of a stream call
/// [Engine::finish] and emit the frames it reports. The next push then starts a new stream.
///
/// Every call takes all of a frame's planes, luma first, as [Geometry] lays them out. A plane group the
/// engine does not clean is written out as it came in, word for word.
///
/// Any GPU error leaves the engine refusing calls with [Error::NeedsReset] until [Engine::reset].
pub trait Engine: Send {
    /// Ingests one frame and returns how many frames are ready to emit.
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error>;

    /// [Engine::push] with the frame's planes taken from `source` one plane group at a time.
    ///
    /// The luma pass is recorded before the chroma planes are asked for, so the device can start on
    /// luma while they upload.
    fn push_from(&mut self, source: &mut dyn PlaneSource) -> Result<usize, Error>;

    /// Ingests one frame as context only, producing no output from this call.
    ///
    /// A stream that starts with this picks up mid-clip instead of at a scene start. Context frames
    /// still fill the window, so a later push or [Engine::finish] may emit them, and `finish` alone
    /// reports at most [Engine::max_held_frames] of them. It returns [Error::ContextAfterPush] once the
    /// stream has had a real push.
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

    /// Reads back the luma film grain measured since the last call, in frame order.
    fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error>;

    /// Device time each pass has spent so far.
    ///
    /// Device time is only gathered after [Engine::sample_device_time] turns it on.
    fn timing(&mut self) -> PassTimings;

    /// Samples the device time of one frame in every `every`, or of none at 0.
    fn sample_device_time(&mut self, every: u32);
}
