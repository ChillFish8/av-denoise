mod depth;
pub(crate) mod io;
mod options;
mod pending;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use av_denoise_core::{DevicePlane, Engine, GrainChunk, WindowSpan};
use cubecl::server::Handle;

pub use self::depth::{Depth, UnsupportedDepthError};
use self::io::PlaneIo;
pub use self::options::{Algorithm, DenoiserOptions};
pub use self::pending::{Pending, TryWait};
use crate::backend::accelerate::Accelerator;
use crate::backend::{Device, build_engine};

/// How many readbacks a [HostDenoiser] keeps in flight at once.
pub const MAX_PENDING: usize = 2;

/// How many output plane sets a [HostDenoiser] rotates through.
const OUTPUT_SETS: usize = 2;

/// Each frame in flight reads its own output set, so a set is never rewritten before its readback lands.
const _: () = assert!(OUTPUT_SETS >= MAX_PENDING);

/// Errors reported by a [HostDenoiser].
#[derive(Debug, thiserror::Error)]
pub enum DenoiserError {
    /// [MAX_PENDING] frames are waiting to be collected.
    ///
    /// Call [HostDenoiser::recv] or [HostDenoiser::try_recv], then retry the same push.
    #[error("denoiser queue is full, collect the pending frame before pushing more")]
    QueueFull,
    /// An earlier call failed, so later output would not line up with its input.
    ///
    /// Call [HostDenoiser::reset_stream] to start a fresh stream, or drop the denoiser.
    #[error("denoiser failed earlier, reset the stream before using it again")]
    Poisoned,
    /// None of the accelerators in the priority list could be started.
    #[error("no accelerator from the priority list is available")]
    NoAcceleratorAvailable,
    /// The engine rejected its options, its planes or a call.
    #[error(transparent)]
    Engine(#[from] av_denoise_core::Error),
    /// Anything else, such as a failed readback.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// A stateful denoiser that cleans a stream of frames held as wire bytes.
///
/// Push frames in order with [Self::push] and collect the cleaned ones with [Self::recv] or
/// [Self::try_recv]. At the end of the stream call [Self::flush] to drain the temporal tail.
///
/// Each frame is one byte buffer per plane. Luma takes Y, chroma takes U and V, and YUV takes all
/// three at one size. Every output frame comes back in the same layout.
///
/// ```no_run
/// use av_denoise::accelerate::Accelerator;
/// use av_denoise::{ChannelMode, DenoiserOptions, DenoisingMode, Device, HostDenoiser};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let options = DenoiserOptions::builder()
///     .channel_mode(ChannelMode::Luma)
///     .mode(DenoisingMode::Temporal { radius: 2 })
///     .build();
///
/// let mut denoiser = HostDenoiser::create(&[Accelerator::Vulkan], &Device::Default, 1920, 1080, options)?;
///
/// let frame = vec![128u8; 1920 * 1080];
/// let mut cleaned = Vec::new();
///
/// for _ in 0..10 {
///     denoiser.push(&[&frame])?;
///
///     if let Some(planes) = denoiser.recv()? {
///         cleaned.push(planes);
///     }
/// }
///
/// denoiser.flush(|planes| cleaned.push(planes))?;
/// # Ok(())
/// # }
/// ```
pub struct HostDenoiser {
    engine: Box<dyn Engine>,
    io: Box<dyn PlaneIo>,
    accelerator: Accelerator,
    width: u32,
    height: u32,
    temporal_radius: u32,
    plane_count: usize,
    /// The wire byte length of one plane.
    plane_length: usize,
    /// The bytes one device plane takes up, rounded up to whole words.
    device_plane_bytes: usize,
    /// Output plane sets, allocated on first use.
    outputs: Vec<Vec<Handle>>,
    next_output: usize,
    pending: VecDeque<Pending>,
    /// Set once any call other than a `QueueFull` push has failed.
    poisoned: bool,
}

impl HostDenoiser {
    /// Builds a denoiser on the first accelerator in `accelerators` that works.
    ///
    /// `device` picks a non-default device on the chosen runtime.
    ///
    /// # Thread stack size
    ///
    /// cubecl runs kernel codegen on its own worker thread, which gets Rust's default stack. A
    /// `search_radius` above 4 can overflow it, so callers using one should call
    /// [raise_codegen_stack_limit](crate::raise_codegen_stack_limit) at the top of `main`.
    pub fn create(
        accelerators: &[Accelerator],
        device: &Device,
        width: u32,
        height: u32,
        options: DenoiserOptions,
    ) -> Result<Self, DenoiserError> {
        let temporal_radius = options.temporal_radius();
        let is_nl4d = matches!(options.algorithm, Algorithm::Nl4d(_));

        if is_nl4d && temporal_radius == 0 {
            let error = anyhow::anyhow!(
                "nl4d needs a temporal window, set DenoiserOptions::mode to DenoisingMode::Temporal"
            );
            return Err(DenoiserError::Other(error));
        }

        let spec = options.algorithm.engine_spec(&options, width, height);
        let built = build_engine(accelerators, device, spec)?;

        let format = options.depth.sample_format();
        let pixels = width as usize * height as usize;
        let device_plane_bytes = format.plane_bytes(pixels as u64) as usize;
        let plane_length = pixels * options.depth.bytes_per_sample();
        let plane_count = options.channel_mode.count() as usize;

        Ok(Self {
            engine: built.engine,
            io: built.io,
            accelerator: built.accelerator,
            width,
            height,
            temporal_radius,
            plane_count,
            plane_length,
            device_plane_bytes,
            outputs: Vec::with_capacity(OUTPUT_SETS),
            next_output: 0,
            pending: VecDeque::with_capacity(MAX_PENDING),
            poisoned: false,
        })
    }

    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn temporal_radius(&self) -> u32 {
        self.temporal_radius
    }

    /// How many frames behind and ahead of a target frame are needed to denoise it.
    pub fn window_span(&self) -> WindowSpan {
        self.engine.window_span()
    }

    /// Uploads one frame and starts denoising any frame it completes.
    ///
    /// Returns [DenoiserError::QueueFull] once [MAX_PENDING] frames wait to be collected, which does
    /// not poison the denoiser. Any other failure poisons it until [Self::reset_stream].
    pub fn push(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        if self.pending.len() >= MAX_PENDING {
            return Err(DenoiserError::QueueFull);
        }

        let result = self.push_inner(planes);
        self.poison_on_error(result)
    }

    fn push_inner(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        let handles = self.upload(planes)?;
        let device_planes = device_planes(&handles, self.width, self.height);
        let ready = self.engine.push(&device_planes)?;
        debug_assert!(
            ready <= 1,
            "a streaming push readies at most one frame, got {ready}"
        );

        for _ in 0..ready {
            let slot = self.next_output;
            self.next_output = (slot + 1) % OUTPUT_SETS;

            let pending = self.emit_into_slot(slot)?;
            self.pending.push_back(pending);
        }

        Ok(())
    }

    /// Uploads one frame as context only, without producing output.
    ///
    /// A stream that starts with this picks up mid-clip, so nl4d runs no head passes for it.
    pub fn push_priming(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let result = self.push_priming_inner(planes);
        self.poison_on_error(result)
    }

    fn push_priming_inner(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        let handles = self.upload(planes)?;
        let device_planes = device_planes(&handles, self.width, self.height);
        self.engine.push_context(&device_planes)?;

        Ok(())
    }

    /// Blocks until the oldest frame in flight lands and returns it, one buffer per plane.
    ///
    /// Returns `Ok(None)` when nothing is in flight.
    pub fn recv(&mut self) -> Result<Option<Vec<Vec<u8>>>, DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let result = self.recv_inner();
        self.poison_on_error(result)
    }

    fn recv_inner(&mut self) -> Result<Option<Vec<Vec<u8>>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };

        let planes = pending.wait()?;
        Ok(Some(planes))
    }

    /// Polls the oldest frame in flight once.
    ///
    /// Returns `Ok(None)` both when nothing is in flight and when the readback has not landed. Only
    /// the wgpu backends avoid blocking here.
    ///
    /// Once polled, a frame's readback has to finish. Dropping the denoiser or calling
    /// [Self::reset_stream] before it lands blocks until it does.
    pub fn try_recv(&mut self) -> Result<Option<Vec<Vec<u8>>>, DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let result = self.try_recv_inner();
        self.poison_on_error(result)
    }

    fn try_recv_inner(&mut self) -> Result<Option<Vec<Vec<u8>>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };

        match pending.try_wait()? {
            TryWait::Ready(planes) => Ok(Some(planes)),
            TryWait::NotReady(pending) => {
                self.pending.push_front(pending);
                Ok(None)
            },
        }
    }

    /// Drains the frames in flight and the temporal tail, handing each frame to `sink`.
    ///
    /// The denoiser is ready for a fresh stream afterwards.
    pub fn flush(&mut self, sink: impl FnMut(Vec<Vec<u8>>)) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let result = self.flush_inner(sink);
        self.poison_on_error(result)
    }

    fn flush_inner(&mut self, mut sink: impl FnMut(Vec<Vec<u8>>)) -> Result<(), DenoiserError> {
        while let Some(planes) = self.recv_inner()? {
            sink(planes);
        }

        let tail = self.engine.finish()?;

        for _ in 0..tail {
            let pending = self.emit_into_slot(0)?;
            let planes = pending.wait()?;
            sink(planes);
        }

        self.next_output = 0;

        Ok(())
    }

    /// Abandons the current stream, keeping every GPU allocation, and clears any poison.
    ///
    /// Frames in flight are dropped unread, except one [Self::try_recv] already polled, which is
    /// settled first.
    pub fn reset_stream(&mut self) {
        self.pending.clear();
        self.engine.reset();
        self.next_output = 0;
        self.poisoned = false;
    }

    /// Reads back the grain chunks measured since the last call, in frame order.
    ///
    /// Empty unless grain export is on.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let drained = self.engine.drain_grain_chunks();
        let drained = drained.map_err(DenoiserError::from);

        self.poison_on_error(drained)
    }

    #[cfg(test)]
    pub(crate) fn poison_for_test(&mut self) {
        self.poisoned = true;
    }

    fn poison_on_error<T>(&mut self, result: Result<T, DenoiserError>) -> Result<T, DenoiserError> {
        if result.is_err() {
            self.poisoned = true;
        }

        result
    }

    /// Uploads each plane, padded to whole words.
    fn upload(&self, planes: &[&[u8]]) -> Result<Vec<Handle>, DenoiserError> {
        if planes.len() != self.plane_count {
            let message = format!("expected {} planes, got {}", self.plane_count, planes.len());
            let error = av_denoise_core::Error::PlaneMismatch(message);
            return Err(DenoiserError::Engine(error));
        }

        for (index, plane) in planes.iter().enumerate() {
            if plane.len() != self.plane_length {
                let error = anyhow::anyhow!(
                    "plane {index} holds {} bytes, expected {}",
                    plane.len(),
                    self.plane_length
                );
                return Err(DenoiserError::Other(error));
            }
        }

        let mut handles = Vec::with_capacity(planes.len());

        for plane in planes {
            let handle = if plane.len() == self.device_plane_bytes {
                self.io.upload(plane)
            } else {
                let mut padded = plane.to_vec();
                padded.resize(self.device_plane_bytes, 0);
                self.io.upload(&padded)
            };
            handles.push(handle);
        }

        Ok(handles)
    }

    /// Writes the oldest ready frame into output set `slot` and starts reading it back.
    fn emit_into_slot(&mut self, slot: usize) -> Result<Pending, DenoiserError> {
        while self.outputs.len() <= slot {
            let set = (0..self.plane_count)
                .map(|_| self.io.allocate(self.device_plane_bytes))
                .collect();
            self.outputs.push(set);
        }

        let handles = self.outputs[slot].clone();
        let device_planes = device_planes(&handles, self.width, self.height);
        self.engine.emit_into(&device_planes)?;

        let future = self.io.read(handles);
        let plane_lengths = vec![self.plane_length; self.plane_count];

        Ok(Pending::new(future, plane_lengths))
    }
}

fn device_planes(handles: &[Handle], width: u32, height: u32) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect()
}
