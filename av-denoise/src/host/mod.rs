mod depth;
pub(crate) mod io;
mod options;
mod pending;
mod timing;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use av_denoise_core::{ChromaLayout, DevicePlane, Engine, EngineSpec, GrainChunk, WindowSpan};
use cubecl::prelude::Client;
use cubecl::server::Handle;

pub use self::depth::{Depth, UnsupportedDepthError};
use self::io::PlaneIo;
pub use self::options::{Algorithm, ChannelIntent, DenoiserOptions};
pub(crate) use self::pending::Pending;
pub use self::timing::{PlanarTiming, PlaneTiming};
use crate::backend::accelerate::Accelerator;
use crate::backend::{Device, DevicePlacement, build_engine};

/// How many readbacks a [HostDenoiser] keeps in flight at once.
pub const MAX_PENDING: usize = 2;

/// How many output frames a [HostDenoiser] rotates through, each holding every plane.
const OUTPUT_SETS: usize = 2;

/// Each frame in flight reads its own output set, so a set is never rewritten before its readback lands.
const _: () = assert!(OUTPUT_SETS >= MAX_PENDING);

/// Errors reported by a [HostDenoiser].
#[derive(Debug, thiserror::Error)]
pub enum DenoiserError {
    /// [MAX_PENDING] frames are waiting to be collected.
    ///
    /// Call [HostDenoiser::recv], then retry the same push.
    #[error("denoiser queue is full, collect the pending frame before pushing more")]
    QueueFull,
    /// An earlier failure means later output would not line up with its input.
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

impl DenoiserError {
    /// Whether this error leaves the stream unusable until [HostDenoiser::reset_stream].
    ///
    /// Rejected planes are caught before any state changes and never reach this check. A full queue
    /// and a context frame sent after a push change no state either. Any other failure may leave the
    /// host partly advanced, so it poisons.
    fn poisons_stream(&self) -> bool {
        !matches!(
            self,
            DenoiserError::QueueFull
                | DenoiserError::Poisoned
                | DenoiserError::Engine(av_denoise_core::Error::ContextAfterPush)
        )
    }
}

/// A stateful denoiser that cleans a stream of frames held as wire bytes.
///
/// Push frames in order with [Self::push] and collect the cleaned ones with [Self::recv]. At the end
/// of the stream call [Self::flush] to drain the temporal tail.
///
/// Each frame is one byte buffer per plane, Y then U and V, each at its own size. A gray frame has
/// only Y. Every output frame comes back in the same layout.
///
/// ```no_run
/// use av_denoise::accelerate::Accelerator;
/// use av_denoise::{ChannelIntent, ChromaLayout, DenoiserOptions, DenoisingMode, Device, HostDenoiser};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let options = DenoiserOptions::builder()
///     .intent(ChannelIntent::Luma)
///     .chroma(ChromaLayout::Gray)
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
    /// The device the engine was asked for, which names its stored tuning results.
    selected: cubecl::Device,
    client: Client,
    spec: EngineSpec,
    /// Each plane's width and height, luma first.
    plane_dims: Vec<(u32, u32)>,
    /// Each plane's wire byte length.
    plane_lengths: Vec<usize>,
    /// The bytes each device plane takes up, rounded up to whole words.
    device_plane_bytes: Vec<usize>,
    /// Output plane sets, one per frame in flight, allocated on first use.
    outputs: Vec<Vec<Handle>>,
    next_output: usize,
    pending: VecDeque<Pending>,
    /// Set once a call fails with an error that [DenoiserError::poisons_stream].
    poisoned: bool,
    /// Host time spent inside this denoiser's public calls.
    call_time: Duration,
    /// Host time spent blocked until a denoised frame's readback landed.
    output_wait: Duration,
    /// Host time spent in the caller's flush sink, which [Self::timing] takes off the call clock.
    sink_time: Duration,
    /// Frames handed back to the caller.
    frames_emitted: u64,
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
        let placement = DevicePlacement::shared();
        Self::create_with(accelerators, device, &placement, width, height, options)
    }

    /// [Self::create] on the client `placement` gives.
    pub(crate) fn create_with(
        accelerators: &[Accelerator],
        device: &Device,
        placement: &DevicePlacement,
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

        let spec = options.engine_spec(width, height);
        let built = build_engine(
            accelerators,
            device,
            placement,
            spec,
            options.tuning,
            options.deterministic,
        )?;

        let format = options.depth.sample_format();
        let plane_dims = frame_plane_dims(width, height, options.chroma);
        let plane_pixels: Vec<usize> = plane_dims
            .iter()
            .map(|&(plane_width, plane_height)| plane_width as usize * plane_height as usize)
            .collect();
        let device_plane_bytes = plane_pixels
            .iter()
            .map(|&pixels| format.plane_bytes(pixels as u64) as usize)
            .collect();
        let bytes_per_sample = options.depth.bytes_per_sample();
        let plane_lengths = plane_pixels
            .iter()
            .map(|&pixels| pixels * bytes_per_sample)
            .collect();

        Ok(Self {
            engine: built.engine,
            io: built.io,
            accelerator: built.accelerator,
            selected: built.selected,
            client: built.client,
            spec,
            plane_dims,
            plane_lengths,
            device_plane_bytes,
            outputs: Vec::with_capacity(OUTPUT_SETS),
            next_output: 0,
            pending: VecDeque::with_capacity(MAX_PENDING),
            poisoned: false,
            call_time: Duration::ZERO,
            output_wait: Duration::ZERO,
            sink_time: Duration::ZERO,
            frames_emitted: 0,
        })
    }

    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    pub(crate) fn selected(&self) -> &cubecl::Device {
        &self.selected
    }

    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    pub(crate) fn spec(&self) -> EngineSpec {
        self.spec
    }

    /// How many frames behind and ahead of a target frame are needed to denoise it.
    pub fn window_span(&self) -> WindowSpan {
        self.engine.window_span()
    }

    /// Uploads one frame, every plane at its own size, and starts denoising any frame it completes.
    ///
    /// Returns [DenoiserError::QueueFull] once [MAX_PENDING] frames wait to be collected, which does
    /// not poison the denoiser. A rejected plane leaves it usable too. Any other failure poisons it
    /// until [Self::reset_stream].
    pub fn push(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        if self.pending.len() >= MAX_PENDING {
            return Err(DenoiserError::QueueFull);
        }

        self.check_planes(planes)?;

        let result = self.clocked(|denoiser| denoiser.push_inner(planes));
        self.poison_on_error(result)
    }

    fn push_inner(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        let handles = self.upload_frame(planes);
        let device_planes = device_planes(&handles, &self.plane_dims);
        let ready = self.engine.push(&device_planes)?;

        // Emitting past the free output sets would overwrite a set whose readback has not landed.
        let in_flight = self.pending.len();
        if in_flight + ready > OUTPUT_SETS {
            let error = anyhow::anyhow!(
                "push readied {ready} frames with {in_flight} in flight, past the {OUTPUT_SETS} output sets"
            );
            return Err(DenoiserError::Other(error));
        }

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

        self.check_planes(planes)?;

        let result = self.clocked(|denoiser| denoiser.push_priming_inner(planes));
        self.poison_on_error(result)
    }

    fn push_priming_inner(&mut self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        let handles = self.upload_frame(planes);
        let device_planes = device_planes(&handles, &self.plane_dims);
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

        let result = self.clocked(Self::recv_inner);
        self.poison_on_error(result)
    }

    fn recv_inner(&mut self) -> Result<Option<Vec<Vec<u8>>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };

        let planes = self.wait_for_output(pending)?;
        Ok(Some(planes))
    }

    /// Drains the frames in flight and the temporal tail, handing each frame to `sink`.
    ///
    /// The denoiser is ready for a fresh stream afterwards.
    pub fn flush(&mut self, sink: impl FnMut(Vec<Vec<u8>>)) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }

        let result = self.clocked(|denoiser| denoiser.flush_inner(sink));
        self.poison_on_error(result)
    }

    fn flush_inner(&mut self, mut sink: impl FnMut(Vec<Vec<u8>>)) -> Result<(), DenoiserError> {
        while let Some(planes) = self.recv_inner()? {
            self.hand_to_sink(&mut sink, planes);
        }

        let tail = self.engine.finish()?;

        for _ in 0..tail {
            let pending = self.emit_into_slot(0)?;
            let planes = self.wait_for_output(pending)?;
            self.hand_to_sink(&mut sink, planes);
        }

        self.next_output = 0;

        Ok(())
    }

    /// Abandons the current stream, keeping every GPU allocation, and clears any poison.
    ///
    /// Frames in flight are dropped unread.
    pub fn reset_stream(&mut self) {
        self.pending.clear();
        self.engine.reset();
        self.next_output = 0;
        self.poisoned = false;
    }

    /// Samples the device time of one frame in every `every`, or of none at 0.
    pub fn sample_device_time(&mut self, every: u32) {
        self.engine.sample_device_time(every);
    }

    /// Where each denoised plane's time went so far.
    ///
    /// It waits for any device-time spans still in flight.
    pub fn timing(&mut self) -> PlanarTiming {
        let passes = self.engine.timing();
        let calls = self.call_time.saturating_sub(self.sink_time);

        PlanarTiming::new(calls, self.output_wait, self.frames_emitted, passes)
    }

    /// Reads back the luma grain chunks measured since the last call, in frame order.
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

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn poison_for_test(&mut self) {
        self.poisoned = true;
    }

    /// Runs `call`, adding its host time to the call clock.
    fn clocked<T>(&mut self, call: impl FnOnce(&mut Self) -> T) -> T {
        let started = Instant::now();
        let result = call(self);
        self.call_time += started.elapsed();

        result
    }

    /// Blocks until `pending` lands, adding the wait to the output clock and counting the frame.
    fn wait_for_output(&mut self, pending: Pending) -> Result<Vec<Vec<u8>>, DenoiserError> {
        let planes = self.output_clocked(|| pending.wait())?;
        self.frames_emitted += 1;

        Ok(planes)
    }

    /// Hands one frame to `sink`, adding the time it takes to the sink clock.
    fn hand_to_sink(&mut self, sink: &mut impl FnMut(Vec<Vec<u8>>), planes: Vec<Vec<u8>>) {
        let started = Instant::now();
        sink(planes);
        self.sink_time += started.elapsed();
    }

    /// Runs `wait`, adding its host time to the output-wait clock.
    fn output_clocked<T>(&mut self, wait: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = wait();
        self.output_wait += started.elapsed();

        result
    }

    fn poison_on_error<T>(&mut self, result: Result<T, DenoiserError>) -> Result<T, DenoiserError> {
        if let Err(error) = &result
            && error.poisons_stream()
        {
            self.poisoned = true;
        }

        result
    }

    /// Rejects a frame with the wrong number of planes or a plane of the wrong length.
    fn check_planes(&self, planes: &[&[u8]]) -> Result<(), DenoiserError> {
        let plane_count = self.plane_lengths.len();
        if planes.len() != plane_count {
            let message = format!("expected {plane_count} planes, got {}", planes.len());
            let error = av_denoise_core::Error::PlaneMismatch(message);
            return Err(DenoiserError::Engine(error));
        }

        for (index, (plane, &expected)) in planes.iter().zip(&self.plane_lengths).enumerate() {
            if plane.len() != expected {
                let message = format!("plane {index} holds {} bytes, expected {expected}", plane.len());
                let error = av_denoise_core::Error::PlaneMismatch(message);
                return Err(DenoiserError::Engine(error));
            }
        }

        Ok(())
    }

    /// Uploads each plane, padded to whole words.
    pub(crate) fn upload_frame(&self, planes: &[&[u8]]) -> Vec<Handle> {
        let mut handles = Vec::with_capacity(planes.len());

        for (plane, &device_bytes) in planes.iter().zip(&self.device_plane_bytes) {
            let handle = if plane.len() == device_bytes {
                self.io.upload(plane)
            } else {
                let mut padded = plane.to_vec();
                padded.resize(device_bytes, 0);
                self.io.upload(&padded)
            };
            handles.push(handle);
        }

        handles
    }

    /// Writes the oldest ready frame into output set `slot` and starts reading it back.
    fn emit_into_slot(&mut self, slot: usize) -> Result<Pending, DenoiserError> {
        while self.outputs.len() <= slot {
            let set = self
                .device_plane_bytes
                .iter()
                .map(|&bytes| self.io.allocate(bytes))
                .collect();
            self.outputs.push(set);
        }

        let handles = self.outputs[slot].clone();
        let device_planes = device_planes(&handles, &self.plane_dims);
        self.engine.emit_into(&device_planes)?;

        let future = self.io.read(handles);
        let plane_lengths = self.plane_lengths.clone();
        let pending = Pending::new(future, plane_lengths);

        Ok(pending)
    }
}

/// Each plane's width and height in a `width` by `height` frame, luma first.
fn frame_plane_dims(width: u32, height: u32, chroma: ChromaLayout) -> Vec<(u32, u32)> {
    let mut dims = vec![(width, height)];

    if let Some(chroma_dims) = chroma.chroma_dims(width, height) {
        dims.push(chroma_dims);
        dims.push(chroma_dims);
    }

    dims
}

fn device_planes<'a>(handles: &'a [Handle], dims: &[(u32, u32)]) -> Vec<DevicePlane<'a>> {
    handles
        .iter()
        .zip(dims)
        .map(|(handle, &(width, height))| DevicePlane::new(handle, width, height))
        .collect()
}
