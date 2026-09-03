use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use super::kernels::gpu_pack_wire;
use super::{BLOCK_1D, Depth, MAX_GRID_1D};
use crate::denoiser::{FrameOutput, OutputFormat};

pub(crate) type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// A denoise that is still in flight.
///
/// The kernels are queued, the GPU may still be working on them, and the
/// readback to the host has not finished.
pub struct Pending<R: Runtime> {
    pub(super) fut: ReadFuture,
    pub(super) channels: u32,
    pub(super) stored_ch: u32,
    pub(super) pixels: usize,
    pub(super) format: OutputFormat,
    pub(super) _marker: PhantomData<R>,
}

impl<R: Runtime> Pending<R> {
    /// Wraps an in-flight readback future into a `Pending`.
    ///
    /// `fut` is the future returned by reading back a single output
    /// buffer. `channels` is how many channels the caller wants out,
    /// `stored_ch` is how many are actually laid out per pixel in that
    /// buffer, and `pixels` is the frame's pixel count. `format` is what
    /// the buffer holds, which is `f32` samples with padding lanes for
    /// [`OutputFormat::F32`] and already-packed wire bytes for
    /// [`OutputFormat::Wire`].
    ///
    /// In `F32` mode `wait` and `wait_into` strip the padding lanes
    /// between `channels` and `stored_ch`. In `Wire` mode the pack kernel
    /// has already done that on the GPU.
    pub(crate) fn new(
        fut: ReadFuture,
        channels: u32,
        stored_ch: u32,
        pixels: usize,
        format: OutputFormat,
    ) -> Self {
        Self {
            fut,
            channels,
            stored_ch,
            pixels,
            format,
            _marker: PhantomData,
        }
    }

    /// Blocks until the readback finishes and returns the denoised frame
    /// in a fresh buffer of this `Pending`'s output format.
    ///
    /// In `F32` mode the YUV padding lanes are stripped, so the buffer
    /// holds exactly `pixels * channels` values.
    pub fn wait(self) -> Result<FrameOutput, anyhow::Error> {
        let mut out = empty_output(self.pixels, self.channels, self.format);
        self.wait_into(&mut out)?;
        Ok(out)
    }

    /// Blocks until the readback finishes and writes the result into `dst`,
    /// which is cleared first.
    ///
    /// `dst` keeps its allocation when it already holds this `Pending`'s
    /// output format, so a caller can reuse one buffer when running frame
    /// after frame.
    pub fn wait_into(self, dst: &mut FrameOutput) -> Result<(), anyhow::Error> {
        let (pixels, channels, stored_ch, format) = (self.pixels, self.channels, self.stored_ch, self.format);
        let bytes = cubecl::future::block_on(self.fut)?.remove(0);
        unpack_into(&bytes, pixels, channels, stored_ch, format, dst);
        Ok(())
    }

    /// Polls the readback once.
    ///
    /// `TryWait::NotReady` hands the same `Pending` back unchanged, so a
    /// caller that gets it can only poll again by calling `try_wait` on
    /// that returned value. There is no way to poll a future that has
    /// already produced its result. The poll uses a no-op waker, so
    /// nothing ever wakes a caller when the readback lands. A caller
    /// that wants the frame has to keep calling `try_wait` again itself,
    /// on whatever `NotReady` the previous call returned.
    ///
    /// This only avoids blocking on the wgpu backends, meaning Vulkan and Metal,
    /// where readiness is external state a discarded wakeup does not lose.
    ///
    /// On CUDA and ROCm the readback future's first poll runs a blocking driver wait
    /// internally, so `try_wait` blocks for the full kernel and readback latency there
    /// and `NotReady` is never actually returned.
    pub fn try_wait(mut self) -> Result<TryWait<R>, anyhow::Error> {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match self.fut.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(mut bytes)) => {
                let bytes = bytes.remove(0);
                let mut out = empty_output(self.pixels, self.channels, self.format);
                unpack_into(
                    &bytes,
                    self.pixels,
                    self.channels,
                    self.stored_ch,
                    self.format,
                    &mut out,
                );
                Ok(TryWait::Ready(out))
            },
            Poll::Ready(Err(e)) => Err(e.into()),
            Poll::Pending => Ok(TryWait::NotReady(self)),
        }
    }
}

/// An empty buffer of `format`, sized for one frame.
fn empty_output(pixels: usize, channels: u32, format: OutputFormat) -> FrameOutput {
    let samples = pixels * channels as usize;
    match format {
        OutputFormat::F32 => FrameOutput::F32(Vec::with_capacity(samples)),
        OutputFormat::Wire { depth } => {
            FrameOutput::Wire(Vec::with_capacity(samples * depth.bytes_per_sample()))
        },
    }
}

/// Turns a raw readback buffer into `dst`.
///
/// This is the shared step behind `wait_into` and `try_wait`, so the
/// blocking and non-blocking paths cannot drift apart. A `dst` that
/// already holds the right variant keeps its allocation, and one that
/// does not is replaced.
fn unpack_into(
    bytes: &Bytes,
    pixels: usize,
    channels: u32,
    stored_ch: u32,
    format: OutputFormat,
    dst: &mut FrameOutput,
) {
    let samples = pixels * channels as usize;
    match (format, dst) {
        (OutputFormat::F32, FrameOutput::F32(out)) => {
            unpack_bytes_into(bytes, pixels, channels, stored_ch, out);
        },
        (OutputFormat::Wire { depth }, FrameOutput::Wire(out)) => {
            unpack_wire_into(bytes, samples * depth.bytes_per_sample(), out);
        },
        (OutputFormat::F32, dst) => {
            let mut out = Vec::new();
            unpack_bytes_into(bytes, pixels, channels, stored_ch, &mut out);
            *dst = FrameOutput::F32(out);
        },
        (OutputFormat::Wire { depth }, dst) => {
            let mut out = Vec::new();
            unpack_wire_into(bytes, samples * depth.bytes_per_sample(), &mut out);
            *dst = FrameOutput::Wire(out);
        },
    }
}

/// The outcome of polling a [`Pending`] without blocking.
pub enum TryWait<R: Runtime> {
    /// The readback landed. This is the denoised frame.
    Ready(FrameOutput),
    /// The readback has not landed. This is the same `Pending`, unchanged
    /// and still in flight.
    NotReady(Pending<R>),
}

/// Starts the readback of a denoised frame sitting in `handle`, in
/// whichever format the caller asked for.
///
/// In `Wire` mode this first queues [`gpu_pack_wire`] against a fresh
/// word buffer and reads that back instead, so the quantised bytes cross
/// the bus rather than four times as many `f32`s.
///
/// No sync sits between the pack launch and the read. `read_async`
/// submits its copy descriptors on the same stream as the launches ahead
/// of it, so it is already the sync point and the pack is just one more
/// queued operation.
///
/// Both algorithms build their `Pending` through here, so their readback
/// paths cannot drift apart.
pub(crate) fn start_readback<R: Runtime>(
    client: &ComputeClient<R>,
    handle: Handle,
    channels: u32,
    stored_ch: u32,
    pixels: usize,
    format: OutputFormat,
) -> Pending<R> {
    let handle = match format {
        OutputFormat::F32 => handle,
        OutputFormat::Wire { depth } => pack_wire(client, &handle, channels, stored_ch, pixels, depth),
    };

    // The future is wrapped in an `async move` that owns a cloned
    // `ComputeClient`, which is cheap because it shares its internals.
    // That owned client lives inside the future, so the future is
    // genuinely `'static` and the `Pending` can outlive the denoiser
    // without any lifetime tricks.
    let client = client.clone();
    let fut = Box::pin(async move { client.read_async(vec![handle]).await });

    Pending::new(fut, channels, stored_ch, pixels, format)
}

/// Queues [`gpu_pack_wire`] over `src` and returns the word buffer it
/// writes.
fn pack_wire<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    channels: u32,
    stored_ch: u32,
    pixels: usize,
    depth: Depth,
) -> Handle {
    let pack = depth.wire_pack();
    let samples = pixels as u32 * channels;
    let words = samples.div_ceil(pack.samples_per_word());

    let split_planes = wire_splits_planes(channels);
    let outer = if split_planes { pixels as u32 } else { channels };

    let grid = words.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    let dst = client.empty(words as usize * size_of::<u32>());

    unsafe {
        gpu_pack_wire::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(src.clone(), pixels * stored_ch as usize),
            ArrayArg::from_raw_parts(dst.clone(), words as usize),
            pack.max(),
            pixels as u32,
            channels,
            stored_ch,
            outer,
            split_planes,
            pack.samples_per_word(),
            words,
            total_threads,
        );
    }

    dst
}

/// Whether a frame with this many channels goes out as one contiguous
/// region per channel rather than interleaved.
///
/// Only the chroma pair splits. A luma frame has nothing to split, and a
/// packed YUV frame stays interleaved because that is the layout its
/// consumer already reads.
pub(crate) fn wire_splits_planes(channels: u32) -> bool {
    channels == 2
}

/// Quantises a denoised `f32` frame into the same wire bytes
/// [`gpu_pack_wire`] would produce for it.
///
/// This is the host fallback for the trailing tail frames, which the
/// algorithms read back themselves rather than through a [`Pending`].
pub(crate) fn f32_frame_to_wire(frame: &[f32], channels: u32, depth: Depth) -> Vec<u8> {
    if wire_splits_planes(channels) {
        let (u, v) = crate::frame::unpack_uv_from_f32(frame, frame.len() / 2, depth);
        return u.into_iter().chain(v).collect();
    }
    crate::frame::f32_to_plane(frame, depth)
}

/// Copies the first `len` bytes of a readback buffer into `dst`, which is
/// cleared first.
///
/// The pack kernel writes whole `u32` words, so the buffer can run up to
/// three bytes past the frame. Everything the caller wants is already
/// quantised and free of padding lanes.
fn unpack_wire_into(bytes: &Bytes, len: usize, dst: &mut Vec<u8>) {
    dst.clear();
    dst.extend_from_slice(&bytes[..len]);
}

/// Unpacks a raw readback buffer straight into `dst`, stripping the padding lanes
/// between `channels` and `stored_ch`.
fn unpack_bytes_into(bytes: &Bytes, pixels: usize, channels: u32, stored_ch: u32, dst: &mut Vec<f32>) {
    let data = f32::from_bytes(bytes);
    unpack_frame(data, pixels, channels as usize, stored_ch as usize, dst);
}

/// Copies `channels` values out of every pixel in `data` into `dst`,
/// skipping any padding lanes `stored_ch` added beyond that.
///
/// `data` holds `pixels * stored_ch` values. When `channels` and
/// `stored_ch` are equal this is a plain copy. `dst` is cleared first.
pub(super) fn unpack_frame(
    data: &[f32],
    pixels: usize,
    channels: usize,
    stored_ch: usize,
    dst: &mut Vec<f32>,
) {
    dst.clear();
    if channels == stored_ch {
        dst.extend_from_slice(data);
    } else {
        dst.reserve(pixels * channels);
        for pixel in 0..pixels {
            let src = pixel * stored_ch;
            dst.extend_from_slice(&data[src..src + channels]);
        }
    }
}
