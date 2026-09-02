use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::ServerError;

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
    pub(super) _marker: PhantomData<R>,
}

impl<R: Runtime> Pending<R> {
    /// Wraps an in-flight readback future into a `Pending`.
    ///
    /// `fut` is the future returned by reading back a single output
    /// buffer. `channels` is how many channels the caller wants out,
    /// `stored_ch` is how many are actually laid out per pixel in that
    /// buffer, and `pixels` is the frame's pixel count. `wait` and
    /// `wait_into` strip the padding lanes between `channels` and
    /// `stored_ch`.
    ///
    /// This exists so code outside `nlmeans` that reads back its own GPU
    /// buffer, such as the collaborative filter that runs after this
    /// denoiser, can hand callers the same `Pending` type this module
    /// produces.
    pub(crate) fn new(fut: ReadFuture, channels: u32, stored_ch: u32, pixels: usize) -> Self {
        Self {
            fut,
            channels,
            stored_ch,
            pixels,
            _marker: PhantomData,
        }
    }

    /// Blocks until the readback finishes and returns the denoised frame
    /// in a fresh `Vec`.
    ///
    /// YUV padding lanes are stripped, so the buffer holds exactly
    /// `pixels * channels` values.
    pub fn wait(self) -> Result<Vec<f32>, anyhow::Error> {
        let mut out = Vec::with_capacity(self.pixels * self.channels as usize);
        self.wait_into(&mut out)?;
        Ok(out)
    }

    /// Blocks until the readback finishes and writes the result into `dst`, 
    /// which is cleared first.
    ///
    /// This lets a caller reuse one allocation when running frame after frame.
    pub fn wait_into(self, dst: &mut Vec<f32>) -> Result<(), anyhow::Error> {
        let bytes = cubecl::future::block_on(self.fut)?.remove(0);
        unpack_bytes_into(&bytes, self.pixels, self.channels, self.stored_ch, dst);
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
                let mut out = Vec::with_capacity(self.pixels * self.channels as usize);
                unpack_bytes_into(&bytes, self.pixels, self.channels, self.stored_ch, &mut out);
                Ok(TryWait::Ready(out))
            },
            Poll::Ready(Err(e)) => Err(e.into()),
            Poll::Pending => Ok(TryWait::NotReady(self)),
        }
    }
}

/// The outcome of polling a [`Pending`] without blocking.
pub enum TryWait<R: Runtime> {
    /// The readback landed. This is the denoised frame.
    Ready(Vec<f32>),
    /// The readback has not landed. This is the same `Pending`, unchanged
    /// and still in flight.
    NotReady(Pending<R>),
}

/// Unpacks a raw readback buffer straight into `dst`, stripping the padding lanes 
/// between `channels` and `stored_ch`.
///
/// This is the shared step behind `wait_into` and `try_wait`, so the blocking and 
/// non-blocking paths cannot drift apart.
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
