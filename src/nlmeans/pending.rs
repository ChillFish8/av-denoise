//! In-flight readback handle. Returned by
//! [`super::NlmDenoiser::denoise_submit`]; consume with
//! [`Pending::wait`] or [`Pending::wait_into`] to retrieve the
//! denoised frame.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::ServerError;

pub(super) type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// In-flight denoise: kernels are queued, the GPU may still be working,
/// and the host-side readback hasn't completed yet.
pub struct Pending<R: Runtime> {
    pub(super) fut: ReadFuture,
    pub(super) channels: u32,
    pub(super) stored_ch: u32,
    pub(super) pixels: usize,
    pub(super) _marker: PhantomData<R>,
}

impl<R: Runtime> Pending<R> {
    /// Block until the readback completes, returning the denoised frame
    /// as a freshly allocated `Vec`. YUV padding lanes are stripped so
    /// the returned buffer is densely packed (`pixels * channels` f32s).
    pub fn wait(self) -> Result<Vec<f32>, anyhow::Error> {
        let mut out = Vec::with_capacity(self.pixels * self.channels as usize);
        self.wait_into(&mut out)?;
        Ok(out)
    }

    /// Block until the readback completes, writing the result into
    /// `dst` (cleared first). Lets the caller reuse an allocation when
    /// running synchronously frame-after-frame.
    pub fn wait_into(self, dst: &mut Vec<f32>) -> Result<(), anyhow::Error> {
        let bytes = cubecl::future::block_on(self.fut)?.remove(0);
        let data = f32::from_bytes(&bytes);
        let channels = self.channels as usize;
        let stored_ch = self.stored_ch as usize;

        dst.clear();
        if channels == stored_ch {
            dst.extend_from_slice(data);
        } else {
            dst.reserve(self.pixels * channels);
            for pixel in 0..self.pixels {
                let src = pixel * stored_ch;
                dst.extend_from_slice(&data[src..src + channels]);
            }
        }
        Ok(())
    }
}
