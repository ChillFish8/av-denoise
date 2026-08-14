use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::ServerError;

pub(super) type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

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

    /// Blocks until the readback finishes and writes the result into
    /// `dst`, which is cleared first.
    ///
    /// This lets a caller reuse one allocation when running frame after
    /// frame.
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
