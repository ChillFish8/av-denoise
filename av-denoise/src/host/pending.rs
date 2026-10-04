use std::task::{Context, Poll, Waker};

use cubecl::bytes::Bytes;

use super::io::ReadFuture;

/// A denoised frame whose readback has not finished.
///
/// The readback starts on the first poll. Dropping a `Pending` that was polled but has not landed
/// blocks until it lands, see the `Drop` impl.
pub struct Pending {
    future: ReadFuture,
    /// Set while `future` has been polled but has not produced its result.
    polled: bool,
    /// The byte length of each plane, which trims the word padding off each buffer.
    plane_lengths: Vec<usize>,
}

/// The outcome of polling a [Pending] without blocking.
pub enum TryWait {
    /// The readback landed. This is the frame, one buffer per plane.
    Ready(Vec<Vec<u8>>),
    /// The readback has not landed. Dropping it blocks until it does.
    NotReady(Pending),
}

impl Pending {
    pub(crate) fn new(future: ReadFuture, plane_lengths: Vec<usize>) -> Self {
        Self {
            future,
            polled: false,
            plane_lengths,
        }
    }

    /// Blocks until the readback finishes and returns one buffer per plane.
    pub fn wait(mut self) -> Result<Vec<Vec<u8>>, anyhow::Error> {
        let result = cubecl::future::block_on(self.future.as_mut());
        self.polled = false;

        let buffers = result?;
        let planes = trim_planes(buffers, &self.plane_lengths);

        Ok(planes)
    }

    /// Polls the readback once.
    ///
    /// Only the wgpu backends avoid blocking here. On CUDA and ROCm the first poll waits for the
    /// whole readback.
    pub fn try_wait(mut self) -> Result<TryWait, anyhow::Error> {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        self.polled = true;
        let poll = self.future.as_mut().poll(&mut context);
        if poll.is_ready() {
            self.polled = false;
        }

        match poll {
            Poll::Ready(Ok(buffers)) => {
                let planes = trim_planes(buffers, &self.plane_lengths);
                Ok(TryWait::Ready(planes))
            },
            Poll::Ready(Err(error)) => Err(error.into()),
            Poll::Pending => Ok(TryWait::NotReady(self)),
        }
    }
}

impl Drop for Pending {
    /// Settles a readback that was polled but never landed.
    ///
    /// On the wgpu backends the first poll maps a staging buffer, and only dropping the finished
    /// readback's bytes unmaps it. A still-mapped buffer back in the device's pool makes the next
    /// submit on that device fail. Blocking here finishes the readback and drops its bytes.
    fn drop(&mut self) {
        if !self.polled || std::thread::panicking() {
            return;
        }

        let _ = cubecl::future::block_on(self.future.as_mut());
    }
}

fn trim_planes(buffers: Vec<Bytes>, plane_lengths: &[usize]) -> Vec<Vec<u8>> {
    buffers
        .iter()
        .zip(plane_lengths)
        .map(|(buffer, &length)| buffer[..length].to_vec())
        .collect()
}
