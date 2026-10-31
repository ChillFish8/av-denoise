use cubecl::bytes::Bytes;

use super::io::ReadFuture;

/// A denoised frame whose readback has not finished.
///
/// The readback starts on the first poll.
pub struct Pending {
    future: ReadFuture,
    /// The byte length of each plane, which trims the word padding off each buffer.
    plane_lengths: Vec<usize>,
}

impl Pending {
    pub(crate) fn new(future: ReadFuture, plane_lengths: Vec<usize>) -> Self {
        Self {
            future,
            plane_lengths,
        }
    }

    /// Blocks until the readback finishes and returns one buffer per plane.
    pub fn wait(mut self) -> Result<Vec<Vec<u8>>, anyhow::Error> {
        let future = self.future.as_mut();
        let result = cubecl::future::block_on(future);

        let buffers = result?;
        let planes = trim_planes(buffers, &self.plane_lengths);

        Ok(planes)
    }
}

fn trim_planes(buffers: Vec<Bytes>, plane_lengths: &[usize]) -> Vec<Vec<u8>> {
    buffers
        .iter()
        .zip(plane_lengths)
        .map(|(buffer, &length)| buffer[..length].to_vec())
        .collect()
}
