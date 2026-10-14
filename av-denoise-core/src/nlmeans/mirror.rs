use cubecl::prelude::*;
use cubecl::server::Handle;

/// A small device buffer the host rewrites only when its contents change.
///
/// The host keeps the bytes it last wrote. An update with the same bytes keeps the current buffer
/// and uploads nothing. Changed bytes upload into a fresh buffer of the same size, and kernels
/// already queued keep reading the old one.
pub(crate) struct MirroredBuffer {
    handle: Handle,
    contents: Vec<u8>,
    uploads: u64,
}

impl MirroredBuffer {
    pub(crate) fn new<R: Runtime>(client: &ComputeClient<R>, bytes: &[u8]) -> Self {
        let handle = client.create_from_slice(bytes);
        let contents = bytes.to_vec();

        Self {
            handle,
            contents,
            uploads: 1,
        }
    }

    /// Makes the buffer hold `bytes`, uploading only when they differ from what it holds.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is not the size the buffer was built with.
    pub(crate) fn update<R: Runtime>(&mut self, client: &ComputeClient<R>, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            self.contents.len(),
            "a mirrored buffer keeps its size"
        );

        if bytes == self.contents.as_slice() {
            return;
        }

        self.handle = client.create_from_slice(bytes);
        self.contents.copy_from_slice(bytes);
        self.uploads += 1;
    }

    pub(crate) fn handle(&self) -> &Handle {
        &self.handle
    }

    /// How many times the buffer was uploaded, the first upload included.
    #[cfg(test)]
    pub(crate) fn uploads(&self) -> u64 {
        self.uploads
    }
}
