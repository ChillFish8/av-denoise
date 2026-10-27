use std::future::Future;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::prelude::Client;
use cubecl::server::{Handle, ServerError};

/// A readback in flight, resolving to one buffer per handle read.
pub(crate) type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// Uploads and reads back planes on whichever runtime an engine was built on.
pub(crate) trait PlaneIo: Send {
    fn upload(&self, bytes: &[u8]) -> Handle;

    fn allocate(&self, bytes: usize) -> Handle;

    /// Starts reading `handles` back to the host.
    ///
    /// Nothing is read until the returned future is first polled.
    fn read(&self, handles: Vec<Handle>) -> ReadFuture;
}

pub(crate) struct ClientIo {
    client: Client,
}

impl ClientIo {
    pub(crate) fn new(client: Client) -> Self {
        Self { client }
    }
}

impl PlaneIo for ClientIo {
    fn upload(&self, bytes: &[u8]) -> Handle {
        self.client.create_from_slice(bytes)
    }

    fn allocate(&self, bytes: usize) -> Handle {
        self.client.empty(bytes)
    }

    fn read(&self, handles: Vec<Handle>) -> ReadFuture {
        // The future owns its own client, so it stays valid after the denoiser that started it is dropped.
        let client = self.client.clone();
        let future = async move { client.read_async(handles).await };

        Box::pin(future)
    }
}
