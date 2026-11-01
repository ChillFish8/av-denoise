use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{DevicePlane, copy_words_into};

/// The source planes of recent frames, kept on the device for a plane group no pass cleans.
///
/// Stream frame `n` lives in slot `n % slots`.
pub(crate) struct PassthroughRing {
    client: Client,
    /// One handle per plane in each slot.
    slots: Vec<Vec<Handle>>,
    words: usize,
}

/// One slot's handles, `planes` planes of `bytes` bytes each.
fn slot_planes(client: &Client, planes: usize, bytes: u64) -> Vec<Handle> {
    (0..planes).map(|_| client.empty(bytes as usize)).collect()
}

impl PassthroughRing {
    /// A ring of `slots` frames, each `planes` planes of `bytes` bytes.
    pub(crate) fn new(client: &Client, slots: usize, planes: usize, bytes: u64) -> Self {
        let slots = (0..slots).map(|_| slot_planes(client, planes, bytes)).collect();

        Self {
            client: client.clone(),
            slots,
            words: (bytes / 4) as usize,
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots.len()
    }

    fn slot(&self, frame: u32) -> &[Handle] {
        let index = frame as usize % self.slots.len();
        &self.slots[index]
    }

    /// Queues a copy of `planes` into the slot of stream frame `frame`.
    pub(crate) fn store(&self, frame: u32, planes: &[DevicePlane<'_>]) {
        let slot = self.slot(frame);

        for (plane, stored) in planes.iter().zip(slot) {
            copy_words_into(&self.client, plane.handle(), stored, self.words);
        }
    }

    /// Queues a copy of stream frame `frame` into `planes`.
    pub(crate) fn load(&self, frame: u32, planes: &[DevicePlane<'_>]) {
        let slot = self.slot(frame);

        for (stored, plane) in slot.iter().zip(planes) {
            copy_words_into(&self.client, stored, plane.handle(), self.words);
        }
    }
}
