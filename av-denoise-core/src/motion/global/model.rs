use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::tune::gpu::zeroed_words;

/// The `f32` words of one pair model, `a, b, tx, ty, valid`.
///
/// The model maps a luma position in the older frame to the newer one with
/// `x' = a*x - b*y + tx` and `y' = b*x + a*y + ty`. A zero `valid` word means no motion.
pub const MODEL_WORDS: u32 = 5;

/// A device ring of camera models, one per adjacent frame pair.
#[derive(Debug, Clone)]
pub struct ModelRing {
    pub(crate) handle: Handle,
    pub(crate) depth: u32,
}

impl ModelRing {
    /// A ring of `depth` slots, every model starting invalid.
    pub fn new(client: &Client, depth: u32) -> Self {
        let words = (depth * MODEL_WORDS) as usize;
        let handle = zeroed_words(client, words);

        Self { handle, depth }
    }

    /// The slot holding the pair from stream frame `frame - 1` to `frame`.
    pub fn slot(&self, frame: u32) -> u32 {
        frame % self.depth
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// The `f32` words the ring holds.
    pub fn words(&self) -> usize {
        (self.depth * MODEL_WORDS) as usize
    }
}
