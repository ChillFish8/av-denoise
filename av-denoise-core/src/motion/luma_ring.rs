use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::chroma_map::{FieldLayout, Subsampling};

/// The luma pass's window motion fields, mapped onto the chroma block grid for the chroma pass.
///
/// The luma pass maps its final window fields into the slot of the centre's stream frame, and the
/// chroma pass of the same frame reads them back. One slot per frame of the window is enough,
/// because luma never runs more passes ahead of chroma than that. Each slot is tagged with its
/// frame, so a read never returns another frame's fields.
///
/// The fields hold vectors only. Both passes must run the same motion search at the same temporal
/// radius, which [EngineSpec::validate](crate::engine::EngineSpec::validate) enforces.
#[derive(Clone)]
pub(crate) struct LumaMotionRing {
    fields: Arc<[Handle]>,
    /// The stream frame each slot holds, `None` for an empty slot.
    tags: Arc<Mutex<Vec<Option<u32>>>>,
    chroma: FieldLayout,
    subsampling: Subsampling,
}

impl fmt::Debug for LumaMotionRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LumaMotionRing")
            .field("slots", &self.fields.len())
            .field("subsampling", &self.subsampling)
            .finish()
    }
}

impl LumaMotionRing {
    /// A ring for chroma pass motion of `chroma` geometry over a window `temporal_radius` either side.
    pub(crate) fn new(
        client: &Client,
        chroma: &MotionCtx,
        temporal_radius: u32,
        subsampling: Subsampling,
    ) -> Self {
        let slots = 2 * temporal_radius + 1;
        let neighbours = u64::from(2 * temporal_radius);
        let slot_bytes = neighbours * chroma.mv_field_bytes_per_neighbour();
        let fields: Arc<[Handle]> = (0..slots).map(|_| client.empty(slot_bytes as usize)).collect();
        let tags = vec![None; slots as usize];

        Self {
            fields,
            tags: Arc::new(Mutex::new(tags)),
            chroma: FieldLayout::window(chroma),
            subsampling,
        }
    }

    /// The window slot of centre stream frame `frame`, tagged for it, for the luma pass to map into.
    pub(crate) fn window_slot(&self, frame: u32) -> Handle {
        let index = self.slot_index(frame);
        let mut tags = self.tags.lock().unwrap_or_else(PoisonError::into_inner);
        tags[index] = Some(frame);

        self.fields[index].clone()
    }

    /// The window fields of centre stream frame `frame`, or `None` if its slot holds another frame.
    pub(crate) fn read_window(&self, frame: u32) -> Option<Handle> {
        let index = self.slot_index(frame);
        let tags = self.tags.lock().unwrap_or_else(PoisonError::into_inner);
        let holds_frame = tags[index] == Some(frame);

        holds_frame.then(|| self.fields[index].clone())
    }

    /// Empties every slot.
    pub(crate) fn clear(&self) {
        let mut tags = self.tags.lock().unwrap_or_else(PoisonError::into_inner);
        tags.fill(None);
    }

    /// The chroma pass's window field layout the slots hold.
    pub(crate) fn chroma_layout(&self) -> FieldLayout {
        self.chroma
    }

    pub(crate) fn subsampling(&self) -> Subsampling {
        self.subsampling
    }

    fn slot_index(&self, frame: u32) -> usize {
        frame as usize % self.fields.len()
    }
}

/// The part a pass plays in a [LumaMotionRing].
#[derive(Debug, Clone)]
pub(crate) enum RingRole {
    /// The luma pass, which maps its window fields into the ring.
    Writer(LumaMotionRing),
    /// The chroma pass, which refines the ring's fields instead of searching its own.
    Reader(LumaMotionRing),
}
