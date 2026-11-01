use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::chroma_map::{FieldLayout, Subsampling};

/// The luma pass's motion fields, mapped onto the chroma block grid for the chroma pass.
///
/// The luma pass maps its final window fields into the slot of the centre's stream frame, and the
/// chroma pass of the same frame reads them back. One slot per frame of the window is enough,
/// because luma never runs more passes ahead of chroma than that. The noise look-ahead's pair
/// fields share one more slot, since the luma pass writes a pushed frame's pair in the same push the
/// chroma pass reads it.
/// Each slot is tagged with its frame, so a read never returns another frame's fields.
///
/// The fields hold vectors only. Both passes must run the same motion search at the same temporal
/// radius and hold the same frames, so either both have a look-ahead or neither does.
/// [EngineSpec::validate](crate::engine::EngineSpec::validate) enforces both.
#[derive(Clone)]
pub(crate) struct LumaMotionRing {
    fields: Arc<[Handle]>,
    /// Both directions of the newest look-ahead pair.
    lookahead: Handle,
    tags: Arc<Mutex<Tags>>,
    window: FieldLayout,
    pair: FieldLayout,
    subsampling: Subsampling,
}

/// The stream frame each slot holds, `None` for an empty slot.
struct Tags {
    window: Vec<Option<u32>>,
    lookahead: Option<u32>,
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
        let lookahead = client.empty(chroma.pair_slot_bytes() as usize);
        let tags = Tags {
            window: vec![None; slots as usize],
            lookahead: None,
        };

        Self {
            fields,
            lookahead,
            tags: Arc::new(Mutex::new(tags)),
            window: FieldLayout::window(chroma),
            pair: FieldLayout::pair(chroma),
            subsampling,
        }
    }

    /// The window slot of centre stream frame `frame`, tagged for it, for the luma pass to map into.
    pub(crate) fn window_slot(&self, frame: u32) -> Handle {
        let index = self.slot_index(frame);
        let mut tags = self.lock_tags();
        tags.window[index] = Some(frame);

        self.fields[index].clone()
    }

    /// The window fields of centre stream frame `frame`, or `None` if its slot holds another frame.
    pub(crate) fn read_window(&self, frame: u32) -> Option<Handle> {
        let index = self.slot_index(frame);
        let tags = self.lock_tags();
        let holds_frame = tags.window[index] == Some(frame);

        holds_frame.then(|| self.fields[index].clone())
    }

    /// The look-ahead slot, tagged for stream frame `frame`, for the luma pass to map its pair into.
    pub(crate) fn lookahead_slot(&self, frame: u32) -> Handle {
        let mut tags = self.lock_tags();
        tags.lookahead = Some(frame);

        self.lookahead.clone()
    }

    /// The look-ahead pair of stream frame `frame`, or `None` if the slot holds another frame.
    pub(crate) fn read_lookahead(&self, frame: u32) -> Option<Handle> {
        let tags = self.lock_tags();
        let holds_frame = tags.lookahead == Some(frame);

        holds_frame.then(|| self.lookahead.clone())
    }

    /// Empties every slot.
    pub(crate) fn clear(&self) {
        let mut tags = self.lock_tags();
        tags.window.fill(None);
        tags.lookahead = None;
    }

    /// The chroma pass's window field layout the window slots hold.
    pub(crate) fn window_layout(&self) -> FieldLayout {
        self.window
    }

    /// The chroma pass's pair field layout the look-ahead slot holds.
    pub(crate) fn pair_layout(&self) -> FieldLayout {
        self.pair
    }

    pub(crate) fn subsampling(&self) -> Subsampling {
        self.subsampling
    }

    fn slot_index(&self, frame: u32) -> usize {
        frame as usize % self.fields.len()
    }

    fn lock_tags(&self) -> MutexGuard<'_, Tags> {
        self.tags.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The part a pass plays in a [LumaMotionRing].
#[derive(Debug, Clone)]
pub(crate) enum RingRole {
    /// The luma pass, which maps its window and look-ahead fields into the ring.
    Writer(LumaMotionRing),
    /// The chroma pass, which refines the ring's fields instead of searching its own.
    Reader(LumaMotionRing),
}
