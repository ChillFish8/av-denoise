//! Position helpers the patch search shares with the kernels
//! downstream of it.
//!
//! A group is a list of patch top-left positions, and both the search
//! that builds one and the filter that reads one back need the same
//! packing and the same clamping. Those live here so the two agree by
//! construction.

use cubecl::prelude::*;

/// Packs a patch's top-left position into one `u32`, x in the low half
/// and y in the high half.
///
/// A patch position never needs more than 16 bits per axis for any frame
/// this filter runs on, so the pair packs into a single integer. That
/// makes the top-K arrays and the dedup check simple comparisons instead
/// of pairs of comparisons.
#[cube]
pub fn pack_pos(x: u32, y: u32) -> u32 {
    (y << 16) | x
}

/// The host-side mirror of [`pack_pos`], for building expected values in
/// tests without a GPU round trip.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_host(x: u32, y: u32) -> u32 {
    (y << 16) | x
}

/// The host-side mirror of unpacking a position [`pack_pos`] produced.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_pos_host(packed: u32) -> (u32, u32) {
    (packed & 0xFFFF, packed >> 16)
}

/// Clamps a candidate top-left coordinate to `[0, max_pos]`.
///
/// Every candidate patch position on every axis goes through this before
/// anything reads from it. That guarantees a patch read at that position
/// always starts and ends inside the frame, so no kernel that later
/// consumes a group needs to clamp its own reads.
///
/// `pub(crate)` because [`crate::collab::kernels::group_temporal`] reuses
/// it for the same purpose against both the spatial window and each
/// neighbour frame's refine window.
#[cube]
pub(crate) fn clamp_top_left(v: i32, max_pos: u32) -> u32 {
    let mut result = v;
    if result < 0 {
        result = 0;
    } else if result > max_pos as i32 {
        result = max_pos as i32;
    }
    result as u32
}
