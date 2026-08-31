use cubecl::prelude::*;

/// Packs a patch's top-left position into one `u32`, x in the low 13
/// bits and y in the next 13.
///
/// Both axes are 13 bits because a wider x would leave y too narrow to
/// cover a 4K-tall frame, and a position that overflows its field
/// corrupts the one beside it silently.
///
/// Packing both coordinates into one word rather than carrying x and y
/// as a pair is what makes the top-K arrays and the duplicate check
/// single comparisons instead of two.
#[cube]
pub(crate) fn pack_pos(x: u32, y: u32) -> u32 {
    (y << 13) | x
}

/// The host-side mirror of [`pack_pos`], for building expected values in
/// tests without a GPU round trip.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_host(x: u32, y: u32) -> u32 {
    (y << 13) | x
}

/// The host-side mirror of unpacking a position [`pack_pos`] produced.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_pos_host(packed: u32) -> (u32, u32) {
    (packed & 0x1FFF, (packed >> 13) & 0x1FFF)
}

/// Packs a candidate position and the neighbour it came from into one
/// word.
///
/// The axes are 13 bits each rather than 16 and 11 because a 16-bit `x`
/// would leave `y` too narrow. `y` at 11 bits stops at 2047, which is
/// below a 2160-line frame, and a position that overflows its field
/// corrupts the one beside it silently. `x` takes bits 0-12 and `y` bits
/// 13-25, which leaves bits 26-31 for `t`, room for 63 neighbours
/// against the 16 a temporal radius of 8 asks for.
///
/// `t` is 0 for a centre-frame position and `neighbour_index + 1`
/// otherwise, so a member's frame and its motion-block confidence are
/// both recoverable from the packed word alone. Keeping them here
/// rather than in a second array retires one register per slot in the
/// matching loop and one array in the filter stage.
///
/// The coordinates sit exactly where [`pack_pos`] puts them, so
/// [`unpack_pos_host`] reads a packed-with-`t` word correctly.
#[cube]
pub(crate) fn pack_pos_t(x: u32, y: u32, t: u32) -> u32 {
    (t << 26u32) | (y << 13u32) | x
}

/// The neighbour field [`pack_pos_t`] wrote.
#[cube]
pub(crate) fn unpack_t(packed: u32) -> u32 {
    packed >> 26u32
}

/// The host-side mirror of [`pack_pos_t`].
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_t_host(x: u32, y: u32, t: u32) -> u32 {
    (t << 26) | (y << 13) | x
}

/// The host-side mirror of [`unpack_t`].
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_t_host(packed: u32) -> u32 {
    packed >> 26
}

/// Clamps a candidate top-left coordinate to `[0, max_pos]`.
///
/// Every candidate patch position on every axis goes through this before
/// anything reads from it, so a patch read at that position always starts
/// and ends inside the frame. No kernel that later consumes a group has
/// to clamp its own reads.
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
