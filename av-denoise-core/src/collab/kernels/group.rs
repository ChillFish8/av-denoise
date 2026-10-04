use cubecl::prelude::*;

/// Packs a patch's top-left position into one `u32`, x in the low 13 bits and y in the next 13.
///
/// Both axes are 13 bits so y covers a 4K-tall frame, because a position that overflows its field
/// silently corrupts the other. One word makes the top-K arrays and the duplicate check single
/// comparisons.
#[cube]
pub(crate) fn pack_pos(x: u32, y: u32) -> u32 {
    (y << 13) | x
}

/// Host mirror of `pack_pos`, for building expected values in tests.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_host(x: u32, y: u32) -> u32 {
    (y << 13) | x
}

/// Host mirror of unpacking a `pack_pos` word.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_pos_host(packed: u32) -> (u32, u32) {
    (packed & 0x1FFF, (packed >> 13) & 0x1FFF)
}

/// Packs a candidate position and the neighbour it came from into one word.
///
/// x takes bits 0-12, y bits 13-25 and `t` bits 26-31, which holds up to 63 neighbours. Thirteen
/// bits per axis keeps y above a 2160-line frame, because a position that overflows its field
/// silently corrupts the one beside it. `t` is 0 for a centre-frame position and
/// `neighbour_index + 1` otherwise, so a member's frame and motion-block confidence are recoverable
/// from the word alone. The coordinates sit where `pack_pos` puts them, so `unpack_pos_host` reads
/// this word too.
#[cube]
pub(crate) fn pack_pos_t(x: u32, y: u32, t: u32) -> u32 {
    (t << 26u32) | (y << 13u32) | x
}

/// The neighbour field `pack_pos_t` wrote.
#[cube]
pub(crate) fn unpack_t(packed: u32) -> u32 {
    packed >> 26u32
}

/// Host mirror of `pack_pos_t`.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_t_host(x: u32, y: u32, t: u32) -> u32 {
    (t << 26) | (y << 13) | x
}

/// Host mirror of `unpack_t`.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_t_host(packed: u32) -> u32 {
    packed >> 26
}

/// Clamps a candidate top-left coordinate to `0..=max_pos`.
///
/// Every candidate position goes through this before anything reads it, so a patch read always
/// stays inside the frame and no later kernel has to clamp its own reads.
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
