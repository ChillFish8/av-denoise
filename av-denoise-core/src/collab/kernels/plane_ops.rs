use cubecl::prelude::*;

/// The plane-local index of the first lane in the calling lane's
/// 8-lane group.
///
/// Every shuffle in this module addresses lanes relative to this, so a
/// group never reads a lane belonging to another group. Cubes are
/// launched 1-D, which makes the thread-to-plane mapping linear, so
/// `UNIT_POS_PLANE % 8` and `UNIT_POS_X % 8` agree at both wave32 and
/// wave64.
#[cube]
pub(crate) fn group_base() -> u32 {
    UNIT_POS_PLANE - UNIT_POS_PLANE % 8u32
}

/// The sum of `partial` across the calling lane's 8-lane group,
/// returned to every lane in it.
///
/// Three XOR shuffles fold 8 values into 8 copies of their sum. The
/// masks are 1, 2 and 4, all below 8, so a lane index XORed with any of
/// them stays inside its own group whatever the plane's width.
///
/// Each lane holds the squared differences for one patch column, so
/// this is what completes a candidate's distance.
#[cube]
pub(crate) fn plane_ssd_reduce8(partial: f32) -> f32 {
    let mut acc = partial;
    acc += plane_shuffle_xor(acc, 1u32);
    acc += plane_shuffle_xor(acc, 2u32);
    acc += plane_shuffle_xor(acc, 4u32);
    acc
}

/// Inserts one candidate into the group's sorted top-8.
///
/// The eight best candidates seen so far live one per lane, ascending,
/// slot 0 in the group's first lane. A new candidate shifts every slot
/// it beats one lane along and drops out the eighth. Nothing is stored
/// in shared memory and no second pass is needed.
///
/// A candidate that ties an incumbent does not displace it, so the
/// first candidate seen at a given distance keeps its slot. Exactly
/// equal distances are common on flat content, so this is what fixes
/// which member a group keeps rather than leaving it to scheduling.
///
/// `plane_shuffle_up` at the group's first lane returns a value from
/// the previous group. The `sub == 0` term discards it before it can be
/// used.
///
/// Exactly two shuffles run, one per carried value. Whether the previous
/// lane also beat `d` is not shuffled, because it follows from `prev_d`.
/// The slots ascend, so the previous lane holds `prev_d` and beat `d`
/// exactly when `d < prev_d`. Each shuffle is an LDS crossbar operation
/// on every candidate, so deriving this rather than shuffling a flag is
/// worth the line of algebra.
#[cube]
pub(crate) fn shift_insert8(best_d: &mut f32, best_pos: &mut u32, d: f32, packed: u32, sub: u32) {
    let prev_d = plane_shuffle_up(*best_d, 1u32);
    let prev_pos = plane_shuffle_up(*best_pos, 1u32);

    if d < *best_d {
        let first = sub == 0u32 || d >= prev_d;
        if first {
            *best_d = d;
            *best_pos = packed;
        } else {
            *best_d = prev_d;
            *best_pos = prev_pos;
        }
    }
}

/// [`shift_insert8`] with the shuffles skipped when the candidate
/// cannot place.
///
/// The group's eighth-best distance sits in its last lane. A candidate
/// that does not beat it changes nothing, so the two shuffles the insert
/// costs are skipped. Every lane holds the same `d` and reads the same
/// broadcast, so the branch is uniform across the group and no lane sits
/// out a shuffle another lane takes.
///
/// The broadcast costs nothing. It fuses into the compare that sets the
/// execution mask, one `v_cmpx_gt_f32` with a `dpp8` modifier, while
/// each shuffle it skips is an LDS crossbar operation. That is why this
/// is the form the kernel uses rather than a later optimisation.
///
/// This is a compute saving and never an admission decision. The eight
/// slots it produces are the same ones [`shift_insert8`] produces.
#[cube]
pub(crate) fn shift_insert8_gated(
    best_d: &mut f32,
    best_pos: &mut u32,
    d: f32,
    packed: u32,
    sub: u32,
    base: u32,
) {
    let worst = plane_shuffle(*best_d, base + 7u32);
    if d < worst {
        shift_insert8(best_d, best_pos, d, packed, sub);
    }
}

/// Transposes one 8x8 block held one column per lane into one row per
/// lane, through `buf`.
///
/// `v` holds the calling lane's 8 values on entry and its transposed 8
/// on return. `slot` is the calling lane's group index, which picks the
/// group's own 65-float region of `buf`, and the stride is padded by
/// one past 64 so that eight lanes writing eight consecutive rows never
/// collide on a bank.
///
/// The spatial row pass needs a row and a lane owns a column, so this
/// runs once before it and once after its inverse.
#[cube]
pub(crate) fn transpose8(buf: &mut SharedMemory<f32>, v: &mut Array<f32>, sub: u32, slot: u32) {
    let base = slot * 65u32;
    #[unroll]
    for i in 0..8u32 {
        buf[(base + i * 8u32 + sub) as usize] = v[i as usize];
    }
    sync_cube();
    #[unroll]
    for i in 0..8u32 {
        v[i as usize] = buf[(base + sub * 8u32 + i) as usize];
    }
    sync_cube();
}
