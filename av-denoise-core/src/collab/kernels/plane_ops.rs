use cubecl::prelude::*;

/// The plane-local index of the first lane in the calling lane's 8-lane group.
///
/// Every shuffle here addresses lanes relative to this, so a group never reads another group's
/// lane. Cubes launch 1-D, so `UNIT_POS_PLANE % 8` and `UNIT_POS_X % 8` agree at wave32 and wave64.
#[cube]
pub(crate) fn group_base() -> u32 {
    UNIT_POS_PLANE - UNIT_POS_PLANE % 8u32
}

/// Sums `partial` across the calling lane's 8-lane group and returns the sum to every lane.
///
/// The XOR masks 1, 2 and 4 are all below 8, so the shuffles stay inside the group whatever the
/// plane width.
#[cube]
pub(crate) fn plane_ssd_reduce8(partial: f32) -> f32 {
    let mut sum = partial;
    sum += plane_shuffle_xor(sum, 1u32);
    sum += plane_shuffle_xor(sum, 2u32);
    sum += plane_shuffle_xor(sum, 4u32);
    sum
}

/// Inserts one candidate into the group's sorted top-8.
///
/// The eight best candidates live one per lane, ascending, with slot 0 in the group's first lane.
/// A new candidate shifts every slot it beats one lane along and drops the eighth. A tie never
/// displaces an incumbent, so on flat content the first candidate seen keeps its slot rather than
/// leaving it to scheduling.
///
/// `plane_shuffle_up` at the group's first lane reads the previous group, so the `sub == 0` term
/// discards it. The slots ascend, so the previous lane beat `distance` exactly when
/// `distance < prev_d`, which saves shuffling a flag.
#[cube]
pub(crate) fn shift_insert8(best_d: &mut f32, best_pos: &mut u32, distance: f32, packed: u32, sub: u32) {
    let prev_d = plane_shuffle_up(*best_d, 1u32);
    let prev_pos = plane_shuffle_up(*best_pos, 1u32);

    if distance < *best_d {
        let lands_here = sub == 0u32 || distance >= prev_d;
        if lands_here {
            *best_d = distance;
            *best_pos = packed;
        } else {
            *best_d = prev_d;
            *best_pos = prev_pos;
        }
    }
}

/// `shift_insert8` with the shuffles skipped when the candidate cannot place.
///
/// The group's eighth-best distance sits in its last lane, and a candidate that does not beat it
/// changes nothing. Every lane holds the same `distance` and reads the same broadcast, so the branch is
/// uniform across the group and no lane skips a shuffle another lane takes. The broadcast fuses
/// into the compare, while each skipped shuffle is an LDS crossbar operation. It keeps the same
/// eight slots as `shift_insert8`.
#[cube]
pub(crate) fn shift_insert8_gated(
    best_d: &mut f32,
    best_pos: &mut u32,
    distance: f32,
    packed: u32,
    sub: u32,
    base: u32,
) {
    let worst = plane_shuffle(*best_d, base + 7u32);
    if distance < worst {
        shift_insert8(best_d, best_pos, distance, packed, sub);
    }
}

/// Transposes an 8x8 block held one column per lane into one row per lane, through `buf`.
///
/// `v` holds the lane's 8 values on entry and its transposed 8 on return. `slot` picks the group's
/// own 65-float region of `buf`, padded one past 64 so eight lanes writing consecutive rows never
/// collide on a bank. It contains two `sync_cube()` barriers, so every lane of the cube must call
/// it.
#[cube]
pub(crate) fn transpose8(buf: &mut SharedMemory<f32>, v: &mut Array<f32>, sub: u32, slot: u32) {
    let region_start = slot * 65u32;
    #[unroll]
    for i in 0..8u32 {
        buf[(region_start + i * 8u32 + sub) as usize] = v[i as usize];
    }
    sync_cube();

    #[unroll]
    for i in 0..8u32 {
        v[i as usize] = buf[(region_start + sub * 8u32 + i) as usize];
    }
    sync_cube();
}
