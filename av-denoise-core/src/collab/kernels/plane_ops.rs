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

/// The distance an unfilled match slot holds.
///
/// It sits far above the largest real distance of 192 and below `f32::MAX`, so any real candidate
/// displaces it.
pub(crate) const UNFILLED_DISTANCE: f32 = 3.0e38;

/// `shift_insert8` with the shuffles skipped when the candidate cannot place.
///
/// `worst` holds the group's eighth-best distance. It changes only when a candidate places, so one
/// broadcast after each insert keeps it current. Every lane holds the same `distance` and `worst`,
/// so the branch is uniform across the group. It keeps the same eight slots as `shift_insert8`.
#[cube]
pub(crate) fn shift_insert8_cached(
    best_d: &mut f32,
    best_pos: &mut u32,
    worst: &mut f32,
    distance: f32,
    packed: u32,
    sub: u32,
    base: u32,
) {
    if distance < *worst {
        shift_insert8(best_d, best_pos, distance, packed, sub);
        *worst = plane_shuffle(*best_d, base + 7u32);
    }
}

/// Whether `(distance, packed)` ranks before `(held_d, held_pos)`.
///
/// Distances compare first and positions break ties. A centre-frame `pack_pos_t` word grows with
/// the row-major index, so a tie goes to the earlier row-major position whatever order the
/// candidates arrive in.
#[cube]
pub(crate) fn ranks_before(distance: f32, packed: u32, held_d: f32, held_pos: u32) -> bool {
    let closer = distance < held_d;
    let same_distance = distance == held_d;
    let earlier = packed < held_pos;

    closer || (same_distance && earlier)
}

/// Inserts one candidate into the group's sorted top-8, ordered by [ranks_before].
///
/// It shifts slots like [shift_insert8] but breaks ties by position, so it keeps the same eight
/// slots for any visit order. The previous lane ranks before this one, so beating it implies
/// beating this slot too.
#[cube]
pub(crate) fn shift_insert8_ordered(
    best_d: &mut f32,
    best_pos: &mut u32,
    distance: f32,
    packed: u32,
    sub: u32,
) {
    let prev_d = plane_shuffle_up(*best_d, 1u32);
    let prev_pos = plane_shuffle_up(*best_pos, 1u32);
    let beats_slot = ranks_before(distance, packed, *best_d, *best_pos);

    if beats_slot {
        let beats_previous = ranks_before(distance, packed, prev_d, prev_pos);
        let lands_here = sub == 0u32 || !beats_previous;

        if lands_here {
            *best_d = distance;
            *best_pos = packed;
        } else {
            *best_d = prev_d;
            *best_pos = prev_pos;
        }
    }
}

/// [shift_insert8_ordered] with the shuffles skipped when the candidate cannot place.
///
/// `worst_d` and `worst_pos` hold the group's eighth slot and are refreshed after each insert that
/// places. It keeps the same eight slots as [shift_insert8_ordered].
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a per-lane register the search already holds"
)]
pub(crate) fn shift_insert8_ordered_cached(
    best_d: &mut f32,
    best_pos: &mut u32,
    worst_d: &mut f32,
    worst_pos: &mut u32,
    distance: f32,
    packed: u32,
    sub: u32,
    base: u32,
) {
    let places = ranks_before(distance, packed, *worst_d, *worst_pos);

    if places {
        shift_insert8_ordered(best_d, best_pos, distance, packed, sub);
        *worst_d = plane_shuffle(*best_d, base + 7u32);
        *worst_pos = plane_shuffle(*best_pos, base + 7u32);
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

/// Sums each of 8 per-lane partials across the group and leaves candidate `sub`'s total in lane
/// `sub`.
///
/// Each stage keeps the half of the candidates whose index bit matches the lane's, and swaps the
/// other half with its partner at XOR 1, 2 and then 4. That is 7 shuffles for 8 totals. The partners
/// pair up as in [plane_ssd_reduce8], so every total has the same bits.
#[cube]
pub(crate) fn scatter_sums8(partials: &Array<f32>, sub: u32) -> f32 {
    let odd_lane = (sub & 1u32) == 1u32;
    let mut pairs = Array::<f32>::new(4usize);

    #[unroll]
    for k in 0..4u32 {
        let even = partials[(2u32 * k) as usize];
        let odd = partials[(2u32 * k + 1u32) as usize];
        let kept = select(odd_lane, odd, even);
        let sent = select(odd_lane, even, odd);
        let received = plane_shuffle_xor(sent, 1u32);
        pairs[k as usize] = kept + received;
    }

    let upper_pair = ((sub >> 1u32) & 1u32) == 1u32;
    let mut quads = Array::<f32>::new(2usize);

    #[unroll]
    for k in 0..2u32 {
        let low = pairs[(2u32 * k) as usize];
        let high = pairs[(2u32 * k + 1u32) as usize];
        let kept = select(upper_pair, high, low);
        let sent = select(upper_pair, low, high);
        let received = plane_shuffle_xor(sent, 2u32);
        quads[k as usize] = kept + received;
    }

    let upper_quad = ((sub >> 2u32) & 1u32) == 1u32;
    let kept = select(upper_quad, quads[1usize], quads[0usize]);
    let sent = select(upper_quad, quads[0usize], quads[1usize]);
    let received = plane_shuffle_xor(sent, 4u32);

    kept + received
}

/// Whether any lane of the calling lane's 8-lane group raised `flag`, returned to every lane.
#[cube]
pub(crate) fn group_any(flag: bool) -> bool {
    let mut votes = select(flag, 1u32, 0u32);
    votes |= plane_shuffle_xor(votes, 1u32);
    votes |= plane_shuffle_xor(votes, 2u32);
    votes |= plane_shuffle_xor(votes, 4u32);

    votes != 0u32
}
