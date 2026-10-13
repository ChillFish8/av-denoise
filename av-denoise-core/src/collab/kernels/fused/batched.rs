use cubecl::prelude::*;

use super::centre::{SearchRect, centre_partial, pin_self_match};
use super::variant::{CollabVariant, Walk};
use crate::collab::kernels::group::pack_pos_t;
use crate::collab::kernels::plane_ops::{group_any, scatter_sums8, shift_insert8};

/// Candidates one batch scores, one per lane of a group.
const BATCH: u32 = 8;

/// Walks the centre rectangle 8 candidates at a time and keeps the best eight positions.
///
/// The batches follow the walk's own order, row-major over the clipped rectangle or over the full
/// span, and their totals are inserted in that order, so the slots match the unbatched walk. The
/// clipped walk skips a batch's inserts when no candidate beats its group's eighth best. The
/// uniform walk votes across the plane, so it skips a batch only when no group in the plane needs
/// it, and skips the inserts only when no group in the plane has a winner.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
pub(crate) fn batched_centre_walk<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    rect: &SearchRect,
    rx: u32,
    ry: u32,
    tile_x0: u32,
    tile_y0: u32,
    centre_slot: u32,
    sub: u32,
    base: u32,
    scale: f32,
    best_d: &mut f32,
    best_pos: &mut u32,
    #[comptime] variant: CollabVariant,
    #[comptime] f16_search: bool,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) {
    let walk = comptime!(variant.walk);
    let span = comptime!(2 * spatial_radius + 1);
    let rect_width = rect.right - rect.left + 1u32;
    let rect_count = rect_width * (rect.bottom - rect.top + 1u32);

    // The uniform walk steps through the full span so every group takes the same turns.
    let walk_width = if comptime!(walk == Walk::Uniform) {
        span.runtime()
    } else {
        rect_width
    };
    let walk_count = if comptime!(walk == Walk::Uniform) {
        comptime!(span * span).runtime()
    } else {
        rect_count
    };

    let batches = walk_count.div_ceil(BATCH);
    let mut batch = 0u32;

    while batch < batches {
        let first = batch * BATCH;

        // Lane `sub` owns candidate `sub` of the batch for the vote and the insert.
        let own_index = first + sub;
        let own_x = rect.left + own_index % walk_width;
        let own_y = rect.top + own_index / walk_width;
        let own_live = own_index < walk_count && own_x <= rect.right && own_y <= rect.bottom;

        // Every group of a clipped walk scores each of its own batches. The uniform walk's vote is
        // the same in every lane, so the plane never splits around a shuffle.
        let mut needed = true;
        if comptime!(walk == Walk::Uniform) {
            needed = plane_any(own_live);
        }

        if needed {
            score_batch(
                ring,
                search_ring,
                tile,
                current,
                reference,
                rect,
                rx,
                ry,
                tile_x0,
                tile_y0,
                centre_slot,
                sub,
                base,
                scale,
                best_d,
                best_pos,
                first,
                walk_width,
                own_x,
                own_y,
                own_live,
                variant,
                f16_search,
                spatial_radius,
                width,
                height,
                channels,
                stored_ch,
            );
        }

        batch += 1u32;
    }
}

/// Scores the batch starting at walk index `first` and inserts its totals in walk order.
///
/// A padding candidate past the walk is pinned inside the rectangle, so every read stays in range,
/// and carries the `3.0e38` an unfilled slot holds.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the batch needs"
)]
fn score_batch<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    rect: &SearchRect,
    rx: u32,
    ry: u32,
    tile_x0: u32,
    tile_y0: u32,
    centre_slot: u32,
    sub: u32,
    base: u32,
    scale: f32,
    best_d: &mut f32,
    best_pos: &mut u32,
    first: u32,
    walk_width: u32,
    own_x: u32,
    own_y: u32,
    own_live: bool,
    #[comptime] variant: CollabVariant,
    #[comptime] f16_search: bool,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) {
    let walk = comptime!(variant.walk);
    let mut partials = Array::<f32>::new(BATCH as usize);

    #[unroll]
    for k in 0..BATCH {
        let index = first + k;
        let wanted_x = rect.left + index % walk_width;
        let wanted_y = rect.top + index / walk_width;
        let candidate_x = u32::min(wanted_x, rect.right);
        let candidate_y = u32::min(wanted_y, rect.bottom);
        partials[k as usize] = centre_partial(
            ring,
            search_ring,
            tile,
            current,
            reference,
            candidate_x,
            candidate_y,
            tile_x0,
            tile_y0,
            centre_slot,
            sub,
            variant,
            f16_search,
            spatial_radius,
            width,
            height,
            channels,
            stored_ch,
        );
    }

    let total = scatter_sums8(&partials, sub);
    let candidate_x = u32::min(own_x, rect.right);
    let candidate_y = u32::min(own_y, rect.bottom);
    let scored = total * scale;
    let masked = select(own_live, scored, 3.0e38f32);
    let dist = pin_self_match(masked, own_live, candidate_x, candidate_y, rx, ry);

    let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
    let worst = plane_shuffle(*best_d, base + 7u32);
    let beats = dist < worst;
    let voted = if comptime!(walk == Walk::Uniform) {
        plane_any(beats)
    } else {
        group_any(beats)
    };

    if voted {
        #[unroll]
        for k in 0..BATCH {
            let candidate_d = plane_shuffle(dist, base + k);
            let candidate_pos = plane_shuffle(packed, base + k);
            shift_insert8(best_d, best_pos, candidate_d, candidate_pos, sub);
        }
    }
}
