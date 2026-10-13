use cubecl::prelude::*;

use super::batched::batched_centre_walk;
use super::distance::{candidate_partial, pack_reference};
use super::tile::tile_partial;
use super::variant::{Centre, CollabVariant, Walk, tile_width};
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{plane_ssd_reduce8, shift_insert8, shift_insert8_gated};

/// The candidate rectangle of one reference's centre search, clipped to the frame.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct SearchRect {
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) top: u32,
    pub(crate) bottom: u32,
}

/// Scores the centre frame's `spatial_radius` rectangle against the reference patch and keeps the
/// best eight positions, one per lane, in ascending order.
///
/// The reference position scores below every real distance, so it always holds slot 0. Returns
/// the number of positions the clipped rectangle holds.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn centre_search<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
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
) -> u32 {
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let reference = pack_reference::<S>(current, f16_search, channels, stored_ch);

    let left = clamp_top_left(rx as i32 - spatial_radius as i32, max_x);
    let right = clamp_top_left(rx as i32 + spatial_radius as i32, max_x);
    let top = clamp_top_left(ry as i32 - spatial_radius as i32, max_y);
    let bottom = clamp_top_left(ry as i32 + spatial_radius as i32, max_y);
    let rect = SearchRect {
        left,
        right,
        top,
        bottom,
    };

    let walk = comptime!(variant.walk);
    let batched = comptime!(variant.batched);

    if comptime!(batched) {
        batched_centre_walk(
            ring,
            search_ring,
            tile,
            current,
            &reference,
            &rect,
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
            variant,
            f16_search,
            spatial_radius,
            width,
            height,
            channels,
            stored_ch,
        );
    } else if comptime!(walk == Walk::Uniform) {
        uniform_centre_walk(
            ring,
            search_ring,
            tile,
            current,
            &reference,
            &rect,
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
            variant,
            f16_search,
            spatial_radius,
            width,
            height,
            channels,
            stored_ch,
        );
    } else {
        clipped_centre_walk(
            ring,
            search_ring,
            tile,
            current,
            &reference,
            &rect,
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
            variant,
            f16_search,
            spatial_radius,
            width,
            height,
            channels,
            stored_ch,
        );
    }

    (right - left + 1u32) * (bottom - top + 1u32)
}

/// Walks the full `2 * spatial_radius + 1` span around the reference and masks the positions past
/// the clipped rectangle.
///
/// The span is comptime, so every group in a plane takes the same turns. A plane-wide vote skips a
/// position no group needs, and a second vote skips the insert when no group's eighth best is
/// beaten. Each vote is the same in every lane, so the plane never splits around a shuffle. A
/// masked turn carries the `3.0e38` an unfilled slot holds, so it never displaces a match.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
pub(crate) fn uniform_centre_walk<S: Float, N: Size>(
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
    // The unclipped span covers every position the clipped walk visits, in the same order.
    let span = comptime!(2 * spatial_radius + 1);

    for dy in 0..span {
        for dx in 0..span {
            let wanted_y = rect.top + dy;
            let wanted_x = rect.left + dx;
            let live_pos = wanted_x <= rect.right && wanted_y <= rect.bottom;

            if plane_any(live_pos) {
                // A masked turn still reads, so it is pinned to the last live position.
                let candidate_x = u32::min(wanted_x, rect.right);
                let candidate_y = u32::min(wanted_y, rect.bottom);
                let scored = centre_distance(
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
                    scale,
                    variant,
                    f16_search,
                    spatial_radius,
                    width,
                    height,
                    channels,
                    stored_ch,
                );
                let masked = select(live_pos, scored, 3.0e38f32);
                let dist = pin_self_match(masked, live_pos, candidate_x, candidate_y, rx, ry);

                let worst = plane_shuffle(*best_d, base + 7u32);

                if plane_any(dist < worst) {
                    let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
                    shift_insert8(best_d, best_pos, dist, packed, sub);
                }
            }
        }
    }
}

/// Walks the clipped rectangle row by row and keeps the best eight positions.
///
/// Each group leaves the rectangle as soon as it is done, and the gated insert skips its shuffles
/// for a candidate that cannot place.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
pub(crate) fn clipped_centre_walk<S: Float, N: Size>(
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
    let mut candidate_y = rect.top;
    while candidate_y <= rect.bottom {
        let mut candidate_x = rect.left;
        while candidate_x <= rect.right {
            let scored = centre_distance(
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
                scale,
                variant,
                f16_search,
                spatial_radius,
                width,
                height,
                channels,
                stored_ch,
            );
            let dist = pin_self_match(scored, true, candidate_x, candidate_y, rx, ry);

            let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
            shift_insert8_gated(best_d, best_pos, dist, packed, sub, base);
            candidate_x += 1u32;
        }
        candidate_y += 1u32;
    }
}

/// Swaps `dist` for a sentinel below every real distance when `live` and the candidate is the
/// reference itself.
///
/// The self-match then holds slot 0 even on flat content where every candidate ties. A masked
/// turn is not live, so one pinned onto the reference never plants a second self-match.
#[cube]
pub(crate) fn pin_self_match(
    dist: f32,
    live: bool,
    candidate_x: u32,
    candidate_y: u32,
    rx: u32,
    ry: u32,
) -> f32 {
    let mut pinned = dist;

    if live && candidate_x == rx && candidate_y == ry {
        pinned = -1.0e38f32;
    }

    pinned
}

/// One lane's share of the distance to the centre candidate at `(x, y)`, read from the tile or the
/// ring as `variant` picks.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn centre_partial<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    x: u32,
    y: u32,
    tile_x0: u32,
    tile_y0: u32,
    centre_slot: u32,
    sub: u32,
    #[comptime] variant: CollabVariant,
    #[comptime] f16_search: bool,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let centre = comptime!(variant.centre);
    let tile_w = comptime!(tile_width(variant.groups, spatial_radius));

    if comptime!(centre == Centre::Tiled) {
        let column = x - tile_x0 + sub;
        let row = y - tile_y0;
        tile_partial(
            tile, current, reference, column, row, f16_search, tile_w, channels, stored_ch,
        )
    } else {
        candidate_partial(
            ring,
            search_ring,
            current,
            reference,
            x,
            y,
            centre_slot,
            sub,
            f16_search,
            width,
            height,
            channels,
            stored_ch,
        )
    }
}

/// The distance to the centre candidate at `(x, y)`, completed across the group with shuffles.
///
/// Every lane of the group must call it, even when the result is discarded.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn centre_distance<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    x: u32,
    y: u32,
    tile_x0: u32,
    tile_y0: u32,
    centre_slot: u32,
    sub: u32,
    scale: f32,
    #[comptime] variant: CollabVariant,
    #[comptime] f16_search: bool,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let partial = centre_partial(
        ring,
        search_ring,
        tile,
        current,
        reference,
        x,
        y,
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
    let distance = plane_ssd_reduce8(partial);

    distance * scale
}
