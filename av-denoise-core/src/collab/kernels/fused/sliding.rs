use cubecl::prelude::*;

use super::centre::{SearchRect, pin_self_match};
use super::distance::{pair_squared_difference, pair_sum_total};
use super::shape::{SearchCursor, SearchShape};
use super::trajectory::{NO_COORDINATE, in_searched_span, is_held_coordinate};
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::pack_pos_t;
use crate::collab::kernels::plane_ops::{
    UNFILLED_DISTANCE,
    plane_ssd_reduce8,
    ranks_before,
    shift_insert8_ordered,
    shift_insert8_ordered_cached,
};
use crate::engine::kernels::read_line;

/// One row of a lane's candidate column, in the search ring's precision.
///
/// With `f16_search` unset the row comes from the f32 `ring` and `S` is f32, so the cast changes
/// nothing.
#[cube]
pub(crate) fn search_line<S: Float, N: Size>(
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    x: u32,
    y: u32,
    slot: u32,
    #[comptime] shape: SearchShape,
) -> Vector<S, N> {
    let f16_search = comptime!(shape.f16_search);
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);

    if comptime!(f16_search) {
        read_line(search_ring, x, y, slot, width, height)
    } else {
        let full = read_line(ring, x, y, slot, width, height);
        Vector::<S, N>::cast_from(full)
    }
}

/// The f16 partial over a window, pairing row `r` with `r + 4` for one stored channel and
/// neighbouring channels within a row otherwise.
#[cube]
fn window_packed_partial<S: Float, N: Size>(
    window: &[Vector<S, N>],
    reference: &[Vector<S, Const<2>>],
    #[comptime] shape: SearchShape,
) -> f32 {
    let stored_ch = comptime!(shape.stored_ch);
    let zero = S::new(0.0f32);
    let mut sum = Vector::<S, Const<2>>::new(zero);

    if comptime!(stored_ch == 1) {
        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let top = window[r as usize];
            let bottom = window[comptime!(r + PATCH_SIZE / 2) as usize];
            let top_lane = top.extract(0usize);
            let bottom_lane = bottom.extract(0usize);
            let squared = pair_squared_difference(reference, top_lane, bottom_lane, r);
            sum += squared;
        }
    } else {
        let pairs_per_row = comptime!(stored_ch / 2);

        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = window[r as usize];

            #[unroll]
            for p in 0..pairs_per_row {
                let first = line.extract(comptime!(2 * p) as usize);
                let second = line.extract(comptime!(2 * p + 1) as usize);
                let pair = comptime!(r * pairs_per_row + p);
                let squared = pair_squared_difference(reference, first, second, pair);
                sum += squared;
            }
        }
    }

    pair_sum_total(sum)
}

/// The distance from the reference to the candidate whose eight rows `window` holds.
///
/// Rows sum in order 0..7, each lane over its own column. Every lane of the group must call it.
#[cube]
pub(crate) fn window_distance<S: Float, N: Size>(
    window: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    scale: f32,
    #[comptime] shape: SearchShape,
) -> f32 {
    let f16_search = comptime!(shape.f16_search);
    let channels = comptime!(shape.channels);
    let mut partial = 0.0f32;

    if comptime!(f16_search) {
        partial = window_packed_partial(window, reference, shape);
    } else {
        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = window[r as usize];

            #[unroll]
            for c in 0..channels {
                let lane = line.extract(c as usize);
                let pixel = f32::cast_from(lane);
                let diff = current[(r * channels + c) as usize] - pixel;
                partial += diff * diff;
            }
        }
    }

    let distance = plane_ssd_reduce8(partial);

    distance * scale
}

/// Moves the window down one row, dropping row 0 and appending `next`.
#[cube]
pub(crate) fn slide_window<S: Float, N: Size>(window: &mut [Vector<S, N>], next: Vector<S, N>) {
    #[unroll]
    for r in 0..comptime!(PATCH_SIZE - 1) {
        window[r as usize] = window[comptime!(r + 1) as usize];
    }

    window[comptime!(PATCH_SIZE - 1) as usize] = next;
}

/// Fills the window with the eight rows of the candidate column starting at row `top`.
#[cube]
fn fill_window<S: Float, N: Size>(
    window: &mut [Vector<S, N>],
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    column: u32,
    top: u32,
    slot: u32,
    #[comptime] shape: SearchShape,
) {
    #[unroll]
    for r in 0..PATCH_SIZE {
        window[r as usize] = search_line(ring, search_ring, column, top + r, slot, shape);
    }
}

/// The centre walk of [centre_search](crate::collab::kernels::fused::centre::centre_search) taken
/// column by column.
///
/// Each lane keeps its eight rows of the current candidate and loads one row per step down. The
/// insert ranks by distance and then position, so the kept slots do not depend on the visit order.
/// With `uniform` the walk runs the full comptime span, masks positions past the clipped rectangle
/// and votes across the plane before each column, score and insert, so every group in a plane takes
/// the same turns.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
pub(crate) fn sliding_centre_walk<S: Float, N: Size>(
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    rect: &SearchRect,
    cursor: &SearchCursor,
    best_d: &mut f32,
    best_pos: &mut u32,
    #[comptime] uniform: bool,
    #[comptime] shape: SearchShape,
) {
    if comptime!(uniform) {
        uniform_sliding_walk(
            ring,
            search_ring,
            current,
            reference,
            rect,
            cursor,
            best_d,
            best_pos,
            shape,
        );
    } else {
        clipped_sliding_walk(
            ring,
            search_ring,
            current,
            reference,
            rect,
            cursor,
            best_d,
            best_pos,
            shape,
        );
    }
}

/// Walks the clipped rectangle column by column, leaving each column at the rectangle's bottom.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
fn clipped_sliding_walk<S: Float, N: Size>(
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    rect: &SearchRect,
    cursor: &SearchCursor,
    best_d: &mut f32,
    best_pos: &mut u32,
    #[comptime] shape: SearchShape,
) {
    let last_row = comptime!(shape.height - 1);
    let mut worst_d = UNFILLED_DISTANCE.runtime();
    let mut worst_pos = 0u32;
    let mut window = Array::<Vector<S, N>>::new(PATCH_SIZE as usize);
    let mut candidate_x = rect.left;

    while candidate_x <= rect.right {
        let column = candidate_x + cursor.sub;
        fill_window(
            window.as_mut_slice(),
            ring,
            search_ring,
            column,
            rect.top,
            cursor.centre_slot,
            shape,
        );

        let mut candidate_y = rect.top;

        while candidate_y <= rect.bottom {
            let scored = window_distance(window.as_slice(), current, reference, cursor.scale, shape);
            let dist = pin_self_match(scored, true, candidate_x, candidate_y, cursor.rx, cursor.ry);

            let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
            shift_insert8_ordered_cached(
                best_d,
                best_pos,
                &mut worst_d,
                &mut worst_pos,
                dist,
                packed,
                cursor.sub,
                cursor.base,
            );

            // The last step's row is clamped and never scored.
            let wanted_row = candidate_y + PATCH_SIZE;
            let next_row = u32::min(wanted_row, last_row);
            let next = search_line(ring, search_ring, column, next_row, cursor.centre_slot, shape);
            slide_window(window.as_mut_slice(), next);
            candidate_y += 1u32;
        }

        candidate_x += 1u32;
    }
}

/// Walks the full `2 * spatial_radius + 1` span column by column and masks the positions past the
/// clipped rectangle.
///
/// A masked turn reads a clamped column or row and carries
/// [UNFILLED_DISTANCE](crate::collab::kernels::plane_ops::UNFILLED_DISTANCE), so it never places.
/// Every vote is the same in every lane, so the plane never splits around a shuffle.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the walk needs"
)]
fn uniform_sliding_walk<S: Float, N: Size>(
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    rect: &SearchRect,
    cursor: &SearchCursor,
    best_d: &mut f32,
    best_pos: &mut u32,
    #[comptime] shape: SearchShape,
) {
    let span = comptime!(2 * shape.spatial_radius + 1);
    let last_row = comptime!(shape.height - 1);
    let mut worst_d = UNFILLED_DISTANCE.runtime();
    let mut worst_pos = 0u32;
    let mut window = Array::<Vector<S, N>>::new(PATCH_SIZE as usize);

    for dx in 0..span {
        let wanted_x = rect.left + dx;
        let column_live = wanted_x <= rect.right;

        if plane_any(column_live) {
            let candidate_x = u32::min(wanted_x, rect.right);
            let column = candidate_x + cursor.sub;
            fill_window(
                window.as_mut_slice(),
                ring,
                search_ring,
                column,
                rect.top,
                cursor.centre_slot,
                shape,
            );

            #[unroll]
            for dy in 0..span {
                let wanted_y = rect.top + dy;
                let live_pos = column_live && wanted_y <= rect.bottom;

                if plane_any(live_pos) {
                    let candidate_y = u32::min(wanted_y, rect.bottom);
                    let scored = window_distance(window.as_slice(), current, reference, cursor.scale, shape);
                    let masked = select(live_pos, scored, UNFILLED_DISTANCE);
                    let dist =
                        pin_self_match(masked, live_pos, candidate_x, candidate_y, cursor.rx, cursor.ry);

                    let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
                    let places = ranks_before(dist, packed, worst_d, worst_pos);

                    if plane_any(places) {
                        shift_insert8_ordered(best_d, best_pos, dist, packed, cursor.sub);
                        worst_d = plane_shuffle(*best_d, cursor.base + 7u32);
                        worst_pos = plane_shuffle(*best_pos, cursor.base + 7u32);
                    }
                }

                // Past the rectangle's bottom the row is clamped and its distance is masked.
                if comptime!(dy + 1 < span) {
                    let wanted_row = wanted_y + PATCH_SIZE;
                    let next_row = u32::min(wanted_row, last_row);
                    let next = search_line(ring, search_ring, column, next_row, cursor.centre_slot, shape);
                    slide_window(window.as_mut_slice(), next);
                }
            }
        }
    }
}

/// Works out the y span each searched rectangle excludes in column `column_x`.
///
/// A rectangle the column crosses excludes its y span, and one it misses gets the empty span `1..=0`.
#[cube]
fn column_spans(
    seen_left: &[u32],
    seen_right: &[u32],
    seen_top: &[u32],
    seen_bot: &[u32],
    column_x: u32,
    skip_top: &mut [u32],
    skip_bot: &mut [u32],
    #[comptime] max_rects: u32,
) {
    #[unroll]
    for s in 0..max_rects {
        let right_of_left = column_x >= seen_left[s as usize];
        let left_of_right = column_x <= seen_right[s as usize];
        let column_inside = right_of_left && left_of_right;
        skip_top[s as usize] = select(column_inside, seen_top[s as usize], 1u32);
        skip_bot[s as usize] = select(column_inside, seen_bot[s as usize], 0u32);
    }
}

/// Works out the row each held member takes in column `column_x` of frame `packed_t`.
///
/// A member in another column or frame gets
/// [NO_COORDINATE](crate::collab::kernels::fused::trajectory::NO_COORDINATE).
#[cube]
fn column_held_rows(
    member_pos: &[u32],
    column_x: u32,
    packed_t: u32,
    held_y: &mut [u32],
    #[comptime] first: u32,
) {
    let column_key = pack_pos_t(column_x, 0u32, packed_t);

    #[unroll]
    for k in 0..first {
        let held = member_pos[k as usize];
        // Keeps the word's frame and x fields.
        let held_column = held & 0xFC00_1FFFu32;
        let held_row = (held >> 13u32) & 0x1FFFu32;
        let in_column = held_column == column_key;
        held_y[k as usize] = select(in_column, held_row, NO_COORDINATE);
    }
}

/// Whether a candidate ranks before a frame's current match.
///
/// Distance compares first, then the ordinal of the rectangle the candidate was scored in, then its
/// packed position.
#[cube]
fn ranks_before_in_frame(
    distance: f32,
    rect: u32,
    packed: u32,
    held_d: f32,
    held_rect: u32,
    held_pos: u32,
) -> bool {
    let closer = distance < held_d;
    let same_distance = distance == held_d;
    let earlier_rect = rect < held_rect;
    let same_rect = rect == held_rect;
    let earlier_position = packed < held_pos;
    let earlier = earlier_rect || (same_rect && earlier_position);

    closer || (same_distance && earlier)
}

/// Replaces the frame's match with the candidate when it ranks before it.
#[cube]
fn keep_frame_match(
    distance: f32,
    rect: u32,
    packed: u32,
    frame_d: &mut f32,
    frame_rect: &mut u32,
    frame_pos: &mut u32,
) {
    let better = ranks_before_in_frame(distance, rect, packed, *frame_d, *frame_rect, *frame_pos);
    *frame_d = select(better, distance, *frame_d);
    *frame_rect = select(better, rect, *frame_rect);
    *frame_pos = select(better, packed, *frame_pos);
}

/// Scores one refine rectangle of a neighbour frame column by column, keeping the frame's best match.
///
/// Each lane keeps its eight rows of the current candidate and loads one row per step down. A
/// position inside an earlier searched rectangle, or held by a member, is skipped. The match ranks
/// by distance, then `rect_index`, then position, so it does not depend on the visit order. With
/// `uniform` the walk runs the full comptime span, masks positions past the clipped rectangle or in
/// a block that is not `block_scored`, and votes across the plane before each column and score.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, per-group array or comptime shape the walk needs"
)]
pub(crate) fn sliding_rect_walk<S: Float, N: Size>(
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    anchor: &[f32],
    reference: &[Vector<S, Const<2>>],
    seen_left: &[u32],
    seen_right: &[u32],
    seen_top: &[u32],
    seen_bot: &[u32],
    member_pos: &[u32],
    bounds: &SearchRect,
    slot: u32,
    sub: u32,
    scale: f32,
    packed_t: u32,
    block_scored: bool,
    frame_d: &mut f32,
    frame_rect: &mut u32,
    frame_pos: &mut u32,
    #[comptime] rect_index: u32,
    #[comptime] uniform: bool,
    #[comptime] max_rects: u32,
    #[comptime] first: u32,
    #[comptime] refine: u32,
    #[comptime] shape: SearchShape,
) {
    let span = comptime!(2 * refine + 1);
    let last_row = comptime!(shape.height - 1);
    let mut window = Array::<Vector<S, N>>::new(PATCH_SIZE as usize);
    let mut skip_top = Array::<u32>::new(max_rects as usize);
    let mut skip_bot = Array::<u32>::new(max_rects as usize);
    let mut held_y = Array::<u32>::new(first as usize);

    if comptime!(uniform) {
        for dx in 0..span {
            let wanted_x = bounds.left + dx;
            let column_live = block_scored && wanted_x <= bounds.right;

            if plane_any(column_live) {
                let candidate_x = u32::min(wanted_x, bounds.right);
                column_spans(
                    seen_left,
                    seen_right,
                    seen_top,
                    seen_bot,
                    candidate_x,
                    skip_top.as_mut_slice(),
                    skip_bot.as_mut_slice(),
                    max_rects,
                );
                column_held_rows(member_pos, candidate_x, packed_t, held_y.as_mut_slice(), first);

                let column = candidate_x + sub;
                fill_window(
                    window.as_mut_slice(),
                    ring,
                    search_ring,
                    column,
                    bounds.top,
                    slot,
                    shape,
                );

                for dy in 0..span {
                    let wanted_y = bounds.top + dy;
                    let candidate_y = u32::min(wanted_y, bounds.bottom);
                    let in_span =
                        in_searched_span(candidate_y, skip_top.as_slice(), skip_bot.as_slice(), max_rects);
                    let held = is_held_coordinate(candidate_y, held_y.as_slice(), first);
                    let in_rect = column_live && wanted_y <= bounds.bottom;
                    let live_pos = in_rect && !in_span && !held;

                    if plane_any(live_pos) {
                        let scored = window_distance(window.as_slice(), anchor, reference, scale, shape);
                        let dist = select(live_pos, scored, UNFILLED_DISTANCE);
                        let packed = pack_pos_t(candidate_x, candidate_y, packed_t);
                        keep_frame_match(dist, rect_index, packed, frame_d, frame_rect, frame_pos);
                    }

                    // Past the rectangle's bottom the row is clamped and its distance is masked.
                    if dy + 1 < span {
                        let wanted_row = wanted_y + PATCH_SIZE;
                        let next_row = u32::min(wanted_row, last_row);
                        let next = search_line(ring, search_ring, column, next_row, slot, shape);
                        slide_window(window.as_mut_slice(), next);
                    }
                }
            }
        }
    } else {
        let mut candidate_x = bounds.left;

        while candidate_x <= bounds.right {
            column_spans(
                seen_left,
                seen_right,
                seen_top,
                seen_bot,
                candidate_x,
                skip_top.as_mut_slice(),
                skip_bot.as_mut_slice(),
                max_rects,
            );
            column_held_rows(member_pos, candidate_x, packed_t, held_y.as_mut_slice(), first);

            let column = candidate_x + sub;
            fill_window(
                window.as_mut_slice(),
                ring,
                search_ring,
                column,
                bounds.top,
                slot,
                shape,
            );

            let mut candidate_y = bounds.top;

            while candidate_y <= bounds.bottom {
                let in_span =
                    in_searched_span(candidate_y, skip_top.as_slice(), skip_bot.as_slice(), max_rects);
                let held = is_held_coordinate(candidate_y, held_y.as_slice(), first);

                if !in_span && !held {
                    let dist = window_distance(window.as_slice(), anchor, reference, scale, shape);
                    let packed = pack_pos_t(candidate_x, candidate_y, packed_t);
                    keep_frame_match(dist, rect_index, packed, frame_d, frame_rect, frame_pos);
                }

                // The last step's row is clamped and never scored.
                let wanted_row = candidate_y + PATCH_SIZE;
                let next_row = u32::min(wanted_row, last_row);
                let next = search_line(ring, search_ring, column, next_row, slot, shape);
                slide_window(window.as_mut_slice(), next);
                candidate_y += 1u32;
            }

            candidate_x += 1u32;
        }
    }
}
