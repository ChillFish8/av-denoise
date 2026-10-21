use cubecl::prelude::*;

use super::{IntCentre, IntCursor, IntShape, IntegerVariant};
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::fused::centre::SearchRect;
use crate::collab::kernels::fused::trajectory::{
    covering_lo,
    in_searched_span,
    is_held_coordinate,
    row_held_columns,
    row_spans,
};
use crate::collab::kernels::fused::{Walk, tile_width};
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{
    SELF_MATCH_SSD,
    UNFILLED_SSD,
    plane_ssd_reduce8_int,
    ranks_before_int,
    shift_insert8_ordered_int,
    shift_insert8_ordered_int_cached,
};

/// Column `sub` of the patch at `(x, y)` in frame `slot` of the code ring, every channel, row by
/// row.
///
/// Codes are held as f32, which is exact for every code up to 2^24.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the load reads"
)]
pub(crate) fn load_code_column<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> Array<f32> {
    let mut column = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);

    #[unroll]
    for r in 0..PATCH_SIZE {
        let index = (slot * height + y + r) * width + x + sub;
        let pixel = codes[index as usize];
        #[unroll]
        for c in 0..channels {
            column[(r * channels + c) as usize] = f32::cast_from(pixel[c as usize]);
        }
    }

    column
}

/// Converts each channel's f32 sum to i32 and adds them in i32.
///
/// Each channel's sum is exact in f32 while it stays below
/// [F32_EXACT_LIMIT](crate::tune::codes::F32_EXACT_LIMIT), but the channels together may not be.
#[cube]
pub(crate) fn exact_channel_total(sums: &Array<f32>, #[comptime] channels: u32) -> i32 {
    let mut total = 0i32;

    #[unroll]
    for c in 0..channels {
        total += i32::cast_from(sums[c as usize]);
    }

    total
}

/// One f32 sum per channel, each starting at zero.
#[cube]
pub(crate) fn zeroed_channel_sums(#[comptime] channels: u32) -> Array<f32> {
    let mut sums = Array::<f32>::new(channels as usize);

    #[unroll]
    for c in 0..channels {
        sums[c as usize] = 0.0f32;
    }

    sums
}

/// The exact SSD from the reference column to the candidate with top-left `(x, y)` in frame
/// `slot`, completed across the group with shuffles.
///
/// Each channel sums its column of squared differences on f32 multiply-adds, which stay exact
/// below [F32_EXACT_LIMIT](crate::tune::codes::F32_EXACT_LIMIT). The channels and lanes then add
/// in i32. Every lane of the group must call it, even when the result is discarded.
#[cube]
pub(crate) fn int_candidate_ssd<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    current: &Array<f32>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] shape: IntShape,
) -> i32 {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let mut sums = zeroed_channel_sums(channels);

    #[unroll]
    for r in 0..PATCH_SIZE {
        let index = (slot * height + y + r) * width + x + sub;
        let pixel = codes[index as usize];
        #[unroll]
        for c in 0..channels {
            let code = f32::cast_from(pixel[c as usize]);
            let diff = current[(r * channels + c) as usize] - code;
            sums[c as usize] += diff * diff;
        }
    }

    let partial = exact_channel_total(&sums, channels);
    plane_ssd_reduce8_int(partial)
}

/// Loads the cube's centre tile, `tile_width` by `tile_height` pixels from `(origin_x, origin_y)`.
///
/// Pixels are row-major with the summed channels interleaved as f32 codes, each read clamped to
/// the frame. Every lane of the cube loads a strided share.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the load needs"
)]
pub(crate) fn load_int_centre_tile<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &mut SharedMemory<f32>,
    origin_x: u32,
    origin_y: u32,
    centre_slot: u32,
    #[comptime] tile_width: u32,
    #[comptime] tile_height: u32,
    #[comptime] lanes: u32,
    #[comptime] shape: IntShape,
) {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let pixels = comptime!(tile_width * tile_height);
    let last_x = comptime!(width - 1);
    let last_y = comptime!(height - 1);
    let mut index = UNIT_POS_X;

    while index < pixels {
        let tile_x = index % tile_width;
        let tile_y = index / tile_width;
        let source_x = u32::min(origin_x + tile_x, last_x);
        let source_y = u32::min(origin_y + tile_y, last_y);
        let source = (centre_slot * height + source_y) * width + source_x;
        let pixel = codes[source as usize];
        let first_lane = index * channels;

        #[unroll]
        for c in 0..channels {
            tile[(first_lane + c) as usize] = f32::cast_from(pixel[c as usize]);
        }

        index += lanes;
    }
}

/// One lane's share of the SSD to the candidate whose column `column` and top row `row` sit in
/// the tile.
///
/// It sums the same codes in the same order as [int_candidate_ssd] does before its reduction.
#[cube]
pub(crate) fn int_tile_partial(
    tile: &SharedMemory<f32>,
    current: &Array<f32>,
    column: u32,
    row: u32,
    #[comptime] tile_width: u32,
    #[comptime] channels: u32,
) -> i32 {
    let mut sums = zeroed_channel_sums(channels);

    #[unroll]
    for r in 0..PATCH_SIZE {
        let first_lane = ((row + r) * tile_width + column) * channels;
        #[unroll]
        for c in 0..channels {
            let code = tile[(first_lane + c) as usize];
            let diff = current[(r * channels + c) as usize] - code;
            sums[c as usize] += diff * diff;
        }
    }

    exact_channel_total(&sums, channels)
}

/// The exact SSD to the centre candidate with top-left `(x, y)`, read from the tile when the
/// variant has one and from the code ring otherwise.
///
/// Every lane of the group must call it, even when the result is discarded.
#[cube]
pub(crate) fn int_centre_ssd<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &SharedMemory<f32>,
    current: &Array<f32>,
    x: u32,
    y: u32,
    cursor: &IntCursor,
    #[comptime] variant: IntegerVariant,
    #[comptime] shape: IntShape,
) -> i32 {
    if comptime!(variant.centre == IntCentre::Tiled) {
        let tile_w = comptime!(tile_width(variant.cube_refs_x, shape.spatial_radius));
        let channels = comptime!(shape.channels);
        let column = x - cursor.tile_x0 + cursor.sub;
        let row = y - cursor.tile_y0;
        let partial = int_tile_partial(tile, current, column, row, tile_w, channels);
        plane_ssd_reduce8_int(partial)
    } else {
        int_candidate_ssd(codes, current, x, y, cursor.centre_slot, cursor.sub, shape)
    }
}

/// Swaps `ssd` for [SELF_MATCH_SSD](crate::collab::kernels::plane_ops::SELF_MATCH_SSD) when `live`
/// and the candidate is the reference itself.
#[cube]
pub(crate) fn pin_self_match_int(
    ssd: i32,
    live: bool,
    candidate_x: u32,
    candidate_y: u32,
    rx: u32,
    ry: u32,
) -> i32 {
    let is_reference = candidate_x == rx && candidate_y == ry;
    let pinned = live && is_reference;

    select(pinned, SELF_MATCH_SSD, ssd)
}

/// Scores the centre frame's `spatial_radius` rectangle against the reference patch and keeps the
/// best eight positions, one per lane, ranked by SSD and then position.
///
/// The self-match holds slot 0. Returns the number of positions the clipped rectangle holds.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, slot or comptime shape the search reads"
)]
pub(crate) fn int_centre_search<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &SharedMemory<f32>,
    current: &Array<f32>,
    cursor: &IntCursor,
    best_ssd: &mut i32,
    best_pos: &mut u32,
    #[comptime] variant: IntegerVariant,
    #[comptime] shape: IntShape,
) -> u32 {
    let walk = comptime!(variant.walk);
    let spatial_radius = comptime!(shape.spatial_radius);
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);

    let left = clamp_top_left(cursor.rx as i32 - spatial_radius as i32, max_x);
    let right = clamp_top_left(cursor.rx as i32 + spatial_radius as i32, max_x);
    let top = clamp_top_left(cursor.ry as i32 - spatial_radius as i32, max_y);
    let bottom = clamp_top_left(cursor.ry as i32 + spatial_radius as i32, max_y);
    let rect = SearchRect {
        left,
        right,
        top,
        bottom,
    };

    if comptime!(walk == Walk::Uniform) {
        int_uniform_centre_walk(
            codes, tile, current, &rect, cursor, best_ssd, best_pos, variant, shape,
        );
    } else {
        int_clipped_centre_walk(
            codes, tile, current, &rect, cursor, best_ssd, best_pos, variant, shape,
        );
    }

    (right - left + 1u32) * (bottom - top + 1u32)
}

/// Walks the full `2 * spatial_radius + 1` span around the reference and masks the positions past
/// the clipped rectangle.
///
/// A plane-wide vote skips a position no group needs, and a second vote skips the insert when no
/// group's eighth slot is beaten. A masked turn carries
/// [UNFILLED_SSD](crate::collab::kernels::plane_ops::UNFILLED_SSD), so it never places.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, slot or comptime shape the walk reads"
)]
fn int_uniform_centre_walk<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &SharedMemory<f32>,
    current: &Array<f32>,
    rect: &SearchRect,
    cursor: &IntCursor,
    best_ssd: &mut i32,
    best_pos: &mut u32,
    #[comptime] variant: IntegerVariant,
    #[comptime] shape: IntShape,
) {
    let mut worst_ssd = UNFILLED_SSD.runtime();
    let mut worst_pos = 0u32;
    let span = comptime!(2 * shape.spatial_radius + 1);

    for dy in 0..span {
        for dx in 0..span {
            let wanted_y = rect.top + dy;
            let wanted_x = rect.left + dx;
            let live_pos = wanted_x <= rect.right && wanted_y <= rect.bottom;

            if plane_any(live_pos) {
                // A masked turn still reads, so it is pinned to the last live position.
                let candidate_x = u32::min(wanted_x, rect.right);
                let candidate_y = u32::min(wanted_y, rect.bottom);
                let scored = int_centre_ssd(
                    codes,
                    tile,
                    current,
                    candidate_x,
                    candidate_y,
                    cursor,
                    variant,
                    shape,
                );
                let masked = select(live_pos, scored, UNFILLED_SSD);
                let ssd =
                    pin_self_match_int(masked, live_pos, candidate_x, candidate_y, cursor.rx, cursor.ry);
                let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
                let places = ranks_before_int(ssd, packed, worst_ssd, worst_pos);

                if plane_any(places) {
                    shift_insert8_ordered_int(best_ssd, best_pos, ssd, packed, cursor.sub);
                    worst_ssd = plane_shuffle(*best_ssd, cursor.base + 7u32);
                    worst_pos = plane_shuffle(*best_pos, cursor.base + 7u32);
                }
            }
        }
    }
}

/// Walks the clipped rectangle row by row and keeps the best eight positions.
///
/// Each group leaves the rectangle as soon as it is done, and the cached insert skips its shuffles
/// for a candidate that cannot place.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, slot or comptime shape the walk reads"
)]
fn int_clipped_centre_walk<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    tile: &SharedMemory<f32>,
    current: &Array<f32>,
    rect: &SearchRect,
    cursor: &IntCursor,
    best_ssd: &mut i32,
    best_pos: &mut u32,
    #[comptime] variant: IntegerVariant,
    #[comptime] shape: IntShape,
) {
    let mut worst_ssd = UNFILLED_SSD.runtime();
    let mut worst_pos = 0u32;
    let mut candidate_y = rect.top;

    while candidate_y <= rect.bottom {
        let mut candidate_x = rect.left;
        while candidate_x <= rect.right {
            let scored = int_centre_ssd(
                codes,
                tile,
                current,
                candidate_x,
                candidate_y,
                cursor,
                variant,
                shape,
            );
            let ssd = pin_self_match_int(scored, true, candidate_x, candidate_y, cursor.rx, cursor.ry);

            let packed = pack_pos_t(candidate_x, candidate_y, 0u32);
            shift_insert8_ordered_int_cached(
                best_ssd,
                best_pos,
                &mut worst_ssd,
                &mut worst_pos,
                ssd,
                packed,
                cursor.sub,
                cursor.base,
            );
            candidate_x += 1u32;
        }

        candidate_y += 1u32;
    }
}

/// Builds one volume's frames, the anchor patch's best match in each neighbour frame.
///
/// It runs [int_trajectory_frame] for every neighbour frame in turn.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn int_trajectory_search<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    anchor: &Array<f32>,
    anchor_x: u32,
    anchor_y: u32,
    sub: u32,
    c_min: f32,
    member_ssd: &mut Array<i32>,
    member_pos: &mut Array<u32>,
    #[comptime] first: u32,
    #[comptime] tail: u32,
    #[comptime] walk: Walk,
    #[comptime] radius: u32,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] shape: IntShape,
) {
    let n_neighbours = comptime!(2 * radius);
    let mut t = 0u32;

    while t < n_neighbours {
        int_trajectory_frame(
            codes,
            mv_field,
            confidence,
            neighbour_slots,
            anchor,
            anchor_x,
            anchor_y,
            sub,
            c_min,
            member_ssd,
            member_pos,
            t,
            first,
            tail,
            walk,
            refine,
            mv_stride,
            conf_stride,
            blk_step,
            blksize,
            blocks_x,
            blocks_y,
            shape,
        );

        t += 1u32;
    }
}

/// Scores neighbour frame `t` for the anchor and inserts its best match into the volume.
///
/// The match is the best position, by [ranks_before_int], inside the refine rectangles of the
/// motion blocks covering the anchor, each around where that block's vector moves the anchor. It
/// joins `member_ssd` and `member_pos` at `first..first + tail` in ascending SSD order, and a tie
/// keeps the earlier frame.
///
/// A position already held in `member_pos[..first]` is skipped, as is a block below `c_min`, and a
/// position reached by two covering blocks is scored once. The clipped walk skips any rectangle
/// that lies wholly inside one it already searched. The uniform walk covers every rectangle at its
/// full comptime span and masks the skipped positions, so both walks keep the same match.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn int_trajectory_frame<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    anchor: &Array<f32>,
    anchor_x: u32,
    anchor_y: u32,
    sub: u32,
    c_min: f32,
    member_ssd: &mut Array<i32>,
    member_pos: &mut Array<u32>,
    t: u32,
    #[comptime] first: u32,
    #[comptime] tail: u32,
    #[comptime] walk: Walk,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] shape: IntShape,
) {
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);
    let covers = comptime!(blksize.div_ceil(blk_step));
    let max_rects = comptime!(covers * covers);

    let bx_hi = (anchor_x / blk_step).min(blocks_x - 1);
    let by_hi = (anchor_y / blk_step).min(blocks_y - 1);
    let bx_lo_covering = covering_lo(anchor_x, blksize, blk_step);
    let by_lo_covering = covering_lo(anchor_y, blksize, blk_step);
    let bx_lo = u32::min(bx_lo_covering, bx_hi);
    let by_lo = u32::min(by_lo_covering, by_hi);

    let slot = neighbour_slots[t as usize];
    let packed_t = t + 1u32;

    let mut frame_ssd = UNFILLED_SSD.runtime();
    let mut frame_pos = 0u32;

    let mut seen_left = Array::<u32>::new(max_rects as usize);
    let mut seen_right = Array::<u32>::new(max_rects as usize);
    let mut seen_top = Array::<u32>::new(max_rects as usize);
    let mut seen_bot = Array::<u32>::new(max_rects as usize);
    #[unroll]
    for s in 0..max_rects {
        seen_left[s as usize] = 1u32;
        seen_right[s as usize] = 0u32;
        seen_top[s as usize] = 1u32;
        seen_bot[s as usize] = 0u32;
    }

    #[unroll]
    for iy in 0..covers {
        #[unroll]
        for ix in 0..covers {
            let wanted_bx = bx_lo + ix;
            let wanted_by = by_lo + iy;
            let block_live = wanted_bx <= bx_hi && wanted_by <= by_hi;

            if comptime!(walk == Walk::Uniform) {
                let clamped_bx = u32::min(wanted_bx, bx_hi);
                let clamped_by = u32::min(wanted_by, by_hi);
                let block = clamped_by * blocks_x + clamped_bx;
                let block_confidence = confidence[(t * conf_stride + block) as usize];
                let block_scored = block_live && block_confidence >= c_min;

                let mv_index = (t * mv_stride + block * 2u32) as usize;
                let predicted_x = anchor_x as i32 + mv_field[mv_index];
                let predicted_y = anchor_y as i32 + mv_field[mv_index + 1];

                let t_left = clamp_top_left(predicted_x - refine as i32, max_x);
                let t_right = clamp_top_left(predicted_x + refine as i32, max_x);
                let t_top = clamp_top_left(predicted_y - refine as i32, max_y);
                let t_bot = clamp_top_left(predicted_y + refine as i32, max_y);

                let span = comptime!(2 * refine + 1);
                for dy in 0..span {
                    let wanted_y = t_top + dy;
                    let candidate_y = u32::min(wanted_y, t_bot);

                    for dx in 0..span {
                        let wanted_x = t_left + dx;
                        let in_rect = wanted_x <= t_right && wanted_y <= t_bot;
                        let candidate_x = u32::min(wanted_x, t_right);
                        let packed = pack_pos_t(candidate_x, candidate_y, packed_t);

                        let mut skipped = false;
                        #[unroll]
                        for s in 0..max_rects {
                            if candidate_x >= seen_left[s as usize]
                                && candidate_x <= seen_right[s as usize]
                                && candidate_y >= seen_top[s as usize]
                                && candidate_y <= seen_bot[s as usize]
                            {
                                skipped = true;
                            }
                        }

                        #[unroll]
                        for k in 0..first {
                            if member_pos[k as usize] == packed {
                                skipped = true;
                            }
                        }

                        let live_pos = block_scored && in_rect && !skipped;

                        if plane_any(live_pos) {
                            let scored =
                                int_candidate_ssd(codes, anchor, candidate_x, candidate_y, slot, sub, shape);
                            let ssd = select(live_pos, scored, UNFILLED_SSD);
                            let better = ranks_before_int(ssd, packed, frame_ssd, frame_pos);
                            frame_ssd = select(better, ssd, frame_ssd);
                            frame_pos = select(better, packed, frame_pos);
                        }
                    }
                }

                let rect = (iy * covers + ix) as usize;
                seen_left[rect] = select(block_scored, t_left, 1u32);
                seen_right[rect] = select(block_scored, t_right, 0u32);
                seen_top[rect] = select(block_scored, t_top, 1u32);
                seen_bot[rect] = select(block_scored, t_bot, 0u32);
            } else if block_live {
                let block = wanted_by * blocks_x + wanted_bx;
                let block_confidence = confidence[(t * conf_stride + block) as usize];
                if block_confidence >= c_min {
                    let mv_index = (t * mv_stride + block * 2u32) as usize;
                    let predicted_x = anchor_x as i32 + mv_field[mv_index];
                    let predicted_y = anchor_y as i32 + mv_field[mv_index + 1];

                    let t_left = clamp_top_left(predicted_x - refine as i32, max_x);
                    let t_right = clamp_top_left(predicted_x + refine as i32, max_x);
                    let t_top = clamp_top_left(predicted_y - refine as i32, max_y);
                    let t_bot = clamp_top_left(predicted_y + refine as i32, max_y);

                    // An empty `seen_*` slot spans 1..=0, which no rectangle fits inside.
                    let mut contained = false;
                    #[unroll]
                    for s in 0..max_rects {
                        let inside_x = t_left >= seen_left[s as usize] && t_right <= seen_right[s as usize];
                        let inside_y = t_top >= seen_top[s as usize] && t_bot <= seen_bot[s as usize];
                        if inside_x && inside_y {
                            contained = true;
                        }
                    }

                    if !contained {
                        let mut skip_left = Array::<u32>::new(max_rects as usize);
                        let mut skip_right = Array::<u32>::new(max_rects as usize);
                        let mut held_x = Array::<u32>::new(first as usize);

                        let mut candidate_y = t_top;
                        while candidate_y <= t_bot {
                            row_spans(
                                &seen_left,
                                &seen_right,
                                &seen_top,
                                &seen_bot,
                                candidate_y,
                                &mut skip_left,
                                &mut skip_right,
                                max_rects,
                            );
                            row_held_columns(member_pos, candidate_y, packed_t, &mut held_x, first);

                            let mut candidate_x = t_left;
                            while candidate_x <= t_right {
                                let in_span =
                                    in_searched_span(candidate_x, &skip_left, &skip_right, max_rects);
                                let held = is_held_coordinate(candidate_x, &held_x, first);
                                if !in_span && !held {
                                    let packed = pack_pos_t(candidate_x, candidate_y, packed_t);
                                    let ssd = int_candidate_ssd(
                                        codes,
                                        anchor,
                                        candidate_x,
                                        candidate_y,
                                        slot,
                                        sub,
                                        shape,
                                    );
                                    if ranks_before_int(ssd, packed, frame_ssd, frame_pos) {
                                        frame_ssd = ssd;
                                        frame_pos = packed;
                                    }
                                }

                                candidate_x += 1u32;
                            }

                            candidate_y += 1u32;
                        }
                    }

                    let rect = (iy * covers + ix) as usize;
                    seen_left[rect] = t_left;
                    seen_right[rect] = t_right;
                    seen_top[rect] = t_top;
                    seen_bot[rect] = t_bot;
                }
            }
        }
    }

    // The frame's match joins the volume's ascending list and pushes the worst out.
    let mut carry_ssd = frame_ssd;
    let mut carry_pos = frame_pos;
    #[unroll]
    for j in 0..tail {
        let slot_index = comptime!(first + j) as usize;
        let held_ssd = member_ssd[slot_index];
        let held_pos = member_pos[slot_index];
        let smaller = carry_ssd < held_ssd;
        member_ssd[slot_index] = select(smaller, carry_ssd, held_ssd);
        member_pos[slot_index] = select(smaller, carry_pos, held_pos);
        carry_ssd = select(smaller, held_ssd, carry_ssd);
        carry_pos = select(smaller, held_pos, carry_pos);
    }
}
