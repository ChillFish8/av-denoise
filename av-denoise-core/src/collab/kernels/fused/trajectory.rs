use cubecl::prelude::*;

use super::centre::SearchRect;
use super::distance::{candidate_distance, pack_reference};
use super::shape::SearchShape;
use super::sliding::sliding_rect_walk;
use super::variant::{NeighbourWalk, Walk};
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::UNFILLED_DISTANCE;

/// The lowest block index whose span contains the patch at `patch_start` on one axis.
///
/// Block `b` spans `b * step..b * step + blksize`, so it contains `patch_start..patch_start + PATCH_SIZE`
/// when `b * step + blksize >= patch_start + PATCH_SIZE`. The caller clamps the result to the highest
/// covering block, `patch_start / step`. It mirrors `covering_blocks` in `nl4d/harness/score.rs`.
#[cube]
pub(crate) fn covering_lo(patch_start: u32, #[comptime] blksize: u32, #[comptime] step: u32) -> u32 {
    let overhang = u32::max(patch_start + PATCH_SIZE, blksize) - blksize;
    overhang.div_ceil(step)
}

/// Host mirror of `covering_lo`.
#[cfg(test)]
fn covering_lo_host(patch_start: u32, blksize: u32, step: u32) -> u32 {
    let overhang = u32::max(patch_start + PATCH_SIZE, blksize) - blksize;
    overhang.div_ceil(step)
}

/// A coordinate no candidate can have, since positions use 13 bits.
pub(crate) const NO_COORDINATE: u32 = 0xFFFF_FFFF;

/// Works out the x span each searched rectangle excludes on row `row_y`.
///
/// A rectangle the row crosses excludes its x span, and one it misses gets the empty span `1..=0`.
#[cube]
fn row_spans(
    seen_left: &Array<u32>,
    seen_right: &Array<u32>,
    seen_top: &Array<u32>,
    seen_bot: &Array<u32>,
    row_y: u32,
    skip_left: &mut Array<u32>,
    skip_right: &mut Array<u32>,
    #[comptime] max_rects: u32,
) {
    #[unroll]
    for s in 0..max_rects {
        let below_top = row_y >= seen_top[s as usize];
        let above_bottom = row_y <= seen_bot[s as usize];
        let row_inside = below_top && above_bottom;
        skip_left[s as usize] = select(row_inside, seen_left[s as usize], 1u32);
        skip_right[s as usize] = select(row_inside, seen_right[s as usize], 0u32);
    }
}

/// Works out the column each held member takes on row `row_y` of frame `packed_t`.
///
/// A member on another row or frame gets [NO_COORDINATE].
#[cube]
fn row_held_columns(
    member_pos: &Array<u32>,
    row_y: u32,
    packed_t: u32,
    held_x: &mut Array<u32>,
    #[comptime] first: u32,
) {
    let row_key = pack_pos_t(0u32, row_y, packed_t);

    #[unroll]
    for k in 0..first {
        let held = member_pos[k as usize];
        let held_row = held & 0xFFFF_E000u32;
        let held_column = held & 0x1FFFu32;
        let on_row = held_row == row_key;
        held_x[k as usize] = select(on_row, held_column, NO_COORDINATE);
    }
}

/// Whether `coordinate` falls inside one of the spans `skip_start[s]..=skip_end[s]`.
#[cube]
pub(crate) fn in_searched_span(
    coordinate: u32,
    skip_start: &Array<u32>,
    skip_end: &Array<u32>,
    #[comptime] max_rects: u32,
) -> bool {
    let mut inside_any = false;

    #[unroll]
    for s in 0..max_rects {
        let after_start = coordinate >= skip_start[s as usize];
        let before_end = coordinate <= skip_end[s as usize];
        let inside = after_start && before_end;
        inside_any = inside_any || inside;
    }

    inside_any
}

/// Whether `coordinate` is one of the held members' coordinates in `held`.
#[cube]
pub(crate) fn is_held_coordinate(coordinate: u32, held: &Array<u32>, #[comptime] first: u32) -> bool {
    let mut held_any = false;

    #[unroll]
    for k in 0..first {
        let matches = held[k as usize] == coordinate;
        held_any = held_any || matches;
    }

    held_any
}

/// Builds one volume's frames, the anchor patch's best match in each neighbour frame.
///
/// A neighbour's match is the lowest-distance position inside the refine rectangles of the motion
/// blocks covering the anchor, each around where that block's vector moves the anchor. The
/// `tail` lowest of those per-frame matches land in `member_d` and `member_pos` at
/// `first..first + tail`, ascending. A slot no frame filled keeps the
/// [UNFILLED_DISTANCE](crate::collab::kernels::plane_ops::UNFILLED_DISTANCE) it starts with.
///
/// A position already held in `member_pos[..first]` is skipped, so no patch enters the group
/// twice. A block below `c_min` is skipped too, and a position reached by two covering blocks
/// is scored once. The clipped walk skips any rectangle that lies wholly inside one it already
/// searched.
///
/// With [Walk::Uniform](crate::collab::kernels::fused::Walk::Uniform) the rectangles are walked
/// at their full comptime span and skipped positions are masked rather than branched around, so
/// every group in a plane takes the same turns. With
/// [NeighbourWalk::Sliding](crate::collab::kernels::fused::NeighbourWalk::Sliding) each rectangle is
/// walked column by column. Every walk scores the same positions in the same rectangles and keeps
/// the same matches.
///
/// A plane-wide vote skips a position no group in the plane needs, so a turn that is masked in
/// every group costs one vote.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn trajectory_search<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    anchor: &Array<f32>,
    anchor_x: u32,
    anchor_y: u32,
    sub: u32,
    scale: f32,
    c_min: f32,
    member_d: &mut Array<f32>,
    member_pos: &mut Array<u32>,
    #[comptime] first: u32,
    #[comptime] tail: u32,
    #[comptime] walk: Walk,
    #[comptime] neighbour_walk: NeighbourWalk,
    #[comptime] radius: u32,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] shape: SearchShape,
) {
    let max_x = comptime!(shape.width - PATCH_SIZE);
    let max_y = comptime!(shape.height - PATCH_SIZE);
    let n_neighbours = comptime!(2 * radius);
    let covers = comptime!(blksize.div_ceil(blk_step));
    let max_rects = comptime!(covers * covers);
    let sliding = comptime!(neighbour_walk == NeighbourWalk::Sliding);
    let reference = pack_reference::<S>(anchor, shape);

    let bx_hi = (anchor_x / blk_step).min(blocks_x - 1);
    let by_hi = (anchor_y / blk_step).min(blocks_y - 1);
    let bx_lo_covering = covering_lo(anchor_x, blksize, blk_step);
    let by_lo_covering = covering_lo(anchor_y, blksize, blk_step);
    let bx_lo = u32::min(bx_lo_covering, bx_hi);
    let by_lo = u32::min(by_lo_covering, by_hi);

    let mut t = 0u32;
    while t < n_neighbours {
        let slot = neighbour_slots[t as usize];
        let packed_t = t + 1u32;

        let mut frame_d = UNFILLED_DISTANCE.runtime();
        let mut frame_pos = 0u32;
        let mut frame_rect = 0u32;

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

                    if comptime!(sliding) {
                        let rect_index = comptime!(iy * covers + ix);
                        let bounds = SearchRect {
                            left: t_left,
                            right: t_right,
                            top: t_top,
                            bottom: t_bot,
                        };
                        sliding_rect_walk(
                            ring,
                            search_ring,
                            anchor,
                            &reference,
                            &seen_left,
                            &seen_right,
                            &seen_top,
                            &seen_bot,
                            member_pos,
                            &bounds,
                            slot,
                            sub,
                            scale,
                            packed_t,
                            block_scored,
                            &mut frame_d,
                            &mut frame_rect,
                            &mut frame_pos,
                            rect_index,
                            true,
                            max_rects,
                            first,
                            refine,
                            shape,
                        );
                    } else {
                        let span = comptime!(2 * refine + 1);
                        for dy in 0..span {
                            let wanted_y = t_top + dy;
                            let candidate_y = u32::min(wanted_y, t_bot);

                            for dx in 0..span {
                                let wanted_x = t_left + dx;
                                let in_rect = wanted_x <= t_right && wanted_y <= t_bot;
                                let candidate_x = u32::min(wanted_x, t_right);
                                let packed = pack_pos_t(candidate_x, candidate_y, packed_t);

                                // The per-row skip spans run slower on this walk than testing each
                                // position against every rectangle and member.
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
                                    let scored = candidate_distance(
                                        ring,
                                        search_ring,
                                        anchor,
                                        &reference,
                                        candidate_x,
                                        candidate_y,
                                        slot,
                                        sub,
                                        scale,
                                        shape,
                                    );
                                    let dist = select(live_pos, scored, UNFILLED_DISTANCE);
                                    let better = dist < frame_d;
                                    frame_d = select(better, dist, frame_d);
                                    frame_pos = select(better, packed, frame_pos);
                                }
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
                            let inside_x =
                                t_left >= seen_left[s as usize] && t_right <= seen_right[s as usize];
                            let inside_y = t_top >= seen_top[s as usize] && t_bot <= seen_bot[s as usize];
                            if inside_x && inside_y {
                                contained = true;
                            }
                        }

                        if !contained {
                            if comptime!(sliding) {
                                let rect_index = comptime!(iy * covers + ix);
                                let bounds = SearchRect {
                                    left: t_left,
                                    right: t_right,
                                    top: t_top,
                                    bottom: t_bot,
                                };
                                sliding_rect_walk(
                                    ring,
                                    search_ring,
                                    anchor,
                                    &reference,
                                    &seen_left,
                                    &seen_right,
                                    &seen_top,
                                    &seen_bot,
                                    member_pos,
                                    &bounds,
                                    slot,
                                    sub,
                                    scale,
                                    packed_t,
                                    true,
                                    &mut frame_d,
                                    &mut frame_rect,
                                    &mut frame_pos,
                                    rect_index,
                                    false,
                                    max_rects,
                                    first,
                                    refine,
                                    shape,
                                );
                            } else {
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
                                            let dist = candidate_distance(
                                                ring,
                                                search_ring,
                                                anchor,
                                                &reference,
                                                candidate_x,
                                                candidate_y,
                                                slot,
                                                sub,
                                                scale,
                                                shape,
                                            );
                                            if dist < frame_d {
                                                frame_d = dist;
                                                frame_pos = packed;
                                            }
                                        }

                                        candidate_x += 1u32;
                                    }

                                    candidate_y += 1u32;
                                }
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
        let mut carry_d = frame_d;
        let mut carry_pos = frame_pos;
        #[unroll]
        for j in 0..tail {
            let slot_index = comptime!(first + j) as usize;
            let held_d = member_d[slot_index];
            let held_pos = member_pos[slot_index];
            let smaller = carry_d < held_d;
            member_d[slot_index] = select(smaller, carry_d, held_d);
            member_pos[slot_index] = select(smaller, carry_pos, held_pos);
            carry_d = select(smaller, held_d, carry_d);
            carry_pos = select(smaller, held_pos, carry_pos);
        }

        t += 1u32;
    }
}

#[cfg(test)]
mod tests {
    use super::covering_lo_host;
    use crate::nl4d::harness::covering_blocks;

    #[test]
    fn covering_lo_matches_the_harness_across_a_range_of_geometries() {
        for (blksize, overlap) in [(16u32, 8u32), (16, 12), (32, 24), (8, 4), (16, 0)] {
            let step = blksize - overlap;
            let blocks = 8u32;
            for patch_start in (0..blocks * step).step_by(3) {
                let (expected_first, last) = covering_blocks(patch_start, blksize, step, blocks);
                let got_first = covering_lo_host(patch_start, blksize, step).min(last);
                assert_eq!(
                    got_first, expected_first,
                    "blksize={blksize} step={step} p={patch_start}: covering_lo disagrees with the \
                     harness's covering_blocks"
                );
            }
        }
    }
}
