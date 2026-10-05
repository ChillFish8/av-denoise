use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{plane_ssd_reduce8, shift_insert8, shift_insert8_gated};
use crate::nlmeans::kernels::helpers::read_line;

/// The lowest block index whose span contains the patch at `p` on one axis.
///
/// Block `b` spans `b * step..b * step + blksize`, so it contains `p..p + PATCH_SIZE` when
/// `b * step + blksize >= p + PATCH_SIZE`. The caller clamps the result to the highest covering
/// block, `p / step`. It mirrors `covering_blocks` in `nl4d/harness/score.rs`.
#[cube]
pub(crate) fn covering_lo(p: u32, #[comptime] blksize: u32, #[comptime] step: u32) -> u32 {
    let overhang = u32::max(p + PATCH_SIZE, blksize) - blksize;
    overhang.div_ceil(step)
}

/// Host mirror of `covering_lo`.
#[cfg(test)]
fn covering_lo_host(p: u32, blksize: u32, step: u32) -> u32 {
    let overhang = u32::max(p + PATCH_SIZE, blksize) - blksize;
    overhang.div_ceil(step)
}

/// The distance from the reference patch to the candidate with top-left `(x, y)` in frame `slot`.
///
/// Each lane sums its own column and `plane_ssd_reduce8` completes the distance with shuffles, so
/// every lane of the group must call this, even when the result is discarded.
#[cube]
pub(crate) fn candidate_distance<N: Size>(
    ring: &Array<Vector<f32, N>>,
    current: &Array<f32>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    scale: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> f32 {
    let mut partial = 0.0f32;
    #[unroll]
    for r in 0..PATCH_SIZE {
        let pixel = read_line(ring, x + sub, y + r, slot, width, height);
        #[unroll]
        for c in 0..channels {
            let diff = current[(r * channels + c) as usize] - pixel[c as usize];
            partial += diff * diff;
        }
    }
    plane_ssd_reduce8(partial) * scale
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
pub(crate) fn spatial_search<N: Size>(
    ring: &Array<Vector<f32, N>>,
    current: &Array<f32>,
    rx: u32,
    ry: u32,
    centre_slot: u32,
    sub: u32,
    base: u32,
    scale: f32,
    best_d: &mut f32,
    best_pos: &mut u32,
    #[comptime] warp_uniform: bool,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> u32 {
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);

    let s_left = clamp_top_left(rx as i32 - spatial_radius as i32, max_x);
    let s_right = clamp_top_left(rx as i32 + spatial_radius as i32, max_x);
    let s_top = clamp_top_left(ry as i32 - spatial_radius as i32, max_y);
    let s_bot = clamp_top_left(ry as i32 + spatial_radius as i32, max_y);

    // The self-match takes a sentinel below every real distance, so it holds slot 0 even on flat
    // content where every candidate ties.
    if warp_uniform {
        // The unclipped span covers every position the clipped walk visits, in the same order, and
        // its comptime size gives every group in the warp the same number of turns.
        let span = comptime!(2 * spatial_radius + 1);
        for dy in 0..span {
            for dx in 0..span {
                let wanted_y = s_top + dy;
                let wanted_x = s_left + dx;
                let live_pos = wanted_x <= s_right && wanted_y <= s_bot;

                // A masked turn still reads, so it is pinned to the last live position.
                let candidate_x = u32::min(wanted_x, s_right);
                let candidate_y = u32::min(wanted_y, s_bot);

                let scored = candidate_distance(
                    ring,
                    current,
                    candidate_x,
                    candidate_y,
                    centre_slot,
                    sub,
                    scale,
                    width,
                    height,
                    channels,
                );

                // The gated insert branches on a group-local distance before it shuffles, which is
                // the divergence this path avoids.
                let mut dist = select(live_pos, scored, 3.0e38f32);

                // A masked turn pinned onto the reference would otherwise plant a second
                // self-match.
                if live_pos && candidate_x == rx && candidate_y == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8(
                    best_d,
                    best_pos,
                    dist,
                    pack_pos_t(candidate_x, candidate_y, 0u32),
                    sub,
                );
            }
        }
    } else {
        let mut candidate_y = s_top;
        while candidate_y <= s_bot {
            let mut candidate_x = s_left;
            while candidate_x <= s_right {
                let mut dist = candidate_distance(
                    ring,
                    current,
                    candidate_x,
                    candidate_y,
                    centre_slot,
                    sub,
                    scale,
                    width,
                    height,
                    channels,
                );
                if candidate_x == rx && candidate_y == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8_gated(
                    best_d,
                    best_pos,
                    dist,
                    pack_pos_t(candidate_x, candidate_y, 0u32),
                    sub,
                    base,
                );
                candidate_x += 1u32;
            }
            candidate_y += 1u32;
        }
    }

    (s_right - s_left + 1u32) * (s_bot - s_top + 1u32)
}

/// Builds one volume's frames, the anchor patch's best match in each neighbour frame.
///
/// A neighbour's match is the lowest-distance position inside the refine rectangles of the motion
/// blocks covering the anchor, each around where that block's vector moves the anchor. The
/// `tail` lowest of those per-frame matches land in `member_d` and `member_pos` at
/// `first..first + tail`, ascending. A slot no frame filled keeps the `3.0e38` it starts with.
///
/// A position already held in `member_pos[..first]` is skipped, so no patch enters the group
/// twice. A block below `c_min` is skipped too, and a position reached by two covering blocks
/// is scored once.
///
/// With `warp_uniform` the rectangles are walked at their full comptime span and skipped
/// positions are masked rather than branched around, so every group in a warp takes the same
/// turns. Both walks score the same positions in the same order and keep the same matches.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn trajectory_search<N: Size>(
    ring: &Array<Vector<f32, N>>,
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
    #[comptime] warp_uniform: bool,
    #[comptime] radius: u32,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let n_neighbours = comptime!(2 * radius);
    let covers = comptime!(blksize.div_ceil(blk_step));
    let max_rects = comptime!(covers * covers);

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

        let mut frame_d = 3.0e38f32;
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

                if warp_uniform {
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
                        for dx in 0..span {
                            let wanted_y = t_top + dy;
                            let wanted_x = t_left + dx;
                            let in_rect = wanted_x <= t_right && wanted_y <= t_bot;
                            let candidate_x = u32::min(wanted_x, t_right);
                            let candidate_y = u32::min(wanted_y, t_bot);
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
                            let scored = candidate_distance(
                                ring,
                                anchor,
                                candidate_x,
                                candidate_y,
                                slot,
                                sub,
                                scale,
                                width,
                                height,
                                channels,
                            );
                            let dist = select(live_pos, scored, 3.0e38f32);
                            let better = dist < frame_d;
                            frame_d = select(better, dist, frame_d);
                            frame_pos = select(better, packed, frame_pos);
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

                        let mut candidate_y = t_top;
                        while candidate_y <= t_bot {
                            let mut candidate_x = t_left;
                            while candidate_x <= t_right {
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

                                if !skipped {
                                    let dist = candidate_distance(
                                        ring,
                                        anchor,
                                        candidate_x,
                                        candidate_y,
                                        slot,
                                        sub,
                                        scale,
                                        width,
                                        height,
                                        channels,
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
    use super::{PATCH_SIZE, covering_lo_host};

    /// A copy of the harness's `covering_blocks`, which its private `score` module keeps out of reach.
    fn covering_blocks_host(patch_start: u32, blksize: u32, step: u32, blocks: u32) -> (u32, u32) {
        let last_block = (patch_start / step).min(blocks - 1);
        let first_block = if patch_start + PATCH_SIZE <= blksize {
            0
        } else {
            (patch_start + PATCH_SIZE - blksize).div_ceil(step)
        };

        (first_block.min(last_block), last_block)
    }

    #[test]
    fn covering_lo_matches_the_harness_across_a_range_of_geometries() {
        for (blksize, overlap) in [(16u32, 8u32), (16, 12), (32, 24), (8, 4), (16, 0)] {
            let step = blksize - overlap;
            let blocks = 8u32;
            for patch_start in (0..blocks * step).step_by(3) {
                let (expected_first, last) = covering_blocks_host(patch_start, blksize, step, blocks);
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
