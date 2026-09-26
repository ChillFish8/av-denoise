use cubecl::prelude::*;

use super::subpel::{already_claimed, refine_subpel};
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::{clamp_top_left, clamp_top_left_within, pack_pos_t};
use crate::collab::kernels::plane_ops::{plane_ssd_reduce8, shift_insert8, shift_insert8_gated};
use crate::nlmeans::kernels::helpers::read_line;

/// The lowest block index whose span contains the patch at `p` on one axis.
///
/// Block `b` spans `b * step..b * step + blksize`, so the patch
/// `p..p + PATCH_SIZE` needs `b * step + blksize >= p + PATCH_SIZE`.
/// The highest such block is `p / step`, which the caller clamps to the
/// grid and uses as the low end's ceiling.
///
/// This mirrors `covering_blocks` in the `mc_accuracy` bench's harness
/// module (`av-denoise-core/benches/harness/score.rs`), which the tests
/// below reproduce on the host to check the two stay in step.
#[cube]
pub(crate) fn covering_lo(p: u32, #[comptime] blksize: u32, #[comptime] step: u32) -> u32 {
    let past = u32::max(p + PATCH_SIZE, blksize) - blksize;
    past.div_ceil(step)
}

/// The host mirror of [covering_lo], for tests that cannot launch a
/// kernel.
#[cfg(test)]
fn covering_lo_host(p: u32, blksize: u32, step: u32) -> u32 {
    let past = u32::max(p + PATCH_SIZE, blksize) - blksize;
    past.div_ceil(step)
}

/// The distance from the reference patch to the candidate whose
/// top-left pixel is `(x, y)` in frame `slot`.
///
/// Each lane holds one column of the reference patch and reads the
/// matching column of the candidate, so the eight per-lane partials
/// only become a whole-patch distance through
/// [plane_ssd_reduce8]. That reduction shuffles, so every lane of the
/// group has to reach it. Callers that end up discarding the result
/// still call this and drop the value afterwards rather than branching
/// around it.
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
        let px = read_line(ring, x + sub, y + r, slot, width, height);
        #[unroll]
        for c in 0..channels {
            let d = current[(r * channels + c) as usize] - px[c as usize];
            partial += d * d;
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

    // The reference patch scores the lowest distance there is, which on
    // textured content is enough to reach slot 0 on its own. On flat
    // content every candidate scores that same distance, and
    // `shift_insert8` leaves a tie with whichever candidate reached the
    // slot first. A sentinel below every real distance pins the
    // self-match whatever ties around it.
    if warp_uniform {
        // The clipped rectangle is never wider than the unclipped one,
        // so walking the unclipped span covers every position the other
        // path visits, in the same order, and the rest are masked. The
        // span is comptime, so every group in the warp takes the same
        // number of turns.
        let span = comptime!(2 * spatial_radius + 1);
        for dy in 0..span {
            for dx in 0..span {
                let wanted_y = s_top + dy;
                let wanted_x = s_left + dx;
                let live_pos = wanted_x <= s_right && wanted_y <= s_bot;
                // A masked turn still reads, so it is pinned to the last
                // live position rather than left to run off the frame.
                let cx = u32::min(wanted_x, s_right);
                let cy = u32::min(wanted_y, s_bot);

                let scored = candidate_distance(
                    ring,
                    current,
                    cx,
                    cy,
                    centre_slot,
                    sub,
                    scale,
                    width,
                    height,
                    channels,
                );
                // Only the branchless part of the insert is shared. The
                // gated form tests a group-local distance before it
                // shuffles, which is exactly the divergence this path
                // exists to avoid.
                let mut dist = select(live_pos, scored, 3.0e38f32);
                // A masked turn can land on the reference's own position
                // once it has been pinned, so `live_pos` has to gate the
                // sentinel too, or a dead turn would plant a second
                // self-match in the group.
                if live_pos && cx == rx && cy == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8(best_d, best_pos, dist, pack_pos_t(cx, cy, 0u32), sub);
            }
        }
    } else {
        let mut cy = s_top;
        while cy <= s_bot {
            let mut cx = s_left;
            while cx <= s_right {
                let mut dist = candidate_distance(
                    ring,
                    current,
                    cx,
                    cy,
                    centre_slot,
                    sub,
                    scale,
                    width,
                    height,
                    channels,
                );
                if cx == rx && cy == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8_gated(best_d, best_pos, dist, pack_pos_t(cx, cy, 0u32), sub, base);
                cx += 1u32;
            }
            cy += 1u32;
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
/// A position already held in `member_pos[..first]` at the same phase is skipped, so no patch
/// enters the group twice. A block below `c_min` is skipped too, and a position reached by two
/// covering blocks is scored once.
///
/// With `subpel` above 0 each frame's match is then refined through
/// [refine_subpel](crate::collab::kernels::fused::subpel::refine_subpel), and its phase lands in
/// `member_phase` beside its position. The refine rectangles keep one pixel clear of every edge,
/// so each interpolated read stays in frame.
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
    phase_ring: &Array<Vector<f32, N>>,
    phase_gain: &Array<f32>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    anchor: &Array<f32>,
    anchor_x: u32,
    anchor_y: u32,
    sub: u32,
    scale: f32,
    c_min: f32,
    noise_px: f32,
    member_d: &mut Array<f32>,
    member_pos: &mut Array<u32>,
    member_phase: &mut Array<u32>,
    #[comptime] first: u32,
    #[comptime] tail: u32,
    #[comptime] warp_uniform: bool,
    #[comptime] subpel: u32,
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
    let edge = comptime!(if subpel > 0 { 1u32 } else { 0u32 });
    let rect_max_x = comptime!(max_x - edge);
    let rect_max_y = comptime!(max_y - edge);

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
        let mut frame_phase = 0u32;

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
                    let cbx = u32::min(wanted_bx, bx_hi);
                    let cby = u32::min(wanted_by, by_hi);
                    let block = cby * blocks_x + cbx;
                    let conf = confidence[(t * conf_stride + block) as usize];
                    let block_scored = block_live && conf >= c_min;

                    let mv = (t * mv_stride + block * 2u32) as usize;
                    let px0 = anchor_x as i32 + mv_field[mv];
                    let py0 = anchor_y as i32 + mv_field[mv + 1];

                    let t_left = clamp_top_left_within(px0 - refine as i32, edge, rect_max_x);
                    let t_right = clamp_top_left_within(px0 + refine as i32, edge, rect_max_x);
                    let t_top = clamp_top_left_within(py0 - refine as i32, edge, rect_max_y);
                    let t_bot = clamp_top_left_within(py0 + refine as i32, edge, rect_max_y);

                    let span = comptime!(2 * refine + 1);
                    for dy in 0..span {
                        for dx in 0..span {
                            let wanted_y = t_top + dy;
                            let wanted_x = t_left + dx;
                            let in_rect = wanted_x <= t_right && wanted_y <= t_bot;
                            let nx = u32::min(wanted_x, t_right);
                            let ny = u32::min(wanted_y, t_bot);
                            let packed = pack_pos_t(nx, ny, packed_t);

                            let mut skipped = false;
                            #[unroll]
                            for s in 0..max_rects {
                                if nx >= seen_left[s as usize]
                                    && nx <= seen_right[s as usize]
                                    && ny >= seen_top[s as usize]
                                    && ny <= seen_bot[s as usize]
                                {
                                    skipped = true;
                                }
                            }
                            if already_claimed(member_pos, member_phase, packed, 0u32, first) {
                                skipped = true;
                            }

                            let live_pos = block_scored && in_rect && !skipped;
                            let scored = candidate_distance(
                                ring, anchor, nx, ny, slot, sub, scale, width, height, channels,
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
                    let conf = confidence[(t * conf_stride + block) as usize];
                    if conf >= c_min {
                        let mv = (t * mv_stride + block * 2u32) as usize;
                        let px0 = anchor_x as i32 + mv_field[mv];
                        let py0 = anchor_y as i32 + mv_field[mv + 1];

                        let t_left = clamp_top_left_within(px0 - refine as i32, edge, rect_max_x);
                        let t_right = clamp_top_left_within(px0 + refine as i32, edge, rect_max_x);
                        let t_top = clamp_top_left_within(py0 - refine as i32, edge, rect_max_y);
                        let t_bot = clamp_top_left_within(py0 + refine as i32, edge, rect_max_y);

                        let mut ny = t_top;
                        while ny <= t_bot {
                            let mut nx = t_left;
                            while nx <= t_right {
                                let packed = pack_pos_t(nx, ny, packed_t);

                                let mut skipped = false;
                                #[unroll]
                                for s in 0..max_rects {
                                    if nx >= seen_left[s as usize]
                                        && nx <= seen_right[s as usize]
                                        && ny >= seen_top[s as usize]
                                        && ny <= seen_bot[s as usize]
                                    {
                                        skipped = true;
                                    }
                                }
                                if already_claimed(member_pos, member_phase, packed, 0u32, first) {
                                    skipped = true;
                                }

                                if !skipped {
                                    let dist = candidate_distance(
                                        ring, anchor, nx, ny, slot, sub, scale, width, height, channels,
                                    );
                                    if dist < frame_d {
                                        frame_d = dist;
                                        frame_pos = packed;
                                    }
                                }

                                nx += 1u32;
                            }
                            ny += 1u32;
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

        if comptime!(subpel > 0) {
            let found = frame_d < 1.0e38f32;
            let inner_x = u32::max(anchor_x, 1u32);
            let inner_y = u32::max(anchor_y, 1u32);
            let safe_x = u32::min(inner_x, comptime!(max_x - 1));
            let safe_y = u32::min(inner_y, comptime!(max_y - 1));
            refine_subpel(
                phase_ring,
                phase_gain,
                anchor,
                member_pos,
                member_phase,
                found,
                safe_x,
                safe_y,
                slot,
                packed_t,
                sub,
                scale,
                noise_px,
                &mut frame_d,
                &mut frame_pos,
                &mut frame_phase,
                first,
                subpel,
                width,
                height,
                channels,
            );
        }

        // The frame's match joins the volume's ascending list and pushes the worst out.
        let mut carry_d = frame_d;
        let mut carry_pos = frame_pos;
        let mut carry_phase = frame_phase;
        #[unroll]
        for j in 0..tail {
            let slot_index = comptime!(first + j) as usize;
            let held_d = member_d[slot_index];
            let held_pos = member_pos[slot_index];
            let held_phase = member_phase[slot_index];
            let smaller = carry_d < held_d;
            member_d[slot_index] = select(smaller, carry_d, held_d);
            member_pos[slot_index] = select(smaller, carry_pos, held_pos);
            member_phase[slot_index] = select(smaller, carry_phase, held_phase);
            carry_d = select(smaller, held_d, carry_d);
            carry_pos = select(smaller, held_pos, carry_pos);
            carry_phase = select(smaller, held_phase, carry_phase);
        }

        t += 1u32;
    }
}

#[cfg(test)]
mod tests {
    use super::covering_lo_host;

    /// The host mirror of `covering_blocks` in the `mc_accuracy` bench's
    /// harness (`benches/harness/score.rs`). Reproduced here, rather than
    /// imported, because that module lives outside the crate as bench-only
    /// code and cannot be a test dependency of the library.
    ///
    /// This pins the kernel's arithmetic against the harness's read of the
    /// same geometry rather than launching a real kernel, so it catches the
    /// two formulas drifting apart on paper but says nothing about whether
    /// [super::covering_lo] compiles or runs correctly on a GPU; the
    /// integration tests in `nl4d::tests` cover that by driving the whole
    /// pipeline.
    fn covering_blocks_host(p: u32, blksize: u32, step: u32, blocks: u32) -> (u32, u32) {
        let hi = (p / step).min(blocks - 1);
        let lo = if p + super::PATCH_SIZE <= blksize {
            0
        } else {
            (p + super::PATCH_SIZE - blksize).div_ceil(step)
        };
        (lo.min(hi), hi)
    }

    #[test]
    fn covering_lo_matches_the_harness_across_a_range_of_geometries() {
        for (blksize, overlap) in [(16u32, 8u32), (16, 12), (32, 24), (8, 4), (16, 0)] {
            let step = blksize - overlap;
            let blocks = 8u32;
            for p in (0..blocks * step).step_by(3) {
                let (expect_lo, hi) = covering_blocks_host(p, blksize, step, blocks);
                let got_lo = covering_lo_host(p, blksize, step).min(hi);
                assert_eq!(
                    got_lo, expect_lo,
                    "blksize={blksize} step={step} p={p}: covering_lo disagrees with the \
                     harness's covering_blocks"
                );
            }
        }
    }
}
