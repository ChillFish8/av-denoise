use cubecl::prelude::*;

use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{plane_ssd_reduce8, shift_insert8, shift_insert8_gated};
use crate::collab::PATCH_SIZE;
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
    noise_floor: f32,
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
    plane_ssd_reduce8(partial) * scale - noise_floor
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
    noise_floor: f32,
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
                    noise_floor,
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
                    noise_floor,
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
