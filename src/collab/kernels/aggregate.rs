use cubecl::prelude::*;
use cubecl::terminate;

use crate::collab::kernels::transforms::safe_reciprocal;
use crate::collab::{PATCH_AREA, PATCH_SIZE, STEP};

/// Blends every reference patch's filtered output back onto one frame
/// plane, weighted by how much its group agreed on it.
///
/// Reference patches sit on a grid with stride `STEP` but are
/// `PATCH_SIZE` pixels wide, so most pixels fall inside more than one
/// reference patch. One thread owns one output pixel, finds every
/// reference whose patch covers it, and writes the weighted mean of
/// their filtered values there. Each covering reference contributes its
/// filtered pixel scaled by `group_weight`, the inverse-variance weight
/// [`crate::collab::kernels::filter_ht::collab_filter_ht`] computed for
/// that group, so a group whose content agreed more strongly counts for
/// more of the output.
///
/// # Finding the covering references
///
/// The host-side `geometry::ref_pos` places reference `i` at `(i *
/// STEP).min(dim - PATCH_SIZE)` on each axis, clamping the last
/// reference so its patch never runs past the frame edge. Away from that
/// edge, the references covering a pixel `x` form a short contiguous run
/// computed directly from `x`, at most two indices wide because `STEP`
/// is half of `PATCH_SIZE`. The clamp breaks that regularity at the far
/// edge. Pulling the last reference back to stay inside the frame can
/// leave it closer to its neighbour than `STEP` pixels, so it can cover
/// pixels the contiguous run never reaches. This kernel always tests the
/// clamped last index on top of that run, and skips it when the run
/// already reached it, so a reference covering a pixel through both
/// paths at once is never added twice.
///
/// Every candidate index, found either way, still has to pass the real
/// coverage test, its clamped patch position at or before the pixel and
/// its far edge past it, before it contributes anything. A candidate
/// that turns out not to cover the pixel simply adds nothing.
///
/// # Why the weight sum is never zero
///
/// Every pixel is covered by one to three references per axis, so at
/// most nine in two dimensions, and never zero, for any frame at least
/// `PATCH_SIZE` pixels on a side. This kernel divides by the accumulated
/// weight relying on that.
///
/// # Buffers
///
/// `filtered` holds `refs * PATCH_AREA` lines, one 8x8 patch per
/// reference in raster order, the layout `collab_filter_ht` writes.
/// `group_weight` holds one weight per reference, also written by
/// `collab_filter_ht`. `output` is a single frame plane of `width *
/// height` lines, not a ring buffer, so writing to it needs no frame
/// index.
#[cube(launch_unchecked)]
pub fn collab_aggregate<N: Size>(
    filtered: &Array<Vector<f32, N>>,
    group_weight: &Array<f32>,
    output: &mut Array<Vector<f32, N>>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] refs_x: u32,
    #[comptime] refs_y: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    // Up to three candidate reference indices per axis, in the layout
    // the doc above describes: the contiguous run's low and high ends,
    // then the clamped last index when the run doesn't already reach
    // it. An unused slot holds `refs_x`/`refs_y`, one past the last
    // valid index, so the coverage check below skips it without a
    // separate validity flag.
    //
    // The `.into()` calls below are what let cubecl unify an if/else
    // branch built from a runtime expression with one built from a
    // `#[comptime]` value, since both arms have to expand to the same
    // `NativeExpand<u32>`. Clippy cannot see that requirement.
    #[allow(clippy::useless_conversion)]
    let ix_lo = if x >= PATCH_SIZE - 1 {
        (x - (PATCH_SIZE - 1)).div_ceil(STEP)
    } else {
        0u32.into()
    };
    let ix_hi = (x / STEP).min(refs_x - 1);

    let mut cand_x = Array::<u32>::new(3usize);
    cand_x[0] = ix_lo;
    #[allow(clippy::useless_conversion)]
    let cand_x1 = if ix_lo < ix_hi {
        ix_lo + 1u32
    } else {
        refs_x.into()
    };
    #[allow(clippy::useless_conversion)]
    let cand_x2: u32 = if refs_x - 1u32 > ix_hi {
        (refs_x - 1u32).into()
    } else {
        refs_x.into()
    };
    cand_x[1] = cand_x1;
    cand_x[2] = cand_x2;

    #[allow(clippy::useless_conversion)]
    let iy_lo = if y >= PATCH_SIZE - 1 {
        (y - (PATCH_SIZE - 1)).div_ceil(STEP)
    } else {
        0u32.into()
    };
    let iy_hi = (y / STEP).min(refs_y - 1);

    let mut cand_y = Array::<u32>::new(3usize);
    cand_y[0] = iy_lo;
    #[allow(clippy::useless_conversion)]
    let cand_y1 = if iy_lo < iy_hi {
        iy_lo + 1u32
    } else {
        refs_y.into()
    };
    #[allow(clippy::useless_conversion)]
    let cand_y2: u32 = if refs_y - 1u32 > iy_hi {
        (refs_y - 1u32).into()
    } else {
        refs_y.into()
    };
    cand_y[1] = cand_y1;
    cand_y[2] = cand_y2;

    let mut acc = Vector::<f32, N>::empty();
    let mut wsum = 0.0f32;

    #[unroll]
    for xi in 0..3u32 {
        let ix = cand_x[xi as usize];
        if ix < refs_x {
            let rx = (ix * STEP).min(width - PATCH_SIZE);
            if rx <= x && x < rx + PATCH_SIZE {
                #[unroll]
                for yi in 0..3u32 {
                    let iy = cand_y[yi as usize];
                    if iy < refs_y {
                        let ry = (iy * STEP).min(height - PATCH_SIZE);
                        if ry <= y && y < ry + PATCH_SIZE {
                            let ref_idx = iy * refs_x + ix;
                            let w = group_weight[ref_idx as usize];
                            let patch_idx = ref_idx * PATCH_AREA + (y - ry) * PATCH_SIZE + (x - rx);
                            let line_w = Vector::<f32, N>::empty().fill(w);
                            acc += filtered[patch_idx as usize] * line_w;
                            wsum += w;
                        }
                    }
                }
            }
        }
    }

    // wsum is a sum of one to nine group_weight values, each already
    // guaranteed finite and non-negative by collab_filter_ht and
    // collab_filter_wiener's own guards, and the doc above establishes
    // it is never exactly zero either. safe_reciprocal still checks
    // explicitly rather than trusting those guarantees to survive
    // whatever a given GPU driver does with a NaN operand to `f32::max`,
    // so this stays finite even if that chain of guarantees is ever
    // broken upstream.
    let inv = safe_reciprocal(wsum, 1e-12f32);
    let line_inv = Vector::<f32, N>::empty().fill(inv);
    output[(y * width + x) as usize] = acc * line_inv;
}
