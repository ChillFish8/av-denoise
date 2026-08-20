use cubecl::prelude::*;

use super::group::{clamp_top_left, pack_pos};
use crate::collab::{PATCH_AREA, PATCH_SIZE, STEP};
use crate::nlmeans::kernels::helpers::{channel_scale, read_line};

// A compile-time tie from the `stride = 32u32` literal in the argmin
// reduction below back to `PATCH_AREA`. That literal can't be written as
// `PATCH_AREA / 2` directly (see the comment at its use site), so this is
// what catches it going stale if `PATCH_SIZE` ever changes.
const _: () = assert!(
    PATCH_AREA / 2 == 32,
    "update the `stride = 32u32` literal in collab_group_temporal to PATCH_AREA / 2"
);

// The distance a candidate carries once it is out of the running is the
// literal `3.0e38`, written out at each use below rather than named.
// Every real distance is a sum of at most `PATCH_AREA` squared
// differences between values in `[0, 1]`, scaled by at most 3, so it
// never exceeds 192. `3.0e38` sits far above that and just below
// `f32::MAX`, which leaves it comparing greater than any live candidate
// without overflowing the argmin reduction. It is spelled as a literal
// rather than a `const`, `f32::MAX`, or `f32::INFINITY` because cubecl
// treats all three as compile-time-only, and the selection loop needs a
// genuine mutable runtime variable to start from.

/// The extra per-member variance a temporal candidate's motion-block
/// confidence implies, the value [`collab_group_temporal`] writes into
/// `member_sig2` for that member.
///
/// A poorly matched motion block is treated as a noisier observation of
/// the true patch rather than a different patch, so its confidence `c`
/// turns into extra variance instead of an admission decision. `thsad`
/// is the same SAD threshold that block's confidence score was derived
/// from, in normalised SAD units (see
/// [`crate::nlmeans::motion::thsad`]), and `blksize` is the motion
/// block's side length in pixels.
///
/// ```text
/// E^2      = thsad^2 * (1 - c) / (1 + c)
/// eps      = E / blksize^2
/// sigma_m2 = (pi / 2) * eps^2
/// ```
///
/// `c = 1`, a perfect match, gives `sigma_m2 = 0` exactly. Lower
/// confidence inflates it.
///
/// A per-plane scale on this result used to be configurable. It is not,
/// because no value other than 1.0 is usable. A ladder over 1, 2, 4, 8
/// and 16, recorded in `data/nl4d_mismatch_ladder/README.md`, found the
/// filter corrupts frames above 2, collapsing high-contrast edges toward
/// black through a cause that is still unexplained, and found the whole
/// usable range too narrow to matter. Turning the mechanism off is
/// [`crate::collab::kernels::filter_ht::collab_filter_ht`]'s
/// `use_member_sigma = false` path.
///
/// This function never runs for a centre-frame member. Those carry
/// `sigma_m2 = 0` exactly because they are not motion-predicted, so
/// there is no mismatch to model, and [`collab_group_temporal`] takes
/// that branch before ever calling this.
#[cube]
fn mismatch_sigma2(confidence: f32, thsad: f32, blksize_area: f32) -> f32 {
    let ratio = (1.0f32 - confidence) / (1.0f32 + confidence);
    let e2 = thsad * thsad * ratio;
    let eps = f32::sqrt(e2) / blksize_area;
    std::f32::consts::FRAC_PI_2 * eps * eps
}

/// Finds the K most similar patches to each reference patch, searching
/// the centre frame spatially and each neighbour frame in the ring
/// around where motion compensation predicts the reference patch moved.
///
/// One cube owns one reference patch, and its `CubeDim::new_2d(8, 8)`
/// threads score the search space one candidate per thread. A thread
/// walks its own candidate's 64 pixels start to finish on its own, so
/// scoring needs no barrier and no cross-thread reduction at all. The
/// candidates outnumber the threads, so each thread takes a strided
/// slice of the search space and scores several in turn.
///
/// The reference patch is staged in shared memory first, because all 64
/// threads read all 64 of its pixels. Candidate pixels are read straight
/// from global memory, which measured faster than staging the window in
/// shared memory. The search windows of neighbouring reference patches
/// overlap heavily at a step of 4, so the cache already serves those
/// reads well, and a shared-memory tile only costs occupancy.
///
/// # Candidates
///
/// A candidate's index `ci` runs from `0` to `n_cand`, the centre
/// frame's `(2 * spatial_radius + 1)^2` window followed by a
/// `(2 * refine + 1)^2` refine window for each of the `2 * radius`
/// neighbour frames. `ci < n_spatial` reads a position in the centre
/// frame, the reference patch's own position plus an offset inside the
/// spatial window. `ci >= n_spatial` divides the remainder into
/// `n_refine`-sized blocks, one per neighbour, and searches around that
/// neighbour's predicted position for the reference patch, `rx, ry` plus
/// the motion vector at the block containing the reference patch, plus
/// an offset inside the refine window.
///
/// # Distance
///
/// A candidate's distance is the channel-scaled sum of squared pixel
/// differences over the whole patch, with `noise_floor` subtracted.
/// `noise_floor` is the distance two noisy copies of the same content
/// are expected to show by chance, so a genuine match isn't penalised
/// for the noise it carries. The floored value is never clamped to zero.
/// Subtracting a constant from every candidate shifts them all by the
/// same amount and does not change their relative order, so the
/// selection below stays a genuine similarity ranking even when
/// `noise_floor` is large enough to push most candidates' floored
/// distance negative.
///
/// # No admission gate
///
/// Every candidate that is not the pinned self-match stays in the
/// running whatever its distance, so the group always fills to `k_max`.
/// `c_min` is not an admission threshold on individual candidates, it
/// gates a whole neighbour frame at a block's worth of granularity, as a
/// compute saving. A neighbour block whose confidence sits below `c_min`
/// never runs the pixel comparison that would score its candidates, and
/// every one of them is retired outright, without the search ever
/// discovering that block is a poor match.
///
/// # Members carry a frame
///
/// A member is no longer just a packed position. Two frames can share a
/// physical `(x, y)`, and a patch matched in one is not the same member
/// as the same coordinates matched in another, so `member_frame` is
/// written alongside `member_pos`, and the whole selection, and its
/// duplicate check, compares `(pos, frame)` as a pair rather than
/// position alone. The self-match is still pinned into slot 0, with
/// `centre_slot` as its frame, and dedup against it, and against every
/// other clamped duplicate, follows the same pair rule.
///
/// # Selection
///
/// Once every candidate has a distance, the group's remaining members
/// are picked by `k_max - 1` rounds of argmin across the whole search
/// space. Each round every thread finds the best candidate in its own
/// slice, a tree reduction over shared memory folds those into the
/// round's winner, and that winner is retired before the next round
/// runs. Ties break toward the lower candidate index, which is the order
/// a linear scan would visit them in. Exactly-equal distances are common
/// on flat content, so this is what fixes which member a group keeps
/// rather than leaving it to thread scheduling.
///
/// # Clamped duplicates
///
/// Candidate positions are patch top-left coordinates clamped to `[0,
/// dim - 8]` on each axis, which is what keeps every member read fully
/// inside the frame. That clamping also means two different search
/// offsets near an edge can land on the same clamped position. Admitting
/// both would let one physical patch count twice toward the group, which
/// would look like stronger agreement than the group actually has.
/// Retiring a winner retires every candidate sharing its `(pos, frame)`
/// pair, and the self-match's pair is retired before the first round, so
/// no member is ever kept twice.
///
/// # Group size
///
/// The final member count is rounded down to the nearest power of two,
/// capped at `k_max`. The stack transform a filtered group later passes
/// through is only defined for power-of-two stack sizes, so a count of
/// 5, 6, or 7 members still only keeps 4 of them.
///
/// # Per-member mismatch variance
///
/// `member_sig2` is written alongside `member_pos`/`member_frame`, the
/// same `ref_idx * k_max + j` layout, and holds the extra variance
/// [`crate::collab::kernels::filter_ht::collab_filter_ht`] adds to
/// `sigma[c]^2` for that member when its own `use_member_sigma` is set.
/// A centre-frame member carries `0.0` exactly. A temporal member's
/// value comes from [`mismatch_sigma2`], run against the same
/// confidence that decided whether its neighbour block was gated,
/// `thsad` passed straight through, and `blksize` squared into the
/// block area `mismatch_sigma2` divides by.
///
/// Every candidate carries its source block's confidence alongside its
/// distance, in a `cand_conf` array shaped like `dist`/`posn`/`frm`.
/// The selection rounds below already read `posn`/`frm` off the
/// winning index once a round's argmin has picked it, so `cand_conf`
/// is read the same way, into a `top_c` array shaped like `top_p`/
/// `top_f`, and it is that saved confidence [`mismatch_sigma2`] runs
/// on, not a later re-read of `confidence` itself.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn collab_group_temporal<N: Size>(
    ring: &Array<Vector<f32, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    member_pos: &mut Array<u32>,
    member_frame: &mut Array<u32>,
    member_count: &mut Array<u32>,
    member_sig2: &mut Array<f32>,
    centre_slot: u32,
    neighbour_slots: &Array<u32>,
    noise_floor: f32,
    c_min: f32,
    thsad: f32,
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
    #[comptime] k_max: u32,
    #[comptime] spatial_radius: u32,
    #[comptime] refs_x: u32,
) {
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let tid = local_y * PATCH_SIZE + local_x;
    let ref_idx = CUBE_POS_Y * refs_x + CUBE_POS_X;

    let max_x = width - PATCH_SIZE;
    let max_y = height - PATCH_SIZE;
    let rx = (CUBE_POS_X * STEP).min(max_x);
    let ry = (CUBE_POS_Y * STEP).min(max_y);

    let window_side = comptime!(2 * spatial_radius + 1);
    let n_spatial = comptime!((2 * spatial_radius + 1) * (2 * spatial_radius + 1));
    let refine_side = comptime!(2 * refine + 1);
    let n_refine = comptime!((2 * refine + 1) * (2 * refine + 1));
    let n_cand = comptime!(n_spatial + 2 * radius * n_refine);

    let mut ref_patch = SharedMemory::<f32>::new(comptime!(PATCH_AREA * channels) as usize);
    let mut dist = SharedMemory::<f32>::new(n_cand as usize);
    let mut posn = SharedMemory::<u32>::new(n_cand as usize);
    let mut frm = SharedMemory::<u32>::new(n_cand as usize);
    let mut cand_conf = SharedMemory::<f32>::new(n_cand as usize);
    let mut red_d = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut red_i = SharedMemory::<u32>::new(PATCH_AREA as usize);
    let mut top_p = SharedMemory::<u32>::new(k_max as usize);
    let mut top_f = SharedMemory::<u32>::new(k_max as usize);
    let mut top_c = SharedMemory::<f32>::new(k_max as usize);
    let mut found = SharedMemory::<u32>::new(1usize);

    // Stage the reference patch from the centre frame, one thread per
    // pixel.
    let centre = read_line(ring, rx + local_x, ry + local_y, centre_slot, width, height);
    #[unroll]
    for c in 0..channels {
        ref_patch[(tid * channels + c) as usize] = centre[c as usize];
    }

    let self_packed = pack_pos(rx, ry);
    if tid == 0u32 {
        top_p[0] = self_packed;
        top_f[0] = centre_slot;
        // Never read, the centre slot on `top_f[0]` always forces
        // `mismatch_sigma2` to be skipped below, but every slot this
        // kernel could write gets a defined value here regardless.
        top_c[0] = 0.0f32;
        found[0] = 1u32;
    }
    sync_cube();

    // The block a temporal candidate reads its motion vector and
    // confidence from depends only on `rx`/`ry`, which are the same for
    // every candidate this whole cube scores, so it is worked out once
    // here rather than once per candidate.
    let bx = (rx / blk_step).min(blocks_x - 1);
    let by = (ry / blk_step).min(blocks_y - 1);
    let block = by * blocks_x + bx;

    // Score: each thread owns a strided slice of the candidates and
    // walks each of its candidates' pixels on its own.
    let scale = channel_scale(channels);
    // `mismatch_sigma2` divides by this rather than `blksize` itself,
    // see its own doc comment, and it is the same for every temporal
    // candidate this whole cube scores, so it is computed once here.
    let blksize_area = comptime!(blksize * blksize) as f32;

    // A thread's own `ci` only ever increases as the loop below strides
    // through it, so the neighbour index `t` a temporal candidate maps
    // to is non-decreasing across one thread's iterations even though
    // consecutive iterations are not consecutive `j` within the same
    // `t`. That makes it safe to cache the last `t` this thread scored
    // and only re-read `confidence`/`mv_field`, both of which depend on
    // `t` alone, when `t` actually changes.
    let mut cached_t = 0u32;
    let mut cached_valid = false;
    let mut cached_c = 0.0f32;
    let mut cached_mv0 = 0i32;
    let mut cached_mv1 = 0i32;

    let mut ci = tid;
    while ci < n_cand {
        let mut cx = 0u32;
        let mut cy = 0u32;
        let mut frame_val = centre_slot;
        let mut gated = false;
        // Never read for a spatial candidate, `frame_val` stays
        // `centre_slot` for one of those and the write-out below never
        // calls `mismatch_sigma2` on a centre-frame member, but every
        // slot `cand_conf` could be indexed at still gets a defined
        // value here.
        let mut cand_c = 1.0f32;

        if ci < n_spatial {
            cx = clamp_top_left(
                rx as i32 + (ci % window_side) as i32 - spatial_radius as i32,
                max_x,
            );
            cy = clamp_top_left(
                ry as i32 + (ci / window_side) as i32 - spatial_radius as i32,
                max_y,
            );
        } else {
            let off = ci - n_spatial;
            let t = off / n_refine;
            let j = off % n_refine;
            let slot = neighbour_slots[t as usize];
            frame_val = slot;

            if !cached_valid || t != cached_t {
                cached_c = confidence[(t * conf_stride + block) as usize];
                let mv = (t * mv_stride + block * 2) as usize;
                cached_mv0 = mv_field[mv];
                cached_mv1 = mv_field[mv + 1];
                cached_t = t;
                cached_valid = true;
            }

            let px = rx as i32 + cached_mv0 + (j % refine_side) as i32 - refine as i32;
            let py = ry as i32 + cached_mv1 + (j / refine_side) as i32 - refine as i32;
            cx = clamp_top_left(px, max_x);
            cy = clamp_top_left(py, max_y);
            cand_c = cached_c;

            if cached_c < c_min {
                gated = true;
            }
        }

        let packed = pack_pos(cx, cy);
        let mut kept = 3.0e38f32;
        if !gated {
            let mut acc = 0.0f32;
            let mut py = 0u32;
            while py < PATCH_SIZE {
                let mut px = 0u32;
                while px < PATCH_SIZE {
                    let cand = read_line(ring, cx + px, cy + py, frame_val, width, height);
                    let slot_off = (py * PATCH_SIZE + px) * channels;
                    #[unroll]
                    for c in 0..channels {
                        let d = ref_patch[(slot_off + c) as usize] - cand[c as usize];
                        acc += d * d;
                    }
                    px += 1u32;
                }
                py += 1u32;
            }
            kept = acc * scale - noise_floor;
        }
        // Retire on the spot the one candidate that can never be
        // selected on its own merit: the self-match, already pinned
        // into slot 0. Every other candidate stays live, however poor
        // its distance, so the rounds below always fill k_max slots.
        if packed == self_packed && frame_val == centre_slot {
            kept = 3.0e38f32;
        }
        dist[ci as usize] = kept;
        posn[ci as usize] = packed;
        frm[ci as usize] = frame_val;
        cand_conf[ci as usize] = cand_c;
        ci += PATCH_AREA;
    }
    sync_cube();

    // Selection: one round of argmin per remaining slot. The dedup
    // below compares the `(position, frame)` pair, not position alone.
    let mut slot = found[0];
    while slot < k_max {
        let mut best_d = 3.0e38f32;
        let mut best_i = 0u32;
        let mut si = tid;
        while si < n_cand {
            if dist[si as usize] < best_d {
                best_d = dist[si as usize];
                best_i = si;
            }
            si += PATCH_AREA;
        }
        red_d[tid as usize] = best_d;
        red_i[tid as usize] = best_i;
        sync_cube();

        // A literal starting value, not `PATCH_AREA / 2`, because a
        // value built entirely from `#[comptime]`/const inputs gets
        // treated as compile-time-only, and this loop needs `stride`
        // to be a genuine mutable runtime variable. The module-level
        // `const _: () = assert!(...)` above ties this literal back to
        // `PATCH_AREA` at compile time, so it can't drift silently.
        let mut stride = 32u32;
        while stride > 0u32 {
            if tid < stride {
                let other_d = red_d[(tid + stride) as usize];
                let other_i = red_i[(tid + stride) as usize];
                let cur_d = red_d[tid as usize];
                let cur_i = red_i[tid as usize];
                if other_d < cur_d || (other_d == cur_d && other_i < cur_i) {
                    red_d[tid as usize] = other_d;
                    red_i[tid as usize] = other_i;
                }
            }
            sync_cube();
            stride /= 2u32;
        }

        let win_d = red_d[0];
        let win_p = posn[red_i[0] as usize];
        let win_f = frm[red_i[0] as usize];
        let win_c = cand_conf[red_i[0] as usize];
        if win_d < 3.0e38f32 {
            if tid == 0u32 {
                top_p[slot as usize] = win_p;
                top_f[slot as usize] = win_f;
                top_c[slot as usize] = win_c;
                found[0] += 1u32;
            }
            // Retire the winner along with every other candidate
            // sharing its `(position, frame)` pair, whether that pair
            // came from a clamped duplicate offset in the same frame or
            // from two refine windows overlapping in the same
            // neighbour.
            let mut di = tid;
            while di < n_cand {
                if posn[di as usize] == win_p && frm[di as usize] == win_f {
                    dist[di as usize] = 3.0e38f32;
                }
                di += PATCH_AREA;
            }
        }
        sync_cube();
        slot += 1u32;
    }

    if tid == 0u32 {
        let cur_found = found[0];
        let mut k = 1u32;
        while k * 2u32 <= cur_found && k * 2u32 <= k_max {
            k *= 2u32;
        }
        member_count[ref_idx as usize] = k;
        let mut j = 0u32;
        while j < k {
            let frame_j = top_f[j as usize];
            member_pos[(ref_idx * k_max + j) as usize] = top_p[j as usize];
            member_frame[(ref_idx * k_max + j) as usize] = frame_j;
            let mut sig2 = 0.0f32;
            if frame_j != centre_slot {
                sig2 = mismatch_sigma2(top_c[j as usize], thsad, blksize_area);
            }
            member_sig2[(ref_idx * k_max + j) as usize] = sig2;
            j += 1u32;
        }
    }
}
