use cubecl::prelude::*;

use crate::collab::{PATCH_AREA, PATCH_SIZE, STEP};
use crate::nlmeans::kernels::helpers::{channel_scale, read_line};

// A compile-time tie from the `stride = 32u32` literal in the argmin
// reduction below back to `PATCH_AREA`. That literal can't be written as
// `PATCH_AREA / 2` directly (see the comment at its use site), so this is
// what catches it going stale if `PATCH_SIZE` ever changes.
const _: () = assert!(
    PATCH_AREA / 2 == 32,
    "update the `stride = 32u32` literal in collab_group_spatial to PATCH_AREA / 2"
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

/// Packs a patch's top-left position into one `u32`, x in the low half
/// and y in the high half.
///
/// A patch position never needs more than 16 bits per axis for any frame
/// this filter runs on, so the pair packs into a single integer. That
/// makes the top-K arrays and the dedup check simple comparisons instead
/// of pairs of comparisons.
#[cube]
pub fn pack_pos(x: u32, y: u32) -> u32 {
    (y << 16) | x
}

/// The host-side mirror of [`pack_pos`], for building expected values in
/// tests without a GPU round trip.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn pack_pos_host(x: u32, y: u32) -> u32 {
    (y << 16) | x
}

/// The host-side mirror of unpacking a position [`pack_pos`] produced.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn unpack_pos_host(packed: u32) -> (u32, u32) {
    (packed & 0xFFFF, packed >> 16)
}

/// Clamps a candidate top-left coordinate to `[0, max_pos]`.
///
/// Every candidate patch position on every axis goes through this before
/// anything reads from it. That guarantees a patch read at that position
/// always starts and ends inside the frame, so no kernel that later
/// consumes a group needs to clamp its own reads.
///
/// `pub(crate)` because [`crate::collab::kernels::group_temporal`] reuses
/// it for the same purpose against both the spatial window and each
/// neighbour frame's refine window.
#[cube]
pub(crate) fn clamp_top_left(v: i32, max_pos: u32) -> u32 {
    let mut result = v;
    if result < 0 {
        result = 0;
    } else if result > max_pos as i32 {
        result = max_pos as i32;
    }
    result as u32
}

/// Finds the K most similar patches to each reference patch in a spatial
/// window around it.
///
/// One cube owns one reference patch, and its `CubeDim::new_2d(8, 8)`
/// threads score the search window one candidate per thread. A thread
/// walks its own candidate's 64 pixels start to finish on its own, so
/// scoring needs no barrier and no cross-thread reduction at all. The
/// candidates outnumber the threads, so each thread takes a strided
/// slice of the window and scores several in turn.
///
/// The reference patch is staged in shared memory first, because all 64
/// threads read all 64 of its pixels. Candidate pixels are read straight
/// from global memory, which measured faster than staging the window in
/// shared memory. The search windows of neighbouring reference patches
/// overlap heavily at a step of 4, so the cache already serves those
/// reads well, and a shared-memory tile only costs occupancy.
///
/// # Distance and admission
///
/// A candidate's distance is the channel-scaled sum of squared pixel
/// differences over the whole patch, with `noise_floor` subtracted.
/// `noise_floor` is the distance two noisy copies of the same content
/// are expected to show by chance, so a genuine match isn't penalised
/// for the noise it carries. A candidate is admitted only when what's
/// left is at most `tau_admit`. The floored value is never clamped to
/// zero before ranking. Subtracting a constant from every candidate
/// shifts them all by the same amount and does not change their relative
/// order, so the top-K selection below stays a genuine similarity
/// ranking even when `noise_floor` is large enough to push most
/// candidates' floored distance negative.
///
/// # The self-match seed
///
/// The reference patch's own position is written into slot 0 before
/// selection starts, and selection only ever fills slots 1 and above. A
/// group therefore always contains its own reference patch, whatever
/// distances the search finds.
///
/// # Selection
///
/// Once every candidate has a distance, the group's remaining members
/// are picked by `k_max - 1` rounds of argmin across the whole window.
/// Each round every thread finds the best candidate in its own slice,
/// a tree reduction over shared memory folds those into the round's
/// winner, and that winner is retired before the next round runs.
///
/// Ties break toward the lower candidate index, which is the order a
/// linear scan of the window would visit them in. Exactly-equal
/// distances are common on flat content, so this is what fixes which
/// member a group keeps rather than leaving it to thread scheduling.
///
/// # Clamped duplicates
///
/// Candidate positions are patch top-left coordinates clamped to `[0,
/// dim - 8]` on each axis, which is what keeps every member read fully
/// inside the frame. That clamping also means two different search
/// offsets near an edge can land on the same clamped position. Admitting
/// both would let one physical patch count twice toward the group, which
/// would look like stronger agreement than the group actually has.
/// Retiring a winner retires every candidate sharing its position, and
/// the self-match's position is retired before the first round, so no
/// position is ever kept twice.
///
/// # Group size
///
/// The final member count is rounded down to the nearest power of two,
/// capped at `k_max`. The stack transform a filtered group later passes
/// through is only defined for power-of-two stack sizes, so a count of
/// 5, 6, or 7 admitted candidates still only keeps 4 of them.
#[cube(launch_unchecked)]
pub fn collab_group_spatial<N: Size>(
    reference: &Array<Vector<f32, N>>,
    member_pos: &mut Array<u32>,
    member_count: &mut Array<u32>,
    frame: u32,
    noise_floor: f32,
    tau_admit: f32,
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
    let n_cand = comptime!((2 * spatial_radius + 1) * (2 * spatial_radius + 1));

    let mut ref_patch = SharedMemory::<f32>::new(comptime!(PATCH_AREA * channels) as usize);
    let mut dist = SharedMemory::<f32>::new(n_cand as usize);
    let mut posn = SharedMemory::<u32>::new(n_cand as usize);
    let mut red_d = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut red_i = SharedMemory::<u32>::new(PATCH_AREA as usize);
    let mut top_p = SharedMemory::<u32>::new(k_max as usize);
    let mut found = SharedMemory::<u32>::new(1usize);

    // Stage the reference patch, one thread per pixel.
    let centre = read_line(reference, rx + local_x, ry + local_y, frame, width, height);
    #[unroll]
    for c in 0..channels {
        ref_patch[(tid * channels + c) as usize] = centre[c as usize];
    }

    let self_packed = pack_pos(rx, ry);
    if tid == 0u32 {
        top_p[0] = self_packed;
        found[0] = 1u32;
    }
    sync_cube();

    // Score: each thread owns a strided slice of the window and walks
    // each of its candidates' pixels on its own.
    let scale = channel_scale(channels);
    let mut ci = tid;
    while ci < n_cand {
        let cx = clamp_top_left(
            rx as i32 + (ci % window_side) as i32 - spatial_radius as i32,
            max_x,
        );
        let cy = clamp_top_left(
            ry as i32 + (ci / window_side) as i32 - spatial_radius as i32,
            max_y,
        );

        let mut acc = 0.0f32;
        let mut py = 0u32;
        while py < PATCH_SIZE {
            let mut px = 0u32;
            while px < PATCH_SIZE {
                let cand = read_line(reference, cx + px, cy + py, frame, width, height);
                let slot = (py * PATCH_SIZE + px) * channels;
                #[unroll]
                for c in 0..channels {
                    let d = ref_patch[(slot + c) as usize] - cand[c as usize];
                    acc += d * d;
                }
                px += 1u32;
            }
            py += 1u32;
        }

        let scored = acc * scale - noise_floor;
        // Retire on the spot whatever can never be selected, so the
        // rounds below only ever compare live candidates. That is
        // anything over `tau_admit`, and the self-match, which is
        // already pinned into slot 0.
        let packed = pack_pos(cx, cy);
        let mut kept = scored;
        if scored > tau_admit || packed == self_packed {
            kept = 3.0e38f32;
        }
        dist[ci as usize] = kept;
        posn[ci as usize] = packed;
        ci += PATCH_AREA;
    }
    sync_cube();

    // Selection: one round of argmin per remaining slot. `slot` is
    // seeded from the shared-memory read rather than a literal 1, so
    // cubecl treats it as a genuine runtime variable. `found[0]` is
    // exactly 1 here, the pinned self-match.
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
        if win_d < 3.0e38f32 {
            if tid == 0u32 {
                top_p[slot as usize] = win_p;
                found[0] += 1u32;
            }
            // Retire the winner along with every other candidate that
            // clamped onto the same physical position.
            let mut di = tid;
            while di < n_cand {
                if posn[di as usize] == win_p {
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
            member_pos[(ref_idx * k_max + j) as usize] = top_p[j as usize];
            j += 1u32;
        }
    }
}
