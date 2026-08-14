use cubecl::prelude::*;

use crate::collab::{PATCH_AREA, PATCH_SIZE, STEP};
use crate::nlmeans::kernels::helpers::{channel_scale, line_sum_sq, read_line};

// A compile-time tie from the `stride = 32u32` literal in the reduction
// loop below back to `PATCH_AREA`. That literal can't be written as
// `PATCH_AREA / 2` directly (see the comment at its use site), so this is
// what catches it going stale if `PATCH_SIZE` ever changes.
const _: () = assert!(
    PATCH_AREA / 2 == 32,
    "update the `stride = 32u32` literal in collab_group_spatial to PATCH_AREA / 2"
);

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
#[cube]
fn clamp_top_left(v: i32, max_pos: u32) -> u32 {
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
/// One cube owns one reference patch. Its `CubeDim::new_2d(8, 8)`
/// threads map one-to-one onto the patch's 64 pixels, and every thread
/// in the cube takes part in scoring every candidate the search visits.
///
/// # Distance and admission
///
/// A candidate's distance is the channel-scaled sum of squared pixel
/// differences over the whole patch, with `noise_floor` subtracted and
/// clamped at zero. `noise_floor` is the distance two noisy copies of
/// the same content are expected to show by chance, so a genuine match
/// isn't penalised for the noise it carries. A candidate is admitted
/// only when what's left is at most `tau_admit`.
///
/// Each thread contributes its own pixel's squared difference, and a
/// tree reduction over shared memory folds the 64 contributions into one
/// sum before thread 0 decides whether the candidate is admitted.
///
/// # The self-match seed
///
/// The reference patch's own position is seeded into slot 0 with
/// distance 0 before the search starts, and the insertion step never
/// touches slot 0 again. A group therefore always contains its own
/// reference patch, whatever distances the search finds.
///
/// # Clamped duplicates
///
/// Candidate positions are patch top-left coordinates clamped to `[0,
/// dim - 8]` on each axis, which is what keeps every member read fully
/// inside the frame. That clamping also means two different search
/// offsets near an edge can land on the same clamped position. Admitting
/// both would let one physical patch count twice toward the group,
/// which would look like stronger agreement than the group actually has.
/// Every admitted candidate is checked against the positions already
/// kept, including the seeded self-match, before it is inserted, and a
/// repeat is dropped.
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

    let centre = read_line(reference, rx + local_x, ry + local_y, frame, width, height);
    let scale = channel_scale(channels);

    let mut red = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut top_d = SharedMemory::<f32>::new(k_max as usize);
    let mut top_p = SharedMemory::<u32>::new(k_max as usize);
    let mut found = SharedMemory::<u32>::new(1usize);

    if tid == 0u32 {
        top_d[0] = 0.0f32;
        top_p[0] = pack_pos(rx, ry);
        found[0] = 1u32;
    }
    sync_cube();

    let window_side = comptime!(2 * spatial_radius + 1);
    for dyi in 0..window_side {
        for dxi in 0..window_side {
            let dy = dyi as i32 - spatial_radius as i32;
            let dx = dxi as i32 - spatial_radius as i32;
            let cx = clamp_top_left(rx as i32 + dx, max_x);
            let cy = clamp_top_left(ry as i32 + dy, max_y);

            let candidate = read_line(reference, cx + local_x, cy + local_y, frame, width, height);
            red[tid as usize] = line_sum_sq(centre - candidate, channels) * scale;
            sync_cube();

            // A literal starting value, not `PATCH_AREA / 2`, because a
            // value built entirely from `#[comptime]`/const inputs gets
            // treated as compile-time-only, and this loop needs `stride`
            // to be a genuine mutable runtime variable. The module-level
            // `const _: () = assert!(...)` above ties this literal back
            // to `PATCH_AREA` at compile time, so it can't drift silently.
            let mut stride = 32u32;
            while stride > 0u32 {
                if tid < stride {
                    red[tid as usize] += red[(tid + stride) as usize];
                }
                sync_cube();
                stride /= 2u32;
            }

            if tid == 0u32 {
                let mut d = red[0] - noise_floor;
                if d < 0.0f32 {
                    d = 0.0f32;
                }
                if d <= tau_admit {
                    let packed = pack_pos(cx, cy);
                    let cur_found = found[0];

                    let mut dup = false;
                    let mut i = 0u32;
                    while i < cur_found {
                        if top_p[i as usize] == packed {
                            dup = true;
                        }
                        i += 1u32;
                    }

                    if !dup {
                        if cur_found < k_max {
                            let mut pos = cur_found;
                            top_d[pos as usize] = d;
                            top_p[pos as usize] = packed;
                            while pos > 1u32 && top_d[(pos - 1) as usize] > top_d[pos as usize] {
                                let swap_d = top_d[(pos - 1) as usize];
                                let swap_p = top_p[(pos - 1) as usize];
                                top_d[(pos - 1) as usize] = top_d[pos as usize];
                                top_p[(pos - 1) as usize] = top_p[pos as usize];
                                top_d[pos as usize] = swap_d;
                                top_p[pos as usize] = swap_p;
                                pos -= 1u32;
                            }
                            found[0] = cur_found + 1u32;
                        } else {
                            let worst = k_max - 1;
                            if d < top_d[worst as usize] {
                                top_d[worst as usize] = d;
                                top_p[worst as usize] = packed;
                                // `cur_found - 1`, not `worst`, even though
                                // the two are numerically equal here. `worst`
                                // is built entirely from the `#[comptime]`
                                // `k_max`, which makes it compile-time-only,
                                // and this loop needs a genuine mutable
                                // runtime variable. `cur_found` comes from a
                                // shared-memory read, so it carries a real
                                // runtime value through the subtraction.
                                let mut pos = cur_found - 1u32;
                                while pos > 1u32 && top_d[(pos - 1) as usize] > top_d[pos as usize] {
                                    let swap_d = top_d[(pos - 1) as usize];
                                    let swap_p = top_p[(pos - 1) as usize];
                                    top_d[(pos - 1) as usize] = top_d[pos as usize];
                                    top_p[(pos - 1) as usize] = top_p[pos as usize];
                                    top_d[pos as usize] = swap_d;
                                    top_p[pos as usize] = swap_p;
                                    pos -= 1u32;
                                }
                            }
                        }
                    }
                }
            }
            sync_cube();
        }
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
