use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::collab::kernels::group::pack_pos_t;
use crate::collab::kernels::plane_ops::plane_ssd_reduce8;
use crate::nlmeans::kernels::helpers::read_line;

/// How far a fractional candidate must undercut the incumbent, as a
/// fraction of the noise floor `noise_px`, before it replaces it.
///
/// In flat content every phase has the same expected cost, so without a
/// margin noise alone would pick fractional positions.
pub(crate) const SUBPEL_ACCEPT_MARGIN: f32 = 0.25;

/// The phase-ring sample at half-pel coordinates `(hx, hy)`.
///
/// The low bit of each coordinate picks the plane, and the rest is the
/// integer pixel that plane is read at.
#[cube]
pub(crate) fn half_sample<N: Size>(
    phase_ring: &Array<Vector<f32, N>>,
    hx: u32,
    hy: u32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<f32, N> {
    let plane = (hx & 1u32) + 2u32 * (hy & 1u32);
    let frame = slot * 4u32 + plane;
    read_line(phase_ring, hx >> 1u32, hy >> 1u32, frame, width, height)
}

/// The phase-ring sample at quarter-pel coordinates `(qx, qy)`.
///
/// It averages the two nearest half-grid samples. A position already on
/// the half grid reads the same sample twice. When both axes are odd the
/// pair lies on the anti-diagonal. The pair comes from `select` rather
/// than a branch so the reads compile under cubecl's value numbering.
#[cube]
pub(crate) fn quarter_sample<N: Size>(
    phase_ring: &Array<Vector<f32, N>>,
    qx: u32,
    qy: u32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Vector<f32, N> {
    let odd_x = qx & 1u32;
    let odd_y = qy & 1u32;
    let both_odd = (odd_x & odd_y) == 1u32;

    let low_y = qy >> 1u32;
    let high_y = (qy + odd_y) >> 1u32;
    let first_y = select(both_odd, high_y, low_y);
    let second_y = select(both_odd, low_y, high_y);

    let first = half_sample(phase_ring, qx >> 1u32, first_y, slot, width, height);
    let second = half_sample(phase_ring, (qx + odd_x) >> 1u32, second_y, slot, width, height);
    let half = Vector::<f32, N>::empty().fill(0.5f32);
    (first + second) * half
}

/// The distance from the reference patch to the candidate at integer
/// top-left `(x, y)` shifted by `phase`, in frame `slot`.
///
/// It matches [candidate_distance](super::search::candidate_distance)
/// except that every pixel is read through [quarter_sample].
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn subpel_candidate_distance<N: Size>(
    phase_ring: &Array<Vector<f32, N>>,
    current: &Array<f32>,
    x: u32,
    y: u32,
    phase: u32,
    slot: u32,
    sub: u32,
    scale: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> f32 {
    let phase_x = phase & 3u32;
    let phase_y = phase >> 2u32;
    let qx = 4u32 * (x + sub) + phase_x;

    let mut partial = 0.0f32;
    #[unroll]
    for r in 0..PATCH_SIZE {
        let qy = 4u32 * (y + r) + phase_y;
        let px = quarter_sample(phase_ring, qx, qy, slot, width, height);
        #[unroll]
        for c in 0..channels {
            let d = current[(r * channels + c) as usize] - px[c as usize];
            partial += d * d;
        }
    }
    plane_ssd_reduce8(partial) * scale
}

/// Whether an earlier volume already holds this position at this phase.
#[cube]
pub(crate) fn already_claimed(
    member_pos: &Array<u32>,
    member_phase: &Array<u32>,
    packed: u32,
    phase: u32,
    #[comptime] first: u32,
) -> bool {
    let mut claimed = false;
    #[unroll]
    for k in 0..first {
        if member_pos[k as usize] == packed && member_phase[k as usize] == phase {
            claimed = true;
        }
    }
    claimed
}

/// Refines one neighbour frame's whole-pixel match to half-pel, then
/// optionally quarter-pel, precision.
///
/// Each candidate is scored on its distance plus the noise its
/// interpolation removed, so every phase sits at the same expected floor
/// as a whole-pixel match. A candidate replaces the incumbent only when
/// it undercuts it by [SUBPEL_ACCEPT_MARGIN] of `noise_px`.
///
/// A frame with no match still runs every read, pinned to `safe_x` and
/// `safe_y`, so every lane reaches every shuffle. The result is then
/// discarded.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn refine_subpel<N: Size>(
    phase_ring: &Array<Vector<f32, N>>,
    phase_gain: &Array<f32>,
    anchor: &Array<f32>,
    member_pos: &Array<u32>,
    member_phase: &Array<u32>,
    found: bool,
    safe_x: u32,
    safe_y: u32,
    slot: u32,
    packed_t: u32,
    sub: u32,
    scale: f32,
    noise_px: f32,
    frame_d: &mut f32,
    frame_pos: &mut u32,
    frame_phase: &mut u32,
    #[comptime] first: u32,
    #[comptime] subpel: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let margin = SUBPEL_ACCEPT_MARGIN * noise_px;
    let whole_x = select(found, *frame_pos & 0x1FFFu32, safe_x);
    let whole_y = select(found, (*frame_pos >> 13u32) & 0x1FFFu32, safe_y);

    let mut best_qx = 4u32 * whole_x;
    let mut best_qy = 4u32 * whole_y;
    let mut best_cost = *frame_d;

    // The half-pel ring around the whole-pixel match. Offsets are in
    // quarter-pel units.
    let centre_qx = best_qx;
    let centre_qy = best_qy;
    #[unroll]
    for oy in 0..3u32 {
        #[unroll]
        for ox in 0..3u32 {
            if comptime!(!(ox == 1 && oy == 1)) {
                let qx = centre_qx + 2u32 * ox - 2u32;
                let qy = centre_qy + 2u32 * oy - 2u32;
                offer_candidate(
                    phase_ring,
                    phase_gain,
                    anchor,
                    member_pos,
                    member_phase,
                    found,
                    qx,
                    qy,
                    slot,
                    packed_t,
                    sub,
                    scale,
                    noise_px,
                    margin,
                    &mut best_qx,
                    &mut best_qy,
                    &mut best_cost,
                    first,
                    width,
                    height,
                    channels,
                );
            }
        }
    }

    // The quarter-pel ring around whichever position won above.
    if comptime!(subpel == 2) {
        let half_qx = best_qx;
        let half_qy = best_qy;
        #[unroll]
        for oy in 0..3u32 {
            #[unroll]
            for ox in 0..3u32 {
                if comptime!(!(ox == 1 && oy == 1)) {
                    let qx = half_qx + ox - 1u32;
                    let qy = half_qy + oy - 1u32;
                    offer_candidate(
                        phase_ring,
                        phase_gain,
                        anchor,
                        member_pos,
                        member_phase,
                        found,
                        qx,
                        qy,
                        slot,
                        packed_t,
                        sub,
                        scale,
                        noise_px,
                        margin,
                        &mut best_qx,
                        &mut best_qy,
                        &mut best_cost,
                        first,
                        width,
                        height,
                        channels,
                    );
                }
            }
        }
    }

    let refined_pos = pack_pos_t(best_qx >> 2u32, best_qy >> 2u32, packed_t);
    let refined_phase = (best_qy & 3u32) * 4u32 + (best_qx & 3u32);
    *frame_pos = select(found, refined_pos, *frame_pos);
    *frame_phase = select(found, refined_phase, 0u32);
    *frame_d = select(found, best_cost, *frame_d);
}

/// Scores one quarter-pel candidate and keeps it when it clears the
/// margin.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
fn offer_candidate<N: Size>(
    phase_ring: &Array<Vector<f32, N>>,
    phase_gain: &Array<f32>,
    anchor: &Array<f32>,
    member_pos: &Array<u32>,
    member_phase: &Array<u32>,
    found: bool,
    qx: u32,
    qy: u32,
    slot: u32,
    packed_t: u32,
    sub: u32,
    scale: f32,
    noise_px: f32,
    margin: f32,
    best_qx: &mut u32,
    best_qy: &mut u32,
    best_cost: &mut f32,
    #[comptime] first: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let base_x = qx >> 2u32;
    let base_y = qy >> 2u32;
    let phase = (qy & 3u32) * 4u32 + (qx & 3u32);
    let packed = pack_pos_t(base_x, base_y, packed_t);

    let distance = subpel_candidate_distance(
        phase_ring, anchor, base_x, base_y, phase, slot, sub, scale, width, height, channels,
    );
    let gain = phase_gain[phase as usize];
    let cost = distance + noise_px * (1.0f32 - gain);

    let claimed = already_claimed(member_pos, member_phase, packed, phase, first);
    let better = found && !claimed && cost + margin < *best_cost;
    *best_qx = select(better, qx, *best_qx);
    *best_qy = select(better, qy, *best_qy);
    *best_cost = select(better, cost, *best_cost);
}
