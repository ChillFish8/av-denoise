use cubecl::prelude::*;

use super::helpers::{
    BLK_STEP,
    R,
    RingFixture,
    deterministic_texture,
    make_client,
    noisy_ring,
    planted_ring,
};
use crate::collab::STEP;
use crate::collab::geometry::{
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    refs_along,
};
use crate::collab::kernels::group::{pack_pos_host, unpack_pos_host};
use crate::collab::kernels::group_temporal::collab_group_temporal;

/// The motion block side length these fixtures score confidence and
/// mismatch variance against, distinct from [`BLK_STEP`], which stays
/// at `PATCH_SIZE` so a block boundary lines up with a patch boundary.
pub(super) const BLKSIZE: u32 = 16;

/// Launches [`collab_group_temporal`] over a fixture and reads back all
/// four output buffers.
///
/// Luma always stores one channel per line, so the kernel's `Size`
/// selector is fixed at 1 here rather than threaded through as an
/// argument.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_group_temporal(
    fx: &RingFixture,
    noise_floor: f32,
    c_min: f32,
    thsad: f32,
    refine: u32,
    k_max: u32,
    spatial_radius: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let client = make_client();
    let w = fx.width;
    let h = fx.height;
    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let refs = ref_count(w, h);
    let pos_len = member_buf_len(w, h, k_max);
    let frame_len = member_frame_buf_len(w, h, k_max);
    let sig2_len = member_sig2_buf_len(w, h, k_max);

    let ring_buf = client.create_from_slice(f32::as_bytes(&fx.ring));
    let mv_buf = client.create_from_slice(i32::as_bytes(&fx.mv_field));
    let conf_buf = client.create_from_slice(f32::as_bytes(&fx.confidence));
    let neighbour_slots_buf = client.create_from_slice(u32::as_bytes(&fx.neighbour_slots));
    let member_pos_buf = client.empty(pos_len * size_of::<u32>());
    let member_frame_buf = client.empty(frame_len * size_of::<u32>());
    let member_count_buf = client.empty(refs * size_of::<u32>());
    let member_sig2_buf = client.empty(sig2_len * size_of::<f32>());

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_group_temporal::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(ring_buf, fx.ring.len()),
            ArrayArg::from_raw_parts(mv_buf, fx.mv_field.len()),
            ArrayArg::from_raw_parts(conf_buf, fx.confidence.len()),
            ArrayArg::from_raw_parts(member_pos_buf.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame_buf.clone(), frame_len),
            ArrayArg::from_raw_parts(member_count_buf.clone(), refs),
            ArrayArg::from_raw_parts(member_sig2_buf.clone(), sig2_len),
            fx.centre_slot,
            ArrayArg::from_raw_parts(neighbour_slots_buf, fx.neighbour_slots.len()),
            noise_floor,
            c_min,
            thsad,
            fx.radius,
            refine,
            fx.mv_stride,
            fx.conf_stride,
            BLK_STEP,
            BLKSIZE,
            fx.blocks_x,
            fx.blocks_y,
            w,
            h,
            1u32,
            k_max,
            spatial_radius,
            refs_x,
        );
    }

    let pos_bytes = client
        .read_one(member_pos_buf)
        .expect("member_pos readback failed");
    let frame_bytes = client
        .read_one(member_frame_buf)
        .expect("member_frame readback failed");
    let count_bytes = client
        .read_one(member_count_buf)
        .expect("member_count readback failed");
    let sig2_bytes = client
        .read_one(member_sig2_buf)
        .expect("member_sig2 readback failed");

    (
        u32::from_bytes(&pos_bytes)[..pos_len].to_vec(),
        u32::from_bytes(&frame_bytes)[..frame_len].to_vec(),
        u32::from_bytes(&count_bytes)[..refs].to_vec(),
        f32::from_bytes(&sig2_bytes)[..sig2_len].to_vec(),
    )
}

/// `thsad(BLKSIZE, 1.0)` in normalised SAD units, the same value real
/// callers get from `NlmDenoiser::thsad_value` at this block size and
/// the default `thsad_scale`. Duplicated here rather than imported,
/// since `motion::thsad` is crate-private to `nlmeans`.
pub(super) const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;

const REFINE: u32 = 2;
const K_MAX: u32 = 8;
const SPATIAL_RADIUS: u32 = 4;

#[test]
fn temporal_members_are_found_at_the_mv_prediction() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(7);
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);

    let (member_pos, member_frame, member_count, _member_sig2) =
        run_group_temporal(&fx, 0.0, 0.05, THSAD, REFINE, K_MAX, SPATIAL_RADIUS);

    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    assert_eq!(
        member_count[ref_idx], 8,
        "expected the group at the planted reference to fill to k_max"
    );

    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }
        let expected_slot = (k + radius as i32) as u32;
        let expected_x = (ref_pos.0 as i32 + 3 * k) as u32;
        let expected_y = ref_pos.1;

        let found = (0..K_MAX as usize).any(|j| {
            let idx = ref_idx * K_MAX as usize + j;
            if member_frame[idx] != expected_slot {
                return false;
            }
            let (px, py) = unpack_pos_host(member_pos[idx]);
            px.abs_diff(expected_x) <= REFINE && py.abs_diff(expected_y) <= REFINE
        });
        assert!(
            found,
            "expected a member in frame slot {expected_slot} (k={k}) near \
             ({expected_x}, {expected_y}); frames present: {:?}",
            (0..K_MAX as usize)
                .map(|j| member_frame[ref_idx * K_MAX as usize + j])
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn low_confidence_neighbours_contribute_no_candidates() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(11);
    // Confidence 0.0 for k = +1 and +2, 1.0 for k = -1 and -2.
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |k| {
        if k > 0 { 0.0 } else { 1.0 }
    });
    let c_min = 0.05f32;

    let (_member_pos, member_frame, member_count, _member_sig2) =
        run_group_temporal(&fx, 0.0, c_min, THSAD, REFINE, K_MAX, SPATIAL_RADIUS);

    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let gated_slots: Vec<u32> = [1i32, 2].iter().map(|&k| (k + radius as i32) as u32).collect();
    let count = member_count[ref_idx] as usize;
    for j in 0..count {
        let frame = member_frame[ref_idx * K_MAX as usize + j];
        assert!(
            !gated_slots.contains(&frame),
            "member {j} carries gated slot {frame}, expected only ungated slots"
        );
    }
}

#[test]
fn no_admission_gate_means_the_group_always_fills() {
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let fx = noisy_ring(w, h, radius, 1.0);

    let (_, _, member_count, _) = run_group_temporal(&fx, 0.0, 0.05, THSAD, REFINE, K_MAX, SPATIAL_RADIUS);

    for (i, &count) in member_count.iter().enumerate() {
        assert_eq!(
            count, K_MAX,
            "ref {i}: expected the group to always fill to k_max with no admission gate"
        );
    }
}

#[test]
fn the_reference_patch_is_always_member_zero() {
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let fx = noisy_ring(w, h, radius, 1.0);

    let (member_pos, member_frame, _member_count, _member_sig2) =
        run_group_temporal(&fx, 0.0, 0.05, THSAD, REFINE, K_MAX, SPATIAL_RADIUS);

    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    for ry_idx in (0..refs_y).step_by(3) {
        for rx_idx in (0..refs_x).step_by(3) {
            let ref_idx = (ry_idx * refs_x + rx_idx) as usize;
            let rx = (rx_idx * STEP).min(w - 8);
            let ry = (ry_idx * STEP).min(h - 8);

            assert_eq!(
                member_pos[ref_idx * K_MAX as usize],
                pack_pos_host(rx, ry),
                "ref {ref_idx}: member 0 must unpack to the reference position"
            );
            assert_eq!(
                member_frame[ref_idx * K_MAX as usize],
                fx.centre_slot,
                "ref {ref_idx}: member 0 must carry the centre slot"
            );
        }
    }
}
