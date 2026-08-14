use std::collections::HashSet;

use cubecl::prelude::*;

use super::helpers::{
    R,
    deterministic_texture,
    make_client,
    make_unique_frame,
    noisy_field_over,
    plant_patch,
};
use crate::collab::geometry::{member_buf_len, ref_count, refs_along};
use crate::collab::kernels::group::{collab_group_spatial, pack_pos_host, unpack_pos_host};

/// Launches [`collab_group_spatial`] over a luma frame and reads back
/// both output buffers.
///
/// Luma always stores one channel per line, so the kernel's `Size`
/// selector is fixed at 1 here rather than threaded through as an
/// argument.
#[allow(clippy::too_many_arguments)]
fn run_group(
    frame: &[f32],
    width: u32,
    height: u32,
    noise_floor: f32,
    tau_admit: f32,
    spatial_radius: u32,
    k_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(frame.len(), (width * height) as usize);

    let client = make_client();
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pos_len = member_buf_len(width, height, k_max);

    let reference_buf = client.create_from_slice(f32::as_bytes(frame));
    let member_pos_buf = client.empty(pos_len * size_of::<u32>());
    let member_count_buf = client.empty(refs * size_of::<u32>());

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_group_spatial::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(reference_buf, frame.len()),
            ArrayArg::from_raw_parts(member_pos_buf.clone(), pos_len),
            ArrayArg::from_raw_parts(member_count_buf.clone(), refs),
            0u32,
            noise_floor,
            tau_admit,
            width,
            height,
            1u32,
            k_max,
            spatial_radius,
            refs_x,
        );
    }

    let pos_bytes = client
        .read_one(member_pos_buf)
        .expect("member_pos readback failed");
    let count_bytes = client
        .read_one(member_count_buf)
        .expect("member_count readback failed");

    (
        u32::from_bytes(&pos_bytes)[..pos_len].to_vec(),
        u32::from_bytes(&count_bytes)[..refs].to_vec(),
    )
}

#[test]
fn flat_frame_fills_the_stack() {
    let (w, h) = (32u32, 32u32);
    let frame = vec![0.5f32; (w * h) as usize];
    let refs_x = refs_along(w);

    let (member_pos, member_count) = run_group(&frame, w, h, 0.0, 1e-6, 4, 8);

    for (i, &count) in member_count.iter().enumerate() {
        assert_eq!(count, 8, "ref {i}: expected a full stack on a flat frame");

        let rx_idx = (i as u32) % refs_x;
        let ry_idx = (i as u32) / refs_x;
        let rx = (rx_idx * crate::collab::STEP).min(w - 8);
        let ry = (ry_idx * crate::collab::STEP).min(h - 8);

        assert_eq!(
            member_pos[i * 8],
            pack_pos_host(rx, ry),
            "ref {i}: member 0 must be the reference itself"
        );
    }
}

#[test]
fn distinct_content_admits_only_the_self_match() {
    let (w, h) = (32u32, 32u32);
    let frame = make_unique_frame(w, h);
    let refs = ref_count(w, h);

    let (_member_pos, member_count) = run_group(&frame, w, h, 0.0, 1e-6, 9, 8);

    assert_eq!(member_count.len(), refs);
    for (i, &count) in member_count.iter().enumerate() {
        assert_eq!(
            count, 1,
            "ref {i}: unique content should admit only the self-match"
        );
    }
}

#[test]
fn a_planted_twin_is_found() {
    let (w, h) = (32u32, 32u32);
    let mut frame = vec![0.2f32; (w * h) as usize];
    let texture = deterministic_texture(7);
    plant_patch(&mut frame, w, 4, 4, &texture);
    plant_patch(&mut frame, w, 16, 12, &texture);

    let (member_pos, member_count) = run_group(&frame, w, h, 0.0, 1e-6, 12, 8);

    let refs_x = refs_along(w);
    let ref_idx = (4 / crate::collab::STEP + (4 / crate::collab::STEP) * refs_x) as usize;

    assert_eq!(
        member_count[ref_idx], 2,
        "the reference at (4,4) should find exactly its twin"
    );
    let (px, py) = unpack_pos_host(member_pos[ref_idx * 8 + 1]);
    assert_eq!(
        (px, py),
        (16, 12),
        "member 1 should be the planted twin at (16,12)"
    );
}

#[test]
fn noise_floor_rescues_noisy_matches() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.02f32;
    let frame = noisy_field_over(w, h, 0.5, sigma);

    // The distance two independent noisy copies of the same flat content
    // are expected to show: channel_scale (3, luma) times the patch area
    // (64) times the per-pixel variance of a difference of two
    // independent noise samples (2 * sigma^2).
    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;

    // A tau small next to the floor's own sampling spread, so a
    // correctly-floored comparison comfortably admits a real noisy
    // match, restricted to a tau this is small enough that reusing it
    // as an absolute cutoff on the raw, un-floored distance rejects
    // almost everything except the seeded self-match and its immediate,
    // heavily-overlapping neighbours.
    let tau_admit = floor * 0.15;

    let (_, with_floor) = run_group(&frame, w, h, floor, tau_admit, 9, 8);
    let (_, without_floor) = run_group(&frame, w, h, 0.0, tau_admit, 9, 8);

    let mean_with: f64 = with_floor.iter().map(|&c| c as f64).sum::<f64>() / with_floor.len() as f64;
    let mean_without: f64 = without_floor.iter().map(|&c| c as f64).sum::<f64>() / without_floor.len() as f64;

    assert!(
        mean_with >= mean_without + 4.0,
        "expected the floored config's mean count ({mean_with}) to beat the unfloored one \
         ({mean_without}) by at least 4"
    );
}

/// A reference patch pinned to the left edge (`rx == 0`) sees several
/// distinct search offsets clamp onto the same physical position, and not
/// only onto the seeded self-match: at `dy == -3` for instance, `dx ==
/// -4, -3, -2, -1, 0` all clamp (or naturally land, for `dx == 0`) on `cx
/// == 0`, giving a non-self position hit by five distinct offsets. On a
/// flat frame every candidate has distance 0, so the group's ascending
/// insertion sort never swaps (`0 > 0` is always false), which makes the
/// visit order of `collab_group_spatial`'s nested nearest-offset-first
/// scan fully predictable: the kept members are exactly the first 8
/// distinct positions the scan encounters, in the order it encounters
/// them. That's hand-traced below.
#[test]
fn corner_clamp_collisions_are_deduplicated() {
    let (w, h) = (32u32, 32u32);
    let frame = vec![0.4f32; (w * h) as usize];
    let refs_x = refs_along(w);

    // Reference index (0, 3): rx = 0 (left edge), ry = 3 * STEP = 12,
    // comfortably clear of the top and bottom edges at radius 4 (window
    // rows 8..=16), so only the x axis clamps.
    let ref_idx = (3 * refs_x) as usize;

    let (member_pos, member_count) = run_group(&frame, w, h, 0.0, 1e-6, 4, 8);

    assert_eq!(
        member_count[ref_idx], 8,
        "expected the corner ref's stack to saturate at k_max"
    );

    let members: Vec<(u32, u32)> = (0..8)
        .map(|j| unpack_pos_host(member_pos[ref_idx * 8 + j]))
        .collect();

    // The direct statement of the dedup property: no physical position
    // appears twice among the kept members, whatever order they came in.
    let distinct: HashSet<(u32, u32)> = members.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        members.len(),
        "expected all 8 kept members to be distinct positions, got {members:?}"
    );

    // The exact, hand-traced visit order: self first, then row dy=-4
    // (cy=8) contributing cx=0,1,2,3,4 in order (five slots, since dx=-4
    // clamps to cx=0 and dx=-3,-2,-1,0 are then dropped as duplicates of
    // it), then row dy=-3 (cy=9) contributing cx=0 then cx=1 before the
    // stack is full at 8 and every later candidate (also distance 0) is
    // not strictly better than the current worst, so nothing more is
    // replaced.
    assert_eq!(
        members,
        vec![(0, 12), (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (0, 9), (1, 9)],
        "expected the exact hand-traced member order for the corner ref"
    );
}
