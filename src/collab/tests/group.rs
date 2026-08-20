use std::collections::HashSet;

use cubecl::prelude::*;

use super::helpers::{R, deterministic_texture, make_client, plant_patch};
use crate::collab::STEP;
use crate::collab::geometry::{
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    refs_along,
};
use crate::collab::kernels::group::{clamp_top_left, pack_pos, pack_pos_host, unpack_pos_host};
use crate::collab::kernels::group_temporal::collab_group_temporal;

/// Runs [`pack_pos`] and [`clamp_top_left`] on the GPU, one input per
/// thread, so the host mirrors below are checked against the kernels
/// that actually consume them rather than against themselves.
#[cube(launch_unchecked)]
fn group_helpers_kernel(
    xs: &Array<u32>,
    ys: &Array<u32>,
    coords: &Array<i32>,
    max_pos: &Array<u32>,
    packed: &mut Array<u32>,
    clamped: &mut Array<u32>,
    #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < n {
        packed[i as usize] = pack_pos(xs[i as usize], ys[i as usize]);
        clamped[i as usize] = clamp_top_left(coords[i as usize], max_pos[i as usize]);
    }
}

fn run_helpers(xs: &[u32], ys: &[u32], coords: &[i32], max_pos: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = xs.len();
    assert_eq!(ys.len(), n);
    assert_eq!(coords.len(), n);
    assert_eq!(max_pos.len(), n);

    let client = make_client();
    let xs_buf = client.create_from_slice(u32::as_bytes(xs));
    let ys_buf = client.create_from_slice(u32::as_bytes(ys));
    let coords_buf = client.create_from_slice(i32::as_bytes(coords));
    let max_buf = client.create_from_slice(u32::as_bytes(max_pos));
    // These size the kernel's two output buffers, which hold one u32
    // per input coordinate. `size_of_val(xs)` reaches the same number
    // but ties an output's size to an input's slice, which reads as if
    // the buffers held `xs` itself.
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "n is the element count these outputs hold, not xs's byte length"
    )]
    let packed_buf = client.empty(n * size_of::<u32>());
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "n is the element count these outputs hold, not xs's byte length"
    )]
    let clamped_buf = client.empty(n * size_of::<u32>());

    unsafe {
        group_helpers_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(xs_buf, n),
            ArrayArg::from_raw_parts(ys_buf, n),
            ArrayArg::from_raw_parts(coords_buf, n),
            ArrayArg::from_raw_parts(max_buf, n),
            ArrayArg::from_raw_parts(packed_buf.clone(), n),
            ArrayArg::from_raw_parts(clamped_buf.clone(), n),
            n as u32,
        );
    }

    let packed = client.read_one(packed_buf).expect("packed readback failed");
    let clamped = client.read_one(clamped_buf).expect("clamped readback failed");

    (
        u32::from_bytes(&packed)[..n].to_vec(),
        u32::from_bytes(&clamped)[..n].to_vec(),
    )
}

/// Positions spanning both halves of the packed word, including the
/// largest value 16 bits per axis holds.
const POSITIONS: &[(u32, u32)] = &[
    (0, 0),
    (1, 0),
    (0, 1),
    (7, 12),
    (255, 256),
    (1919, 1079),
    (65535, 65535),
];

#[test]
fn packing_a_position_round_trips_through_the_host_mirror() {
    for &(x, y) in POSITIONS {
        let (px, py) = unpack_pos_host(pack_pos_host(x, y));
        assert_eq!((px, py), (x, y), "({x}, {y}) did not survive the round trip");
    }
}

#[test]
fn a_position_packs_x_low_and_y_high() {
    let packed = pack_pos_host(7, 12);
    assert_eq!(packed & 0xFFFF, 7, "x must sit in the low half");
    assert_eq!(packed >> 16, 12, "y must sit in the high half");
}

#[test]
fn distinct_positions_pack_to_distinct_words() {
    let mut seen = std::collections::HashSet::new();
    for &(x, y) in POSITIONS {
        assert!(
            seen.insert(pack_pos_host(x, y)),
            "({x}, {y}) collided with an earlier position"
        );
    }
}

#[test]
fn the_gpu_helpers_match_their_host_mirrors() {
    let xs: Vec<u32> = POSITIONS.iter().map(|&(x, _)| x).collect();
    let ys: Vec<u32> = POSITIONS.iter().map(|&(_, y)| y).collect();
    // One coordinate per position, covering below the range, inside it,
    // and past its top, against a max of 24 (a 32-wide frame's last
    // legal 8x8 patch position).
    let coords: Vec<i32> = vec![-9, -1, 0, 1, 24, 25, 4096];
    let max_pos: Vec<u32> = vec![24; coords.len()];

    let (packed, clamped) = run_helpers(&xs, &ys, &coords, &max_pos);

    for (i, &(x, y)) in POSITIONS.iter().enumerate() {
        assert_eq!(
            packed[i],
            pack_pos_host(x, y),
            "pack_pos disagreed with pack_pos_host at ({x}, {y})"
        );
    }

    assert_eq!(
        clamped,
        vec![0, 0, 0, 1, 24, 24, 24],
        "clamp_top_left must pin every coordinate into [0, 24]"
    );
}

/// Launches [`collab_group_temporal`] over a luma frame and reads back
/// the position and count buffers.
///
/// `radius = 0` leaves the ring one frame wide, so every candidate comes
/// from the spatial window around the reference patch and the motion,
/// confidence, and neighbour-slot buffers are one- or two-element
/// dummies that nothing reads. Luma always stores one channel per line,
/// so the kernel's `Size` selector is fixed at 1 here rather than
/// threaded through as an argument.
fn run_group(
    frame: &[f32],
    width: u32,
    height: u32,
    noise_floor: f32,
    spatial_radius: u32,
    k_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(frame.len(), (width * height) as usize);

    let client = make_client();
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pos_len = member_buf_len(width, height, k_max);
    let frame_len = member_frame_buf_len(width, height, k_max);
    let sig2_len = member_sig2_buf_len(width, height, k_max);

    let reference_buf = client.create_from_slice(f32::as_bytes(frame));
    let mv_dummy = client.create_from_slice(i32::as_bytes(&[0i32, 0i32]));
    let conf_dummy = client.create_from_slice(f32::as_bytes(&[1.0f32]));
    let slots_dummy = client.create_from_slice(u32::as_bytes(&[0u32]));
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
            ArrayArg::from_raw_parts(reference_buf, frame.len()),
            ArrayArg::from_raw_parts(mv_dummy, 2),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(member_pos_buf.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame_buf, frame_len),
            ArrayArg::from_raw_parts(member_count_buf.clone(), refs),
            ArrayArg::from_raw_parts(member_sig2_buf, sig2_len),
            0u32,
            ArrayArg::from_raw_parts(slots_dummy, 1),
            noise_floor,
            0.0f32,
            1.0f32,
            1.0f32,
            0u32,
            0u32,
            2u32,
            1u32,
            8u32,
            8u32,
            1u32,
            1u32,
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

/// The best-ranked member of a group is the patch that genuinely matches
/// the reference best, not merely one the scan reached early.
///
/// One texture is planted twice, at `(4, 4)` and at `(16, 12)`, over a
/// flat background. The reference at `(4, 4)` sees its exact twin at
/// distance 0, while every other candidate in the window either sits on
/// flat background or straddles a texture edge, so all of them score far
/// above 0. `k_max = 2` keeps the self-match and exactly one other, which
/// must therefore be the twin.
#[test]
fn a_planted_twin_is_found() {
    let (w, h) = (32u32, 32u32);
    let mut frame = vec![0.2f32; (w * h) as usize];
    let texture = deterministic_texture(7);
    plant_patch(&mut frame, w, 4, 4, &texture);
    plant_patch(&mut frame, w, 16, 12, &texture);

    let (member_pos, member_count) = run_group(&frame, w, h, 0.0, 12, 2);

    let refs_x = refs_along(w);
    let ref_idx = (4 / STEP + (4 / STEP) * refs_x) as usize;

    assert_eq!(
        member_count[ref_idx], 2,
        "the reference at (4,4) should keep the self-match and one other member"
    );
    assert_eq!(
        unpack_pos_host(member_pos[ref_idx * 2]),
        (4, 4),
        "member 0 must always be the reference itself"
    );
    let (px, py) = unpack_pos_host(member_pos[ref_idx * 2 + 1]);
    assert_eq!(
        (px, py),
        (16, 12),
        "member 1 should be the planted twin at (16,12)"
    );
}

/// A reference patch pinned to the left edge (`rx == 0`) sees several
/// distinct search offsets clamp onto the same physical position, and not
/// only onto the seeded self-match: at `dy == -3` for instance, `dx ==
/// -4, -3, -2, -1, 0` all clamp (or naturally land, for `dx == 0`) on `cx
/// == 0`, giving a non-self position hit by five distinct offsets. On a
/// flat frame every candidate has distance 0, so every argmin round is a
/// tie and the kernel's documented tie-break, toward the lower candidate
/// index, fixes the winner exactly. Candidate indices run in raster order
/// over the window, so the kept members are exactly the first 8 distinct
/// positions that order reaches. That's hand-traced below.
#[test]
fn corner_clamp_collisions_are_deduplicated() {
    let (w, h) = (32u32, 32u32);
    let frame = vec![0.4f32; (w * h) as usize];
    let refs_x = refs_along(w);

    // Reference index (0, 3): rx = 0 (left edge), ry = 3 * STEP = 12,
    // comfortably clear of the top and bottom edges at radius 4 (window
    // rows 8..=16), so only the x axis clamps.
    let ref_idx = (3 * refs_x) as usize;

    let (member_pos, member_count) = run_group(&frame, w, h, 0.0, 4, 8);

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
    // clamps to cx=0 and dx=-3,-2,-1,0 are then retired along with it),
    // then row dy=-3 (cy=9) contributing cx=0 then cx=1, at which point
    // the stack is full.
    assert_eq!(
        members,
        vec![(0, 12), (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (0, 9), (1, 9)],
        "expected the exact hand-traced member order for the corner ref"
    );
}

/// Writes an 8x8 flat block into `frame` with its top-left corner at
/// `(px, py)`.
fn set_flat_block(frame: &mut [f32], w: u32, px: u32, py: u32, value: f32) {
    for row in 0..8u32 {
        for col in 0..8u32 {
            frame[((py + row) * w + (px + col)) as usize] = value;
        }
    }
}

/// The members a group KEEPS are the ones RANKED best, even when
/// `noise_floor` sits far above every candidate's raw distance.
///
/// The kernel subtracts `noise_floor` and never clamps the result at
/// zero. If it did clamp, every candidate whose raw distance sat below
/// the floor would collapse to the same `0.0`, the argmin rounds would
/// all tie, and the tie-break would keep whichever candidates the raster
/// order reached first rather than the closest ones.
///
/// This builds a reference patch and four flat candidate patches at four
/// known, widely separated raw distances (`best1 < best2 < best3 <
/// worst`), all far enough below `noise_floor` that a clamp would flatten
/// every one of them to an identical `0.0`. The candidates are placed so
/// raster order reaches them as `worst, best1, best2, best3` - the
/// reverse of true similarity order - specifically so that a tie-break
/// on scan order would keep `worst` and drop `best3`, which is exactly
/// the defect this test exists to catch.
///
/// Every candidate sits at least 8px of pure background away from every
/// other candidate and from the reference's own patch on at least one
/// axis, so no 8x8 window the kernel scans can ever straddle two of
/// them: every scanned candidate is either wholly one of the four
/// blocks, wholly background, or a block/background mix. The background
/// is `-5.0` against a reference of `0.7`, so anything carrying even one
/// background pixel scores hundreds, three orders of magnitude past the
/// four planted distances, and cannot compete for a slot.
#[test]
fn ranking_survives_a_noise_floor_that_would_clamp_every_candidate_to_zero() {
    let (w, h) = (64u32, 64u32);
    let (rx, ry) = (40u32, 40u32);
    let ref_value = 0.7f32;

    let mut frame = vec![-5.0f32; (w * h) as usize];
    set_flat_block(&mut frame, w, rx, ry, ref_value);

    // Raw distance for a flat candidate offset from the reference by a
    // constant `delta`: channel_scale (3, luma) * patch area (64) *
    // delta^2.
    //   best1 (delta 0.01): raw 0.0192
    //   best2 (delta 0.02): raw 0.0768
    //   best3 (delta 0.03): raw 0.1728
    //   worst (delta 0.15): raw 4.32
    // All four sit comfortably under noise_floor (10.0) below, so a
    // clamp would flatten every one of them to 0.0.
    set_flat_block(&mut frame, w, rx - 8, ry - 16, ref_value + 0.15); // worst
    set_flat_block(&mut frame, w, rx + 8, ry - 16, ref_value + 0.01); // best1
    set_flat_block(&mut frame, w, rx - 8, ry + 16, ref_value + 0.02); // best2
    set_flat_block(&mut frame, w, rx + 8, ry + 16, ref_value + 0.03); // best3

    let noise_floor = 10.0f32;

    let (member_pos, member_count) = run_group(&frame, w, h, noise_floor, 16, 4);

    let refs_x = refs_along(w);
    let ref_idx = ((ry / STEP) * refs_x + (rx / STEP)) as usize;

    assert_eq!(
        member_count[ref_idx], 4,
        "expected the self-match plus all 3 true best candidates to fill the stack"
    );

    let members: Vec<(u32, u32)> = (0..4)
        .map(|j| unpack_pos_host(member_pos[ref_idx * 4 + j]))
        .collect();

    assert_eq!(
        members[0],
        (rx, ry),
        "member 0 must always be the reference itself"
    );
    assert_eq!(
        members[1..],
        [(rx + 8, ry - 16), (rx - 8, ry + 16), (rx + 8, ry + 16)],
        "expected the 3 true best candidates (best1, best2, best3) in similarity order, \
         with the worst candidate evicted; got {members:?}"
    );
}
