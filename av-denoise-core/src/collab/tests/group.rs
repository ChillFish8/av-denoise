use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::kernels::group::{
    clamp_top_left,
    pack_pos,
    pack_pos_host,
    pack_pos_t,
    pack_pos_t_host,
    unpack_pos_host,
    unpack_t_host,
};

/// Runs [pack_pos], [pack_pos_t] and [clamp_top_left] on the GPU, one input per thread.
///
/// This checks the host mirrors against the kernels that consume them rather than against themselves.
#[cube(launch_unchecked)]
fn group_helpers_kernel(
    xs: &Array<u32>,
    ys: &Array<u32>,
    ts: &Array<u32>,
    coords: &Array<i32>,
    max_pos: &Array<u32>,
    packed: &mut Array<u32>,
    packed_t: &mut Array<u32>,
    clamped: &mut Array<u32>,
    #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < n {
        packed[i as usize] = pack_pos(xs[i as usize], ys[i as usize]);
        packed_t[i as usize] = pack_pos_t(xs[i as usize], ys[i as usize], ts[i as usize]);
        clamped[i as usize] = clamp_top_left(coords[i as usize], max_pos[i as usize]);
    }
}

fn run_helpers(
    xs: &[u32],
    ys: &[u32],
    ts: &[u32],
    coords: &[i32],
    max_pos: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let count = xs.len();
    assert_eq!(ys.len(), count);
    assert_eq!(ts.len(), count);
    assert_eq!(coords.len(), count);
    assert_eq!(max_pos.len(), count);

    let client = make_client();
    let xs_bytes = u32::as_bytes(xs);
    let ys_bytes = u32::as_bytes(ys);
    let ts_bytes = u32::as_bytes(ts);
    let coords_bytes = i32::as_bytes(coords);
    let max_bytes = u32::as_bytes(max_pos);
    let xs_buf = client.create_from_slice(xs_bytes);
    let ys_buf = client.create_from_slice(ys_bytes);
    let ts_buf = client.create_from_slice(ts_bytes);
    let coords_buf = client.create_from_slice(coords_bytes);
    let max_buf = client.create_from_slice(max_bytes);
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "count is the element count these outputs hold, not xs's byte length"
    )]
    let packed_buf = client.empty(count * size_of::<u32>());
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "count is the element count these outputs hold, not xs's byte length"
    )]
    let packed_t_buf = client.empty(count * size_of::<u32>());
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "count is the element count these outputs hold, not xs's byte length"
    )]
    let clamped_buf = client.empty(count * size_of::<u32>());

    unsafe {
        group_helpers_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(xs_buf, count),
            ArrayArg::from_raw_parts(ys_buf, count),
            ArrayArg::from_raw_parts(ts_buf, count),
            ArrayArg::from_raw_parts(coords_buf, count),
            ArrayArg::from_raw_parts(max_buf, count),
            ArrayArg::from_raw_parts(packed_buf.clone(), count),
            ArrayArg::from_raw_parts(packed_t_buf.clone(), count),
            ArrayArg::from_raw_parts(clamped_buf.clone(), count),
            count as u32,
        );
    }

    let packed_bytes = client.read_one(packed_buf).expect("packed readback failed");
    let packed_t_bytes = client.read_one(packed_t_buf).expect("packed_t readback failed");
    let clamped_bytes = client.read_one(clamped_buf).expect("clamped readback failed");
    let packed = u32::from_bytes(&packed_bytes)[..count].to_vec();
    let packed_t = u32::from_bytes(&packed_t_bytes)[..count].to_vec();
    let clamped = u32::from_bytes(&clamped_bytes)[..count].to_vec();

    (packed, packed_t, clamped)
}

/// Positions spanning both halves of the packed word, including the largest value each 13-bit field
/// holds.
const POSITIONS: &[(u32, u32)] = &[
    (0, 0),
    (1, 0),
    (0, 1),
    (7, 12),
    (255, 256),
    (1919, 1079),
    (8191, 8191),
];

#[test]
fn packing_a_position_round_trips_through_the_host_mirror() {
    for &(x, y) in POSITIONS {
        let packed = pack_pos_host(x, y);
        let unpacked = unpack_pos_host(packed);
        assert_eq!(unpacked, (x, y), "({x}, {y}) did not survive the round trip");

        // A neighbour index in the field above y must not leak into the coordinates.
        let packed_with_t = pack_pos_t_host(x, y, 4);
        let unpacked_with_t = unpack_pos_host(packed_with_t);
        assert_eq!(
            unpacked_with_t,
            (x, y),
            "t=4 leaked into the coordinates for ({x}, {y})"
        );
    }
}

#[test]
fn pack_pos_t_round_trips_and_leaves_the_coordinates_readable() {
    for &(x, y, t) in &[
        (0u32, 0u32, 0u32),
        (1919, 1079, 0),
        (1912, 1072, 4),
        (7, 3, 1),
        // All three fields at their maximum at once, so none can bleed into another at saturation.
        (8191, 8191, 63),
    ] {
        let packed = pack_pos_t_host(x, y, t);
        let coords = unpack_pos_host(packed);
        let unpacked_t = unpack_t_host(packed);
        assert_eq!(coords, (x, y), "coords for ({x},{y},{t})");
        assert_eq!(unpacked_t, t, "t for ({x},{y},{t})");
    }
}

/// A centre-frame position packs a zero, which lets the filter stage tell it apart from a
/// motion-predicted member without a second array.
#[test]
fn pack_pos_t_agrees_with_pack_pos_at_t_zero() {
    let packed_t = pack_pos_t_host(120, 400, 0);
    let packed = pack_pos_host(120, 400);
    assert_eq!(packed_t, packed);
}

#[test]
fn a_position_packs_x_low_and_y_high() {
    let packed = pack_pos_host(7, 12);
    assert_eq!(packed & 0x1FFF, 7, "x must sit in the low 13 bits");
    assert_eq!(packed >> 13, 12, "y must sit in the next 13 bits");
}

#[test]
fn distinct_positions_pack_to_distinct_words() {
    let mut seen = std::collections::HashSet::new();
    for &(x, y) in POSITIONS {
        let packed = pack_pos_host(x, y);
        assert!(
            seen.insert(packed),
            "({x}, {y}) collided with an earlier position"
        );
    }
}

#[test]
fn the_gpu_helpers_match_their_host_mirrors() {
    let xs: Vec<u32> = POSITIONS.iter().map(|&(x, _)| x).collect();
    let ys: Vec<u32> = POSITIONS.iter().map(|&(_, y)| y).collect();
    // One t per position, covering the centre-frame value and a spread of neighbour indices.
    let ts: Vec<u32> = vec![0, 1, 4, 0, 2, 0, 3];
    // One coordinate per position, covering below the range, inside it and past its top, against a
    // max of 24 (a 32-wide frame's last legal 8x8 patch position).
    let coords: Vec<i32> = vec![-9, -1, 0, 1, 24, 25, 4096];
    let max_pos: Vec<u32> = vec![24; coords.len()];

    let (packed, packed_t, clamped) = run_helpers(&xs, &ys, &ts, &coords, &max_pos);

    for (i, &(x, y)) in POSITIONS.iter().enumerate() {
        let expected = pack_pos_host(x, y);
        let expected_t = pack_pos_t_host(x, y, ts[i]);
        assert_eq!(
            packed[i], expected,
            "pack_pos disagreed with pack_pos_host at ({x}, {y})"
        );
        assert_eq!(
            packed_t[i], expected_t,
            "pack_pos_t disagreed with pack_pos_t_host at ({x}, {y}, {})",
            ts[i]
        );
    }

    assert_eq!(
        clamped,
        vec![0, 0, 0, 1, 24, 24, 24],
        "clamp_top_left must pin every coordinate into [0, 24]"
    );
}
