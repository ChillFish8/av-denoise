use cubecl::prelude::*;

use crate::collab::kernels::fused::subpel::{already_claimed, quarter_sample, subpel_candidate_distance};
use crate::collab::tests::helpers::{R, make_client};
use crate::nl4d::subpel::{half_pair, phase_planes_host};
use crate::nlmeans::kernels::helpers::read_line;

#[cube(launch_unchecked)]
fn quarter_probe(
    phase_ring: &Array<Vector<f32, Const<1>>>,
    out: &mut Array<f32>,
    qx: u32,
    qy: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let sample = quarter_sample(phase_ring, qx, qy, 0u32, width, height);
    out[0] = sample[0];
}

#[cube(launch_unchecked)]
fn distance_probe(
    ring: &Array<Vector<f32, Const<1>>>,
    phase_ring: &Array<Vector<f32, Const<1>>>,
    out: &mut Array<f32>,
    x: u32,
    y: u32,
    phase: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let sub = UNIT_POS_X % 8u32;
    let mut current = Array::<f32>::new(8usize);
    #[unroll]
    for r in 0..8u32 {
        let pixel = read_line(ring, 4u32 + sub, 4u32 + r, 0u32, width, height);
        current[r as usize] = pixel[0];
    }
    let distance = subpel_candidate_distance(
        phase_ring, &current, x, y, phase, 0u32, sub, 1.0f32, width, height, 1u32,
    );
    if UNIT_POS_X == 0u32 {
        out[0] = distance;
    }
}

#[cube(launch_unchecked)]
fn claim_probe(
    member_pos: &Array<u32>,
    member_phase: &Array<u32>,
    out: &mut Array<u32>,
    packed: u32,
    phase: u32,
) {
    let claimed = already_claimed(member_pos, member_phase, packed, phase, 2u32);
    out[0] = select(claimed, 1u32, 0u32);
}

/// A textured frame with values in `[0.25, 0.75]`, small enough that
/// f32 rounding never threatens the tolerances below.
fn textured_frame(width: u32, height: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut hash = idx.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            frame[idx as usize] = 0.25 + (hash as f32 / u32::MAX as f32) * 0.5;
        }
    }
    frame
}

/// The phase-ring sample at half-pel coordinates `(hx, hy)`, read the
/// same way [crate::collab::kernels::fused::subpel::half_sample] does.
fn half_sample_host(planes: &[Vec<f32>; 4], width: u32, hx: u32, hy: u32) -> f32 {
    let plane = (hx & 1) + 2 * (hy & 1);
    let px = hx >> 1;
    let py = hy >> 1;
    planes[plane as usize][(py * width + px) as usize]
}

/// The host mirror of [crate::collab::kernels::fused::subpel::quarter_sample].
fn quarter_sample_host(planes: &[Vec<f32>; 4], width: u32, qx: u32, qy: u32) -> f32 {
    let (ax, ay, bx, by) = half_pair(qx, qy);
    let first = half_sample_host(planes, width, ax, ay);
    let second = half_sample_host(planes, width, bx, by);
    (first + second) * 0.5
}

/// Packs four phase planes back to back, the layout `phase_ring`
/// addresses for one ring slot.
fn pack_phase_ring(planes: &[Vec<f32>; 4]) -> Vec<f32> {
    let mut packed = Vec::with_capacity(planes[0].len() * 4);
    for plane in planes {
        packed.extend_from_slice(plane);
    }
    packed
}

#[test]
fn quarter_sample_averages_the_two_nearest_half_grid_samples() {
    let width = 24u32;
    let height = 20u32;
    let frame = textured_frame(width, height);
    let planes = phase_planes_host(&frame, width, height, 1);
    let phase_ring_data = pack_phase_ring(&planes);
    let phase_ring_len = phase_ring_data.len();

    let client = make_client();
    let phase_ring_buf = client.create_from_slice(f32::as_bytes(&phase_ring_data));

    for qy in 16..24u32 {
        for qx in 16..24u32 {
            let out_buf = client.empty(size_of::<f32>());

            unsafe {
                quarter_probe::launch_unchecked::<R>(
                    &client,
                    CubeCount::new_1d(1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(phase_ring_buf.clone(), phase_ring_len),
                    ArrayArg::from_raw_parts(out_buf.clone(), 1),
                    qx,
                    qy,
                    width,
                    height,
                );
            }

            let out = client.read_one(out_buf).expect("quarter probe readback failed");
            let got = f32::from_bytes(&out)[0];
            let expected = quarter_sample_host(&planes, width, qx, qy);
            assert!(
                (got - expected).abs() < 1e-6,
                "qx={qx} qy={qy}: got {got}, expected {expected}"
            );
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, shape, or geometry the probe kernel binds"
)]
fn run_distance_probe(
    client: &ComputeClient<R>,
    ring_buf: &cubecl::server::Handle,
    ring_len: usize,
    phase_ring_buf: &cubecl::server::Handle,
    phase_ring_len: usize,
    x: u32,
    y: u32,
    phase: u32,
    width: u32,
    height: u32,
) -> f32 {
    let out_buf = client.empty(size_of::<f32>());

    unsafe {
        distance_probe::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(ring_buf.clone(), ring_len),
            ArrayArg::from_raw_parts(phase_ring_buf.clone(), phase_ring_len),
            ArrayArg::from_raw_parts(out_buf.clone(), 1),
            x,
            y,
            phase,
            width,
            height,
        );
    }

    let out = client.read_one(out_buf).expect("distance probe readback failed");
    f32::from_bytes(&out)[0]
}

#[test]
fn subpel_distance_matches_the_host_at_integer_and_half_phase() {
    let width = 24u32;
    let height = 20u32;
    let frame = textured_frame(width, height);
    let planes = phase_planes_host(&frame, width, height, 1);
    let phase_ring_data = pack_phase_ring(&planes);
    let phase_ring_len = phase_ring_data.len();

    let client = make_client();
    let ring_buf = client.create_from_slice(f32::as_bytes(&frame));
    let ring_len = frame.len();
    let phase_ring_buf = client.create_from_slice(f32::as_bytes(&phase_ring_data));

    let self_match = run_distance_probe(
        &client,
        &ring_buf,
        ring_len,
        &phase_ring_buf,
        phase_ring_len,
        4,
        4,
        0,
        width,
        height,
    );
    assert!(self_match.abs() < 1e-5, "self-match distance was {self_match}");

    let mut expected = 0.0f32;
    for row in 0..8u32 {
        for col in 0..8u32 {
            let reference = frame[((4 + row) * width + (4 + col)) as usize];
            let candidate = quarter_sample_host(&planes, width, 4 * (4 + col) + 2, 4 * (4 + row));
            let diff = reference - candidate;
            expected += diff * diff;
        }
    }

    let got = run_distance_probe(
        &client,
        &ring_buf,
        ring_len,
        &phase_ring_buf,
        phase_ring_len,
        4,
        4,
        2,
        width,
        height,
    );
    assert!(
        (got - expected).abs() < 1e-5,
        "phase 2 distance: got {got}, expected {expected}"
    );
}

fn run_claim_probe(member_pos: &[u32; 2], member_phase: &[u32; 2], packed: u32, phase: u32) -> bool {
    let client = make_client();
    let member_pos_buf = client.create_from_slice(u32::as_bytes(member_pos));
    let member_phase_buf = client.create_from_slice(u32::as_bytes(member_phase));
    let out_buf = client.empty(size_of::<u32>());

    unsafe {
        claim_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(member_pos_buf, 2),
            ArrayArg::from_raw_parts(member_phase_buf, 2),
            ArrayArg::from_raw_parts(out_buf.clone(), 1),
            packed,
            phase,
        );
    }

    let out = client.read_one(out_buf).expect("claim probe readback failed");
    u32::from_bytes(&out)[0] == 1
}

#[test]
fn a_position_is_claimed_only_at_the_same_phase() {
    let p = 0x1000u32;
    let q = 0x2000u32;
    let member_pos = [p, q];
    let member_phase = [0u32, 2u32];

    assert!(run_claim_probe(&member_pos, &member_phase, p, 0));
    assert!(!run_claim_probe(&member_pos, &member_phase, p, 2));
    assert!(run_claim_probe(&member_pos, &member_phase, q, 2));
    assert!(!run_claim_probe(&member_pos, &member_phase, q, 0));
}
