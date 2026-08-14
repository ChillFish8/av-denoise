use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::kernels::transforms::*;
use crate::collab::{MAX_K, PATCH_AREA};

/// A deterministic, spatially varied 8x8 patch, so the round trip and
/// Parseval checks exercise every basis row rather than just the DC
/// term.
fn pseudo_random_patch(seed: u32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (idx, v) in out.iter_mut().enumerate() {
        let mut hash = (idx as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EB_CA6B);
        hash ^= hash >> 13;
        *v = (hash as f32 / u32::MAX as f32) - 0.5;
    }
    out
}

#[cube(launch_unchecked)]
fn dct8_roundtrip_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let mut basis = SharedMemory::<f32>::new(64usize);
    let mut a = SharedMemory::<f32>::new(64usize);
    let mut b = SharedMemory::<f32>::new(64usize);
    let tid = UNIT_POS_Y * 8 + UNIT_POS_X;
    fill_dct8_basis(&mut basis, tid);
    a[tid as usize] = input[tid as usize];
    sync_cube();
    // Rows then columns, forward.
    if tid < 8 {
        dct8_line_fwd(&basis, &a, &mut b, tid * 8, 1);
    }
    sync_cube();
    if tid < 8 {
        dct8_line_fwd(&basis, &b, &mut a, tid, 8);
    }
    sync_cube();
    // Columns then rows, inverse.
    if tid < 8 {
        dct8_line_inv(&basis, &a, &mut b, tid, 8);
    }
    sync_cube();
    if tid < 8 {
        dct8_line_inv(&basis, &b, &mut a, tid * 8, 1);
    }
    sync_cube();
    output[tid as usize] = a[tid as usize];
}

/// Just the forward half of `dct8_roundtrip_kernel`, so the output is
/// the 2D DCT coefficients rather than a round-tripped patch.
#[cube(launch_unchecked)]
fn dct8_forward_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let mut basis = SharedMemory::<f32>::new(64usize);
    let mut a = SharedMemory::<f32>::new(64usize);
    let mut b = SharedMemory::<f32>::new(64usize);
    let tid = UNIT_POS_Y * 8 + UNIT_POS_X;
    fill_dct8_basis(&mut basis, tid);
    a[tid as usize] = input[tid as usize];
    sync_cube();
    if tid < 8 {
        dct8_line_fwd(&basis, &a, &mut b, tid * 8, 1);
    }
    sync_cube();
    if tid < 8 {
        dct8_line_fwd(&basis, &b, &mut a, tid, 8);
    }
    sync_cube();
    output[tid as usize] = a[tid as usize];
}

/// Forward then inverse stack Haar for one `k_use`, over a full
/// `MAX_K * PATCH_AREA` buffer. Threads beyond `k_use` never touch
/// slots outside `[0, k_use * PATCH_AREA)`, so the rest of the buffer
/// passes through untouched.
#[cube(launch_unchecked)]
fn haar_roundtrip_kernel(input: &Array<f32>, output: &mut Array<f32>, k_use: u32) {
    let mut stack = SharedMemory::<f32>::new(512usize);
    let pos = UNIT_POS_Y * 8 + UNIT_POS_X;

    #[unroll]
    for k in 0..8u32 {
        if k < k_use {
            stack[(k * 64 + pos) as usize] = input[(k * 64 + pos) as usize];
        }
    }
    sync_cube();

    haar_fwd_stack(&mut stack, pos, k_use);
    sync_cube();
    haar_inv_stack(&mut stack, pos, k_use);
    sync_cube();

    #[unroll]
    for k in 0..8u32 {
        if k < k_use {
            output[(k * 64 + pos) as usize] = stack[(k * 64 + pos) as usize];
        }
    }
}

fn run_dct8_roundtrip(input: &[f32; 64]) -> Vec<f32> {
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    let output_buf = client.empty(64 * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        dct8_roundtrip_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(input_buf, 64),
            ArrayArg::from_raw_parts(output_buf.clone(), 64),
        );
    }

    let bytes = client
        .read_one(output_buf)
        .expect("dct roundtrip readback failed");
    f32::from_bytes(&bytes)[..64].to_vec()
}

fn run_dct8_forward(input: &[f32; 64]) -> Vec<f32> {
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    let output_buf = client.empty(64 * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        dct8_forward_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(input_buf, 64),
            ArrayArg::from_raw_parts(output_buf.clone(), 64),
        );
    }

    let bytes = client.read_one(output_buf).expect("dct forward readback failed");
    f32::from_bytes(&bytes)[..64].to_vec()
}

fn run_haar_roundtrip(input: &[f32], k_use: u32) -> Vec<f32> {
    let len = (MAX_K * PATCH_AREA) as usize;
    assert_eq!(input.len(), len);
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    let output_buf = client.empty(len * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        haar_roundtrip_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(input_buf, len),
            ArrayArg::from_raw_parts(output_buf.clone(), len),
            k_use,
        );
    }

    let bytes = client
        .read_one(output_buf)
        .expect("haar roundtrip readback failed");
    f32::from_bytes(&bytes)[..len].to_vec()
}

#[test]
fn dct8_round_trip_recovers_a_random_patch() {
    let input = pseudo_random_patch(1);
    let output = run_dct8_roundtrip(&input);

    for (idx, (&want, &got)) in input.iter().zip(output.iter()).enumerate() {
        assert!((want - got).abs() < 1e-4, "idx={idx}: want {want} got {got}");
    }
}

#[test]
fn dct8_forward_of_a_flat_patch_has_only_a_dc_coefficient() {
    let v = 0.37f32;
    let input = [v; 64];
    let coeffs = run_dct8_forward(&input);

    let expected_dc = 8.0 * v;
    assert!(
        (coeffs[0] - expected_dc).abs() < 1e-4,
        "dc: got {} want {expected_dc}",
        coeffs[0]
    );
    for (idx, &c) in coeffs.iter().enumerate().skip(1) {
        assert!(c.abs() < 1e-4, "idx={idx}: expected ~0 ac coefficient, got {c}");
    }
}

#[test]
fn dct8_forward_preserves_the_sum_of_squares() {
    let input = pseudo_random_patch(2);
    let coeffs = run_dct8_forward(&input);

    let input_energy: f32 = input.iter().map(|v| v * v).sum();
    let coeff_energy: f32 = coeffs.iter().map(|v| v * v).sum();

    let rel_diff = (input_energy - coeff_energy).abs() / input_energy;
    assert!(
        rel_diff < 1e-3,
        "parseval violated: input energy {input_energy} vs coefficient energy {coeff_energy}"
    );
}

#[test]
fn haar_stack_round_trip_recovers_the_input_for_every_k_use() {
    for k_use in [1u32, 2, 4, 8] {
        let len = (MAX_K * PATCH_AREA) as usize;
        let mut input = vec![0.0f32; len];
        for (idx, v) in input.iter_mut().enumerate() {
            *v = pseudo_random_patch((idx as u32).wrapping_add(k_use))[idx % 64];
        }

        let output = run_haar_roundtrip(&input, k_use);

        let active = (k_use * PATCH_AREA) as usize;
        for idx in 0..active {
            assert!(
                (input[idx] - output[idx]).abs() < 1e-4,
                "k_use={k_use} idx={idx}: want {} got {}",
                input[idx],
                output[idx]
            );
        }
    }
}
