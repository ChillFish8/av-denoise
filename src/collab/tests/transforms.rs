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

#[cube(launch_unchecked)]
fn safe_reciprocal_probe(denom: &Array<f32>, floor: &Array<f32>, out: &mut Array<f32>, n: u32) {
    let tid = ABSOLUTE_POS_X;
    if tid < n {
        out[tid as usize] = safe_reciprocal(denom[tid as usize], floor[tid as usize]);
    }
}

/// Pins `safe_reciprocal`'s own contract directly, rather than only
/// through whatever a caller's `f32::max` happens to do with a `NaN` on
/// this particular GPU backend.
///
/// `collab_filter_ht` and `collab_aggregate` both reach this same
/// function for their own weight and normalisation
/// divisions, so a probe of the function itself covers every call site
/// at once, and does not depend on a real denominator ever going
/// non-finite in one of those larger kernels to exercise the guard.
#[test]
fn safe_reciprocal_is_zero_for_a_non_finite_denominator_and_ordinary_otherwise() {
    let client = make_client();

    let denom = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0f32,
        5.0f32,
        -3.0f32,
    ];
    let floor = vec![1e-12f32; 6];
    let n = denom.len();

    let denom_buf = client.create_from_slice(f32::as_bytes(&denom));
    let floor_buf = client.create_from_slice(f32::as_bytes(&floor));
    let out_buf = client.empty(n * size_of::<f32>());

    unsafe {
        safe_reciprocal_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(denom_buf, n),
            ArrayArg::from_raw_parts(floor_buf, n),
            ArrayArg::from_raw_parts(out_buf.clone(), n),
            n as u32,
        );
    }

    let bytes = client.read_one(out_buf).expect("safe_reciprocal readback failed");
    let out = f32::from_bytes(&bytes)[..n].to_vec();

    assert_eq!(
        out[0], 0.0,
        "a NaN denominator must yield exactly 0, got {}",
        out[0]
    );
    assert_eq!(
        out[1], 0.0,
        "a positive-infinite denominator must yield exactly 0, got {}",
        out[1]
    );
    assert_eq!(
        out[2], 0.0,
        "a negative-infinite denominator must yield exactly 0, got {}",
        out[2]
    );
    assert_eq!(
        out[3], 1e12,
        "an ordinary zero denominator floors to 1e12, got {}",
        out[3]
    );
    assert!(
        (out[4] - 0.2).abs() < 1e-6,
        "1 / max(5, 1e-12) should be 0.2, got {}",
        out[4]
    );
    // A negative but finite denominator is not a case any real caller
    // hits (every caller here sums non-negative terms), but it still
    // has to stay finite rather than flip the sign of the result. The
    // floor outweighs it the same way it outweighs a legitimate zero.
    assert_eq!(
        out[5], 1e12,
        "a negative but finite denominator floors the same way zero does, got {}",
        out[5]
    );
}

#[cube(launch_unchecked)]
fn haar8_roundtrip_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS;
    let mut basis = SharedMemory::<f32>::new(64usize);
    fill_haar8_basis(&mut basis, tid);
    sync_cube();
    let mut line = SharedMemory::<f32>::new(8usize);
    let mut coeff = SharedMemory::<f32>::new(8usize);
    if tid == 0 {
        for i in 0..8u32 {
            line[i as usize] = input[i as usize];
        }
    }
    sync_cube();
    if tid == 0 {
        dct8_line_fwd(&basis, &line, &mut coeff, 0u32, 1u32);
        dct8_line_inv(&basis, &coeff, &mut line, 0u32, 1u32);
        for i in 0..8u32 {
            output[i as usize] = line[i as usize];
        }
    }
}

/// Writes the Haar-8 basis straight to global memory, one entry per
/// thread, so the host side can check its orthonormality directly.
#[cube(launch_unchecked)]
fn haar8_basis_kernel(output: &mut Array<f32>) {
    let tid = UNIT_POS;
    let mut basis = SharedMemory::<f32>::new(64usize);
    fill_haar8_basis(&mut basis, tid);
    sync_cube();
    if tid < 64u32 {
        output[tid as usize] = basis[tid as usize];
    }
}

fn run_haar8_roundtrip(input: &[f32; 8]) -> Vec<f32> {
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    let output_buf = client.empty(8 * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_1d(64);

    unsafe {
        haar8_roundtrip_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(input_buf, 8),
            ArrayArg::from_raw_parts(output_buf.clone(), 8),
        );
    }

    let bytes = client
        .read_one(output_buf)
        .expect("haar8 roundtrip readback failed");
    f32::from_bytes(&bytes)[..8].to_vec()
}

fn run_haar8_basis() -> Vec<f32> {
    let client = make_client();
    let output_buf = client.empty(64 * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_1d(64);

    unsafe {
        haar8_basis_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(output_buf.clone(), 64),
        );
    }

    let bytes = client.read_one(output_buf).expect("haar8 basis readback failed");
    f32::from_bytes(&bytes)[..64].to_vec()
}

/// Round-trips an arbitrary 8-vector through the Haar-8 forward and
/// inverse mat-vec helpers, then reads the basis itself back and checks
/// on the host that it is genuinely orthonormal: `B * B^T` is the
/// identity, and row 0, the DC row, is the constant `1/sqrt(8)`.
#[test]
fn haar8_basis_roundtrips_and_is_orthonormal() {
    let patch = pseudo_random_patch(3);
    let input: [f32; 8] = patch[..8].try_into().expect("patch has at least 8 entries");
    let output = run_haar8_roundtrip(&input);
    for (idx, (&want, &got)) in input.iter().zip(output.iter()).enumerate() {
        assert!((want - got).abs() < 1e-6, "idx={idx}: want {want} got {got}");
    }

    let basis = run_haar8_basis();
    for j in 0..8usize {
        for k in 0..8usize {
            let mut dot = 0.0f32;
            for i in 0..8usize {
                dot += basis[j * 8 + i] * basis[k * 8 + i];
            }
            let expected = if j == k { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "G[{j}][{k}] = {dot}, want {expected}"
            );
        }
    }

    let expected_row0 = 1.0f32 / 8.0f32.sqrt();
    for (i, &v) in basis[..8].iter().enumerate() {
        assert!(
            (v - expected_row0).abs() < 1e-6,
            "row0[{i}] = {v}, want {expected_row0}"
        );
    }
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
