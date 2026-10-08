use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::MAX_K;
use crate::collab::kernels::transforms::*;

#[cube(launch_unchecked)]
fn safe_reciprocal_probe(denom: &Array<f32>, floor: &Array<f32>, out: &mut Array<f32>, count: u32) {
    let tid = ABSOLUTE_POS_X;
    if tid < count {
        out[tid as usize] = safe_reciprocal(denom[tid as usize], floor[tid as usize]);
    }
}

/// Probes `safe_reciprocal` itself, so the test checks the guard without relying on how a caller's
/// `f32::max` treats a `NaN` on this backend.
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
    let count = denom.len();

    let denom_bytes = f32::as_bytes(&denom);
    let floor_bytes = f32::as_bytes(&floor);
    let denom_buf = client.create_from_slice(denom_bytes);
    let floor_buf = client.create_from_slice(floor_bytes);
    let out_buf = client.empty(count * size_of::<f32>());

    unsafe {
        safe_reciprocal_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(count as u32),
            ArrayArg::from_raw_parts(denom_buf, count),
            ArrayArg::from_raw_parts(floor_buf, count),
            ArrayArg::from_raw_parts(out_buf.clone(), count),
            count as u32,
        );
    }

    let out_bytes = client.read_one(out_buf).expect("safe_reciprocal readback failed");
    let reciprocals = f32::from_bytes(&out_bytes)[..count].to_vec();

    assert_eq!(
        reciprocals[0], 0.0,
        "a NaN denominator must yield exactly 0, got {}",
        reciprocals[0]
    );
    assert_eq!(
        reciprocals[1], 0.0,
        "a positive-infinite denominator must yield exactly 0, got {}",
        reciprocals[1]
    );
    assert_eq!(
        reciprocals[2], 0.0,
        "a negative-infinite denominator must yield exactly 0, got {}",
        reciprocals[2]
    );
    assert_eq!(
        reciprocals[3], 1e12,
        "an ordinary zero denominator floors to 1e12, got {}",
        reciprocals[3]
    );
    assert!(
        (reciprocals[4] - 0.2).abs() < 1e-6,
        "1 / max(5, 1e-12) should be 0.2, got {}",
        reciprocals[4]
    );
    // Callers only sum non-negative terms, but a negative finite denominator must still stay finite
    // rather than flip the result's sign. The floor outweighs it the same way it outweighs a zero.
    assert_eq!(
        reciprocals[5], 1e12,
        "a negative but finite denominator floors the same way zero does, got {}",
        reciprocals[5]
    );
}

/// Runs the same three-level variance ladder as
/// [collab_fused](crate::collab::kernels::fused::collab_fused), in the same pairing order.
#[cube(launch_unchecked)]
fn variance_ladder_kernel(input: &Array<f32>, k_use: u32, output: &mut Array<f32>) {
    let mut variances = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..MAX_K {
        variances[k as usize] = input[k as usize];
    }
    if k_use >= 8u32 {
        variance_reg_level(&mut variances, 8u32);
    }
    if k_use >= 4u32 {
        variance_reg_level(&mut variances, 4u32);
    }
    if k_use >= 2u32 {
        variance_reg_level(&mut variances, 2u32);
    }
    #[unroll]
    for k in 0..MAX_K {
        output[k as usize] = variances[k as usize];
    }
}

fn run_variance_ladder(sig2: &[f32; 8], k_use: u32) -> Vec<f32> {
    let client = make_client();
    let input_bytes = f32::as_bytes(sig2);
    let input_buf = client.create_from_slice(input_bytes);
    let output_buf = client.empty(8 * size_of::<f32>());

    unsafe {
        variance_ladder_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_2d(1, 1),
            ArrayArg::from_raw_parts(input_buf, 8),
            k_use,
            ArrayArg::from_raw_parts(output_buf.clone(), 8),
        );
    }

    let output_bytes = client
        .read_one(output_buf)
        .expect("variance ladder readback failed");

    f32::from_bytes(&output_bytes)[..8].to_vec()
}

/// Uniform input sits at a fixed point under any pairing order, so only non-uniform input can catch
/// a level or pairing mismatch against [haar_variance_ladder].
#[test]
fn gpu_variance_ladder_matches_the_host_mirror() {
    let sig2 = [0.7f32, 1.3, 0.2, 2.5, 0.05, 3.0, 1.1, 0.4];

    for k_use in [1u32, 2, 4, 8] {
        let host = haar_variance_ladder(&sig2, k_use);
        let gpu = run_variance_ladder(&sig2, k_use);

        for idx in 0..8usize {
            assert!(
                (host[idx] - gpu[idx]).abs() < 1e-5,
                "k_use={k_use} idx={idx}: host {} gpu {}",
                host[idx],
                gpu[idx]
            );
        }
    }
}

#[cube(launch_unchecked)]
fn dct8_probe(input: &Array<f32>, output: &mut Array<f32>, #[comptime] round_trip: bool) {
    let line_start = UNIT_POS_X * 8u32;
    let mut line = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0..8u32 {
        line[i as usize] = input[(line_start + i) as usize];
    }

    dct8_reg_fwd(&mut line);
    if comptime!(round_trip) {
        dct8_reg_inv(&mut line);
    }

    #[unroll]
    for i in 0..8u32 {
        output[(line_start + i) as usize] = line[i as usize];
    }
}

/// Every impulse, so each basis column is checked on its own, plus three mixed lines.
fn dct8_test_lines() -> Vec<[f32; 8]> {
    let mut lines = Vec::new();
    for position in 0..8 {
        let mut impulse = [0.0f32; 8];
        impulse[position] = 1.0;
        lines.push(impulse);
    }

    let ramp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(|step: f32| step / 7.0);
    lines.push(ramp);
    lines.push([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);
    lines.push([0.31, 0.29, 0.35, 0.62, 0.66, 0.30, 0.28, 0.33]);
    lines
}

fn run_dct8_probe(lines: &[[f32; 8]], round_trip: bool) -> Vec<[f32; 8]> {
    let client = make_client();
    let flat: Vec<f32> = lines.iter().flatten().copied().collect();
    let input_bytes = f32::as_bytes(&flat);
    let input_buf = client.create_from_slice(input_bytes);
    let output_buf = client.empty(flat.len() * size_of::<f32>());

    unsafe {
        dct8_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(lines.len() as u32),
            ArrayArg::from_raw_parts(input_buf, flat.len()),
            ArrayArg::from_raw_parts(output_buf.clone(), flat.len()),
            round_trip,
        );
    }

    let output_bytes = client.read_one(output_buf).expect("dct8 probe readback failed");
    let values = &f32::from_bytes(&output_bytes)[..flat.len()];
    let (chunks, _remainder) = values.as_chunks::<8>();
    chunks.to_vec()
}

fn host_dct8(line: &[f32; 8]) -> [f32; 8] {
    let mut coefficients = [0.0f32; 8];
    for (row, coefficient) in coefficients.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (col, &value) in line.iter().enumerate() {
            let entry = dct8_basis(row as u32, col as u32);
            sum += entry * value as f64;
        }
        *coefficient = sum as f32;
    }
    coefficients
}

#[test]
fn gpu_dct8_forward_matches_the_host_basis_product() {
    let lines = dct8_test_lines();
    let gpu_lines = run_dct8_probe(&lines, false);

    for (line_index, (line, gpu)) in lines.iter().zip(gpu_lines.iter()).enumerate() {
        let expected = host_dct8(line);
        for row in 0..8 {
            assert!(
                (gpu[row] - expected[row]).abs() < 1e-6,
                "line {line_index} coefficient {row}: gpu {} host {}",
                gpu[row],
                expected[row],
            );
        }
    }
}

#[test]
fn gpu_dct8_inverse_undoes_the_forward() {
    let lines = dct8_test_lines();
    let gpu_lines = run_dct8_probe(&lines, true);

    for (line_index, (line, gpu)) in lines.iter().zip(gpu_lines.iter()).enumerate() {
        for position in 0..8 {
            assert!(
                (gpu[position] - line[position]).abs() < 1e-6,
                "line {line_index} position {position}: got {} want {}",
                gpu[position],
                line[position],
            );
        }
    }
}

#[test]
fn dct8_basis_rows_are_orthonormal() {
    for first in 0..8u32 {
        for second in 0..8u32 {
            let mut dot = 0.0f64;
            for col in 0..8u32 {
                let first_entry = dct8_basis(first, col);
                let second_entry = dct8_basis(second, col);
                dot += first_entry * second_entry;
            }

            let expected = if first == second { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "rows {first} and {second}: dot {dot}"
            );
        }
    }
}
