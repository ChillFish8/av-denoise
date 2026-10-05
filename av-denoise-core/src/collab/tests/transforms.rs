use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::MAX_K;
use crate::collab::kernels::transforms::*;

#[cube(launch_unchecked)]
fn safe_reciprocal_probe(denom: &Array<f32>, floor: &Array<f32>, out: &mut Array<f32>, n: u32) {
    let tid = ABSOLUTE_POS_X;
    if tid < n {
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
