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

/// Pins `safe_reciprocal`'s own contract directly, rather than only
/// through whatever a caller's `f32::max` happens to do with a `NaN` on
/// this particular GPU backend.
///
/// `collab_fused` and `collab_aggregate` both reach this same function
/// for their own weight and normalisation divisions, so a probe of the
/// function itself covers every call site at once, and does not depend
/// on a real denominator ever going non-finite in one of those larger
/// kernels to exercise the guard.
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

/// Runs the same three-level variance ladder [`collab_fused`] runs, so
/// the host mirror is checked against the levels and pairing order the
/// shipped kernel actually applies.
///
/// [`collab_fused`]: crate::collab::kernels::fused::collab_fused
#[cube(launch_unchecked)]
fn variance_ladder_kernel(input: &Array<f32>, k_use: u32, output: &mut Array<f32>) {
    let mut v = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..MAX_K {
        v[k as usize] = input[k as usize];
    }
    if k_use >= 8u32 {
        variance_reg_level(&mut v, 8u32);
    }
    if k_use >= 4u32 {
        variance_reg_level(&mut v, 4u32);
    }
    if k_use >= 2u32 {
        variance_reg_level(&mut v, 2u32);
    }
    #[unroll]
    for k in 0..MAX_K {
        output[k as usize] = v[k as usize];
    }
}

fn run_variance_ladder(sig2: &[f32; 8], k_use: u32) -> Vec<f32> {
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(sig2));
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

    let bytes = client
        .read_one(output_buf)
        .expect("variance ladder readback failed");
    f32::from_bytes(&bytes)[..8].to_vec()
}

/// Pins the GPU ladder against the host mirror [`haar_variance_ladder`],
/// for every valid `k_use`, with non-uniform input variances.
///
/// Uniform input would leave the ladder at a fixed point regardless of
/// pairing order, so it could not catch a level or pairing mismatch
/// between the two implementations. Only a non-uniform input can.
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
