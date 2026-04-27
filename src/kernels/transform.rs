use cubecl::cube;
use cubecl::prelude::*;

use super::dct::{dct1d, idct1d};

#[cube]
/// Fast Walsh-Hadamard Transform on a strided view of a tensor.
///
/// `base` is the index of the first element; `stride` is the step between
/// consecutive elements; `n` is the transform length (must be a non-zero power of 2).
/// The transform is performed in place and is orthonormally scaled.
fn fwht_strided<F: Float>(x: &mut Tensor<F>, base: usize, stride: usize, n: usize) {
    if n == 0 || (n & (n - 1)) != 0 {
        terminate!();
    }

    let mut h: usize = 1;

    while h < n {
        for i in range_stepped(0, n, 2 * h) {
            for j in 0..h {
                let left = base + (i + j) * stride;
                let right = base + (i + j + h) * stride;
                let a = x[left];
                let b = x[right];
                x[left] = a + b;
                x[right] = a - b;
            }
        }

        h *= 2;
    }

    let scale = F::cast_from(n).sqrt();

    for idx in 0..n {
        x[base + idx * stride] /= scale;
    }
}

#[cube]
/// Forward 3-D transform of a (k, k, n_blocks) stack.
///
/// 1. Apply normalised 2-D DCT to each block independently (spatial axes).
/// 2. Apply normalised 1-D WHT to every (i, j) fibre along the grouping axis.
///
/// Both sub-transforms are orthonormal, so the combined 3-D transform is
/// also orthonormal: noise variance σ² in the image domain equals σ² in the
/// transform domain. This is what makes the hard threshold λ·σ valid.
///
/// `scratch` must have at least `k` elements (used as a temporary buffer for DCT).
pub(crate) fn transform_3d(
    stack: &mut Tensor<f32>,
    scratch: &mut Tensor<f32>,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let s0 = stack.stride(0);
    let s1 = stack.stride(1);
    let s2 = stack.stride(2);

    for b in 0..n_blocks {
        #[unroll]
        for row in 0..k {
            dct1d::<f32>(stack, row * s0 + b * s2, s1, scratch, k);
        }

        #[unroll]
        for col in 0..k {
            dct1d::<f32>(stack, col * s1 + b * s2, s0, scratch, k);
        }
    }

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            fwht_strided::<f32>(stack, i * s0 + j * s1, s2, n_blocks);
        }
    }
}

#[cube]
/// Inverse 3-D transform. Reverse order: inverse WHT first, then iDCT.
///
/// `scratch` must have at least `k` elements (used as a temporary buffer for iDCT).
pub(crate) fn inverse_transform_3d(
    stack: &mut Tensor<f32>,
    scratch: &mut Tensor<f32>,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let s0 = stack.stride(0);
    let s1 = stack.stride(1);
    let s2 = stack.stride(2);

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            fwht_strided::<f32>(stack, i * s0 + j * s1, s2, n_blocks);
        }
    }

    for b in 0..n_blocks {
        #[unroll]
        for col in 0..k {
            idct1d::<f32>(stack, col * s1 + b * s2, s0, scratch, k);
        }

        #[unroll]
        for row in 0..k {
            idct1d::<f32>(stack, row * s0 + b * s2, s1, scratch, k);
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use std::mem::size_of;

    use cubecl::prelude::*;

    use super::*;
    use crate::kernels::test_util::{
        assert_close,
        cpu_client,
        f32_as_bytes,
        read_f32_allocation,
        tensor_arg_f32,
    };

    const K: usize = 4;
    const N_BLOCKS: usize = 4;

    #[cube(launch)]
    fn transform_3d_test_kernel(
        stack: &mut Tensor<f32>,
        scratch: &mut Tensor<f32>,
        n_blocks: usize,
        #[comptime] k: usize,
    ) {
        transform_3d(stack, scratch, n_blocks, k);
    }

    #[cube(launch)]
    fn inverse_transform_3d_test_kernel(
        stack: &mut Tensor<f32>,
        scratch: &mut Tensor<f32>,
        n_blocks: usize,
        #[comptime] k: usize,
    ) {
        inverse_transform_3d(stack, scratch, n_blocks, k);
    }

    #[test]
    fn transform_3d_matches_host_reference() {
        let input = sample_stack();
        let actual = run_transform(&input);
        let mut expected = input.clone();
        host_transform_3d(&mut expected, K, N_BLOCKS);
        assert_close(&actual, &expected, 1.0e-5);
    }

    #[test]
    fn transform_3d_round_trip_restores_input() {
        let input = sample_stack();
        let transformed = run_transform(&input);
        let restored = run_inverse(&transformed);
        // Accumulated f32 rounding through two DCTs and two WHTs
        assert_close(&restored, &input, 1.0e-4);
    }

    #[test]
    fn transform_3d_is_orthonormal() {
        let input = sample_stack();
        let output = run_transform(&input);

        let in_energy: f32 = input.iter().map(|v| v * v).sum();
        let out_energy: f32 = output.iter().map(|v| v * v).sum();
        let rel_err = (in_energy - out_energy).abs() / in_energy;

        assert!(
            rel_err <= 1.0e-5,
            "energy mismatch: input={in_energy}, output={out_energy}, rel_err={rel_err}"
        );
    }

    fn sample_stack() -> Vec<f32> {
        (0..(K * K * N_BLOCKS))
            .map(|i| (i as f32 + 1.0) * 0.5 - (i as f32 * 0.1).sin())
            .collect()
    }

    fn run_transform(input: &[f32]) -> Vec<f32> {
        let client = cpu_client();
        let shape = [K, K, N_BLOCKS];
        let stack_alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(input)),
            &shape,
            size_of::<f32>(),
        );
        let scratch_alloc = client.empty_tensor(&[K], size_of::<f32>());
        transform_3d_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_f32(&stack_alloc, &shape),
            tensor_arg_f32(&scratch_alloc, &[K]),
            ScalarArg::new(N_BLOCKS),
            K,
        )
        .expect("transform_3d kernel should launch");
        read_f32_allocation(&client, &stack_alloc, &shape)
    }

    fn run_inverse(input: &[f32]) -> Vec<f32> {
        let client = cpu_client();
        let shape = [K, K, N_BLOCKS];
        let stack_alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(input)),
            &shape,
            size_of::<f32>(),
        );
        let scratch_alloc = client.empty_tensor(&[K], size_of::<f32>());
        inverse_transform_3d_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_f32(&stack_alloc, &shape),
            tensor_arg_f32(&scratch_alloc, &[K]),
            ScalarArg::new(N_BLOCKS),
            K,
        )
        .expect("inverse_transform_3d kernel should launch");
        read_f32_allocation(&client, &stack_alloc, &shape)
    }

    fn host_dct1d(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let pi = core::f32::consts::PI;
        let scale0 = 1.0 / (n as f32).sqrt();
        let scale_k = (2.0 / n as f32).sqrt();

        (0..n)
            .map(|k| {
                let scale = if k == 0 { scale0 } else { scale_k };
                let sum: f32 = (0..n)
                    .map(|m| {
                        let angle = pi * k as f32 * (2 * m + 1) as f32 / (2.0 * n as f32);
                        input[m] * angle.cos()
                    })
                    .sum();
                scale * sum
            })
            .collect()
    }

    fn host_fwht(input: &[f32]) -> Vec<f32> {
        let mut values = input.to_vec();
        let n = values.len();
        let mut h = 1usize;

        while h < n {
            let mut next = values.clone();

            for i in (0..n).step_by(2 * h) {
                for j in 0..h {
                    let a = values[i + j];
                    let b = values[i + j + h];
                    next[i + j] = a + b;
                    next[i + j + h] = a - b;
                }
            }

            values = next;
            h *= 2;
        }

        let scale = (n as f32).sqrt();
        values.into_iter().map(|v| v / scale).collect()
    }

    fn host_transform_3d(stack: &mut [f32], k: usize, n_blocks: usize) {
        for b in 0..n_blocks {
            for row in 0..k {
                let mut row_vals: Vec<f32> = (0..k)
                    .map(|col| stack[row * k * n_blocks + col * n_blocks + b])
                    .collect();
                row_vals = host_dct1d(&row_vals);
                for col in 0..k {
                    stack[row * k * n_blocks + col * n_blocks + b] = row_vals[col];
                }
            }
            for col in 0..k {
                let mut col_vals: Vec<f32> = (0..k)
                    .map(|row| stack[row * k * n_blocks + col * n_blocks + b])
                    .collect();
                col_vals = host_dct1d(&col_vals);
                for row in 0..k {
                    stack[row * k * n_blocks + col * n_blocks + b] = col_vals[row];
                }
            }
        }

        for i in 0..k {
            for j in 0..k {
                let mut fibre: Vec<f32> = (0..n_blocks)
                    .map(|b| stack[i * k * n_blocks + j * n_blocks + b])
                    .collect();
                fibre = host_fwht(&fibre);
                for b in 0..n_blocks {
                    stack[i * k * n_blocks + j * n_blocks + b] = fibre[b];
                }
            }
        }
    }
}
