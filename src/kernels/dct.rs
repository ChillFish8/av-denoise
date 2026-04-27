use cubecl::cube;
use cubecl::prelude::*;

#[cube]
/// 1-D DCT-II (orthonormal) applied in-place to a strided view of a tensor.
///
/// Equivalent to `scipy.fft.dct(x, type=2, norm='ortho')`.
///
/// `offset` is the index of the first element; `stride` is the step between
/// consecutive elements; `n` is the transform length (comptime).
/// `scratch` must have at least `n` elements and is used as a temporary buffer.
pub(crate) fn dct1d<F: Float>(
    x: &mut Tensor<F>,
    offset: usize,
    stride: usize,
    scratch: &mut Tensor<F>,
    #[comptime] n: usize,
) {
    let pi = F::new(core::f32::consts::PI);
    let scale0 = F::new(1.0) / F::cast_from(n).sqrt();
    let scale_k = (F::new(2.0) / F::cast_from(n)).sqrt();

    #[unroll]
    for k in 0..n {
        let mut sum = F::new(0.0);

        #[unroll]
        for m in 0..n {
            let angle = pi * F::cast_from(k) * F::cast_from(2 * m + 1)
                / (F::new(2.0) * F::cast_from(n));
            sum += x[offset + m * stride] * angle.cos();
        }

        let scale = if k == 0 { scale0 } else { scale_k };
        scratch[k] = scale * sum;
    }

    #[unroll]
    for k in 0..n {
        x[offset + k * stride] = scratch[k];
    }
}

#[cube]
/// 1-D iDCT-III (orthonormal) applied in-place to a strided view of a tensor.
///
/// Inverse of [dct1d]. Equivalent to `scipy.fft.idct(x, type=2, norm='ortho')`.
pub(crate) fn idct1d<F: Float>(
    x: &mut Tensor<F>,
    offset: usize,
    stride: usize,
    scratch: &mut Tensor<F>,
    #[comptime] n: usize,
) {
    let pi = F::new(core::f32::consts::PI);
    let scale0 = F::new(1.0) / F::cast_from(n).sqrt();
    let scale_k = (F::new(2.0) / F::cast_from(n)).sqrt();

    #[unroll]
    for m in 0..n {
        let mut sum = scale0 * x[offset];

        #[unroll]
        for k in 1..n {
            let angle = pi * F::cast_from(k) * F::cast_from(2 * m + 1)
                / (F::new(2.0) * F::cast_from(n));
            sum += scale_k * x[offset + k * stride] * angle.cos();
        }

        scratch[m] = sum;
    }

    #[unroll]
    for m in 0..n {
        x[offset + m * stride] = scratch[m];
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
        read_1d_f32_allocation,
        tensor_arg_1d_f32,
    };

    const N: usize = 4;

    #[cube(launch)]
    fn dct1d_test_kernel(
        x: &mut Tensor<f32>,
        scratch: &mut Tensor<f32>,
        #[comptime] n: usize,
    ) {
        dct1d::<f32>(x, 0, 1, scratch, n);
    }

    #[cube(launch)]
    fn idct1d_test_kernel(
        x: &mut Tensor<f32>,
        scratch: &mut Tensor<f32>,
        #[comptime] n: usize,
    ) {
        idct1d::<f32>(x, 0, 1, scratch, n);
    }

    #[test]
    fn dct1d_matches_host_reference() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let actual = run_dct(&input);
        let expected = host_dct1d(&input);
        assert_close(&actual, &expected, 1.0e-5);
    }

    #[test]
    fn dct1d_round_trip_restores_input() {
        let input = [1.5f32, -0.5, 2.25, 0.75];
        let transformed = run_dct(&input);
        let restored = run_idct(&transformed);
        assert_close(&restored, &input, 1.0e-5);
    }

    #[test]
    fn dct1d_is_orthonormal() {
        let input = [1.0f32, -2.0, 0.5, 4.0];
        let output = run_dct(&input);

        let in_energy: f32 = input.iter().map(|v| v * v).sum();
        let out_energy: f32 = output.iter().map(|v| v * v).sum();

        assert!(
            (in_energy - out_energy).abs() <= 1.0e-5,
            "energy mismatch: input={in_energy}, output={out_energy}"
        );
    }

    fn run_dct(input: &[f32]) -> Vec<f32> {
        let client = cpu_client();
        let shape = [input.len()];
        let alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(input)),
            &shape,
            size_of::<f32>(),
        );
        let scratch_alloc = client.empty_tensor(&shape, size_of::<f32>());
        dct1d_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_1d_f32(&alloc, &shape),
            tensor_arg_1d_f32(&scratch_alloc, &shape),
            N,
        )
        .expect("dct kernel should launch");
        read_1d_f32_allocation(&client, &alloc, input.len())
    }

    fn run_idct(input: &[f32]) -> Vec<f32> {
        let client = cpu_client();
        let shape = [input.len()];
        let alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(input)),
            &shape,
            size_of::<f32>(),
        );
        let scratch_alloc = client.empty_tensor(&shape, size_of::<f32>());
        idct1d_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_1d_f32(&alloc, &shape),
            tensor_arg_1d_f32(&scratch_alloc, &shape),
            N,
        )
        .expect("idct kernel should launch");
        read_1d_f32_allocation(&client, &alloc, input.len())
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
}
