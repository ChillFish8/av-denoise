use cubecl::cube;
use cubecl::prelude::*;

#[cube]
/// Fast Walsh-Hadamard Transform (orthonormal) of a 1-D scalar tensor.
///
/// The transform is performed in place. The tensor length must be a non-zero
/// power of two.
pub(crate) fn fwht<F: Float>(x: &mut Tensor<F>) {
    let n = x.len();

    if n == 0 || (n & (n - 1)) != 0 {
        terminate!();
    }

    let mut h = 1usize;

    while h < n {
        for i in range_stepped(0usize, n, 2 * h) {
            for j in range(0usize, h) {
                let left = i + j;
                let right = left + h;
                let a = x[left];
                let b = x[right];

                x[left] = a + b;
                x[right] = a - b;
            }
        }

        h *= 2;
    }

    let scale = F::cast_from(n).sqrt();

    for index in range(0usize, n) {
        x[index] /= scale;
    }
}

#[cube]
/// Inverse Walsh-Hadamard Transform.
///
/// The orthonormal Walsh-Hadamard matrix is symmetric, so the inverse is the
/// same operation as the forward transform.
pub(crate) fn ifwht<F: Float>(x: &mut Tensor<F>) {
    fwht::<F>(x);
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

    #[cube(launch)]
    fn fwht_test_kernel(result: &mut Tensor<f32>) {
        fwht::<f32>(result);
    }

    #[cube(launch)]
    fn ifwht_test_kernel(result: &mut Tensor<f32>) {
        ifwht::<f32>(result);
    }

    #[test]
    fn fwht_matches_host_reference() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let values = run_fwht(&input);
        let expected = host_fwht(&input);

        assert_close(&values, &expected, 1.0e-6);
    }

    #[test]
    fn ifwht_round_trip_restores_input() {
        let input = [1.5, -0.5, 0.25, 2.0, -1.0, 3.0, 4.5, -2.5];
        let transformed = run_fwht(&input);
        let restored = run_ifwht(&transformed);

        assert_close(&restored, &input, 1.0e-5);
    }

    #[test]
    fn single_element_is_unchanged() {
        let input = [3.25];
        let values = run_fwht(&input);

        assert_close(&values, &input, 1.0e-6);
    }

    #[test]
    fn fwht_is_orthonormal() {
        let input = [1.0, -2.0, 0.5, 4.0, -1.5, 3.0, -0.25, 2.25];
        let output = run_fwht(&input);

        let input_energy = input.iter().map(|value| value * value).sum::<f32>();
        let output_energy = output.iter().map(|value| value * value).sum::<f32>();

        assert!(
            (input_energy - output_energy).abs() <= 1.0e-5,
            "energy mismatch: input={input_energy}, output={output_energy}"
        );
    }

    fn run_fwht(input: &[f32]) -> Vec<f32> {
        run_kernel(input, fwht_test_kernel::launch)
    }

    fn run_ifwht(input: &[f32]) -> Vec<f32> {
        run_kernel(input, ifwht_test_kernel::launch)
    }

    fn run_kernel(
        input: &[f32],
        launch: impl FnOnce(
            &ComputeClient<cubecl::cpu::CpuRuntime>,
            CubeCount,
            CubeDim,
            TensorArg<'_, cubecl::cpu::CpuRuntime>,
        ) -> Result<(), LaunchError>,
    ) -> Vec<f32> {
        let client = cpu_client();
        let shape = [input.len()];
        let allocation = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(input)),
            &shape,
            size_of::<f32>(),
        );
        let tensor_arg = tensor_arg_1d_f32(&allocation, &shape);

        launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg,
        )
        .expect("walsh kernel should launch");

        read_1d_f32_allocation(&client, &allocation, input.len())
    }

    fn host_fwht(input: &[f32]) -> Vec<f32> {
        let mut values = input.to_vec();
        let n = values.len();

        assert!(
            n >= 1 && (n & (n - 1)) == 0,
            "WHT length must be a power of 2"
        );

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
        values.into_iter().map(|value| value / scale).collect()
    }
}
