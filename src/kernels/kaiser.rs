use cubecl::cube;
use cubecl::prelude::*;

#[cube]
fn bessel_i0<F: Float>(x: F) -> F {
    let ax = x.abs();

    if ax <= F::new(3.75) {
        let y = ax / F::new(3.75);
        let y2 = y * y;

        F::new(1.0)
            + y2 * (F::new(3.515_622_9)
                + y2 * (F::new(3.089_942_4)
                    + y2 * (F::new(1.206_749_2)
                        + y2 * (F::new(0.265_973_2)
                            + y2 * (F::new(0.036_076_8) + y2 * F::new(0.004_581_3))))))
    } else {
        let y = F::new(3.75) / ax;
        let poly = F::new(0.398_942_3)
            + y * (F::new(0.013_285_92)
                + y * (F::new(0.002_253_19)
                    + y * (F::new(-0.001_575_65)
                        + y * (F::new(0.009_162_81)
                            + y * (F::new(-0.020_577_06)
                                + y * (F::new(0.026_355_37)
                                    + y * (F::new(-0.016_476_33)
                                        + y * F::new(0.003_923_77))))))));

        ax.exp() * poly / ax.sqrt()
    }
}

#[cube]
/// Populates the result tensor with the kaiser window for the given `k` and `beta`.
///
/// This is based on NumPy's implementation and matches the behavior.
/// https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html
pub(crate) fn kaiser<F: Float>(result: &mut Tensor<F>, k: usize, beta: F) {
    if k <= 1 {
        for index in range(0usize, k) {
            result[index] = F::new(1.0);
        }
    } else {
        let len_minus_one = k - 1;
        let len_minus_one_f = F::cast_from(len_minus_one);
        let denominator = bessel_i0::<F>(beta);

        for index in range(0usize, k) {
            let index_f = F::cast_from(index);
            let ratio = (F::new(2.0) * index_f) / len_minus_one_f - F::new(1.0);
            let inside = max(F::new(0.0), F::new(1.0) - ratio * ratio);
            let numerator = bessel_i0::<F>(beta * inside.sqrt());

            result[index] = numerator / denominator;
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
        read_1d_f32_allocation,
        tensor_arg_1d_f32,
    };

    #[cube(launch)]
    fn kaiser_test_kernel(result: &mut Tensor<f32>, k: usize, beta: f32) {
        kaiser::<f32>(result, k, beta);
    }

    #[test]
    fn matches_numpy_documented_example() {
        let values = run_kaiser(12, 14.0);
        let expected = [
            7.726_867e-6,
            3.460_092e-3,
            4.652_001_7e-2,
            2.297_371_2e-1,
            5.998_853_4e-1,
            9.456_749e-1,
            9.456_749e-1,
            5.998_853_4e-1,
            2.297_371_2e-1,
            4.652_001_7e-2,
            3.460_092e-3,
            7.726_867e-6,
        ];

        assert_close(&values, &expected, 1.0e-5);
    }

    #[test]
    fn beta_zero_produces_rectangular_window() {
        let values = run_kaiser(8, 0.0);

        assert_close(&values, &[1.0; 8], 1.0e-6);
    }

    #[test]
    fn single_element_window_is_one() {
        let values = run_kaiser(1, 14.0);

        assert_close(&values, &[1.0], 1.0e-6);
    }

    #[test]
    fn matches_host_reference_for_general_case() {
        let values = run_kaiser(17, 8.6);
        let expected = host_kaiser(17, 8.6);

        assert_close(&values, &expected, 1.0e-5);
    }

    fn run_kaiser(len: usize, beta: f32) -> Vec<f32> {
        let client = cpu_client();
        let shape = [len];
        let allocation = client.empty_tensor(&shape, size_of::<f32>());
        let tensor_arg = tensor_arg_1d_f32(&allocation, &shape);

        kaiser_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg,
            ScalarArg::new(len),
            ScalarArg::new(beta),
        )
        .expect("kaiser test kernel should launch");

        read_1d_f32_allocation(&client, &allocation, len)
    }

    fn host_kaiser(len: usize, beta: f32) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }

        if len == 1 {
            return vec![1.0];
        }

        let denominator = host_bessel_i0(beta);
        let len_minus_one = (len - 1) as f32;

        (0..len)
            .map(|index| {
                let ratio = (2.0 * index as f32) / len_minus_one - 1.0;
                let inside = (1.0 - ratio * ratio).max(0.0);
                host_bessel_i0(beta * inside.sqrt()) / denominator
            })
            .collect()
    }

    fn host_bessel_i0(x: f32) -> f32 {
        let ax = x.abs();

        if ax <= 3.75 {
            let y = ax / 3.75;
            let y2 = y * y;

            1.0 + y2
                * (3.515_622_9
                    + y2 * (3.089_942_4
                        + y2 * (1.206_749_2
                            + y2 * (0.265_973_2
                                + y2 * (0.036_076_8 + y2 * 0.004_581_3)))))
        } else {
            let y = 3.75 / ax;
            let poly = 0.398_942_3
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77)))))));

            ax.exp() * poly / ax.sqrt()
        }
    }
}
