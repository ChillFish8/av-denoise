use cubecl::cube;
use cubecl::prelude::*;

#[cube]
/// Estimates the noise standard deviation of a 2D frame using the MAD of the
/// HH (diagonal) wavelet subband (Donoho & Johnstone, 1994).
///
/// `frame` may hold integer pixels (u8/u16) or float values; elements are cast
/// to `f32` before arithmetic.  `scratch` must be pre-allocated with at least
/// `(rows-1) * (cols-1)` elements.
pub(crate) fn estimate_sigma<N: Numeric>(
    frame: &Tensor<N>,
    scratch: &mut Tensor<f32>,
) -> f32 {
    let rows = frame.shape(0);
    let cols = frame.shape(1);

    let row_stride = frame.stride(0);
    let col_stride = frame.stride(1);
    let mut n: usize = 0;

    if rows >= 2 && cols >= 2 {
        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let tl = f32::cast_from(frame[r * row_stride + c * col_stride]);
                let tr = f32::cast_from(frame[r * row_stride + (c + 1) * col_stride]);
                let bl = f32::cast_from(frame[(r + 1) * row_stride + c * col_stride]);
                let br =
                    f32::cast_from(frame[(r + 1) * row_stride + (c + 1) * col_stride]);
                let hh = (tl - tr - bl + br) / f32::new(2.0);
                scratch[n] = hh.abs();
                n += 1;
            }
        }
    }

    if n == 0 {
        f32::new(0.0)
    } else {
        let k = n / 2;
        nth_element::<f32>(scratch, 0, n - 1, k);

        let median = if n.is_multiple_of(2) {
            // After nth_element, scratch[0..k] holds values ≤ scratch[k].
            // Their maximum is the (k-1)-th order statistic (lower median).
            let mut lower = scratch[0];
            for i in 1..k {
                if scratch[i] > lower {
                    lower = scratch[i];
                }
            }
            (lower + scratch[k]) / f32::new(2.0)
        } else {
            scratch[k]
        };

        median / f32::new(0.6745)
    }
}

#[cube]
fn partition<F: Float>(arr: &mut Tensor<F>, lo: usize, hi: usize) -> usize {
    let pivot = arr[hi];
    let mut i = lo;

    for j in lo..hi {
        if arr[j] <= pivot {
            let tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i += 1;
        }
    }

    let tmp = arr[i];
    arr[i] = arr[hi];
    arr[hi] = tmp;

    i
}

#[cube]
fn nth_element<F: Float>(arr: &mut Tensor<F>, lo: usize, hi: usize, k: usize) {
    let mut lo = lo;
    let mut hi = hi;

    while lo < hi {
        let pivot = partition::<F>(arr, lo, hi);

        if pivot == k {
            break;
        } else if pivot < k {
            lo = pivot + 1;
        } else {
            hi = pivot - 1;
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
        read_1d_f32_allocation,
        tensor_arg_1d_f32,
        tensor_arg_f32,
    };

    #[cube(launch)]
    fn estimate_sigma_test_kernel(
        frame: &Tensor<f32>,
        scratch: &mut Tensor<f32>,
        result: &mut Tensor<f32>,
    ) {
        result[0] = estimate_sigma::<f32>(frame, scratch);
    }

    #[test]
    fn constant_frame_gives_zero_sigma() {
        let frame = [1.0f32; 9];
        let sigma = run_estimate_sigma(&frame, 3, 3);

        assert_close(&[sigma], &[0.0], 1.0e-6);
    }

    #[test]
    fn linear_ramp_gives_zero_sigma() {
        let frame: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let sigma = run_estimate_sigma(&frame, 3, 3);

        assert_close(&[sigma], &[0.0], 1.0e-6);
    }

    #[test]
    fn single_hh_value_case() {
        // 2×2 frame → exactly one HH value
        // HH[0,0] = (0 - 1 - 1 + 0) / 2 = -1.0 → abs = 1.0
        // sigma = 1.0 / 0.6745
        let frame = [0.0f32, 1.0, 1.0, 0.0];
        let sigma = run_estimate_sigma(&frame, 2, 2);
        let expected = 1.0 / 0.6745f32;

        assert_close(&[sigma], &[expected], 1.0e-5);
    }

    #[test]
    fn impulse_frame_exercises_even_median() {
        // 3×3 frame with center impulse → 4 HH values all equal to 0.5
        // median of [0.5, 0.5, 0.5, 0.5] = 0.5
        // sigma = 0.5 / 0.6745
        #[rustfmt::skip]
        let frame = [
            0.0f32, 0.0, 0.0,
            0.0,    1.0, 0.0,
            0.0,    0.0, 0.0,
        ];
        let sigma = run_estimate_sigma(&frame, 3, 3);
        let expected = 0.5 / 0.6745f32;

        assert_close(&[sigma], &[expected], 1.0e-5);
    }

    #[test]
    fn matches_host_reference_for_general_case() {
        #[rustfmt::skip]
        let frame = [
            3.0f32,  1.0, 4.0, 1.0,
            5.0,     9.0, 2.0, 6.0,
            5.0,     3.0, 5.0, 8.0,
            9.0,     7.0, 9.0, 3.0,
        ];
        let sigma = run_estimate_sigma(&frame, 4, 4);
        let expected = host_estimate_sigma(&frame, 4, 4);

        assert_close(&[sigma], &[expected], 1.0e-5);
    }

    fn run_estimate_sigma(frame: &[f32], rows: usize, cols: usize) -> f32 {
        let client = cpu_client();
        let frame_shape = [rows, cols];
        let scratch_len = (rows - 1) * (cols - 1);

        let frame_alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(frame)),
            &frame_shape,
            size_of::<f32>(),
        );
        let scratch_alloc = client.empty_tensor(&[scratch_len], size_of::<f32>());
        let result_alloc = client.empty_tensor(&[1], size_of::<f32>());

        estimate_sigma_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_f32(&frame_alloc, &frame_shape),
            tensor_arg_1d_f32(&scratch_alloc, &[scratch_len]),
            tensor_arg_1d_f32(&result_alloc, &[1]),
        )
        .expect("estimate_sigma test kernel should launch");

        read_1d_f32_allocation(&client, &result_alloc, 1)[0]
    }

    fn host_estimate_sigma(frame: &[f32], rows: usize, cols: usize) -> f32 {
        let mut hh_abs: Vec<f32> = Vec::with_capacity((rows - 1) * (cols - 1));

        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let tl = frame[r * cols + c];
                let tr = frame[r * cols + (c + 1)];
                let bl = frame[(r + 1) * cols + c];
                let br = frame[(r + 1) * cols + (c + 1)];
                hh_abs.push(((tl - tr - bl + br) / 2.0).abs());
            }
        }

        hh_abs.sort_unstable_by(f32::total_cmp);

        let n = hh_abs.len();
        let median = if n.is_multiple_of(2) {
            (hh_abs[n / 2 - 1] + hh_abs[n / 2]) / 2.0
        } else {
            hh_abs[n / 2]
        };

        median / 0.6745
    }
}
