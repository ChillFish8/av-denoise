use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::noise::{NoiseCtx, partials_len, run_noise_estimate, sigma_from_abs_sum};

/// Runs both noise-estimate stages over one frame in a one-slot ring.
///
/// `dense` is packed as `pixels * channels` and is padded to `stored_ch` on upload. Returns the four
/// raw per-lane absolute-sum totals.
fn estimate_abs_sums(width: u32, height: u32, channels: u32, stored_ch: u32, dense: &[f32]) -> [f32; 4] {
    let client = make_client();
    let pixels = (width * height) as usize;
    let padded = pad_channels(dense, pixels, channels, stored_ch);

    let input_bytes = f32::as_bytes(&padded);
    let partials_bytes = partials_len(width, height) * size_of::<f32>();
    let input_buf = client.create_from_slice(input_bytes);
    let partials_buf = client.empty(partials_bytes);
    let results_buf = client.empty(4 * size_of::<f32>());

    let ctx = NoiseCtx {
        width,
        height,
        channels,
        stored_ch,
        frame_count: 1,
        frame: 0,
        slot: 0,
        input_buf: &input_buf,
        partials_buf: &partials_buf,
        results_buf: &results_buf,
    };

    run_noise_estimate::<R>(&client, &ctx).expect("noise estimate dispatch failed");

    let bytes = client.read_one(results_buf).expect("readback failed");
    let data = f32::from_bytes(&bytes);
    [data[0], data[1], data[2], data[3]]
}

/// The luma sigma measured over `frame`.
fn estimate_luma_sigma(width: u32, height: u32, frame: &[f32]) -> f32 {
    let sums = estimate_abs_sums(width, height, 1, 1, frame);
    sigma_from_abs_sum(sums[0], width, height)
}

#[test]
fn noise_estimate_recovers_known_sigma() {
    let width = 256;
    let height = 256;
    let true_sigma = 8.0 / 255.0;

    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[true_sigma]);
    let estimated = estimate_luma_sigma(width, height, &frame);

    let rel_err = (estimated - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.20,
        "estimated sigma {estimated} vs true {true_sigma} (rel err {rel_err:.3})"
    );
}

#[test]
fn noise_estimate_zero_on_uniform() {
    let width = 128;
    let height = 128;

    let frame = make_uniform_frame(width, height, 1, 0.5);
    let estimated = estimate_luma_sigma(width, height, &frame);

    assert!(
        estimated < 0.2 / 255.0,
        "expected near-zero estimate on uniform input, got {estimated}"
    );
}

/// The unused 4th lane must read exactly zero because stage 1 zeroes it for every thread.
#[test]
fn noise_estimate_per_channel() {
    let width = 256;
    let height = 256;
    let true_sigma_y = 8.0 / 255.0;
    let true_sigma_uv = 2.0 / 255.0;

    let sigmas = [true_sigma_y, true_sigma_uv, true_sigma_uv];
    let frame = make_noisy_gaussian_frame(width, height, 3, 0.5, &sigmas);
    let sums = estimate_abs_sums(width, height, 3, 4, &frame);

    let estimated_y = sigma_from_abs_sum(sums[0], width, height);
    let estimated_u = sigma_from_abs_sum(sums[1], width, height);
    let estimated_v = sigma_from_abs_sum(sums[2], width, height);

    let rel_err_y = (estimated_y - true_sigma_y).abs() / true_sigma_y;
    let rel_err_u = (estimated_u - true_sigma_uv).abs() / true_sigma_uv;
    let rel_err_v = (estimated_v - true_sigma_uv).abs() / true_sigma_uv;

    assert!(
        rel_err_y <= 0.25,
        "Y: estimated {estimated_y} vs true {true_sigma_y} (rel err {rel_err_y:.3})"
    );
    assert!(
        rel_err_u <= 0.25,
        "U: estimated {estimated_u} vs true {true_sigma_uv} (rel err {rel_err_u:.3})"
    );
    assert!(
        rel_err_v <= 0.25,
        "V: estimated {estimated_v} vs true {true_sigma_uv} (rel err {rel_err_v:.3})"
    );
    assert_eq!(sums[3], 0.0, "padding lane must be exactly zero, got {}", sums[3]);
}

/// The mask is orthogonal to affine content, so this bounds the known content bias on a gradient.
#[test]
fn noise_estimate_gradient_bias_bounded() {
    let width = 256;
    let height = 256;
    let true_sigma = 6.0 / 255.0;

    // Noise is built around mid-grey, clear of the clamp bounds, then re-centred to zero mean before
    // it is layered onto the gradient. The final clamp then never clips the gradient.
    let gradient = make_gradient_frame(width, height, 0.2, 0.8);
    let noise_frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[true_sigma]);
    let frame: Vec<f32> = gradient
        .iter()
        .zip(noise_frame.iter())
        .map(|(&gradient_value, &noise_value)| (gradient_value + (noise_value - 0.5)).clamp(0.0, 1.0))
        .collect();
    let estimated = estimate_luma_sigma(width, height, &frame);

    let rel_err = (estimated - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.30,
        "estimated sigma {estimated} vs true {true_sigma} on gradient content (rel err {rel_err:.3})"
    );
}

/// Quantises a normalised frame to `bits` and back, as a real source of that depth would be.
fn requantise(frame: &[f32], bits: u32) -> Vec<f32> {
    let max = ((1u32 << bits) - 1) as f32;

    frame
        .iter()
        .map(|&value| (value * max).round().clamp(0.0, max) / max)
        .collect()
}

/// Normalising by `(1 << bits) - 1` holds the scale fixed across depths, so calibrated constants
/// stay depth-independent.
#[test]
fn sigma_estimate_agrees_across_bit_depths() {
    let width = 256;
    let height = 256;
    let true_sigma = 8.0 / 255.0;

    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[true_sigma]);

    let eight = requantise(&frame, 8);
    let ten = requantise(&frame, 10);

    let sigma_eight = estimate_luma_sigma(width, height, &eight);
    let sigma_ten = estimate_luma_sigma(width, height, &ten);

    let rel_diff = (sigma_eight - sigma_ten).abs() / sigma_eight;
    assert!(
        rel_diff <= 0.05,
        "8-bit sigma {sigma_eight} vs 10-bit sigma {sigma_ten} (rel diff {rel_diff:.4})"
    );

    // Both must still recover the true sigma, not just agree on a wrong answer.
    for (label, sigma) in [("8-bit", sigma_eight), ("10-bit", sigma_ten)] {
        let err = (sigma - true_sigma).abs() / true_sigma;
        assert!(err <= 0.20, "{label} sigma {sigma} vs true {true_sigma}");
    }
}

/// The sigma is half an 8-bit step, which 10-bit resolves across two of its own steps.
///
/// It sits 5x above `SIGMA_FLOOR` (0.1/255), so a pass also shows the floor does not clip a real
/// fine measurement.
#[test]
fn fine_grain_survives_ten_bit_quantisation() {
    let width = 256;
    let height = 256;
    let true_sigma = 0.5 / 255.0;

    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[true_sigma]);
    let ten = requantise(&frame, 10);
    let eight = requantise(&frame, 8);

    let estimated = estimate_luma_sigma(width, height, &ten);
    let err = (estimated - true_sigma).abs() / true_sigma;

    assert!(
        err <= 0.25,
        "10-bit estimate {estimated} vs true {true_sigma} (rel err {err:.3})"
    );

    // Through 8-bit the grain picks up quantisation noise worth more than half its amplitude, so the
    // estimate inflates. Without that gap the 10-bit check above proves nothing.
    let estimated_eight = estimate_luma_sigma(width, height, &eight);

    assert!(
        estimated_eight > estimated,
        "8-bit estimate {estimated_eight} should inflate above the 10-bit estimate {estimated}"
    );
}
