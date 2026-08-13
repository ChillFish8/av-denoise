use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::noise::{NoiseCtx, partials_len, run_noise_estimate, sigma_from_abs_sum};

/// Uploads `dense` (packed `pixels * ch`) as the padded GPU storage
/// layout and runs both noise-estimate stages for a single frame in a
/// one-slot ring. Returns the four raw per-lane absolute-sum totals.
fn estimate_abs_sums(w: u32, h: u32, ch: u32, stored_ch: u32, dense: &[f32]) -> [f32; 4] {
    let client = make_client();
    let pixels = (w * h) as usize;
    let padded = pad_channels(dense, pixels, ch, stored_ch);

    let input_buf = client.create_from_slice(f32::as_bytes(&padded));
    let partials_buf = client.empty(partials_len(w, h) * size_of::<f32>());
    let results_buf = client.empty(4 * size_of::<f32>());

    let ctx = NoiseCtx {
        width: w,
        height: h,
        channels: ch,
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

#[test]
fn noise_estimate_recovers_known_sigma() {
    let w = 256;
    let h = 256;
    let true_sigma = 8.0 / 255.0;

    let frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[true_sigma]);
    let sums = estimate_abs_sums(w, h, 1, 1, &frame);
    let estimated = sigma_from_abs_sum(sums[0], w, h);

    let rel_err = (estimated - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.20,
        "estimated sigma {estimated} vs true {true_sigma} (rel err {rel_err:.3})"
    );
}

/// A perfectly uniform frame has zero mask response everywhere, so the
/// estimate must land below the noise floor.
#[test]
fn noise_estimate_zero_on_uniform() {
    let w = 128;
    let h = 128;

    let frame = make_uniform_frame(w, h, 1, 0.5);
    let sums = estimate_abs_sums(w, h, 1, 1, &frame);
    let estimated = sigma_from_abs_sum(sums[0], w, h);

    assert!(
        estimated < 0.2 / 255.0,
        "expected near-zero estimate on uniform input, got {estimated}"
    );
}

/// YUV storage. Distinct per-channel sigmas must be recovered
/// independently and the unused 4th (padding) lane must return exactly
/// zero (stage 1 zeroes it explicitly for every thread).
#[test]
fn noise_estimate_per_channel() {
    let w = 256;
    let h = 256;
    let true_sigma_y = 8.0 / 255.0;
    let true_sigma_uv = 2.0 / 255.0;

    let frame = make_noisy_gaussian_frame(w, h, 3, 0.5, &[true_sigma_y, true_sigma_uv, true_sigma_uv]);
    let sums = estimate_abs_sums(w, h, 3, 4, &frame);

    let estimated_y = sigma_from_abs_sum(sums[0], w, h);
    let estimated_u = sigma_from_abs_sum(sums[1], w, h);
    let estimated_v = sigma_from_abs_sum(sums[2], w, h);

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

/// Smooth linear gradient plus noise. The mask is orthogonal to affine
/// content so this mainly documents that gradient content doesn't blow
/// up the estimate, bounding the known content bias rather than hiding
/// it.
#[test]
fn noise_estimate_gradient_bias_bounded() {
    let w = 256;
    let h = 256;
    let true_sigma = 6.0 / 255.0;

    // Noise is generated around a mid-gray base (safely away from the
    // clamp bounds for this sigma) and re-centred to zero-mean before
    // being layered onto the gradient, so the final clamp never clips
    // and doesn't disturb the gradient's linearity.
    let gradient = make_gradient_frame(w, h, 0.2, 0.8);
    let noise_frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[true_sigma]);
    let frame: Vec<f32> = gradient
        .iter()
        .zip(noise_frame.iter())
        .map(|(&g, &n)| (g + (n - 0.5)).clamp(0.0, 1.0))
        .collect();
    let sums = estimate_abs_sums(w, h, 1, 1, &frame);
    let estimated = sigma_from_abs_sum(sums[0], w, h);

    let rel_err = (estimated - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.30,
        "estimated sigma {estimated} vs true {true_sigma} on gradient content (rel err {rel_err:.3})"
    );
}
