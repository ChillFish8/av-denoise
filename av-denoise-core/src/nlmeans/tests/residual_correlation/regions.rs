use cubecl::prelude::ComputeClient;

use super::{Sample, denoised_difference, lag1_horizontal_rect, lag1_vertical_rect, std_dev_rect};
use crate::nlmeans::tests::helpers::*;

/// A rectangle as `(left, top, right, bottom)`, covering `left..right` by `top..bottom`.
type Region = (u32, u32, u32, u32);

/// Builds a frame that is flat left of `split_x` and carries
/// [make_textured_frame](crate::nlmeans::tests::helpers::make_textured_frame)'s sine pattern to
/// the right.
fn make_flat_and_textured_frame(width: u32, height: u32, split_x: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let value = if x < split_x {
                0.5
            } else {
                let fraction_x = x as f32 / width as f32;
                let fraction_y = y as f32 / height as f32;
                let raw = 0.5
                    + 0.2
                        * (fraction_x * 8.0 * std::f32::consts::PI).sin()
                        * (fraction_y * 6.0 * std::f32::consts::PI).cos()
                    + 0.1 * (fraction_x * 20.0 * std::f32::consts::PI).sin();
                raw.clamp(0.05, 0.95)
            };
            frame[(y * width + x) as usize] = value;
        }
    }
    frame
}

/// Measures residual correlation by differencing two denoised realisations, separately over two
/// regions of the frame.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn measure_diff_two_regions(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
    region_a: Region,
    region_b: Region,
) -> (Sample, Sample) {
    let (difference, input_noise) = denoised_difference(
        client,
        width,
        height,
        clean,
        sigma,
        make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
    );
    let sqrt2 = 2.0f64.sqrt();

    let sample_for = |(left, top, right, bottom): Region| {
        let difference_std = std_dev_rect(&difference, width, left, top, right, bottom);
        let input_std = std_dev_rect(&input_noise, width, left, top, right, bottom);
        Sample {
            rho_out_h: lag1_horizontal_rect(&difference, width, left, top, right, bottom),
            rho_out_v: lag1_vertical_rect(&difference, width, left, top, right, bottom),
            sigma_ratio: (difference_std / sqrt2) / input_std,
        }
    };

    (sample_for(region_a), sample_for(region_b))
}

/// Separate frames cannot rule out other differences between two setups, so this places flat and
/// textured regions side by side in one frame.
///
/// Both tiles sit 15 pixels from the seam and every edge, outside the reach of
/// `patch_radius + search_radius` (4).
#[test]
fn nlm_residual_correlation_within_a_single_frame_flat_vs_textured_regions() {
    let client = make_client();
    let width = 200;
    let height = 120;
    let sigma_pre = 0.06f32;
    let search_radius = 2;
    let patch_radius = 2;
    let split_x = 100;

    let clean = make_flat_and_textured_frame(width, height, split_x);
    let flat_region = (15u32, 15u32, 85u32, 105u32);
    let textured_region = (115u32, 15u32, 185u32, 105u32);

    let (flat, textured) = measure_diff_two_regions(
        &client,
        width,
        height,
        &clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
        search_radius,
        0,
        patch_radius,
        flat_region,
        textured_region,
    );

    eprintln!(
        "within-frame flat vs textured region (w={width} h={height} sigma_pre={sigma_pre} \
         search_radius={search_radius} patch_radius={patch_radius}):\n\
         {:<10} {:>8} {:>8} {:>8}",
        "region", "rho_h", "rho_v", "sig_ratio"
    );
    for (label, sample) in [("flat", &flat), ("textured", &textured)] {
        eprintln!(
            "{:<10} {:>8.4} {:>8.4} {:>8.4}",
            label, sample.rho_out_h, sample.rho_out_v, sample.sigma_ratio
        );
    }
    eprintln!(
        "within-frame gap: rho_h {:.4}, rho_v {:.4}",
        (flat.rho_out_h - textured.rho_out_h).abs(),
        (flat.rho_out_v - textured.rho_out_v).abs()
    );

    assert!(
        flat.sigma_ratio < 0.9,
        "flat region sigma_ratio={:.4} too close to 1.0, no real smoothing happened",
        flat.sigma_ratio
    );
    assert!(
        textured.sigma_ratio < 0.9,
        "textured region sigma_ratio={:.4} too close to 1.0, no real smoothing happened",
        textured.sigma_ratio
    );

    // Each 70x90 tile has far fewer pixels than a 160x160 frame, so sampling noise is larger and
    // the tolerance is looser than the 0.08 used across separate frames.
    assert!(
        (flat.rho_out_h - textured.rho_out_h).abs() < 0.1,
        "within one frame, flat region rho_out_h={:.4} and textured region rho_out_h={:.4} \
         diverge by more than the tolerance; a single per-frame correlation profile would not \
         be structurally sound if this fails",
        flat.rho_out_h,
        textured.rho_out_h
    );
}
