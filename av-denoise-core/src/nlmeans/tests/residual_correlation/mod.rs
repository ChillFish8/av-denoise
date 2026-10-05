mod parameters;
mod regions;
mod search_radius;

use cubecl::prelude::ComputeClient;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

/// Residual correlation and smoothing measured for one configuration.
struct Sample {
    rho_out_h: f64,
    rho_out_v: f64,
    /// The residual's standard deviation over the input noise's.
    sigma_ratio: f64,
}

/// Pearson correlation between each pixel and its right neighbour, within `left..right` by
/// `top..bottom`.
///
/// The last column is skipped so no pixel is paired with a clamped edge copy of itself.
fn lag1_horizontal_rect(field: &[f32], width: u32, left: u32, top: u32, right: u32, bottom: u32) -> f64 {
    let mut sum_current = 0.0f64;
    let mut sum_next = 0.0f64;
    let mut sum_product = 0.0f64;
    let mut sum_current_sq = 0.0f64;
    let mut sum_next_sq = 0.0f64;
    let mut count = 0.0f64;
    for y in top..bottom {
        for x in left..(right - 1) {
            let current = field[(y * width + x) as usize] as f64;
            let next = field[(y * width + x + 1) as usize] as f64;
            sum_current += current;
            sum_next += next;
            sum_product += current * next;
            sum_current_sq += current * current;
            sum_next_sq += next * next;
            count += 1.0;
        }
    }

    let mean_current = sum_current / count;
    let mean_next = sum_next / count;
    let covariance = sum_product / count - mean_current * mean_next;
    let variance_current = sum_current_sq / count - mean_current * mean_current;
    let variance_next = sum_next_sq / count - mean_next * mean_next;
    covariance / (variance_current.sqrt() * variance_next.sqrt())
}

/// Pearson correlation between each pixel and the one below it, within `left..right` by
/// `top..bottom`, skipping the last row.
fn lag1_vertical_rect(field: &[f32], width: u32, left: u32, top: u32, right: u32, bottom: u32) -> f64 {
    let mut sum_current = 0.0f64;
    let mut sum_next = 0.0f64;
    let mut sum_product = 0.0f64;
    let mut sum_current_sq = 0.0f64;
    let mut sum_next_sq = 0.0f64;
    let mut count = 0.0f64;
    for y in top..(bottom - 1) {
        for x in left..right {
            let current = field[(y * width + x) as usize] as f64;
            let next = field[((y + 1) * width + x) as usize] as f64;
            sum_current += current;
            sum_next += next;
            sum_product += current * next;
            sum_current_sq += current * current;
            sum_next_sq += next * next;
            count += 1.0;
        }
    }

    let mean_current = sum_current / count;
    let mean_next = sum_next / count;
    let covariance = sum_product / count - mean_current * mean_next;
    let variance_current = sum_current_sq / count - mean_current * mean_current;
    let variance_next = sum_next_sq / count - mean_next * mean_next;
    covariance / (variance_current.sqrt() * variance_next.sqrt())
}

fn lag1_horizontal(field: &[f32], width: u32, height: u32) -> f64 {
    lag1_horizontal_rect(field, width, 0, 0, width, height)
}

fn lag1_vertical(field: &[f32], width: u32, height: u32) -> f64 {
    lag1_vertical_rect(field, width, 0, 0, width, height)
}

fn std_dev(field: &[f32]) -> f64 {
    let count = field.len() as f64;
    let mean: f64 = field.iter().map(|&value| value as f64).sum::<f64>() / count;
    let variance: f64 = field
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / count;
    variance.sqrt()
}

fn std_dev_rect(field: &[f32], width: u32, left: u32, top: u32, right: u32, bottom: u32) -> f64 {
    let count = ((right - left) * (bottom - top)) as f64;

    let mut sum = 0.0f64;
    for y in top..bottom {
        for x in left..right {
            sum += field[(y * width + x) as usize] as f64;
        }
    }

    let mean = sum / count;

    let mut variance = 0.0f64;
    for y in top..bottom {
        for x in left..right {
            let value = field[(y * width + x) as usize] as f64;
            variance += (value - mean).powi(2);
        }
    }

    (variance / count).sqrt()
}

fn subtract(minuend: &[f32], subtrahend: &[f32]) -> Vec<f32> {
    minuend
        .iter()
        .zip(subtrahend.iter())
        .map(|(&left, &right)| left - right)
        .collect()
}

/// Pushes `2 * temporal_radius + 1` noisy frames seeded from `seed_base` through the front end.
///
/// It returns the output of the final push, the first whose window holds no priming duplicate,
/// together with the noisy centre frame that output is built on.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn run_front_end(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
    seed_base: u32,
) -> (Vec<f32>, Vec<f32>) {
    let params = NlmParams {
        temporal_radius,
        search_radius,
        patch_radius,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::with_sigma(sigma)),
    };
    let mut denoiser = NlmDenoiser::<R>::new(client, params, width, height);

    let push_count = 2 * temporal_radius + 1;
    let mut center_noisy: Option<Vec<f32>> = None;
    let mut output: Option<Vec<f32>> = None;
    for i in 0..push_count {
        let frame = make_noise(clean, seed_base + i);
        if i == temporal_radius {
            center_noisy = Some(frame.clone());
        }

        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap();
        if i == push_count - 1 {
            output = result;
        }
    }

    (
        output.expect("a fully real window must emit on its final push"),
        center_noisy.expect("center frame must have been pushed"),
    )
}

/// Measures residual correlation against a flat clean reference.
///
/// A flat field gives NLM no structure to respond to, so `output - clean` is a noise-only residual.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn measure_flat(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
) -> Sample {
    let (output, center_noisy) = run_front_end(
        client,
        width,
        height,
        clean,
        sigma,
        make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        100,
    );
    let residual = subtract(&output, clean);
    let input_noise = subtract(&center_noisy, clean);

    Sample {
        rho_out_h: lag1_horizontal(&residual, width, height),
        rho_out_v: lag1_vertical(&residual, width, height),
        sigma_ratio: std_dev(&residual) / std_dev(&input_noise),
    }
}

/// Denoises two independent noise realisations of `clean` and returns their difference, plus the
/// first realisation's input noise.
///
/// On textured content `output - clean` carries NLM's deterministic response to structure, which
/// cancels in the difference. Two independent, identically distributed outputs differ with the
/// same correlation and `sqrt(2)` times the standard deviation. The seed bases 100 and 500 never
/// overlap for windows of up to 9 frames.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn denoised_difference(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
) -> (Vec<f32>, Vec<f32>) {
    let (output_a, noisy_a) = run_front_end(
        client,
        width,
        height,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        100,
    );
    let (output_b, _noisy_b) = run_front_end(
        client,
        width,
        height,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        500,
    );

    let difference = subtract(&output_a, &output_b);
    let input_noise = subtract(&noisy_a, clean);
    (difference, input_noise)
}

/// Measures residual correlation by differencing two denoised realisations of `clean`.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn measure_diff(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
) -> Sample {
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

    Sample {
        rho_out_h: lag1_horizontal(&difference, width, height),
        rho_out_v: lag1_vertical(&difference, width, height),
        sigma_ratio: (std_dev(&difference) / sqrt2) / std_dev(&input_noise),
    }
}
