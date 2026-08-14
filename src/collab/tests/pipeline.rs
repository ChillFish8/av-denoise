use cubecl::prelude::*;

use super::helpers::{R, make_client, make_gradient_frame, noisy_field_over};
use crate::collab::{CollabParams, CollabPipeline};
use crate::nlmeans::ChannelMode;

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (1.0f64 / mse).log10()
}

/// A flat 64x64 luma field with real i.i.d. noise. Denoising it should
/// visibly improve PSNR against the clean flat field, a low bar any
/// working two-stage filter clears comfortably.
#[test]
fn two_stage_improves_psnr_on_flat_noise() {
    let (w, h) = (64u32, 64u32);
    let sigma = 0.03f32;
    let clean = vec![0.5f32; (w * h) as usize];
    let noisy = noisy_field_over(w, h, 0.5, sigma);

    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Luma,
        ..CollabParams::default()
    };
    let mut pipeline = CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");

    let input = client.create_from_slice(f32::as_bytes(&noisy));
    let output = client.empty(noisy.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[sigma], &output)
        .expect("run_two_stage failed");

    let out_bytes = client.read_one(output).expect("output readback failed");
    let denoised = f32::from_bytes(&out_bytes)[..noisy.len()].to_vec();

    let noisy_psnr = psnr(&noisy, &clean);
    let denoised_psnr = psnr(&denoised, &clean);

    assert!(
        denoised_psnr >= noisy_psnr + 3.0,
        "expected at least a 3 dB PSNR improvement, got noisy={noisy_psnr:.2} dB \
         denoised={denoised_psnr:.2} dB"
    );
}

/// A smooth 64x64 gradient with only a trace of noise. The filter must
/// not visibly disturb it, the detail-preservation guard on the other
/// side of the flat-noise test above.
#[test]
fn two_stage_preserves_a_clean_gradient() {
    let (w, h) = (64u32, 64u32);
    let sigma = 0.001f32;
    let gradient = make_gradient_frame(w, h, 0.1, 0.9);

    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Luma,
        ..CollabParams::default()
    };
    let mut pipeline = CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");

    let input = client.create_from_slice(f32::as_bytes(&gradient));
    let output = client.empty(gradient.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[sigma], &output)
        .expect("run_two_stage failed");

    let out_bytes = client.read_one(output).expect("output readback failed");
    let denoised = f32::from_bytes(&out_bytes)[..gradient.len()].to_vec();

    let max_dev = gradient
        .iter()
        .zip(denoised.iter())
        .map(|(&want, &have)| (want - have).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_dev < 2.0 / 255.0,
        "expected the gradient to come back nearly unchanged, max deviation {max_dev} >= {}",
        2.0 / 255.0
    );
}

/// Running the same input through the same pipeline twice must produce
/// bitwise identical output, since nothing in the two-stage filter reads
/// anything but the GPU buffers it explicitly writes.
#[test]
fn two_stage_is_deterministic() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.02f32;
    let noisy = noisy_field_over(w, h, 0.5, sigma);

    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Luma,
        ..CollabParams::default()
    };
    let mut pipeline = CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");

    let input = client.create_from_slice(f32::as_bytes(&noisy));

    let output_a = client.empty(noisy.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[sigma], &output_a)
        .expect("first run_two_stage failed");
    let bytes_a = client.read_one(output_a).expect("first output readback failed");

    let output_b = client.empty(noisy.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[sigma], &output_b)
        .expect("second run_two_stage failed");
    let bytes_b = client.read_one(output_b).expect("second output readback failed");

    assert_eq!(
        bytes_a, bytes_b,
        "two runs of the same input must produce bitwise identical output"
    );
}

/// A two-channel flat field, `stored_ch = 2`, the first configuration in
/// this tree that runs any collab kernel above `stored_ch = 1`. `u` and
/// `v` are given different flat values, so a truncated or mis-scaled
/// `ArrayArg` read would show up as one channel drifting away from its
/// own flat value, not merely as a panic.
#[test]
fn chroma_mode_runs() {
    let (w, h) = (32u32, 32u32);
    let sigma = 0.001f32;
    let (u_base, v_base) = (0.3f32, 0.7f32);

    let u_plane = noisy_field_over(w, h, u_base, sigma);
    let v_plane = noisy_field_over(w, h, v_base, sigma);
    let mut interleaved = vec![0.0f32; (w * h * 2) as usize];
    for i in 0..(w * h) as usize {
        interleaved[i * 2] = u_plane[i];
        interleaved[i * 2 + 1] = v_plane[i];
    }

    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Chroma,
        ..CollabParams::default()
    };
    let mut pipeline = CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");

    let input = client.create_from_slice(f32::as_bytes(&interleaved));
    let output = client.empty(interleaved.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[sigma, sigma], &output)
        .expect("run_two_stage failed");

    let out_bytes = client.read_one(output).expect("output readback failed");
    let denoised = f32::from_bytes(&out_bytes)[..interleaved.len()].to_vec();

    for i in 0..(w * h) as usize {
        let u = denoised[i * 2];
        let v = denoised[i * 2 + 1];
        assert!((u - u_base).abs() < 1e-3, "pixel {i}: u={u} want near {u_base}");
        assert!((v - v_base).abs() < 1e-3, "pixel {i}: v={v} want near {v_base}");
    }
}
