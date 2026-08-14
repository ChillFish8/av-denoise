use cubecl::prelude::*;

use super::helpers::{R, make_client, make_gradient_frame, noisy_field_over, textured_frame};
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

/// A cascade's second stage runs on already-mostly-clean content, so a
/// small but real sigma over genuinely textured content, not a flat
/// field, is the regime that matters most and the one every other test
/// in this file avoids.
///
/// Sweeps sigma down from a moderate value to exactly 0 over a textured
/// frame. At every step the output must stay finite and stay close to
/// the valid pixel range. As sigma shrinks toward 0 the filter has less
/// and less noise to remove, so its output should converge toward the
/// untouched input rather than drift away from it, and the deviation
/// from the input at sigma 0 should be small. A shrinkage factor or
/// group weight that turned unbounded under a small sigma would fail the
/// finite/range check directly, and a filter that kept over-smoothing
/// even as sigma shrank would fail the convergence check.
#[test]
fn output_stays_sane_and_converges_to_input_as_sigma_shrinks_over_textured_content() {
    let (w, h) = (64u32, 64u32);
    let frame = textured_frame(w, h);
    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Luma,
        ..CollabParams::default()
    };

    let sigmas = [0.05f32, 0.01, 0.001, 0.0001, 0.00001, 0.0];
    let mut prev_max_dev: Option<f32> = None;

    for &sigma in &sigmas {
        let mut pipeline =
            CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");
        let input = client.create_from_slice(f32::as_bytes(&frame));
        let output = client.empty(frame.len() * size_of::<f32>());
        pipeline
            .run_two_stage(&input, &[sigma], &output)
            .expect("run_two_stage failed");

        let out_bytes = client.read_one(output).expect("output readback failed");
        let out = f32::from_bytes(&out_bytes)[..frame.len()].to_vec();

        let mut max_dev = 0.0f32;
        for (idx, (&want, &have)) in frame.iter().zip(out.iter()).enumerate() {
            assert!(have.is_finite(), "sigma={sigma}: out[{idx}]={have} is not finite");
            assert!(
                (-0.1..=1.1).contains(&have),
                "sigma={sigma}: out[{idx}]={have} left the valid pixel range"
            );
            max_dev = max_dev.max((want - have).abs());
        }

        if let Some(prev) = prev_max_dev {
            assert!(
                max_dev <= prev + 1e-4,
                "deviation from the input grew as sigma shrank, from {prev} to {max_dev} at \
                 sigma={sigma}, expected it to hold steady or shrink toward the identity"
            );
        }
        prev_max_dev = Some(max_dev);
    }

    let final_dev = prev_max_dev.expect("the sweep ran at least one sigma");
    assert!(
        final_dev < 0.05,
        "at sigma 0 the two-stage filter should be nearly the identity over textured content, \
         got a maximum deviation of {final_dev} from the input"
    );
}

/// A non-finite sigma reaching `run_two_stage` must not leave the output
/// non-finite or out of range, at the public API boundary rather than
/// inside any one kernel.
///
/// `sigmas` feeds both the admission threshold grouping uses and the
/// noise variance the filter kernels shrink by. This pins the whole
/// path at once, so a future change to either use of `sigmas` that
/// reintroduces an unbounded result would fail here even if it slipped
/// past a narrower test.
///
/// Finite and in-range is not enough on its own. A fully black frame is
/// also finite and in range, and a filter that quietly zeroes everything
/// under a noise level it cannot make sense of has destroyed the frame
/// just as thoroughly as one that blows it up, only more quietly. The
/// variance and correlation checks below catch that. A filter with no
/// real noise level to shrink by has no principled reason to remove
/// signal it cannot evaluate, so the correct fail-safe response is to
/// leave the content close to untouched, not delete it.
#[test]
fn non_finite_sigma_reaching_run_two_stage_stays_bounded() {
    let (w, h) = (64u32, 64u32);
    let frame = textured_frame(w, h);
    let client = make_client();
    let params = CollabParams {
        channels: ChannelMode::Luma,
        ..CollabParams::default()
    };
    let mut pipeline = CollabPipeline::<R>::new(&client, params, w, h).expect("pipeline construction failed");

    let input = client.create_from_slice(f32::as_bytes(&frame));
    let output = client.empty(frame.len() * size_of::<f32>());
    pipeline
        .run_two_stage(&input, &[f32::NAN], &output)
        .expect("run_two_stage failed");

    let out_bytes = client.read_one(output).expect("output readback failed");
    let out = f32::from_bytes(&out_bytes)[..frame.len()].to_vec();

    for (idx, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "out[{idx}]={v} is not finite under a non-finite sigma"
        );
        assert!(
            (-0.1..=1.1).contains(&v),
            "out[{idx}]={v} left the valid pixel range under a non-finite sigma"
        );
    }

    let n = out.len() as f64;
    let in_mean: f64 = frame.iter().map(|&v| v as f64).sum::<f64>() / n;
    let out_mean: f64 = out.iter().map(|&v| v as f64).sum::<f64>() / n;
    let in_var: f64 = frame.iter().map(|&v| (v as f64 - in_mean).powi(2)).sum::<f64>() / n;
    let out_var: f64 = out.iter().map(|&v| (v as f64 - out_mean).powi(2)).sum::<f64>() / n;
    let cov: f64 = out
        .iter()
        .zip(frame.iter())
        .map(|(&o, &i)| (o as f64 - out_mean) * (i as f64 - in_mean))
        .sum::<f64>()
        / n;
    let correlation = cov / (in_var.sqrt() * out_var.sqrt());

    assert!(
        in_var > 1e-6,
        "sanity check: textured_frame must itself carry real variance ({in_var}) for the \
         checks below to mean anything"
    );
    assert!(
        out_var > 0.5 * in_var,
        "output variance {out_var} collapsed relative to the input's {in_var} under a \
         non-finite sigma -- a black or otherwise flattened frame would pass the \
         finite/in-range checks above while failing this one"
    );
    assert!(
        correlation > 0.9,
        "output barely correlates with the input ({correlation}) under a non-finite sigma -- \
         a bounded but scrambled frame would pass the finite/in-range checks above while \
         failing this one"
    );
}
