use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

/// Reads back a single-slot reference ring buffer (`temporal_radius: 0`,
/// so the ring holds exactly one frame and the whole handle is that
/// frame, no byte-offset slicing needed).
fn read_single_slot_reference(denoiser: &NlmDenoiser<R>) -> Vec<f32> {
    let handle = denoiser
        .reference_buf
        .as_ref()
        .expect("reference buffer must exist for NlmSpatial")
        .clone();
    let bytes = denoiser
        .client
        .read_one(handle)
        .expect("reference readback failed");
    f32::from_bytes(&bytes).to_vec()
}

/// Mean absolute horizontal + vertical neighbour difference, a simple
/// roughness proxy for single-channel dense (`stored_ch == 1`) frames.
/// Lower means smoother.
fn mean_abs_neighbour_diff(frame: &[f32], w: u32, h: u32) -> f32 {
    let w = w as usize;
    let h = h as usize;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            let v = frame[y * w + x];
            if x + 1 < w {
                sum += (frame[y * w + x + 1] - v).abs();
                count += 1;
            }
            if y + 1 < h {
                sum += (frame[(y + 1) * w + x] - v).abs();
                count += 1;
            }
        }
    }
    sum / count as f32
}

/// Bilateral prefilter on a uniform image must reproduce the uniform
/// value exactly (weights sum to anything, but the weighted average of
/// identical values is itself).
#[test]
fn bilateral_uniform_image_passthrough() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1.0,
            sigma_r: 0.1,
        },
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {v}");
    }
}

/// Bilateral smoke test on noisy input. Verifies the kernel produces
/// finite, in-range outputs (we trust the kernel correctness from the
/// uniform-image and identity tests).
#[test]
fn bilateral_noisy_image_finite() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.4, 8, 8, 3, 0.8);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 2.0,
            sigma_r: 0.05,
        },
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

fn nlm_spatial_params() -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    }
}

#[test]
fn nlm_spatial_pilot_fills_reference() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[6.0 / 255.0]);

    let mut d = NlmDenoiser::<R>::new(&client, nlm_spatial_params(), w, h);
    d.push_frame(&frame);

    let reference = read_single_slot_reference(&d);

    assert_eq!(reference.len(), frame.len());
    let mut differs = false;
    for (i, (&input, &pilot)) in frame.iter().zip(reference.iter()).enumerate() {
        assert!(pilot.is_finite(), "pixel {i}: non-finite pilot output {pilot}");
        assert!(
            (0.0..=1.0).contains(&pilot),
            "pixel {i}: out-of-range pilot output {pilot}"
        );
        if (input - pilot).abs() > 1e-6 {
            differs = true;
        }
    }
    assert!(differs, "pilot output must differ from the noisy input somewhere");
}

#[test]
fn nlm_spatial_pilot_smooths() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[10.0 / 255.0]);

    let mut d = NlmDenoiser::<R>::new(&client, nlm_spatial_params(), w, h);
    d.push_frame(&frame);

    let reference = read_single_slot_reference(&d);

    let input_roughness = mean_abs_neighbour_diff(&frame, w, h);
    let pilot_roughness = mean_abs_neighbour_diff(&reference, w, h);

    assert!(
        pilot_roughness < input_roughness,
        "expected the pilot to smooth the input: input roughness {input_roughness}, pilot roughness {pilot_roughness}"
    );
}

/// A uniform frame has zero patch distance everywhere, so the pilot's
/// weighted average reproduces the input value exactly.
#[test]
fn nlm_spatial_pilot_uniform_passthrough() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut d = NlmDenoiser::<R>::new(&client, nlm_spatial_params(), w, h);
    d.push_frame(&frame);

    let reference = read_single_slot_reference(&d);

    for (i, &v) in reference.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {v}");
    }
}

/// The main-pass offset must be zeroed for `NlmSpatial` (pilot-vs-pilot
/// distances no longer carry the noise floor) while the pilot-facing
/// input offset keeps the full value the HQ noise-floor math would
/// otherwise apply to the main pass too.
#[test]
fn nlm_spatial_zeros_main_offset_but_keeps_input_offset() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let sigma = 8.0 / 255.0;

    let params = NlmParams {
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        hq: Some(HqParams::with_sigma(sigma)),
        ..nlm_spatial_params()
    };

    let expected_input_offset = params.noise_offset();
    assert!(
        expected_input_offset > 0.0,
        "test setup: expected a nonzero noise floor"
    );

    let denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    assert_eq!(
        denoiser.noise_offset, 0.0,
        "main-pass offset must be zeroed for NlmSpatial"
    );
    assert_eq!(
        denoiser.input_noise_offset, expected_input_offset,
        "pilot-facing offset must keep the full noise floor"
    );
}
