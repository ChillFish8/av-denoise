use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

/// Reads back the reference ring, which holds exactly one frame at `temporal_radius: 0`.
///
/// The single slot is the whole buffer, so the handle needs no byte-offset slicing.
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

/// Mean absolute difference to the right and lower neighbours of a single-channel frame.
///
/// It is a simple roughness proxy, where lower means smoother.
fn mean_abs_neighbour_diff(frame: &[f32], width: u32, height: u32) -> f32 {
    let width = width as usize;
    let height = height as usize;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let centre = frame[y * width + x];
            if x + 1 < width {
                sum += (frame[y * width + x + 1] - centre).abs();
                count += 1;
            }

            if y + 1 < height {
                sum += (frame[(y + 1) * width + x] - centre).abs();
                count += 1;
            }
        }
    }

    sum / count as f32
}

#[test]
fn bilateral_uniform_image_passthrough() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 1, 0.5);

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

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!((value - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {value}");
    }
}

#[test]
fn bilateral_noisy_image_finite() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.4, 8, 8, 3, 0.8);

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

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (-0.01..=1.01).contains(&value),
            "pixel {i}: out-of-range output {value}"
        );
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
    let width = 16;
    let height = 16;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[6.0 / 255.0]);

    let params = nlm_spatial_params();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let reference = read_single_slot_reference(&denoiser);

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
    let width = 32;
    let height = 32;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[10.0 / 255.0]);

    let params = nlm_spatial_params();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let reference = read_single_slot_reference(&denoiser);

    let input_roughness = mean_abs_neighbour_diff(&frame, width, height);
    let pilot_roughness = mean_abs_neighbour_diff(&reference, width, height);

    assert!(
        pilot_roughness < input_roughness,
        "expected the pilot to smooth the input: input roughness {input_roughness}, pilot roughness {pilot_roughness}"
    );
}

#[test]
fn nlm_spatial_pilot_uniform_passthrough() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let params = nlm_spatial_params();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let reference = read_single_slot_reference(&denoiser);

    for (i, &value) in reference.iter().enumerate() {
        assert!((value - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {value}");
    }
}

/// Pilot-vs-pilot distances carry no noise floor, so only the pilot-facing input offset keeps it.
#[test]
fn nlm_spatial_zeros_main_offset_but_keeps_input_offset() {
    let client = make_client();
    let width = 16;
    let height = 16;
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

    let denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    assert_eq!(
        denoiser.noise_offset, 0.0,
        "main-pass offset must be zeroed for NlmSpatial"
    );
    assert_eq!(
        denoiser.input_noise_offset, expected_input_offset,
        "pilot-facing offset must keep the full noise floor"
    );
}
