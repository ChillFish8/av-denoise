//! Prefilter-path tests: externally-supplied reference clip and the
//! built-in GPU bilateral prefilter. Identity tests (reference == input)
//! must reproduce the no-prefilter baseline; smoke tests verify the
//! kernel produces finite, in-range output on noisy input.

use super::helpers::*;
use crate::nlmeans::*;

/// Aliasing the reference to the input must reproduce the no-prefilter
/// baseline exactly. Sanity check on the `_ref` kernel variants.
#[test]
fn external_reference_equals_input_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7);

    let baseline = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame(&frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame_with_reference(&frame, &frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    assert_eq!(baseline.len(), with_ref.len());
    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: baseline={a}, with_ref={b}");
    }
}

/// Separable path (patch_radius > 2) variant of the identity check.
#[test]
fn external_reference_separable_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7);

    let baseline = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame(&frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame_with_reference(&frame, &frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "pixel {i}: baseline={a}, with_ref={b}");
    }
}

/// Temporal rclip with `reference == input` for every pushed frame
/// must also reproduce baseline.
#[test]
fn external_reference_temporal_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frames = [
        make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7),
        make_frame_with_noisy_region(w, h, 1, 0.3, 7, 8, 2, 0.65),
        make_frame_with_noisy_region(w, h, 1, 0.3, 9, 8, 2, 0.75),
    ];

    let baseline = {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        for f in &frames {
            d.push_frame(f);
        }
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
            motion_compensation: MotionCompensationMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        for f in &frames {
            d.push_frame_with_reference(f, f);
        }
        d.denoise().unwrap().unwrap().to_vec()
    };

    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: baseline={a}, with_ref={b}");
    }
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
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

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
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}
