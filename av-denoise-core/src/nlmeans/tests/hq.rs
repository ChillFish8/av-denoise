use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

fn base_params() -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    }
}

/// Pushes five frames with a moving noisy square, then flushes.
///
/// Asserts every output is finite and in range, and that each pushed frame produces one output.
fn run_temporal_smoke(params: NlmParams, width: u32, height: u32) {
    let client = make_client();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    let frames: Vec<Vec<f32>> = (0..5)
        .map(|i| make_frame_with_noisy_region(width, height, 1, 0.5, 6 + i, 8, 2, 0.8))
        .collect();

    let mut emitted = 0usize;
    let check = |frame: &[f32]| {
        for (i, &value) in frame.iter().enumerate() {
            assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
            assert!(
                (0.0..=1.0).contains(&value),
                "pixel {i}: out-of-range output {value}"
            );
        }
    };

    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(result) = denoiser.denoise().unwrap() {
            check(&result);
            emitted += 1;
        }
    }

    denoiser
        .flush(|frame| {
            check(frame);
            emitted += 1;
        })
        .unwrap();

    assert_eq!(emitted, frames.len(), "expected one output per pushed frame");
}

/// With both features off, `effective_strength` and `noise_offset` reduce to the fast path's values.
#[test]
fn hq_disabled_features_match_fast_mode() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);

    let fast_params = base_params();
    let mut fast = NlmDenoiser::<R>::new(&client, fast_params, width, height);
    fast.push_frame(&frame);
    let fast_out = fast.denoise().unwrap().unwrap();

    let hq_params = NlmParams {
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: false,
            sigma_override: Some(8.0 / 255.0),
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };
    let mut hq = NlmDenoiser::<R>::new(&client, hq_params, width, height);
    hq.push_frame(&frame);
    let hq_out = hq.denoise().unwrap().unwrap();

    assert_eq!(
        fast_out, hq_out,
        "disabled HQ features should reproduce the fast path exactly"
    );
}

/// The sigma sits far above realistic noise because the solid block gives only a few discrete
/// patch distances.
///
/// A realistic sigma would sit below every nonzero distance and clamp nothing. This one lands the
/// offset between two of those steps.
#[test]
fn hq_noise_floor_changes_output() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);

    let fast_params = base_params();
    let mut fast = NlmDenoiser::<R>::new(&client, fast_params, width, height);
    fast.push_frame(&frame);
    let fast_out = fast.denoise().unwrap().unwrap();

    let hq_params = NlmParams {
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: true,
            sigma_override: Some(40.0 / 255.0),
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };
    let mut hq = NlmDenoiser::<R>::new(&client, hq_params, width, height);
    hq.push_frame(&frame);
    let hq_out = hq.denoise().unwrap().unwrap();

    let mut max_diff = 0.0f32;
    for (i, (&fast_value, &hq_value)) in fast_out.iter().zip(hq_out.iter()).enumerate() {
        assert!(hq_value.is_finite(), "pixel {i}: non-finite HQ output {hq_value}");
        assert!(
            (0.0..=1.0).contains(&hq_value),
            "pixel {i}: out-of-range HQ output {hq_value}"
        );
        max_diff = max_diff.max((fast_value - hq_value).abs());
    }

    assert!(
        max_diff > 1e-3,
        "expected the noise floor to change the output somewhere, max diff was {max_diff}"
    );
}

#[test]
fn hq_uniform_input_passthrough() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let params = NlmParams {
        hq: Some(HqParams::with_sigma(8.0 / 255.0)),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!((value - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {value}");
    }
}

#[test]
fn hq_temporal_smoke() {
    let params = NlmParams {
        temporal_radius: 1,
        hq: Some(HqParams::with_sigma(6.0 / 255.0)),
        ..base_params()
    };

    run_temporal_smoke(params, 16, 16);
}

/// Uses per-pixel Gaussian noise because the Immerkær estimator reads a solid block close to the
/// noise floor, since only its boundary ring varies.
#[test]
fn hq_auto_sigma_denoises() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[8.0 / 255.0]);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    let mut max_diff = 0.0f32;
    for (i, (&input, &output)) in frame.iter().zip(result.iter()).enumerate() {
        assert!(output.is_finite(), "pixel {i}: non-finite output {output}");
        assert!(
            (0.0..=1.0).contains(&output),
            "pixel {i}: out-of-range output {output}"
        );
        max_diff = max_diff.max((input - output).abs());
    }

    assert!(
        max_diff > 1e-3,
        "expected the auto-estimated sigma to actually denoise the input, max diff was {max_diff}"
    );
}

#[test]
fn hq_auto_sigma_temporal_smoke() {
    let params = NlmParams {
        temporal_radius: 1,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };

    run_temporal_smoke(params, 16, 16);
}

#[test]
fn hq_override_skips_estimation() {
    let client = make_client();
    let width = 16;
    let height = 16;

    let params = NlmParams {
        hq: Some(HqParams::with_sigma(8.0 / 255.0)),
        ..base_params()
    };

    let denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    assert!(
        denoiser.noise_partials.is_none(),
        "sigma_override must skip allocating the partials scratch buffer"
    );
    assert!(
        denoiser.noise_results.is_none(),
        "sigma_override must skip allocating the results buffer"
    );
}

#[test]
fn hq_reset_clears_noise_state() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let noisy = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);
    let low = make_uniform_frame(width, height, 1, 0.5);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params.clone(), width, height);
    denoiser.push_frame(&noisy);
    denoiser.denoise().unwrap();

    denoiser.reset_stream_state();
    denoiser.push_frame(&low);
    denoiser.denoise().unwrap();

    let mut fresh = NlmDenoiser::<R>::new(&client, params, width, height);
    fresh.push_frame(&low);
    fresh.denoise().unwrap();

    assert_eq!(
        denoiser.h2_inv_norm, fresh.h2_inv_norm,
        "reset should clear the EMA so the next estimate starts fresh, not blended with stale state"
    );
    assert_eq!(
        denoiser.noise_offset, fresh.noise_offset,
        "reset should clear the EMA so the next estimate starts fresh, not blended with stale state"
    );
}

#[test]
fn hq_pilot_temporal_end_to_end() {
    let params = NlmParams {
        temporal_radius: 1,
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };

    run_temporal_smoke(params, 16, 16);
}

/// Uses per-pixel Gaussian noise because a solid block only changes patch distances around its
/// boundary ring, whether or not the pilot ran.
#[test]
fn hq_pilot_differs_from_unguided() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[10.0 / 255.0]);

    let hq_params = |prefilter: PrefilterMode| NlmParams {
        prefilter,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..base_params()
    };

    let unguided_params = hq_params(PrefilterMode::None);
    let mut unguided = NlmDenoiser::<R>::new(&client, unguided_params, width, height);
    unguided.push_frame(&frame);
    let unguided_out = unguided.denoise().unwrap().unwrap();

    let piloted_params = hq_params(PrefilterMode::NlmSpatial { strength_scale: 1.0 });
    let mut piloted = NlmDenoiser::<R>::new(&client, piloted_params, width, height);
    piloted.push_frame(&frame);
    let piloted_out = piloted.denoise().unwrap().unwrap();

    let mut max_diff = 0.0f32;
    let pairs = unguided_out.iter().zip(piloted_out.iter()).enumerate();
    for (i, (&unguided_value, &piloted_value)) in pairs {
        assert!(
            piloted_value.is_finite(),
            "pixel {i}: non-finite piloted output {piloted_value}"
        );
        assert!(
            (0.0..=1.0).contains(&piloted_value),
            "pixel {i}: out-of-range piloted output {piloted_value}"
        );
        max_diff = max_diff.max((unguided_value - piloted_value).abs());
    }

    assert!(
        max_diff > 1e-4,
        "expected the pilot to change HQ output somewhere, max diff was {max_diff}"
    );
}

/// HQ temporal params where a uniformly mismatched neighbour sits well past `thsad` while its plain
/// NLM weight stays significant.
///
/// `strength` and `patch_radius` keep the two thresholds apart, so confidence is separable from the
/// Welsch suppression.
fn temporal_conf_params(temporal_confidence: bool) -> NlmParams {
    NlmParams {
        temporal_radius: 1,
        search_radius: 1,
        patch_radius: 1,
        strength: 20.0,
        self_weight: 0.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: false,
            sigma_override: Some(2.0 / 255.0),
            temporal_confidence,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
    }
}

/// The previous neighbour is flat at 0.55, a 0.05 mismatch well past the default threshold.
///
/// Without confidence, plain NLM still gives it a noticeable weight at this strength and the output
/// drifts from 0.5.
#[test]
fn hq_temporal_confidence_suppresses_mismatched_neighbour() {
    let client = make_client();
    let width = 16;
    let height = 16;

    let previous = make_uniform_frame(width, height, 1, 0.55);
    let centre = make_uniform_frame(width, height, 1, 0.5);
    let next = make_uniform_frame(width, height, 1, 0.5);

    let run = |temporal_confidence: bool| {
        let params = temporal_conf_params(temporal_confidence);
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
        denoiser.push_frame(&previous);
        denoiser.push_frame(&centre);
        denoiser.push_frame(&next);
        denoiser.denoise().unwrap().unwrap()
    };

    let off = run(false);
    let on = run(true);

    let off_deviation = (off[(8 * width + 8) as usize] - 0.5).abs();
    let on_deviation = (on[(8 * width + 8) as usize] - 0.5).abs();

    assert!(
        off_deviation > 5e-3,
        "without confidence weighting the mismatched neighbour should pull the \
         output measurably away from 0.5, got deviation {off_deviation}"
    );
    assert!(
        on_deviation < off_deviation * 0.5,
        "confidence weighting should suppress the mismatched neighbour's \
         contribution: off deviation {off_deviation}, on deviation {on_deviation}"
    );
}

/// `thsad_scale` only feeds the confidence threshold, so with confidence off a sweep must leave the
/// output bitwise unchanged.
#[test]
fn hq_temporal_confidence_disabled_ignores_thsad_scale() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frames: Vec<Vec<f32>> = (0..3)
        .map(|i| make_frame_with_noisy_region(width, height, 1, 0.5, 6 + i, 8, 2, 0.8))
        .collect();

    let run = |thsad_scale: f32| {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
            hq: Some(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: Some(6.0 / 255.0),
                temporal_confidence: false,
                thsad_scale,
                sigma_scale: 1.0,
                windowed_noise_estimation: false,
            }),
        };

        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
        for frame in &frames {
            denoiser.push_frame(frame);
        }

        denoiser.denoise().unwrap().unwrap()
    };

    let base = run(1.0);
    let scaled = run(4.0);

    assert_eq!(
        base, scaled,
        "temporal_confidence: false must make thsad_scale inert (confidence buffer unused)"
    );
}

#[test]
fn hq_temporal_mc_confidence_smoke() {
    let params = NlmParams {
        temporal_radius: 1,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: Some(HqParams::with_sigma(6.0 / 255.0)),
        ..base_params()
    };

    run_temporal_smoke(params, 32, 32);
}

/// The EMA's first sample sets its state directly, so doubling `sigma_scale` exactly doubles the
/// folded estimate and quadruples `noise_offset`.
///
/// Checking both consumers from one fold shows the multiply sits before the blend feeds either.
#[test]
fn hq_sigma_scale_multiplies_the_folded_estimate() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[8.0 / 255.0]);

    let run = |sigma_scale: f32| {
        let params = NlmParams {
            hq: Some(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale,
                windowed_noise_estimation: false,
            }),
            ..base_params()
        };

        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
        denoiser.push_frame(&frame);
        denoiser.denoise().unwrap();

        let folded = denoiser
            .noise_estimator
            .current()
            .expect("estimator should hold a value after one push")[0];
        (folded, denoiser.noise_offset)
    };

    let (folded_1x, offset_1x) = run(1.0);
    let (folded_2x, offset_2x) = run(2.0);

    assert!(
        (folded_2x - folded_1x * 2.0).abs() < folded_1x * 1e-4,
        "expected the folded estimate to scale exactly 2x: 1x={folded_1x}, 2x={folded_2x}"
    );
    assert!(
        (offset_2x - offset_1x * 4.0).abs() < offset_1x * 1e-4,
        "expected noise_offset to scale 4x (quadratic in sigma): 1x={offset_1x}, 2x={offset_2x}"
    );
}
