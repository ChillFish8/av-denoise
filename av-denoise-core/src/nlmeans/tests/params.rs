use crate::nlmeans::params::{
    ChannelMode,
    HqParams,
    MAX_TEMPORAL_RADIUS,
    MIN_FRAME_DIM,
    NLM_LEGACY,
    NLM_NORM,
    NlmParams,
    SEPARABLE_THRESHOLD,
    hq_default_strength,
    sigma_eff,
    validate_dimensions,
};
use crate::nlmeans::prefilter::{self, PrefilterMode};

#[test]
fn noise_offset_scales_with_sigma_and_patch_size() {
    let sigma = 4.0 / 255.0;
    let params = NlmParams {
        patch_radius: 4,
        hq: Some(HqParams::with_sigma(sigma)),
        ..NlmParams::default()
    };

    let expected = 6.0 * sigma * sigma * 81.0;
    let got = params.noise_offset();
    assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
}

#[test]
fn noise_offset_zero_without_noise_floor() {
    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: false,
            sigma_override: Some(4.0 / 255.0),
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..NlmParams::default()
    };

    assert_eq!(params.noise_offset(), 0.0);
}

#[test]
fn noise_offset_zero_without_hq() {
    let params = NlmParams::default();
    assert_eq!(params.noise_offset(), 0.0);
}

#[test]
fn h2_inv_norm_with_auto_strength_matches_hand_computed() {
    let sigma = 8.0 / 255.0;
    let params = NlmParams {
        strength: 1.0,
        hq: Some(HqParams::with_sigma(sigma)),
        ..NlmParams::default()
    };

    let patch_area = (2 * params.patch_radius + 1) * (2 * params.patch_radius + 1);
    let effective_strength = 1.0 * sigma * 255.0;
    let expected = NLM_NORM / (NLM_LEGACY * effective_strength * effective_strength * patch_area as f32);

    let got = params.h2_inv_norm();
    assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
}

#[test]
fn validate_rejects_zero_hq_sigma() {
    let params = NlmParams {
        hq: Some(HqParams::with_sigma(0.0)),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_hq_sigma_above_one() {
    let params = NlmParams {
        hq: Some(HqParams::with_sigma(1.5)),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_nan_hq_sigma() {
    let params = NlmParams {
        hq: Some(HqParams::with_sigma(f32::NAN)),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_zero_thsad_scale() {
    let params = NlmParams {
        hq: Some(HqParams {
            thsad_scale: 0.0,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_negative_thsad_scale() {
    let params = NlmParams {
        hq: Some(HqParams {
            thsad_scale: -1.0,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_nan_thsad_scale() {
    let params = NlmParams {
        hq: Some(HqParams {
            thsad_scale: f32::NAN,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_accepts_default_thsad_scale() {
    let params = NlmParams {
        hq: Some(HqParams::default()),
        ..NlmParams::default()
    };
    assert!(params.validate().is_ok());
}

#[test]
fn hq_params_default_sigma_scale_is_one() {
    let defaults = HqParams::default();
    assert_eq!(defaults.sigma_scale, 1.0);
}

#[test]
fn validate_rejects_sigma_scale_below_the_minimum() {
    let params = NlmParams {
        hq: Some(HqParams {
            sigma_scale: 0.05,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    let err = params.validate().expect_err("0.05 is below the 0.1 minimum");
    assert!(
        err.to_string().contains("hq sigma_scale"),
        "error should name the field, got {err}"
    );
}

#[test]
fn validate_rejects_sigma_scale_above_the_maximum() {
    let params = NlmParams {
        hq: Some(HqParams {
            sigma_scale: 10.5,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_nan_sigma_scale() {
    let params = NlmParams {
        hq: Some(HqParams {
            sigma_scale: f32::NAN,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_accepts_sigma_scale_at_the_bounds() {
    let low = NlmParams {
        hq: Some(HqParams {
            sigma_scale: 0.1,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(low.validate().is_ok());

    let high = NlmParams {
        hq: Some(HqParams {
            sigma_scale: 10.0,
            ..HqParams::default()
        }),
        ..NlmParams::default()
    };
    assert!(high.validate().is_ok());
}

#[test]
fn noise_offset_with_handles_distinct_per_channel_sigmas() {
    let sigma_u = 4.0 / 255.0;
    let sigma_v = 10.0 / 255.0;
    let params = NlmParams {
        patch_radius: 4,
        channels: ChannelMode::Chroma,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..NlmParams::default()
    };

    let patch_area = (2 * params.patch_radius + 1) * (2 * params.patch_radius + 1);
    // The chroma scale of 1.5 applies per channel, and each channel keeps its own sigma.
    let expected = 2.0 * 1.5 * (sigma_u * sigma_u + sigma_v * sigma_v) * patch_area as f32;

    let got = params.noise_offset_with(Some(&[sigma_u, sigma_v]));
    assert!((got - expected).abs() < 1e-9, "expected {expected}, got {got}");
}

#[test]
fn sigma_eff_is_rms_over_active_channels() {
    let sigmas = [3.0 / 255.0, 4.0 / 255.0];
    let got = sigma_eff(&sigmas, ChannelMode::Chroma);
    let expected = ((sigmas[0] * sigmas[0] + sigmas[1] * sigmas[1]) / 2.0).sqrt();
    assert!((got - expected).abs() < 1e-9, "expected {expected}, got {got}");
}

#[test]
fn validate_rejects_non_positive_pilot_strength_scale() {
    let zero = NlmParams {
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 0.0 },
        ..NlmParams::default()
    };
    assert!(zero.validate().is_err());

    let nan = NlmParams {
        prefilter: PrefilterMode::NlmSpatial {
            strength_scale: f32::NAN,
        },
        ..NlmParams::default()
    };
    assert!(nan.validate().is_err());
}

#[test]
fn validate_rejects_pilot_with_patch_radius_above_separable_threshold() {
    let params = NlmParams {
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        patch_radius: SEPARABLE_THRESHOLD + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_accepts_pilot_within_limits() {
    let params = NlmParams {
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        patch_radius: SEPARABLE_THRESHOLD,
        ..NlmParams::default()
    };
    assert!(params.validate().is_ok());
}

#[test]
fn validate_rejects_non_positive_bilateral_sigma_r() {
    // The centre tap's range distance is 0, and 0 times an infinite factor is NaN, which poisons
    // every pixel of the reference image.
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.0,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());

    let negative = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: -0.02,
        },
        ..NlmParams::default()
    };
    assert!(negative.validate().is_err());

    let nan = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: f32::NAN,
        },
        ..NlmParams::default()
    };
    assert!(nan.validate().is_err());

    let infinite = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: f32::INFINITY,
        },
        ..NlmParams::default()
    };
    assert!(infinite.validate().is_err());
}

#[test]
fn validate_rejects_non_positive_bilateral_sigma_s() {
    // The centre tap's spatial distance is 0, so an infinite factor poisons it with the same NaN.
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 0.0,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());

    let negative = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: -3.0,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(negative.validate().is_err());

    let nan = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: f32::NAN,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(nan.validate().is_err());

    let infinite = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: f32::INFINITY,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(infinite.validate().is_err());
}

#[test]
fn validate_accepts_positive_finite_bilateral_sigmas() {
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_ok());
}

/// Pins the guard to `<= 0.0`, with a value whose square stays a normal float.
#[test]
fn validate_accepts_a_small_positive_bilateral_sigma_at_the_boundary() {
    let safe_small = 1e-6_f32;
    assert!(
        (safe_small * safe_small).is_normal(),
        "the test value itself must not underflow"
    );

    let small_sigma_s = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: safe_small,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(small_sigma_s.validate().is_ok());

    let small_sigma_r = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: safe_small,
        },
        ..NlmParams::default()
    };
    assert!(small_sigma_r.validate().is_ok());
}

#[test]
fn validate_rejects_a_subnormal_bilateral_sigma_that_underflows_on_squaring() {
    // `f32::MIN_POSITIVE` is finite and above 0, but its square underflows to exactly 0.0, which
    // makes the derived factor infinite.
    let squared = f32::MIN_POSITIVE * f32::MIN_POSITIVE;
    assert_eq!(
        squared, 0.0,
        "this test assumes MIN_POSITIVE underflows on squaring"
    );
    let inverse = prefilter::inv_two_sigma_sq(f32::MIN_POSITIVE);
    assert!(
        !inverse.is_finite(),
        "this test assumes the derived factor is infinite here"
    );

    let sigma_s = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: f32::MIN_POSITIVE,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(
        sigma_s.validate().is_err(),
        "a subnormal sigma_s that underflows to an infinite normalisation factor must be rejected"
    );

    let sigma_r = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: f32::MIN_POSITIVE,
        },
        ..NlmParams::default()
    };
    assert!(
        sigma_r.validate().is_err(),
        "a subnormal sigma_r that underflows to an infinite normalisation factor must be rejected"
    );
}

#[test]
fn validate_rejects_bilateral_sigma_s_above_the_smem_ceiling() {
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 16.0,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    let err = params.validate().expect_err("radius 32 exceeds the 22 ceiling");
    assert!(
        err.to_string().contains("sigma_s"),
        "error should name the field, got {err}"
    );
}

/// A `sigma_s` of 1e9 would overflow the tile-size arithmetic at launch.
#[test]
fn validate_rejects_extreme_bilateral_sigma_s() {
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1e9,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

/// A `sigma_s` of 11.0 gives a radius of 22, exactly the ceiling.
#[test]
fn validate_accepts_bilateral_sigma_s_at_the_smem_ceiling() {
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 11.0,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_ok());
}

/// A `sigma_s` of 11.01 gives a radius of 23, one past the ceiling.
#[test]
fn validate_rejects_bilateral_sigma_s_just_above_the_smem_ceiling() {
    let params = NlmParams {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 11.01,
            sigma_r: 0.02,
        },
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn sigma_eff_ignores_channels_past_the_mode_count() {
    let sigmas = [6.0 / 255.0, 100.0 / 255.0, 200.0 / 255.0];
    let got = sigma_eff(&sigmas, ChannelMode::Luma);
    assert!(
        (got - sigmas[0]).abs() < 1e-9,
        "expected {}, got {got}",
        sigmas[0]
    );
}

#[test]
fn hq_default_strength_matches_the_measured_luma_table() {
    const EXPECTED: [f32; 9] = [0.45, 0.45, 0.42, 0.42, 0.35, 0.35, 0.35, 0.30, 0.30];
    for (radius, &expected) in EXPECTED.iter().enumerate() {
        let got = hq_default_strength(ChannelMode::Luma, radius as u32);
        assert!(
            (got - expected).abs() < f32::EPSILON,
            "at radius {radius} expected {expected}, got {got}"
        );
    }
}

#[test]
fn hq_default_strength_matches_the_measured_chroma_table() {
    const EXPECTED: [f32; 9] = [1.00, 0.85, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70];
    for (radius, &expected) in EXPECTED.iter().enumerate() {
        let got = hq_default_strength(ChannelMode::Chroma, radius as u32);
        assert!(
            (got - expected).abs() < f32::EPSILON,
            "at radius {radius} expected {expected}, got {got}"
        );
    }
}

#[test]
fn hq_default_strength_yuv_reads_the_luma_table() {
    for radius in 0..=8u32 {
        let yuv = hq_default_strength(ChannelMode::Yuv, radius);
        let luma = hq_default_strength(ChannelMode::Luma, radius);
        assert!(
            (yuv - luma).abs() < f32::EPSILON,
            "at radius {radius} yuv is {yuv} but luma is {luma}"
        );
    }
}

#[test]
fn validate_dimensions_rejects_frames_below_the_minimum() {
    let small_width = validate_dimensions(2, 64);
    let small_height = validate_dimensions(64, 2);
    let empty = validate_dimensions(0, 0);
    assert!(small_width.is_err());
    assert!(small_height.is_err());
    assert!(empty.is_err());
}

#[test]
fn validate_dimensions_accepts_the_minimum() {
    let minimum = validate_dimensions(MIN_FRAME_DIM, MIN_FRAME_DIM);
    let full_hd = validate_dimensions(1920, 1080);
    assert!(minimum.is_ok());
    assert!(full_hd.is_ok());
}

#[test]
fn hq_default_strength_clamps_radius_above_the_table() {
    let at_max = hq_default_strength(ChannelMode::Luma, MAX_TEMPORAL_RADIUS);
    let above_max = hq_default_strength(ChannelMode::Luma, MAX_TEMPORAL_RADIUS + 5);
    assert!(
        (at_max - above_max).abs() < f32::EPSILON,
        "expected clamping to hold the last table entry, got {at_max} vs {above_max}"
    );
}
