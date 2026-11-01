use av_denoise::{
    ChannelIntent,
    DenoisingMode,
    Depth,
    MotionCompensationMode,
    Nl4dOptions,
    NlmeansVariant,
    PrefilterMode,
    Preset,
    Subsampling,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
};
use av_denoise_vs::params::{AlgorithmKind, RawFormat, RawParams, layout_from_format, plane_options_from};
use vapoursynth::format::{ColorFamily, SampleType};

fn test_format_yuv(subsampling_w: u8, subsampling_h: u8, bits_per_sample: u8) -> RawFormat {
    RawFormat {
        sample_type: SampleType::Integer,
        bits_per_sample,
        subsampling_w,
        subsampling_h,
        color_family: ColorFamily::YUV,
    }
}

fn test_format_rgb(bits_per_sample: u8) -> RawFormat {
    RawFormat {
        sample_type: SampleType::Integer,
        bits_per_sample,
        subsampling_w: 0,
        subsampling_h: 0,
        color_family: ColorFamily::RGB,
    }
}

fn test_format_yuv_float() -> RawFormat {
    RawFormat {
        sample_type: SampleType::Float,
        bits_per_sample: 32,
        subsampling_w: 0,
        subsampling_h: 0,
        color_family: ColorFamily::YUV,
    }
}

fn test_format_gray(bits_per_sample: u8) -> RawFormat {
    RawFormat {
        sample_type: SampleType::Integer,
        bits_per_sample,
        subsampling_w: 0,
        subsampling_h: 0,
        color_family: ColorFamily::Gray,
    }
}

#[test]
fn channel_mode_yuv_is_unknown() {
    let raw = RawParams {
        channel_mode: Some("yuv".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert_eq!(
        err,
        "unknown channel_mode 'yuv', expected one of luma, chroma, lumachroma"
    );
}

#[test]
fn lumachroma_parses_to_luma_chroma() {
    let raw = RawParams {
        channel_mode: Some("lumachroma".to_string()),
        ..RawParams::default()
    };
    let options = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(options.intent, ChannelIntent::LumaChroma);
}

#[test]
fn accepts_the_three_supported_subsamplings_at_eight_bit() {
    let cases = [
        (test_format_yuv(1, 1, 8), Subsampling::Yuv420),
        (test_format_yuv(1, 0, 8), Subsampling::Yuv422),
        (test_format_yuv(0, 0, 8), Subsampling::Yuv444),
    ];
    for (format, expected) in cases {
        let layout = layout_from_format(format, 160, 120).unwrap();

        assert_eq!(layout.subsampling, expected);
        assert_eq!(layout.depth, Depth::Eight);
    }
}

#[test]
fn rejects_rgb_with_a_clear_message() {
    let format = test_format_rgb(8);
    let err = layout_from_format(format, 160, 120).unwrap_err().to_string();

    assert!(err.to_lowercase().contains("rgb"), "got {err}");
    assert!(
        err.to_lowercase().contains("yuv"),
        "error should say what to convert to, got {err}"
    );
}

#[test]
fn rejects_float_clips_with_a_clear_message() {
    let format = test_format_yuv_float();
    let err = layout_from_format(format, 160, 120).unwrap_err().to_string();

    assert!(err.to_lowercase().contains("float"), "got {err}");
}

#[test]
fn rejects_gray_with_a_clear_message() {
    let format = test_format_gray(8);
    let err = layout_from_format(format, 160, 120).unwrap_err().to_string();

    assert!(err.to_lowercase().contains("gray"), "got {err}");
    assert!(
        err.to_lowercase().contains("yuv"),
        "error should say what is accepted, got {err}"
    );
}

#[test]
fn strength_is_rejected_for_nl4d() {
    let raw = RawParams {
        strength: Some(0.5),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("strength"), "got {err}");
    assert!(
        err.to_lowercase().contains("nl4d"),
        "error should name the algorithm, got {err}"
    );
}

#[test]
fn luma_lambda_ht_is_rejected_for_nlmeans() {
    let raw = RawParams {
        luma_lambda_ht: Some(1.2),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("lambda_ht"), "got {err}");
    assert!(
        err.to_lowercase().contains("nlmeans"),
        "error should name the algorithm, got {err}"
    );
}

#[test]
fn patch_radius_is_rejected_for_nl4d() {
    let raw = RawParams {
        patch_radius: Some(3),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("patch_radius"), "got {err}");
    assert!(
        err.to_lowercase().contains("nl4d"),
        "error should name the algorithm, got {err}"
    );
}

#[test]
fn search_radius_is_rejected_for_nl4d() {
    let raw = RawParams {
        search_radius: Some(3),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("search_radius"), "got {err}");
    assert!(
        err.to_lowercase().contains("nl4d"),
        "error should name the algorithm, got {err}"
    );
}

#[test]
fn the_nl4d_mismatch_error_wins_over_the_stack_guard() {
    // SAFETY: single-threaded test, no denoiser thread exists yet.
    unsafe { std::env::remove_var("RUST_MIN_STACK") };

    let raw = RawParams {
        search_radius: Some(6),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(
        err.to_lowercase().contains("nl4d"),
        "search_radius on nl4d should fail as a mismatched parameter before the stack guard runs, got {err}"
    );
    assert!(
        !err.contains("RUST_MIN_STACK"),
        "the mismatch rejection should run first, so RUST_MIN_STACK should never be mentioned, got {err}"
    );
}

#[test]
fn per_plane_overrides_reach_plane_options() {
    let raw = RawParams {
        luma_strength: Some(0.6),
        chroma_strength: Some(0.3),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(opts.luma_strength, Some(0.6));
    assert_eq!(opts.chroma_strength, Some(0.3));
}

#[test]
fn sigma_reaches_nl4d_options() {
    let raw = RawParams {
        sigma: Some(6.0),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => {
            assert_eq!(nl4d.sigma, Some(6.0_f32));
        },
        other => panic!("expected Nl4d algorithm, got {other:?}"),
    }
}

#[test]
fn sigma_is_rejected_for_nlmeans_fast() {
    let raw = RawParams {
        sigma: Some(6.0),
        variant: Some("fast".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("sigma"), "got {err}");
    assert!(
        err.to_lowercase().contains("fast"),
        "error should name the variant, got {err}"
    );
}

#[test]
fn sigma_reaches_hq_sigma_override_under_variant_hq() {
    let raw = RawParams {
        sigma: Some(6.0),
        variant: Some("hq".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => {
            assert_eq!(hq.hq.sigma_override, Some(6.0_f32));
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn variant_hq_produces_the_hq_algorithm() {
    let raw = RawParams {
        variant: Some("hq".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert!(
        matches!(opts.algorithm, av_denoise::Algorithm::NlmeansHq(_)),
        "got {:?}",
        opts.algorithm
    );
}

#[test]
fn variant_fast_produces_the_fast_algorithm() {
    let raw = RawParams {
        variant: Some("fast".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert!(
        matches!(opts.algorithm, av_denoise::Algorithm::Nlmeans(_)),
        "got {:?}",
        opts.algorithm
    );
}

#[test]
fn no_variant_given_defaults_to_the_hq_algorithm() {
    let raw = RawParams::default();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert!(
        matches!(opts.algorithm, av_denoise::Algorithm::NlmeansHq(_)),
        "got {:?}",
        opts.algorithm
    );
}

#[test]
fn unrecognised_variant_errors_clearly() {
    let raw = RawParams {
        variant: Some("turbo".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.contains("turbo"), "got {err}");
    assert!(
        err.to_lowercase().contains("fast") && err.to_lowercase().contains("hq"),
        "error should name the accepted values, got {err}"
    );
}

#[test]
fn nlmeans_default_temporal_radius_is_two() {
    let raw = RawParams::default();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(opts.mode, DenoisingMode::Temporal { radius: 2 });
}

#[test]
fn nlmeans_explicit_zero_temporal_radius_stays_spacial() {
    let raw = RawParams {
        temporal_radius: Some(0),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(opts.mode, DenoisingMode::Spacial);
}

#[test]
fn a_large_search_radius_is_rejected_when_the_stack_is_not_raised() {
    // SAFETY: single-threaded test, no denoiser thread exists yet.
    unsafe { std::env::remove_var("RUST_MIN_STACK") };

    let raw = RawParams {
        search_radius: Some(6),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.contains("RUST_MIN_STACK"), "got {err}");
}

// --- sigma_scale ---

#[test]
fn sigma_scale_reaches_hq_params_under_variant_hq() {
    let raw = RawParams {
        variant: Some("hq".to_string()),
        sigma_scale: Some(1.5),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => {
            assert!((hq.hq.sigma_scale - 1.5).abs() < f32::EPSILON);
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn sigma_scale_reaches_nl4d_options() {
    let raw = RawParams {
        sigma_scale: Some(1.5),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => {
            assert!((nl4d.sigma_scale - 1.5).abs() < f32::EPSILON);
        },
        other => panic!("expected Nl4d algorithm, got {other:?}"),
    }
}

#[test]
fn sigma_scale_is_rejected_for_nlmeans_fast() {
    let raw = RawParams {
        sigma_scale: Some(1.5),
        variant: Some("fast".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("sigma_scale"), "got {err}");
    assert!(
        err.to_lowercase().contains("fast"),
        "error should name the variant, got {err}"
    );
}

// --- prefilter ---

#[test]
fn prefilter_none_and_empty_parse() {
    for raw_prefilter in ["none", ""] {
        let raw = RawParams {
            prefilter: Some(raw_prefilter.to_string()),
            ..RawParams::default()
        };
        let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

        match opts.algorithm {
            av_denoise::Algorithm::NlmeansHq(hq) => {
                assert!(matches!(hq.nlm.prefilter, PrefilterMode::None));
            },
            other => panic!("expected NlmeansHq algorithm, got {other:?}"),
        }
    }
}

#[test]
fn prefilter_bilateral_parses() {
    let raw = RawParams {
        prefilter: Some("bilateral:3.0,0.02".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => match hq.nlm.prefilter {
            PrefilterMode::Bilateral { sigma_s, sigma_r } => {
                assert!((sigma_s - 3.0).abs() < f32::EPSILON);
                assert!((sigma_r - 0.02).abs() < f32::EPSILON);
            },
            other => panic!("expected Bilateral, got {other:?}"),
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn prefilter_nlm_variants_parse() {
    let raw = RawParams {
        prefilter: Some("nlm:0.8".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => match hq.nlm.prefilter {
            PrefilterMode::NlmSpatial { strength_scale } => {
                assert!((strength_scale - 0.8).abs() < f32::EPSILON);
            },
            other => panic!("expected NlmSpatial, got {other:?}"),
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn prefilter_unknown_string_errors_clearly() {
    let raw = RawParams {
        prefilter: Some("garbage".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("prefilter"), "got {err}");
}

#[test]
fn prefilter_external_cannot_be_produced() {
    let raw = RawParams {
        prefilter: Some("external".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(
        err.to_lowercase().contains("prefilter"),
        "'external' has no string form, so it should fail as an unknown prefilter, got {err}"
    );
}

#[test]
fn prefilter_is_rejected_for_nl4d() {
    let raw = RawParams {
        prefilter: Some("none".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("prefilter"), "got {err}");
    assert!(err.to_lowercase().contains("nl4d"), "got {err}");
}

// --- preset ---

#[test]
fn preset_resolves_the_same_dials_as_core_for_nlmeans() {
    for name in ["veryfast", "fast", "base", "slow", "veryslow"] {
        let preset: Preset = name.parse().unwrap();
        let raw = RawParams {
            preset: Some(name.to_string()),
            ..RawParams::default()
        };
        let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

        let want_variant = nlmeans_variant_for(preset);
        let want_temporal_radius = nlmeans_temporal_radius_for(preset);
        let want_search_radius = nlmeans_search_radius_for(preset);

        let want_mode = if want_temporal_radius == 0 {
            DenoisingMode::Spacial
        } else {
            DenoisingMode::Temporal {
                radius: want_temporal_radius,
            }
        };

        assert_eq!(
            opts.mode, want_mode,
            "preset {name} resolved the wrong temporal radius"
        );

        let got_variant_is_hq = matches!(opts.algorithm, av_denoise::Algorithm::NlmeansHq(_));

        assert_eq!(
            got_variant_is_hq,
            want_variant == NlmeansVariant::Hq,
            "preset {name} resolved the wrong variant"
        );

        if let av_denoise::Algorithm::NlmeansHq(hq) = opts.algorithm {
            assert_eq!(
                hq.nlm.tuning.search_radius,
                Some(want_search_radius),
                "preset {name} resolved the wrong search radius"
            );
        } else if let av_denoise::Algorithm::Nlmeans(nlm) = opts.algorithm {
            assert_eq!(
                nlm.tuning.search_radius,
                Some(want_search_radius),
                "preset {name} resolved the wrong search radius"
            );
        }
    }
}

#[test]
fn nl4d_resolves_the_library_defaults() {
    let raw = RawParams {
        preset: Some("base".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();
    let defaults = Nl4dOptions::default();

    assert_eq!(
        opts.mode,
        DenoisingMode::Temporal {
            radius: defaults.temporal_radius
        }
    );

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.spatial_radius, defaults.spatial_radius),
        other => panic!("expected Nl4d algorithm, got {other:?}"),
    }
}

#[test]
fn nl4d_rejects_presets_other_than_base_and_slow() {
    for name in ["veryfast", "fast", "veryslow"] {
        let raw = RawParams {
            preset: Some(name.to_string()),
            ..RawParams::default()
        };
        let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
            .unwrap_err()
            .to_string();

        assert!(err.contains("base and slow presets"), "{name}: got {err}");
    }
}

fn nl4d_ceiling_for(preset: &str) -> u32 {
    let raw = RawParams {
        preset: Some(preset.to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => nl4d.motion.max_search_escalation,
        other => panic!("expected Nl4d algorithm, got {other:?}"),
    }
}

#[test]
fn nl4d_slow_widens_failing_motion_searches_up_to_eight() {
    let defaults = Nl4dOptions::default();

    assert_eq!(nl4d_ceiling_for("slow"), 8);
    assert_eq!(nl4d_ceiling_for("base"), defaults.motion.search_radius);
}

fn nlmeans_ceiling_for(preset: &str) -> u32 {
    let raw = RawParams {
        preset: Some(preset.to_string()),
        motion_compensation: Some(true),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();
    let motion = match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => hq.nlm.motion_compensation,
        av_denoise::Algorithm::Nlmeans(nlm) => nlm.motion_compensation,
        other => panic!("expected an nlmeans algorithm, got {other:?}"),
    };

    match motion {
        MotionCompensationMode::Mvtools {
            max_search_escalation,
            ..
        } => max_search_escalation,
        other => panic!("expected motion compensation, got {other:?}"),
    }
}

#[test]
fn nlmeans_slow_and_veryslow_widen_failing_motion_searches_up_to_eight() {
    for (preset, expected) in [
        ("veryfast", 4),
        ("fast", 4),
        ("base", 4),
        ("slow", 8),
        ("veryslow", 8),
    ] {
        assert_eq!(nlmeans_ceiling_for(preset), expected, "{preset}");
    }
}

#[test]
fn unset_preset_defaults_to_base() {
    let raw = RawParams::default();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();
    let base_temporal_radius = nlmeans_temporal_radius_for(Preset::Base);

    assert_eq!(
        opts.mode,
        DenoisingMode::Temporal {
            radius: base_temporal_radius
        }
    );
}

#[test]
fn unrecognised_preset_errors_clearly() {
    let raw = RawParams {
        preset: Some("turbo".to_string()),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.contains("turbo"), "got {err}");
}

#[test]
fn explicit_temporal_radius_overrides_the_preset() {
    let raw = RawParams {
        preset: Some("veryslow".to_string()),
        temporal_radius: Some(3),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(opts.mode, DenoisingMode::Temporal { radius: 3 });
}

#[test]
fn explicit_variant_overrides_the_preset() {
    let raw = RawParams {
        // `base` resolves to the `hq` variant, so the explicit `variant` must win.
        preset: Some("base".to_string()),
        variant: Some("fast".to_string()),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert!(
        matches!(opts.algorithm, av_denoise::Algorithm::Nlmeans(_)),
        "got {:?}",
        opts.algorithm
    );
}

// --- motion_compensation ---

#[test]
fn motion_compensation_defaults_to_false() {
    let raw = RawParams::default();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => {
            assert!(matches!(hq.nlm.motion_compensation, MotionCompensationMode::None));
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn motion_compensation_true_turns_it_on() {
    let raw = RawParams {
        motion_compensation: Some(true),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => {
            assert!(matches!(
                hq.nlm.motion_compensation,
                MotionCompensationMode::Mvtools { .. }
            ));
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn motion_compensation_false_stays_off() {
    let raw = RawParams {
        motion_compensation: Some(false),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::NlmeansHq(hq) => {
            assert!(matches!(hq.nlm.motion_compensation, MotionCompensationMode::None));
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn motion_compensation_is_rejected_for_nl4d() {
    let raw = RawParams {
        motion_compensation: Some(true),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.to_lowercase().contains("motion_compensation"), "got {err}");
    assert!(err.to_lowercase().contains("nl4d"), "got {err}");
}

#[test]
fn lambda_ht_scale_reaches_nl4d_options() {
    let raw = RawParams {
        lambda_ht_scale: Some(1.15),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => {
            assert!((nl4d.lambda_ht_scale - 1.15).abs() < 1e-6)
        },
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn shared_lambda_ht_reaches_nl4d_options() {
    let raw = RawParams {
        lambda_ht: Some(4.6),
        ..RawParams::default()
    };
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    match opts.algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.lambda_ht, Some(4.6)),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn spatial_radius_overrides_the_default() {
    let from_preset = {
        let raw = RawParams {
            preset: Some("base".into()),
            ..RawParams::default()
        };
        let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

        match algorithm {
            av_denoise::Algorithm::Nl4d(nl4d) => nl4d.spatial_radius,
            other => panic!("expected Nl4d, got {other:?}"),
        }
    };

    let raw = RawParams {
        preset: Some("base".into()),
        spatial_radius: Some((from_preset + 1) as i64),
        ..RawParams::default()
    };
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.spatial_radius, from_preset + 1),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn refine_reaches_nl4d_options() {
    let raw = RawParams {
        refine: Some(3),
        ..RawParams::default()
    };
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.refine, 3),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn noise_map_defaults_to_on() {
    let raw = RawParams::default();
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert!(nl4d.noise_map),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn noise_map_false_turns_it_off() {
    let raw = RawParams {
        noise_map: Some(false),
        ..RawParams::default()
    };
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert!(!nl4d.noise_map),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn noise_map_is_rejected_for_nlmeans() {
    let raw = RawParams {
        noise_map: Some(true),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.contains("noise_map"), "got {err}");
}

#[test]
fn pooled_threshold_defaults_to_on() {
    let raw = RawParams::default();
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert!(nl4d.pooled_threshold),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn pooled_threshold_false_turns_it_off() {
    let raw = RawParams {
        pooled_threshold: Some(false),
        ..RawParams::default()
    };
    let algorithm = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap().algorithm;

    match algorithm {
        av_denoise::Algorithm::Nl4d(nl4d) => assert!(!nl4d.pooled_threshold),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn pooled_threshold_is_rejected_for_nlmeans() {
    let raw = RawParams {
        pooled_threshold: Some(true),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
        .unwrap_err()
        .to_string();

    assert!(err.contains("pooled_threshold"), "got {err}");
}

#[test]
fn the_four_nl4d_dials_are_rejected_for_nlmeans() {
    let cases = [
        (
            "lambda_ht_scale",
            RawParams {
                lambda_ht_scale: Some(1.1),
                ..RawParams::default()
            },
        ),
        (
            "lambda_ht",
            RawParams {
                lambda_ht: Some(4.6),
                ..RawParams::default()
            },
        ),
        (
            "spatial_radius",
            RawParams {
                spatial_radius: Some(3),
                ..RawParams::default()
            },
        ),
        (
            "refine",
            RawParams {
                refine: Some(3),
                ..RawParams::default()
            },
        ),
    ];
    for (name, raw) in cases {
        let err = plane_options_from(&raw, AlgorithmKind::Nlmeans)
            .unwrap_err()
            .to_string();

        assert!(err.contains(name), "error should name {name}, got {err}");
    }
}

#[test]
fn a_negative_spatial_radius_is_rejected() {
    let raw = RawParams {
        spatial_radius: Some(-1),
        ..RawParams::default()
    };
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(err.contains("spatial_radius"), "got {err}");
}

#[test]
fn tuning_defaults_to_auto_and_not_deterministic() {
    let raw = RawParams::default();

    let options = plane_options_from(&raw, AlgorithmKind::Nl4d).unwrap();

    assert_eq!(options.tuning, av_denoise::TuningMode::Auto);
    assert!(!options.deterministic);
}

#[test]
fn tune_and_deterministic_are_read() {
    let raw = RawParams {
        tune: Some("cached".to_string()),
        deterministic: Some(true),
        ..RawParams::default()
    };

    let options = plane_options_from(&raw, AlgorithmKind::Nlmeans).unwrap();

    assert_eq!(options.tuning, av_denoise::TuningMode::Cached);
    assert!(options.deterministic);
}

#[test]
fn an_unknown_tune_mode_is_rejected() {
    let raw = RawParams {
        tune: Some("sometimes".to_string()),
        ..RawParams::default()
    };

    let error = plane_options_from(&raw, AlgorithmKind::Nl4d)
        .unwrap_err()
        .to_string();

    assert!(error.contains("sometimes"), "got {error}");
}
