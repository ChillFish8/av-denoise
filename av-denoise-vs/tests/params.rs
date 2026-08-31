use av_denoise_core::frame::Subsampling;
use av_denoise_core::{
    DenoisingMode,
    Depth,
    MotionCompensationMode,
    NlmeansVariant,
    PrefilterMode,
    Preset,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
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
fn accepts_the_three_supported_subsamplings_at_eight_bit() {
    for (fmt, expected) in [
        (test_format_yuv(1, 1, 8), Subsampling::Yuv420),
        (test_format_yuv(1, 0, 8), Subsampling::Yuv422),
        (test_format_yuv(0, 0, 8), Subsampling::Yuv444),
    ] {
        let layout = layout_from_format(fmt, 160, 120).unwrap();
        assert_eq!(layout.subsampling, expected);
        assert_eq!(layout.depth, Depth::Eight);
    }
}

#[test]
fn rejects_rgb_with_a_clear_message() {
    let err = layout_from_format(test_format_rgb(8), 160, 120)
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("rgb"), "got {err}");
    assert!(
        err.to_lowercase().contains("yuv"),
        "error should say what to convert to, got {err}"
    );
}

#[test]
fn rejects_float_clips_with_a_clear_message() {
    let err = layout_from_format(test_format_yuv_float(), 160, 120)
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("float"), "got {err}");
}

#[test]
fn rejects_gray_with_a_clear_message() {
    let err = layout_from_format(test_format_gray(8), 160, 120)
        .unwrap_err()
        .to_string();
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert_eq!(opts.luma_strength, Some(0.6));
    assert_eq!(opts.chroma_strength, Some(0.3));
}

#[test]
fn sigma_reaches_nl4d_options() {
    let raw = RawParams {
        sigma: Some(6.0),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::Nl4d(nl4d) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert!(
        matches!(opts.algorithm, av_denoise_core::Algorithm::NlmeansHq(_)),
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert!(
        matches!(opts.algorithm, av_denoise_core::Algorithm::Nlmeans(_)),
        "got {:?}",
        opts.algorithm
    );
}

#[test]
fn no_variant_given_defaults_to_the_hq_algorithm() {
    let raw = RawParams::default();
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert!(
        matches!(opts.algorithm, av_denoise_core::Algorithm::NlmeansHq(_)),
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert_eq!(opts.mode, DenoisingMode::Temporal { radius: 2 });
}

#[test]
fn nlmeans_explicit_zero_temporal_radius_stays_spacial() {
    let raw = RawParams {
        temporal_radius: Some(0),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert_eq!(opts.mode, DenoisingMode::Spacial);
}

#[test]
fn the_hq_arm_sets_windowed_noise_estimation() {
    let raw = RawParams {
        variant: Some("hq".to_string()),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
            assert!(hq.hq.windowed_noise_estimation);
        },
        other => panic!("expected NlmeansHq algorithm, got {other:?}"),
    }
}

#[test]
fn a_large_search_radius_is_rejected_when_the_stack_is_not_raised() {
    // SAFETY: single-threaded test, no denoiser thread exists yet.
    unsafe { std::env::remove_var("RUST_MIN_STACK") };
    let raw = RawParams {
        search_radius: Some(6),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::Nl4d(nl4d) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
        let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
        let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
        match opts.algorithm {
            av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => match hq.nlm.prefilter {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => match hq.nlm.prefilter {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("prefilter"), "got {err}");
}

/// `parse_prefilter`'s string grammar has no form that produces
/// `External`, and the boundary in `plane_options_from` rejects it if it
/// ever did, since this plugin has no way to supply the reference frame
/// `External` needs.
#[test]
fn prefilter_external_cannot_be_produced() {
    let raw = RawParams {
        prefilter: Some("external".to_string()),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
        let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
        let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();

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

        let got_variant_is_hq = matches!(opts.algorithm, av_denoise_core::Algorithm::NlmeansHq(_));
        assert_eq!(
            got_variant_is_hq,
            want_variant == NlmeansVariant::Hq,
            "preset {name} resolved the wrong variant"
        );

        if let av_denoise_core::Algorithm::NlmeansHq(hq) = opts.algorithm {
            assert_eq!(
                hq.nlm.tuning.search_radius,
                Some(want_search_radius),
                "preset {name} resolved the wrong search radius"
            );
        } else if let av_denoise_core::Algorithm::Nlmeans(nlm) = opts.algorithm {
            assert_eq!(
                nlm.tuning.search_radius,
                Some(want_search_radius),
                "preset {name} resolved the wrong search radius"
            );
        }
    }
}

#[test]
fn preset_resolves_the_same_dials_as_core_for_nl4d() {
    for name in ["veryfast", "fast", "base", "slow", "veryslow"] {
        let preset: Preset = name.parse().unwrap();
        let raw = RawParams {
            preset: Some(name.to_string()),
            ..RawParams::default()
        };
        let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
        let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();

        let want_temporal_radius = nl4d_temporal_radius_for(preset);
        let want_spatial_radius = nl4d_spatial_radius_for(preset);

        assert_eq!(
            opts.mode,
            DenoisingMode::Temporal {
                radius: want_temporal_radius
            },
            "preset {name} resolved the wrong temporal radius"
        );

        match opts.algorithm {
            av_denoise_core::Algorithm::Nl4d(nl4d) => {
                assert_eq!(
                    nl4d.spatial_radius, want_spatial_radius,
                    "preset {name} resolved the wrong spatial radius"
                );
            },
            other => panic!("expected Nl4d algorithm, got {other:?}"),
        }
    }
}

#[test]
fn unset_preset_defaults_to_base() {
    let raw = RawParams::default();
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    assert_eq!(
        opts.mode,
        DenoisingMode::Temporal {
            radius: nl4d_temporal_radius_for(Preset::Base)
        }
    );
}

#[test]
fn unrecognised_preset_errors_clearly() {
    let raw = RawParams {
        preset: Some("turbo".to_string()),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    assert_eq!(opts.mode, DenoisingMode::Temporal { radius: 3 });
}

#[test]
fn explicit_variant_overrides_the_preset() {
    let raw = RawParams {
        // `base` resolves to the `hq` variant; an explicit `variant`
        // must win anyway.
        preset: Some("base".to_string()),
        variant: Some("fast".to_string()),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    assert!(
        matches!(opts.algorithm, av_denoise_core::Algorithm::Nlmeans(_)),
        "got {:?}",
        opts.algorithm
    );
}

// --- motion_compensation ---

#[test]
fn motion_compensation_defaults_to_false() {
    let raw = RawParams::default();
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::NlmeansHq(hq) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::Nl4d(nl4d) => {
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let opts = plane_options_from(&raw, AlgorithmKind::Nl4d, layout).unwrap();
    match opts.algorithm {
        av_denoise_core::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.lambda_ht, Some(4.6)),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn spatial_radius_overrides_the_preset() {
    let from_preset = {
        let raw = RawParams {
            preset: Some("base".into()),
            ..RawParams::default()
        };
        let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
        match plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
            .unwrap()
            .algorithm
        {
            av_denoise_core::Algorithm::Nl4d(nl4d) => nl4d.spatial_radius,
            other => panic!("expected Nl4d, got {other:?}"),
        }
    };

    let raw = RawParams {
        preset: Some("base".into()),
        spatial_radius: Some((from_preset + 1) as i64),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    match plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
        .unwrap()
        .algorithm
    {
        av_denoise_core::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.spatial_radius, from_preset + 1),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn refine_reaches_nl4d_options() {
    let raw = RawParams {
        refine: Some(3),
        ..RawParams::default()
    };
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    match plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
        .unwrap()
        .algorithm
    {
        av_denoise_core::Algorithm::Nl4d(nl4d) => assert_eq!(nl4d.refine, 3),
        other => panic!("expected Nl4d, got {other:?}"),
    }
}

#[test]
fn the_four_nl4d_dials_are_rejected_for_nlmeans() {
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    for (name, raw) in [
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
    ] {
        let err = plane_options_from(&raw, AlgorithmKind::Nlmeans, layout)
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
    let layout = layout_from_format(test_format_yuv(1, 1, 8), 160, 120).unwrap();
    let err = plane_options_from(&raw, AlgorithmKind::Nl4d, layout)
        .unwrap_err()
        .to_string();
    assert!(err.contains("spatial_radius"), "got {err}");
}
