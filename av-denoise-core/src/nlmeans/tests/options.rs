use crate::nlmeans::{
    ChannelMode,
    DenoisingMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    NlmParams,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
    hq_default_strength,
    resolve_params,
};

fn fast(options: NlmeansOptions) -> NlmeansAlgorithm {
    NlmeansAlgorithm::Fast(options)
}

fn hq_with(hq: HqParams, mode: DenoisingMode) -> NlmeansAlgorithm {
    let nlm_options = NlmeansOptions {
        mode,
        ..NlmeansOptions::default()
    };

    NlmeansAlgorithm::Hq(NlmeansHqOptions { nlm: nlm_options, hq })
}

#[test]
fn spatial_mode_maps_to_zero_temporal_radius() {
    let options = NlmeansOptions {
        mode: DenoisingMode::Spacial,
        ..NlmeansOptions::default()
    };
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert_eq!(params.temporal_radius, 0);
    assert_eq!(params.channels, ChannelMode::Yuv);
}

#[test]
fn temporal_mode_propagates_radius() {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 3 },
        ..NlmeansOptions::default()
    };
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Luma);

    assert_eq!(params.temporal_radius, 3);
}

#[test]
fn prefilter_passthrough() {
    let options = NlmeansOptions {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        },
        ..NlmeansOptions::default()
    };
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!(matches!(params.prefilter, PrefilterMode::Bilateral { .. }));
}

#[test]
fn hq_unset_prefilter_defaults_to_none() {
    let hq = HqParams::default();
    let algorithm = hq_with(hq, DenoisingMode::Spacial);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!(matches!(params.prefilter, PrefilterMode::None));
}

#[test]
fn fast_unset_prefilter_defaults_to_none() {
    let options = NlmeansOptions::default();
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!(matches!(params.prefilter, PrefilterMode::None));
}

#[test]
fn hq_unset_strength_defaults_to_hq_default_strength() {
    let hq = HqParams::default();
    let algorithm = hq_with(hq, DenoisingMode::Spacial);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    let expected = hq_default_strength(ChannelMode::Yuv, 0);
    assert!((params.strength - expected).abs() < f32::EPSILON);
}

#[test]
fn hq_no_auto_strength_falls_back_to_the_legacy_absolute_default() {
    let hq = HqParams {
        auto_strength: false,
        ..HqParams::default()
    };
    let algorithm = hq_with(hq, DenoisingMode::Spacial);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    let expected = NlmParams::default().strength;
    assert!((params.strength - expected).abs() < f32::EPSILON);
}

#[test]
fn hq_luma_r4_uses_measured_table_value() {
    let mode = DenoisingMode::Temporal { radius: 4 };
    let hq = HqParams::default();
    let algorithm = hq_with(hq, mode);
    let params = resolve_params(&algorithm, ChannelMode::Luma);

    assert!((params.strength - 0.35).abs() < f32::EPSILON);
}

#[test]
fn hq_chroma_r4_uses_measured_table_value() {
    let mode = DenoisingMode::Temporal { radius: 4 };
    let hq = HqParams::default();
    let algorithm = hq_with(hq, mode);
    let params = resolve_params(&algorithm, ChannelMode::Chroma);

    assert!((params.strength - 0.70).abs() < f32::EPSILON);
}

#[test]
fn hq_yuv_r8_uses_measured_table_value() {
    let mode = DenoisingMode::Temporal { radius: 8 };
    let hq = HqParams::default();
    let algorithm = hq_with(hq, mode);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!((params.strength - 0.30).abs() < f32::EPSILON);
}

#[test]
fn hq_spacial_mode_uses_radius_zero_table_values() {
    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let hq = HqParams::default();
        let algorithm = hq_with(hq, DenoisingMode::Spacial);
        let params = resolve_params(&algorithm, channels);

        let expected = hq_default_strength(channels, 0);
        assert!((params.strength - expected).abs() < f32::EPSILON);
    }
}

#[test]
fn hq_explicit_strength_wins_over_the_table_for_every_plane() {
    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let nlm_options = NlmeansOptions {
            mode: DenoisingMode::Temporal { radius: 4 },
            tuning: NlmTuning {
                strength: Some(0.99),
                ..NlmTuning::default()
            },
            ..NlmeansOptions::default()
        };
        let hq = HqParams::default();
        let algorithm = NlmeansAlgorithm::Hq(NlmeansHqOptions { nlm: nlm_options, hq });
        let params = resolve_params(&algorithm, channels);

        assert!((params.strength - 0.99).abs() < f32::EPSILON);
    }
}

#[test]
fn fast_unset_strength_defaults_to_legacy_default() {
    let options = NlmeansOptions::default();
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!((params.strength - 1.2).abs() < f32::EPSILON);
}

#[test]
fn motion_compensation_passthrough() {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        ..NlmeansOptions::default()
    };
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!(matches!(
        params.motion_compensation,
        MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            ..
        }
    ));
}

#[test]
fn motion_compensation_defaults_to_none() {
    let options = NlmeansOptions::default();
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert!(matches!(params.motion_compensation, MotionCompensationMode::None));
}

#[test]
fn nlm_tuning_overrides_individual_fields() {
    let defaults = NlmParams::default();
    let options = NlmeansOptions {
        tuning: NlmTuning {
            search_radius: Some(7),
            patch_radius: None,
            strength: Some(2.5),
            self_weight: None,
        },
        ..NlmeansOptions::default()
    };
    let algorithm = fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Yuv);

    assert_eq!(params.search_radius, 7);
    assert_eq!(params.patch_radius, defaults.patch_radius);
    assert!((params.strength - 2.5).abs() < f32::EPSILON);
    assert!((params.self_weight - defaults.self_weight).abs() < f32::EPSILON);
}
