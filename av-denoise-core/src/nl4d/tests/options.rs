use crate::error::Error;
use crate::nl4d::{
    Nl4dOptions,
    Nl4dParams,
    nl4d_default_lambda_ht,
    nl4d_pool_ratio,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    resolve_params,
};
use crate::nlmeans::{
    ChannelMode,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    NlmParams,
    PrefilterMode,
};
use crate::options::Preset;

const ALL_CHANNELS: [ChannelMode; 3] = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

fn lambda_ht_for(options: &Nl4dOptions, channels: ChannelMode) -> f32 {
    let params = resolve_params(options, channels).expect("the options are valid");
    params.lambda_ht
}

#[test]
fn nl4d_default_lambda_ht_differs_between_luma_and_chroma() {
    let luma = nl4d_default_lambda_ht(ChannelMode::Luma);
    let chroma = nl4d_default_lambda_ht(ChannelMode::Chroma);

    assert!((luma - 4.158).abs() < f32::EPSILON);
    assert!((chroma - 3.234).abs() < f32::EPSILON);
    assert!(
        (chroma - luma).abs() > f32::EPSILON,
        "the two planes should not resolve to the same default"
    );
}

#[test]
fn nl4d_default_lambda_ht_yuv_reads_the_luma_value() {
    let yuv = nl4d_default_lambda_ht(ChannelMode::Yuv);
    let luma = nl4d_default_lambda_ht(ChannelMode::Luma);

    assert!((yuv - luma).abs() < f32::EPSILON);
}

#[test]
fn nl4d_pool_ratio_gives_the_calibrated_threshold_at_each_default_lambda() {
    for channels in ALL_CHANNELS {
        let threshold = nl4d_pool_ratio(channels) * nl4d_default_lambda_ht(channels);
        assert!(
            (threshold - 2.42).abs() < 1.0e-6,
            "{channels:?} gives {threshold}"
        );
    }

    let yuv_ratio = nl4d_pool_ratio(ChannelMode::Yuv);
    let luma_ratio = nl4d_pool_ratio(ChannelMode::Luma);
    assert_eq!(yuv_ratio, luma_ratio);
}

#[test]
fn nl4d_options_default_to_pooling_on() {
    assert!(Nl4dOptions::default().pooled_threshold);
}

#[test]
fn nl4d_options_default_temporal_radius_is_the_base_preset_radius() {
    let options = Nl4dOptions::default();

    assert_eq!(options.temporal_radius, nl4d_temporal_radius_for(Preset::Base));
    assert_eq!(options.temporal_radius, 2);
}

#[test]
fn nl4d_spatial_radius_for_veryfast_is_narrower_than_the_default() {
    let veryfast = nl4d_spatial_radius_for(Preset::Veryfast);
    let base = nl4d_spatial_radius_for(Preset::Base);

    assert_eq!(veryfast, 6);
    assert_eq!(base, Nl4dOptions::default().spatial_radius);
}

#[test]
fn nl4d_options_default_matches_nl4d_params_default() {
    let options = Nl4dOptions::default();
    let params = Nl4dParams::default();

    assert_eq!(options.refine, params.refine);
    assert_eq!(options.spatial_radius, params.spatial_radius);
    assert!((options.c_min - params.c_min).abs() < f32::EPSILON);
    assert_eq!(options.lambda_ht, None);
    assert!((params.lambda_ht - nl4d_default_lambda_ht(ChannelMode::Yuv)).abs() < f32::EPSILON);
}

#[test]
fn resolve_lambda_ht_unset_uses_the_per_plane_default() {
    let options = Nl4dOptions::default();

    let luma = lambda_ht_for(&options, ChannelMode::Luma);
    let chroma = lambda_ht_for(&options, ChannelMode::Chroma);

    assert!((luma - 4.158).abs() < f32::EPSILON, "got {luma}");
    assert!((chroma - 3.234).abs() < f32::EPSILON, "got {chroma}");
}

#[test]
fn resolve_lambda_ht_explicit_value_overrides_every_plane() {
    let options = Nl4dOptions {
        lambda_ht: Some(4.4),
        ..Nl4dOptions::default()
    };

    for channels in ALL_CHANNELS {
        let got = lambda_ht_for(&options, channels);
        assert!(
            (got - 4.4).abs() < f32::EPSILON,
            "channels {channels:?} got {got}"
        );
    }
}

#[test]
fn resolve_lambda_ht_default_scale_leaves_the_value_alone() {
    let options = Nl4dOptions::default();

    for channels in ALL_CHANNELS {
        let got = lambda_ht_for(&options, channels);
        let want = nl4d_default_lambda_ht(channels);
        assert!(
            (got - want).abs() < f32::EPSILON,
            "channels {channels:?} got {got}"
        );
    }
}

#[test]
fn resolve_lambda_ht_scale_multiplies_the_per_plane_default() {
    let options = Nl4dOptions {
        lambda_ht_scale: 1.1,
        ..Nl4dOptions::default()
    };

    for channels in ALL_CHANNELS {
        let got = lambda_ht_for(&options, channels);
        let want = nl4d_default_lambda_ht(channels) * 1.1;
        assert!(
            (got - want).abs() < 1e-5,
            "channels {channels:?} got {got}, want {want}"
        );
    }
}

#[test]
fn resolve_lambda_ht_scale_multiplies_an_explicit_value() {
    let options = Nl4dOptions {
        lambda_ht: Some(4.0),
        lambda_ht_scale: 1.5,
        ..Nl4dOptions::default()
    };

    for channels in ALL_CHANNELS {
        let got = lambda_ht_for(&options, channels);
        assert!((got - 6.0).abs() < 1e-5, "channels {channels:?} got {got}");
    }
}

#[test]
fn resolve_lambda_ht_rejects_an_out_of_range_scale() {
    for bad in [0.0, -1.0, 0.05, 10.5, f32::NAN, f32::INFINITY] {
        let options = Nl4dOptions {
            lambda_ht_scale: bad,
            ..Nl4dOptions::default()
        };
        let result = resolve_params(&options, ChannelMode::Luma);

        let Err(Error::InvalidOptions(message)) = result else {
            panic!("lambda_ht_scale={bad} should be rejected");
        };
        assert!(
            message.contains("lambda_ht_scale"),
            "lambda_ht_scale={bad} gave {message}"
        );
    }
}

#[test]
fn nl4d_builds_the_front_ends_hq_params_from_its_own_fields() {
    let options = Nl4dOptions {
        sigma: Some(0.02),
        sigma_scale: 1.3,
        thsad_scale: 0.8,
        ..Nl4dOptions::default()
    };
    let params = resolve_params(&options, ChannelMode::Yuv).expect("resolve");

    let hq = params.nlm.hq.expect("nl4d always runs the hq front end");
    assert_eq!(hq.sigma_override, Some(0.02));
    assert!((hq.sigma_scale - 1.3).abs() < f32::EPSILON);
    assert!((hq.thsad_scale - 0.8).abs() < f32::EPSILON);
    assert!(
        hq.temporal_confidence,
        "the grouping kernel reads the confidence scores, so this cannot be off"
    );
}

#[test]
fn nl4d_never_builds_a_prefilter() {
    let params = resolve_params(&Nl4dOptions::default(), ChannelMode::Yuv).expect("resolve");

    assert!(matches!(params.nlm.prefilter, PrefilterMode::None));
}

#[test]
fn nl4d_leaves_the_nlm_weighting_knobs_at_their_defaults() {
    let defaults = NlmParams::default();
    let options = Nl4dOptions {
        temporal_radius: 4,
        ..Nl4dOptions::default()
    };
    let params = resolve_params(&options, ChannelMode::Luma).expect("resolve");

    assert!((params.nlm.strength - defaults.strength).abs() < f32::EPSILON);
    assert_eq!(params.nlm.search_radius, defaults.search_radius);
    assert_eq!(params.nlm.patch_radius, defaults.patch_radius);
    assert!((params.nlm.self_weight - defaults.self_weight).abs() < f32::EPSILON);
}

#[test]
fn nl4d_motion_search_becomes_an_active_mvtools_mode() {
    let options = Nl4dOptions {
        motion: MotionSearch {
            blksize: 32,
            overlap: 16,
            search_radius: 6,
            pyramid_levels: 1,
            estimation: MotionEstimation::Direct,
        },
        ..Nl4dOptions::default()
    };
    let params = resolve_params(&options, ChannelMode::Yuv).expect("resolve");

    assert!(matches!(
        params.nlm.motion_compensation,
        MotionCompensationMode::Mvtools {
            blksize: 32,
            overlap: 16,
            search_radius: 6,
            pyramid_levels: 1,
            estimation: MotionEstimation::Direct,
        }
    ));
}

#[test]
fn nl4d_motion_search_defaults_match_the_front_ends_own_defaults() {
    let params = resolve_params(&Nl4dOptions::default(), ChannelMode::Yuv).expect("resolve");
    let defaults = Nl4dParams::default();

    assert_eq!(params.nlm.motion_compensation, defaults.nlm.motion_compensation);
}

#[test]
fn a_zero_temporal_radius_is_rejected() {
    let options = Nl4dOptions {
        temporal_radius: 0,
        ..Nl4dOptions::default()
    };
    let result = resolve_params(&options, ChannelMode::Luma);

    assert!(matches!(result, Err(Error::InvalidOptions(_))));
}

#[test]
fn the_temporal_radius_reaches_both_the_front_end_and_the_grouping_stage() {
    let options = Nl4dOptions {
        temporal_radius: 3,
        ..Nl4dOptions::default()
    };
    let params = resolve_params(&options, ChannelMode::Luma).expect("resolve");

    assert_eq!(params.temporal_radius, 3);
    assert_eq!(params.nlm.temporal_radius, 3);
}
