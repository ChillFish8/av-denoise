use av_denoise_core::NlmeansAlgorithm;

use super::*;
use crate::backend::EngineSpec;

/// A `PlaneOptions` with every field other than the four arguments at a neutral default.
fn base_options(
    mode: DenoisingMode,
    algorithm: Algorithm,
    luma_strength: Option<f32>,
    chroma_strength: Option<f32>,
) -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode,
        algorithm,
        luma_strength,
        chroma_strength,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

#[test]
fn luma_strength_alone_overrides_only_the_luma_plane() {
    let algorithm = Algorithm::default();
    let plane_options = base_options(DenoisingMode::Spacial, algorithm, Some(0.7), None);

    let luma_options = plane_options.denoiser_options(ChannelMode::Luma, Depth::Eight);
    let chroma_options = plane_options.denoiser_options(ChannelMode::Chroma, Depth::Eight);
    let luma = expect_nlmeans(luma_options.algorithm);
    let chroma = expect_nlmeans(chroma_options.algorithm);

    assert!(
        matches!(luma.tuning.strength, Some(strength) if (strength - 0.7).abs() < f32::EPSILON),
        "expected luma tuning.strength = Some(0.7), got {:?}",
        luma.tuning.strength
    );
    assert_eq!(
        chroma.tuning.strength, None,
        "chroma plane should carry no override so the table default applies"
    );
}

#[test]
fn both_per_plane_strengths_set_independently() {
    let algorithm = Algorithm::default();
    let plane_options = base_options(DenoisingMode::Spacial, algorithm, Some(0.7), Some(0.3));

    let luma_options = plane_options.denoiser_options(ChannelMode::Luma, Depth::Eight);
    let chroma_options = plane_options.denoiser_options(ChannelMode::Chroma, Depth::Eight);
    let luma = expect_nlmeans(luma_options.algorithm);
    let chroma = expect_nlmeans(chroma_options.algorithm);

    assert!(
        matches!(luma.tuning.strength, Some(strength) if (strength - 0.7).abs() < f32::EPSILON),
        "expected luma tuning.strength = Some(0.7), got {:?}",
        luma.tuning.strength
    );
    assert!(
        matches!(chroma.tuning.strength, Some(strength) if (strength - 0.3).abs() < f32::EPSILON),
        "expected chroma tuning.strength = Some(0.3), got {:?}",
        chroma.tuning.strength
    );
}

#[test]
fn no_overrides_hq_leaves_strength_to_the_per_plane_table() {
    let hq_options = NlmeansHqOptions::default();
    let plane_options = base_options(
        DenoisingMode::Temporal { radius: 4 },
        Algorithm::NlmeansHq(hq_options),
        None,
        None,
    );

    for channels in [ChannelMode::Luma, ChannelMode::Chroma] {
        let options = plane_options.denoiser_options(channels, Depth::Eight);
        let spec = options.algorithm.engine_spec(&options, 16, 16);

        let EngineSpec::Nlmeans {
            algorithm: NlmeansAlgorithm::Hq(hq),
            geometry,
        } = spec
        else {
            panic!("expected an HQ nlmeans spec for {channels:?}, got {spec:?}");
        };

        assert_eq!(geometry.channels, channels);
        assert_eq!(hq.nlm.mode, DenoisingMode::Temporal { radius: 4 });
        assert_eq!(
            hq.nlm.tuning.strength, None,
            "{channels:?} should use the calibrated table"
        );
    }
}

/// A `PlaneOptions` running `Algorithm::Nl4d` with only the two `lambda_ht` overrides set.
fn nl4d_options(luma_lambda_ht: Option<f32>, chroma_lambda_ht: Option<f32>) -> PlaneOptions {
    let default_nl4d = Nl4dOptions::default();

    PlaneOptions {
        accelerators: vec![],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 2 },
        algorithm: Algorithm::Nl4d(default_nl4d),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht,
        chroma_lambda_ht,
    }
}

fn expect_nlmeans(algorithm: Algorithm) -> NlmeansOptions {
    match algorithm {
        Algorithm::Nlmeans(options) => options,
        other => panic!("expected Algorithm::Nlmeans, got {other:?}"),
    }
}

fn expect_nl4d(algorithm: Algorithm) -> Nl4dOptions {
    match algorithm {
        Algorithm::Nl4d(options) => options,
        other => panic!("expected Algorithm::Nl4d, got {other:?}"),
    }
}

#[test]
fn luma_lambda_ht_alone_overrides_only_the_luma_instance_for_nl4d() {
    let plane_options = nl4d_options(Some(4.0), None);
    let default_lambda_ht = Nl4dOptions::default().lambda_ht;

    let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
    let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
    let luma = expect_nl4d(luma_algorithm);
    let chroma = expect_nl4d(chroma_algorithm);

    assert!((luma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
    assert_eq!(
        chroma.lambda_ht, default_lambda_ht,
        "chroma should stay unresolved here (None), deferred to its own per-plane \
         default at construction, got {:?}",
        chroma.lambda_ht
    );
}

#[test]
fn chroma_lambda_ht_alone_overrides_only_the_chroma_instance_for_nl4d() {
    let plane_options = nl4d_options(None, Some(4.0));
    let default_lambda_ht = Nl4dOptions::default().lambda_ht;

    let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
    let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
    let luma = expect_nl4d(luma_algorithm);
    let chroma = expect_nl4d(chroma_algorithm);

    assert_eq!(
        luma.lambda_ht, default_lambda_ht,
        "luma should stay unresolved here (None), deferred to its own per-plane \
         default at construction, got {:?}",
        luma.lambda_ht
    );
    assert!((chroma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
}

#[test]
fn both_planes_lambda_ht_set_independently_for_nl4d() {
    let plane_options = nl4d_options(Some(2.0), Some(3.5));

    let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
    let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
    let luma = expect_nl4d(luma_algorithm);
    let chroma = expect_nl4d(chroma_algorithm);

    assert!((luma.lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
    assert!((chroma.lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);

    // Every other field stays shared between the two instances even though lambda_ht diverges.
    assert_eq!(luma.refine, chroma.refine);
    assert_eq!(luma.spatial_radius, chroma.spatial_radius);
    assert!((luma.c_min - chroma.c_min).abs() < f32::EPSILON);
}

#[test]
fn unset_nl4d_overrides_resolve_to_different_lambda_ht_per_plane_end_to_end() {
    let plane_options = nl4d_options(None, None);

    let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
    let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
    let luma = expect_nl4d(luma_algorithm);
    let chroma = expect_nl4d(chroma_algorithm);

    // Neither plane has anything set, so both stay unresolved at this layer.
    assert_eq!(luma.lambda_ht, None);
    assert_eq!(chroma.lambda_ht, None);

    // Construction resolves each through `nl4d_default_lambda_ht`, which gives luma and chroma
    // different values.
    let luma_default = crate::nl4d_default_lambda_ht(ChannelMode::Luma);
    let chroma_default = crate::nl4d_default_lambda_ht(ChannelMode::Chroma);
    assert!((luma_default - 4.158).abs() < f32::EPSILON);
    assert!((chroma_default - 3.234).abs() < f32::EPSILON);
    assert!((chroma_default - luma_default).abs() > f32::EPSILON);
}
