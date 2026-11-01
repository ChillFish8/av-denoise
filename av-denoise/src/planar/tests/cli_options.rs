use av_denoise_core::{ChannelMode, Nl4dOptions, NlmeansAlgorithm, NlmeansHqOptions, NlmeansOptions};

use super::*;

fn layout() -> FrameLayout {
    FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

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
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

/// The luma and chroma nlmeans options `plane_options` resolves to.
fn nlmeans_planes(plane_options: &PlaneOptions) -> (NlmeansAlgorithm, NlmeansAlgorithm) {
    let spec = plane_spec(plane_options, layout());
    let EngineSpec::Nlmeans {
        luma: Some(luma),
        chroma: Some(chroma),
        ..
    } = spec
    else {
        panic!("expected an nlmeans spec cleaning both plane groups, got {spec:?}");
    };

    (luma, chroma)
}

/// The luma and chroma nl4d options `plane_options` resolves to.
fn nl4d_planes(plane_options: &PlaneOptions) -> (Nl4dOptions, Nl4dOptions) {
    let spec = plane_spec(plane_options, layout());
    let EngineSpec::Nl4d {
        luma: Some(luma),
        chroma: Some(chroma),
        ..
    } = spec
    else {
        panic!("expected an nl4d spec cleaning both plane groups, got {spec:?}");
    };

    (luma, chroma)
}

fn expect_fast(algorithm: NlmeansAlgorithm) -> NlmeansOptions {
    match algorithm {
        NlmeansAlgorithm::Fast(options) => options,
        other => panic!("expected the fast nlmeans path, got {other:?}"),
    }
}

#[test]
fn luma_strength_alone_overrides_only_the_luma_plane() {
    let algorithm = Algorithm::default();
    let plane_options = base_options(DenoisingMode::Spacial, algorithm, Some(0.7), None);

    let (luma_algorithm, chroma_algorithm) = nlmeans_planes(&plane_options);
    let luma = expect_fast(luma_algorithm);
    let chroma = expect_fast(chroma_algorithm);

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

    let (luma_algorithm, chroma_algorithm) = nlmeans_planes(&plane_options);
    let luma = expect_fast(luma_algorithm);
    let chroma = expect_fast(chroma_algorithm);

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

    let (luma, chroma) = nlmeans_planes(&plane_options);

    for (name, algorithm) in [("luma", luma), ("chroma", chroma)] {
        let NlmeansAlgorithm::Hq(hq) = algorithm else {
            panic!("expected HQ nlmeans for {name}, got {algorithm:?}");
        };

        assert_eq!(hq.nlm.mode, DenoisingMode::Temporal { radius: 4 });
        assert_eq!(
            hq.nlm.tuning.strength, None,
            "{name} should use the calibrated table"
        );
    }
}

#[test]
fn the_spec_follows_the_layouts_chroma_planes() {
    let plane_options = base_options(DenoisingMode::Spacial, Algorithm::default(), None, None);
    let layouts = [
        (Subsampling::Yuv420, ChromaLayout::Yuv420),
        (Subsampling::Yuv422, ChromaLayout::Yuv422),
        (Subsampling::Yuv444, ChromaLayout::Yuv444),
    ];

    for (subsampling, chroma) in layouts {
        let frame_layout = FrameLayout {
            subsampling,
            ..layout()
        };
        let spec = plane_spec(&plane_options, frame_layout);

        assert_eq!(spec.geometry().chroma, chroma);
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
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

#[test]
fn luma_lambda_ht_alone_overrides_only_the_luma_instance_for_nl4d() {
    let plane_options = nl4d_options(Some(4.0), None);
    let default_lambda_ht = Nl4dOptions::default().lambda_ht;

    let (luma, chroma) = nl4d_planes(&plane_options);

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

    let (luma, chroma) = nl4d_planes(&plane_options);

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

    let (luma, chroma) = nl4d_planes(&plane_options);

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

    let (luma, chroma) = nl4d_planes(&plane_options);

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
