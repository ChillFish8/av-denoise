use av_denoise_core::{
    ChromaLayout,
    DenoisingMode,
    EngineSpec,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    SampleFormat,
};

use crate::host::{Algorithm, ChannelIntent, DenoiserOptions, Depth};

fn spec_for(options: &DenoiserOptions) -> EngineSpec {
    options.engine_spec(32, 16)
}

fn expect_luma_nlmeans(spec: EngineSpec) -> NlmeansAlgorithm {
    match spec {
        EngineSpec::Nlmeans {
            luma: Some(algorithm),
            ..
        } => algorithm,
        other => panic!("expected an nlmeans spec cleaning luma, got {other:?}"),
    }
}

fn expect_luma_nl4d(spec: EngineSpec) -> Nl4dOptions {
    match spec {
        EngineSpec::Nl4d {
            luma: Some(options), ..
        } => options,
        other => panic!("expected an nl4d spec cleaning luma, got {other:?}"),
    }
}

fn gray_luma() -> DenoiserOptions {
    DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .build()
}

#[test]
fn the_default_algorithm_is_the_fast_nlmeans_path() {
    let options = gray_luma();
    let expected = Algorithm::Nlmeans(NlmeansOptions::default());

    assert_eq!(options.algorithm, expected);
}

#[test]
fn the_geometry_carries_the_size_layout_and_depth() {
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Chroma)
        .chroma(ChromaLayout::Yuv422)
        .depth(Depth::Ten)
        .build();

    let EngineSpec::Nlmeans {
        luma,
        chroma,
        geometry,
    } = spec_for(&options)
    else {
        panic!("expected an nlmeans spec");
    };

    assert!(luma.is_none(), "a chroma intent leaves luma uncleaned");
    assert!(chroma.is_some(), "a chroma intent cleans chroma");
    assert_eq!(geometry.width, 32);
    assert_eq!(geometry.height, 16);
    assert_eq!(geometry.chroma, ChromaLayout::Yuv422);
    assert_eq!(geometry.input, SampleFormat::U16 { depth: 10 });
    assert_eq!(geometry.output, SampleFormat::U16 { depth: 10 });
}

#[test]
fn each_intent_cleans_its_own_plane_groups() {
    let cases = [
        (ChannelIntent::Luma, true, false),
        (ChannelIntent::Chroma, false, true),
        (ChannelIntent::LumaChroma, true, true),
    ];

    for (intent, cleans_luma, cleans_chroma) in cases {
        let options = DenoiserOptions::builder()
            .intent(intent)
            .chroma(ChromaLayout::Yuv420)
            .build();

        let EngineSpec::Nlmeans { luma, chroma, .. } = spec_for(&options) else {
            panic!("expected an nlmeans spec");
        };

        assert_eq!(luma.is_some(), cleans_luma, "luma for {intent:?}");
        assert_eq!(chroma.is_some(), cleans_chroma, "chroma for {intent:?}");
    }
}

#[test]
fn the_denoising_mode_wins_over_the_fast_options_mode() {
    let fast_options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 5 },
        ..NlmeansOptions::default()
    };
    let algorithm = Algorithm::Nlmeans(fast_options);
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Spacial)
        .algorithm(algorithm)
        .build();

    let spec = spec_for(&options);
    let NlmeansAlgorithm::Fast(fast) = expect_luma_nlmeans(spec) else {
        panic!("expected the fast variant");
    };

    assert_eq!(fast.mode, DenoisingMode::Spacial);
}

#[test]
fn the_denoising_mode_wins_over_the_hq_options_mode() {
    let algorithm = Algorithm::NlmeansHq(NlmeansHqOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(algorithm)
        .build();

    let spec = spec_for(&options);
    let NlmeansAlgorithm::Hq(hq) = expect_luma_nlmeans(spec) else {
        panic!("expected the hq variant");
    };

    assert_eq!(hq.nlm.mode, DenoisingMode::Temporal { radius: 3 });
}

#[test]
fn the_denoising_mode_sets_the_nl4d_temporal_radius() {
    for radius in [1u32, 4, 8] {
        let nl4d_options = Nl4dOptions {
            temporal_radius: 2,
            ..Nl4dOptions::default()
        };
        let algorithm = Algorithm::Nl4d(nl4d_options);
        let options = DenoiserOptions::builder()
            .intent(ChannelIntent::Luma)
            .chroma(ChromaLayout::Gray)
            .mode(DenoisingMode::Temporal { radius })
            .algorithm(algorithm)
            .build();

        let spec = spec_for(&options);
        let nl4d = expect_luma_nl4d(spec);

        assert_eq!(nl4d.temporal_radius, radius);
    }
}

#[test]
fn other_algorithm_options_pass_through_untouched() {
    let nl4d = Nl4dOptions {
        sigma: Some(0.02),
        refine: 3,
        ..Nl4dOptions::default()
    };
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(Algorithm::Nl4d(nl4d))
        .build();

    let spec = spec_for(&options);
    let resolved = expect_luma_nl4d(spec);

    assert_eq!(resolved.sigma, Some(0.02));
    assert_eq!(resolved.refine, 3);
}

#[test]
fn grain_is_only_exported_from_luma() {
    let nl4d = Nl4dOptions {
        grain_export: true,
        ..Nl4dOptions::default()
    };
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::LumaChroma)
        .chroma(ChromaLayout::Yuv420)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(Algorithm::Nl4d(nl4d))
        .build();

    let EngineSpec::Nl4d {
        luma: Some(luma),
        chroma: Some(chroma),
        ..
    } = spec_for(&options)
    else {
        panic!("expected an nl4d spec cleaning both plane groups");
    };

    assert!(luma.grain_export);
    assert!(!chroma.grain_export);
}

#[test]
fn the_grain_report_reaches_both_plane_groups() {
    let nl4d = Nl4dOptions {
        grain_report: true,
        ..Nl4dOptions::default()
    };
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::LumaChroma)
        .chroma(ChromaLayout::Yuv420)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(Algorithm::Nl4d(nl4d))
        .build();

    let EngineSpec::Nl4d {
        luma: Some(luma),
        chroma: Some(chroma),
        ..
    } = spec_for(&options)
    else {
        panic!("expected an nl4d spec cleaning both plane groups");
    };

    assert!(luma.grain_report);
    assert!(chroma.grain_report);
}
