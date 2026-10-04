use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    SampleFormat,
};

use crate::backend::EngineSpec;
use crate::host::{Algorithm, DenoiserOptions, Depth};

fn spec_for(options: &DenoiserOptions) -> EngineSpec {
    options.algorithm.engine_spec(options, 32, 16)
}

fn expect_nlmeans(spec: EngineSpec) -> NlmeansAlgorithm {
    match spec {
        EngineSpec::Nlmeans { algorithm, .. } => algorithm,
        other => panic!("expected an nlmeans spec, got {other:?}"),
    }
}

fn expect_nl4d(spec: EngineSpec) -> Nl4dOptions {
    match spec {
        EngineSpec::Nl4d { options, .. } => options,
        other => panic!("expected an nl4d spec, got {other:?}"),
    }
}

#[test]
fn the_default_algorithm_is_the_fast_nlmeans_path() {
    let options = DenoiserOptions::builder().build();
    assert_eq!(options.algorithm, Algorithm::Nlmeans(NlmeansOptions::default()));
}

#[test]
fn the_geometry_carries_the_size_planes_and_depth() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Chroma)
        .depth(Depth::Ten)
        .build();

    let EngineSpec::Nlmeans { geometry, .. } = spec_for(&options) else {
        panic!("expected an nlmeans spec");
    };

    assert_eq!(geometry.width, 32);
    assert_eq!(geometry.height, 16);
    assert_eq!(geometry.channels, ChannelMode::Chroma);
    assert_eq!(geometry.input, SampleFormat::U16 { depth: 10 });
    assert_eq!(geometry.output, SampleFormat::U16 { depth: 10 });
}

#[test]
fn the_denoising_mode_wins_over_the_fast_options_mode() {
    let algorithm = Algorithm::Nlmeans(NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 5 },
        ..NlmeansOptions::default()
    });
    let options = DenoiserOptions::builder()
        .mode(DenoisingMode::Spacial)
        .algorithm(algorithm)
        .build();

    let spec = spec_for(&options);
    let NlmeansAlgorithm::Fast(fast) = expect_nlmeans(spec) else {
        panic!("expected the fast variant");
    };

    assert_eq!(fast.mode, DenoisingMode::Spacial);
}

#[test]
fn the_denoising_mode_wins_over_the_hq_options_mode() {
    let algorithm = Algorithm::NlmeansHq(NlmeansHqOptions::default());
    let options = DenoiserOptions::builder()
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(algorithm)
        .build();

    let spec = spec_for(&options);
    let NlmeansAlgorithm::Hq(hq) = expect_nlmeans(spec) else {
        panic!("expected the hq variant");
    };

    assert_eq!(hq.nlm.mode, DenoisingMode::Temporal { radius: 3 });
}

#[test]
fn the_denoising_mode_sets_the_nl4d_temporal_radius() {
    for radius in [1u32, 4, 8] {
        let algorithm = Algorithm::Nl4d(Nl4dOptions {
            temporal_radius: 2,
            ..Nl4dOptions::default()
        });
        let options = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius })
            .algorithm(algorithm)
            .build();

        let spec = spec_for(&options);
        let nl4d = expect_nl4d(spec);
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
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(Algorithm::Nl4d(nl4d))
        .build();

    let spec = spec_for(&options);
    let resolved = expect_nl4d(spec);
    assert_eq!(resolved.sigma, Some(0.02));
    assert_eq!(resolved.refine, 3);
}
