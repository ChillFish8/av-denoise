use av_denoise_core::{DenoisingMode, Nl4dOptions};

use crate::backend::Device;
use crate::host::{Algorithm, DenoiserOptions, Depth};
use crate::planar::{ChannelIntent, PlaneOptions};
use crate::tune::{TuningMode, profile_key, resolve_mode, tuning_frame_count};

#[test]
fn modes_parse_case_insensitively_and_print_back() {
    for (text, mode) in [
        ("auto", TuningMode::Auto),
        ("CACHED", TuningMode::Cached),
        ("Off", TuningMode::Off),
    ] {
        let parsed: TuningMode = text.parse().expect("a known mode");
        assert_eq!(parsed, mode);

        let printed = mode.to_string();
        let reparsed: TuningMode = printed.parse().expect("a printed mode");
        assert_eq!(reparsed, mode);
    }

    assert!("sometimes".parse::<TuningMode>().is_err());
}

#[test]
fn the_environment_overrides_the_flag() {
    assert_eq!(resolve_mode(TuningMode::Auto, Some("off")), TuningMode::Off);
    assert_eq!(resolve_mode(TuningMode::Off, Some("cached")), TuningMode::Cached);
    assert_eq!(resolve_mode(TuningMode::Cached, None), TuningMode::Cached);
}

#[test]
fn an_unknown_environment_value_keeps_the_flag() {
    assert_eq!(
        resolve_mode(TuningMode::Cached, Some("bogus")),
        TuningMode::Cached
    );
}

fn plane_options(algorithm: Algorithm, radius: u32) -> PlaneOptions {
    PlaneOptions {
        accelerators: Vec::new(),
        device: Device::Default,
        intent: ChannelIntent::Luma,
        mode: DenoisingMode::Temporal { radius },
        algorithm,
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Auto,
        deterministic: false,
    }
}

#[test]
fn tuning_reads_the_temporal_span_plus_eight_frames() {
    let nlmeans = plane_options(Algorithm::default(), 2);
    let nl4d_options = Nl4dOptions::default();
    let nl4d = plane_options(Algorithm::Nl4d(nl4d_options), 2);

    assert_eq!(tuning_frame_count(&nlmeans), 5 + 8);
    assert_eq!(tuning_frame_count(&nl4d), 9 + 8);
}

#[test]
fn profile_keys_bucket_the_plane_size_and_ignore_tuning_switches() {
    let options = DenoiserOptions::builder().depth(Depth::Ten).build();
    let other_switches = DenoiserOptions {
        tuning: TuningMode::Off,
        deterministic: true,
        ..options.clone()
    };

    let full_hd = profile_key(&options, 1920, 1080);
    let wuxga = profile_key(&options, 1920, 1200);
    let switched = profile_key(&other_switches, 1920, 1080);

    assert_eq!(full_hd, wuxga);
    assert_eq!(full_hd, switched);
    assert!(full_hd.ends_with("|w2048|h2048"), "{full_hd}");
}

#[test]
fn auto_runs_as_cached_after_tuning() {
    assert_eq!(TuningMode::Auto.after_tuning(), TuningMode::Cached);
    assert_eq!(TuningMode::Cached.after_tuning(), TuningMode::Cached);
    assert_eq!(TuningMode::Off.after_tuning(), TuningMode::Off);
}
