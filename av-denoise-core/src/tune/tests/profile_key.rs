use crate::tune::{TUNE_EXTRA_FRAMES, profile_key, tuning_frame_count};
use crate::{
    ChannelMode,
    DenoisingMode,
    EngineSpec,
    Geometry,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    SampleFormat,
};

fn geometry(
    channels: ChannelMode,
    input: SampleFormat,
    output: SampleFormat,
    width: u32,
    height: u32,
) -> Geometry {
    Geometry {
        width,
        height,
        channels,
        input,
        output,
    }
}

fn nl4d_spec(
    channels: ChannelMode,
    radius: u32,
    input: SampleFormat,
    output: SampleFormat,
    width: u32,
    height: u32,
) -> EngineSpec {
    let options = Nl4dOptions {
        temporal_radius: radius,
        ..Nl4dOptions::default()
    };
    let geometry = geometry(channels, input, output, width, height);

    EngineSpec::Nl4d { options, geometry }
}

fn nlmeans_spec(algorithm: NlmeansAlgorithm) -> EngineSpec {
    let geometry = geometry(ChannelMode::Luma, SampleFormat::U8, SampleFormat::U8, 64, 64);

    EngineSpec::Nlmeans { algorithm, geometry }
}

fn temporal_nlm(radius: u32) -> NlmeansOptions {
    NlmeansOptions {
        mode: DenoisingMode::Temporal { radius },
        ..NlmeansOptions::default()
    }
}

#[test]
fn an_eight_bit_nl4d_chroma_key_matches_the_stored_layout() {
    let spec = nl4d_spec(
        ChannelMode::Chroma,
        2,
        SampleFormat::U8,
        SampleFormat::U8,
        960,
        540,
    );

    let key = profile_key(&spec);

    assert!(key.starts_with("Nl4d(Nl4dOptions {"), "{key}");
    assert!(
        key.ends_with("|Chroma|Temporal { radius: 2 }|Eight|w1024|h1024"),
        "{key}"
    );
}

#[test]
fn float_output_gets_a_format_suffix() {
    let spec = nl4d_spec(
        ChannelMode::Luma,
        2,
        SampleFormat::U8,
        SampleFormat::F32,
        1920,
        1080,
    );

    let key = profile_key(&spec);

    assert!(key.ends_with("|Eight|w2048|h2048|in-u8|out-f32"), "{key}");
}

#[test]
fn a_depth_av_denoise_never_builds_gets_a_format_suffix() {
    let sixteen_bit = SampleFormat::U16 { depth: 16 };
    let spec = nl4d_spec(ChannelMode::Luma, 2, sixteen_bit, sixteen_bit, 64, 64);

    let key = profile_key(&spec);

    assert!(key.ends_with("|u16|w64|h64|in-u16|out-u16"), "{key}");
}

#[test]
fn ten_bit_input_with_float_output_names_both_formats() {
    let ten_bit = SampleFormat::U16 { depth: 10 };
    let spec = nl4d_spec(ChannelMode::Luma, 2, ten_bit, SampleFormat::F32, 64, 64);

    let key = profile_key(&spec);

    assert!(key.ends_with("|Ten|w64|h64|in-u10|out-f32"), "{key}");
}

#[test]
fn ten_and_twelve_bit_keys_name_their_depth_without_a_suffix() {
    let ten_bit = SampleFormat::U16 { depth: 10 };
    let twelve_bit = SampleFormat::U16 { depth: 12 };
    let ten_spec = nl4d_spec(ChannelMode::Yuv, 2, ten_bit, ten_bit, 64, 64);
    let twelve_spec = nl4d_spec(ChannelMode::Yuv, 2, twelve_bit, twelve_bit, 64, 64);

    let ten_key = profile_key(&ten_spec);
    let twelve_key = profile_key(&twelve_spec);

    assert!(ten_key.ends_with("|Ten|w64|h64"), "{ten_key}");
    assert!(twelve_key.ends_with("|Twelve|w64|h64"), "{twelve_key}");
}

#[test]
fn fast_and_hq_nlmeans_keep_their_legacy_names() {
    let nlm = temporal_nlm(1);
    let fast = NlmeansAlgorithm::Fast(nlm);
    let hq_options = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };
    let hq = NlmeansAlgorithm::Hq(hq_options);

    let fast_spec = nlmeans_spec(fast);
    let hq_spec = nlmeans_spec(hq);

    let fast_key = profile_key(&fast_spec);
    let hq_key = profile_key(&hq_spec);

    assert!(fast_key.starts_with("Nlmeans(NlmeansOptions {"), "{fast_key}");
    assert!(hq_key.starts_with("NlmeansHq(NlmeansHqOptions {"), "{hq_key}");
    assert!(
        fast_key.ends_with("|Luma|Temporal { radius: 1 }|Eight|w64|h64"),
        "{fast_key}"
    );
    assert!(
        hq_key.ends_with("|Luma|Temporal { radius: 1 }|Eight|w64|h64"),
        "{hq_key}"
    );
}

#[test]
fn a_spatial_nlmeans_key_names_the_spatial_mode() {
    let nlm = NlmeansOptions::default();
    let spec = nlmeans_spec(NlmeansAlgorithm::Fast(nlm));

    let key = profile_key(&spec);

    assert!(key.ends_with("|Luma|Spacial|Eight|w64|h64"), "{key}");
}

#[test]
fn the_frame_count_follows_the_widest_span() {
    let nl4d = nl4d_spec(ChannelMode::Luma, 2, SampleFormat::U8, SampleFormat::U8, 64, 64);
    let nlm = temporal_nlm(1);
    let nlmeans = nlmeans_spec(NlmeansAlgorithm::Fast(nlm));

    let frames = tuning_frame_count(&[nl4d, nlmeans]);

    assert_eq!(frames, 9 + TUNE_EXTRA_FRAMES);
}
