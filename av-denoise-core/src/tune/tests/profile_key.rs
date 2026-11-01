use crate::tune::{TUNE_EXTRA_FRAMES, profile_key, tuning_frame_count};
use crate::{
    ChannelMode,
    DenoisingMode,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PassGeometry,
    PassSpec,
    SampleFormat,
};

fn geometry(
    channels: ChannelMode,
    input: SampleFormat,
    output: SampleFormat,
    width: u32,
    height: u32,
) -> PassGeometry {
    PassGeometry {
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
) -> PassSpec {
    let options = Nl4dOptions {
        temporal_radius: radius,
        ..Nl4dOptions::default()
    };
    let geometry = geometry(channels, input, output, width, height);

    PassSpec::Nl4d { options, geometry }
}

fn nlmeans_spec(algorithm: NlmeansAlgorithm) -> PassSpec {
    let geometry = geometry(ChannelMode::Luma, SampleFormat::U8, SampleFormat::U8, 64, 64);

    PassSpec::Nlmeans { algorithm, geometry }
}

fn temporal_nlm(radius: u32) -> NlmeansOptions {
    NlmeansOptions {
        mode: DenoisingMode::Temporal { radius },
        ..NlmeansOptions::default()
    }
}

#[test]
fn an_eight_bit_nl4d_chroma_key_has_the_uniform_layout() {
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
    assert!(key.ends_with("})|Chroma|in-u8|out-u8|w1024|h1024"), "{key}");
}

#[test]
fn ten_bit_input_with_float_output_names_both_formats() {
    let ten_bit = SampleFormat::U16 { depth: 10 };
    let spec = nl4d_spec(ChannelMode::Luma, 2, ten_bit, SampleFormat::F32, 64, 64);

    let key = profile_key(&spec);

    assert!(key.ends_with("|Luma|in-u10|out-f32|w64|h64"), "{key}");
}

#[test]
fn different_radii_give_different_keys() {
    let radius_one = nl4d_spec(ChannelMode::Luma, 1, SampleFormat::U8, SampleFormat::U8, 64, 64);
    let radius_two = nl4d_spec(ChannelMode::Luma, 2, SampleFormat::U8, SampleFormat::U8, 64, 64);

    let key_one = profile_key(&radius_one);
    let key_two = profile_key(&radius_two);

    assert_ne!(key_one, key_two);
}

#[test]
fn fast_and_hq_nlmeans_give_different_keys() {
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

    assert!(fast_key.starts_with("Nlmeans(Fast("), "{fast_key}");
    assert!(hq_key.starts_with("Nlmeans(Hq("), "{hq_key}");
    assert!(fast_key.ends_with("|Luma|in-u8|out-u8|w64|h64"), "{fast_key}");
    assert!(hq_key.ends_with("|Luma|in-u8|out-u8|w64|h64"), "{hq_key}");
    assert_ne!(fast_key, hq_key);
}

#[test]
fn the_frame_count_follows_the_widest_span() {
    let nl4d = nl4d_spec(ChannelMode::Luma, 2, SampleFormat::U8, SampleFormat::U8, 64, 64);
    let nlm = temporal_nlm(1);
    let nlmeans = nlmeans_spec(NlmeansAlgorithm::Fast(nlm));

    let frames = tuning_frame_count(&[nl4d, nlmeans]);

    assert_eq!(frames, 9 + TUNE_EXTRA_FRAMES);
}
