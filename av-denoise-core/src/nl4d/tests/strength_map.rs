use super::helpers::{R, make_client, unit_noise};
use crate::collab::kernels::fused::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA};
use crate::nl4d::denoiser::strength_map_upload;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    NlmParams,
    PrefilterMode,
    QuarterClass,
    QuarterClasses,
    StrengthMapParams,
};

const LUMA_MAP: StrengthMapParams = StrengthMapParams {
    flat_boost: 1.5,
    shadow_soften: 0.65,
};
const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 12;
/// Columns either side of the texture seam left out of each half's measurement.
const SEAM_MARGIN: u32 = 48;

fn two_quarters() -> QuarterClasses {
    let classes = vec![
        Some(QuarterClass {
            flat: true,
            luma: 0.1,
        }),
        Some(QuarterClass {
            flat: false,
            luma: 0.1,
        }),
    ];
    QuarterClasses::from_classes(2, 1, classes)
}

#[test]
fn the_luma_map_uploads_only_with_a_valid_curve() {
    let classes = two_quarters();

    let with_curve = strength_map_upload(Some(&classes), 1, Some(LUMA_MAP), None);
    let without_curve = strength_map_upload(Some(&classes), 0, Some(LUMA_MAP), None);

    assert_eq!(with_curve, Some((vec![1.5, 0.65], STRENGTH_MAP_LUMA)));
    assert_eq!(without_curve, None);
}

#[test]
fn the_chroma_map_uploads_without_a_curve() {
    let classes = two_quarters();

    let upload = strength_map_upload(Some(&classes), 0, None, Some(1.5));

    assert_eq!(upload, Some((vec![1.5, 1.0], STRENGTH_MAP_ALL)));
}

#[test]
fn nothing_uploads_without_classes_or_a_map() {
    let classes = two_quarters();

    assert_eq!(strength_map_upload(None, 1, Some(LUMA_MAP), None), None);
    assert_eq!(strength_map_upload(Some(&classes), 1, None, None), None);
}

fn ramp_luma(y: u32) -> f32 {
    let fraction = y as f32 / (HEIGHT - 1) as f32;
    0.15 + 0.7 * fraction
}

fn texture_at(x: u32, y: u32) -> f32 {
    if x >= WIDTH / 2 {
        return 0.0;
    }

    let phase_x = x as f32 * std::f32::consts::TAU / 12.0;
    let phase_y = y as f32 * std::f32::consts::TAU / 16.0;
    0.04 * phase_x.sin() * phase_y.cos()
}

/// A static clip, textured on its left half and flat on its right, with grain that grows with
/// brightness, in `channels` interleaved planes.
fn half_textured_clip(channels: u32) -> Vec<Vec<f32>> {
    let mut frames = Vec::new();
    for frame_index in 0..FRAMES {
        let mut frame = vec![0.0f32; (WIDTH * HEIGHT * channels) as usize];
        for y in 0..HEIGHT {
            let luma = ramp_luma(y);
            let noise_std = 0.004 + 0.02 * luma;

            for x in 0..WIDTH {
                let clean = luma + texture_at(x, y);
                for channel in 0..channels {
                    let pixel = y * WIDTH + x;
                    let sample = pixel * channels + channel;
                    let seed = frame_index * channels + channel;
                    let noise = unit_noise(pixel, seed);
                    frame[sample as usize] = (clean + noise * noise_std).clamp(0.0, 1.0);
                }
            }
        }
        frames.push(frame);
    }
    frames
}

fn clip_params(
    channels: ChannelMode,
    flat_boost: f32,
    chroma_flat_boost: f32,
    shadow_soften: f32,
) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    Nl4dParams {
        nlm: NlmParams {
            channels,
            prefilter: PrefilterMode::None,
            hq: Some(HqParams::default()),
            ..defaults.nlm
        },
        flat_boost,
        chroma_flat_boost,
        shadow_soften,
        ..defaults
    }
}

fn build_nl4d(channels: ChannelMode, mut params: Nl4dParams) -> Nl4dDenoiser<R> {
    let client = make_client();
    params.nlm.channels = channels;
    Nl4dDenoiser::new(&client, params, 64, 64).expect("construction failed")
}

fn unit_params(channels: ChannelMode) -> Nl4dParams {
    clip_params(channels, 1.0, 1.0, 1.0)
}

fn default_params(channels: ChannelMode) -> Nl4dParams {
    clip_params(channels, 1.5, 1.5, 0.65)
}

/// A clip's denoised frames, and whether the front end ever held quarter classes.
struct ClipRun {
    outputs: Vec<Vec<f32>>,
    classes_seen: bool,
}

fn denoise_clip(params: Nl4dParams, frames: &[Vec<f32>]) -> ClipRun {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, WIDTH, HEIGHT).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let mut classes_seen = false;
    for frame in frames {
        denoiser.push_frame(frame);
        let pending = denoiser.denoise_submit().expect("denoise_submit failed");
        classes_seen |= denoiser.front_for_test().current_quarter_classes().is_some();

        if let Some(pending) = pending {
            let output = pending.wait().expect("readback failed");
            let output_frame = output.into_f32().expect("f32 output");
            outputs.push(output_frame);
        }
    }

    denoiser
        .flush(|frame| {
            let output_frame = frame.as_f32().expect("f32 denoiser").to_vec();
            outputs.push(output_frame);
        })
        .expect("flush failed");

    ClipRun {
        outputs,
        classes_seen,
    }
}

/// The std of what was removed from channel 0 inside columns `x_range` and rows `y_range`.
fn removed_std(
    inputs: &[Vec<f32>],
    outputs: &[Vec<f32>],
    channels: u32,
    x_range: std::ops::Range<u32>,
    y_range: std::ops::Range<u32>,
) -> f64 {
    removed_channel_std(inputs, outputs, channels, 0, x_range, y_range)
}

/// The std of what was removed from `channel` inside columns `x_range` and rows `y_range`.
fn removed_channel_std(
    inputs: &[Vec<f32>],
    outputs: &[Vec<f32>],
    channels: u32,
    channel: u32,
    x_range: std::ops::Range<u32>,
    y_range: std::ops::Range<u32>,
) -> f64 {
    let mut residuals = Vec::new();
    for (input, output) in inputs.iter().zip(outputs) {
        for y in y_range.clone() {
            for x in x_range.clone() {
                let sample = ((y * WIDTH + x) * channels + channel) as usize;
                residuals.push(input[sample] as f64 - output[sample] as f64);
            }
        }
    }

    let count = residuals.len() as f64;
    let mean = residuals.iter().sum::<f64>() / count;
    let variance = residuals.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count;
    variance.sqrt()
}

/// Every sample of `channel` across `frames`, in order.
fn channel_samples(frames: &[Vec<f32>], channels: u32, channel: u32) -> Vec<f32> {
    frames
        .iter()
        .flat_map(|frame| frame.iter().skip(channel as usize).step_by(channels as usize))
        .copied()
        .collect()
}

/// Rows whose luma sits below 128 of 255, where the soften applies in full.
fn dark_rows() -> std::ops::Range<u32> {
    0..HEIGHT * 2 / 5
}

#[test]
fn the_defaults_differ_from_a_map_of_ones() {
    let frames = half_textured_clip(1);

    let defaults = denoise_clip(default_params(ChannelMode::Luma), &frames);
    let ones = denoise_clip(unit_params(ChannelMode::Luma), &frames);

    assert!(defaults.classes_seen, "the clip should form quarter classes");
    assert_ne!(defaults.outputs, ones.outputs);
}

#[test]
fn shadow_soften_removes_less_from_textured_darks() {
    let frames = half_textured_clip(1);

    let softened = denoise_clip(clip_params(ChannelMode::Luma, 1.0, 1.0, 0.65), &frames);
    let ones = denoise_clip(unit_params(ChannelMode::Luma), &frames);

    let left = 0..WIDTH / 2 - SEAM_MARGIN;
    let softened_std = removed_std(&frames, &softened.outputs, 1, left.clone(), dark_rows());
    let ones_std = removed_std(&frames, &ones.outputs, 1, left, dark_rows());
    assert!(softened_std < ones_std, "{softened_std} vs {ones_std}");
}

#[test]
fn flat_boost_removes_more_from_flat_grain() {
    let frames = half_textured_clip(1);

    let boosted = denoise_clip(clip_params(ChannelMode::Luma, 1.5, 1.0, 1.0), &frames);
    let ones = denoise_clip(unit_params(ChannelMode::Luma), &frames);

    let right = WIDTH / 2 + SEAM_MARGIN..WIDTH;
    let boosted_std = removed_std(&frames, &boosted.outputs, 1, right.clone(), 0..HEIGHT);
    let ones_std = removed_std(&frames, &ones.outputs, 1, right, 0..HEIGHT);
    assert!(boosted_std > ones_std, "{boosted_std} vs {ones_std}");
}

#[test]
fn the_noise_map_off_ignores_the_strength_map() {
    let frames = half_textured_clip(1);
    let mut defaults = default_params(ChannelMode::Luma);
    defaults.noise_map = false;
    let mut ones = unit_params(ChannelMode::Luma);
    ones.noise_map = false;

    let with_defaults = denoise_clip(defaults, &frames);
    let with_ones = denoise_clip(ones, &frames);

    assert!(!with_defaults.classes_seen);
    assert_eq!(with_defaults.outputs, with_ones.outputs);
}

#[test]
fn a_pinned_sigma_never_applies_a_map() {
    let frames = half_textured_clip(1);
    let mut defaults = default_params(ChannelMode::Luma);
    defaults.nlm.hq = Some(HqParams::with_sigma(0.02));
    let mut ones = unit_params(ChannelMode::Luma);
    ones.nlm.hq = Some(HqParams::with_sigma(0.02));

    let with_defaults = denoise_clip(defaults, &frames);
    let with_ones = denoise_clip(ones, &frames);

    assert!(!with_defaults.classes_seen);
    assert_eq!(with_defaults.outputs, with_ones.outputs);
}

#[test]
fn a_chroma_denoiser_applies_its_flat_boost() {
    let frames = half_textured_clip(2);

    let boosted = denoise_clip(default_params(ChannelMode::Chroma), &frames);
    let ones = denoise_clip(unit_params(ChannelMode::Chroma), &frames);

    assert!(
        boosted.classes_seen,
        "a boosting chroma denoiser should class its quarters"
    );
    assert!(
        !ones.classes_seen,
        "a chroma denoiser at 1.0 should build no classes"
    );
    assert_ne!(boosted.outputs, ones.outputs);
}

#[test]
fn a_chroma_denoiser_with_the_noise_map_off_builds_nothing() {
    let frames = half_textured_clip(2);
    let mut defaults = default_params(ChannelMode::Chroma);
    defaults.noise_map = false;
    let mut ones = unit_params(ChannelMode::Chroma);
    ones.noise_map = false;

    let with_defaults = denoise_clip(defaults, &frames);
    let with_ones = denoise_clip(ones, &frames);

    assert!(!with_defaults.classes_seen);
    assert_eq!(with_defaults.outputs, with_ones.outputs);
}

#[test]
fn a_chroma_denoiser_boosts_its_second_channel() {
    let frames = half_textured_clip(2);

    let boosted = denoise_clip(default_params(ChannelMode::Chroma), &frames);
    let unboosted = denoise_clip(clip_params(ChannelMode::Chroma, 1.5, 1.0, 0.65), &frames);

    let boosted_samples = channel_samples(&boosted.outputs, 2, 1);
    let unboosted_samples = channel_samples(&unboosted.outputs, 2, 1);
    assert_ne!(boosted_samples, unboosted_samples);

    let right = WIDTH / 2 + SEAM_MARGIN..WIDTH;
    let boosted_std = removed_channel_std(&frames, &boosted.outputs, 2, 1, right.clone(), 0..HEIGHT);
    let unboosted_std = removed_channel_std(&frames, &unboosted.outputs, 2, 1, right, 0..HEIGHT);
    assert!(boosted_std > unboosted_std, "{boosted_std} vs {unboosted_std}");
}

#[test]
fn the_luma_denoiser_passes_the_texture_cut_to_its_front() {
    let denoiser = build_nl4d(ChannelMode::Luma, Nl4dParams::default());

    assert_eq!(denoiser.front_for_test().flat_texture_cut(), Some(0.21));
}

#[test]
fn the_fused_yuv_denoiser_passes_the_texture_cut_to_its_front() {
    let denoiser = build_nl4d(ChannelMode::Yuv, Nl4dParams::default());

    assert_eq!(denoiser.front_for_test().flat_texture_cut(), Some(0.21));
}

#[test]
fn the_chroma_denoiser_never_gets_a_texture_cut() {
    let denoiser = build_nl4d(ChannelMode::Chroma, Nl4dParams::default());

    assert_eq!(denoiser.front_for_test().flat_texture_cut(), None);
}

#[test]
fn a_texture_cut_of_one_is_not_passed_on() {
    let params = Nl4dParams {
        flat_texture_cut: 1.0,
        ..Nl4dParams::default()
    };
    let denoiser = build_nl4d(ChannelMode::Luma, params);

    assert_eq!(denoiser.front_for_test().flat_texture_cut(), None);
}
