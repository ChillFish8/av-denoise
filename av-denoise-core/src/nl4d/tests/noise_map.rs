use super::helpers::{R, make_client, unit_noise};
use crate::bench_api::HostIo;
use crate::nl4d::denoiser::noise_curve_upload;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{ChannelMode, HqParams, NOISE_CURVE_BINS, NlmParams, PrefilterMode};

// Large enough that each luma bin the ramp crosses gathers the blocks a curve needs. At 320x240
// no curve forms.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 12;
const RAMP_TOP: f32 = 0.15;
const RAMP_BOTTOM: f32 = 0.85;

/// A clip's denoised frames, and whether the front end built a noise curve during any pass.
struct ClipRun {
    outputs: Vec<Vec<f32>>,
    curve_seen: bool,
}

fn ramp_luma(y: u32) -> f32 {
    let fraction = y as f32 / (HEIGHT - 1) as f32;
    RAMP_TOP + (RAMP_BOTTOM - RAMP_TOP) * fraction
}

/// The noise std at `luma`, which grows with brightness.
fn noise_std_at(luma: f32) -> f32 {
    0.004 + 0.02 * luma
}

/// A static vertical brightness ramp with fresh noise per frame, in `channels` interleaved planes
/// that all carry the same ramp.
fn ramp_clip(channels: u32) -> Vec<Vec<f32>> {
    let mut frames = Vec::new();
    for frame_index in 0..FRAMES {
        let mut frame = vec![0.0f32; (WIDTH * HEIGHT * channels) as usize];
        for y in 0..HEIGHT {
            let luma = ramp_luma(y);
            let noise_std = noise_std_at(luma);

            for x in 0..WIDTH {
                for channel in 0..channels {
                    let pixel = y * WIDTH + x;
                    let sample = pixel * channels + channel;
                    let seed = frame_index * channels + channel;
                    let noise = unit_noise(pixel, seed);
                    frame[sample as usize] = (luma + noise * noise_std).clamp(0.0, 1.0);
                }
            }
        }

        frames.push(frame);
    }

    frames
}

fn ramp_params(channels: ChannelMode, noise_map: bool, sigma_scale: f32) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let hq = HqParams {
        sigma_scale,
        ..HqParams::default()
    };
    let nlm = NlmParams {
        channels,
        prefilter: PrefilterMode::None,
        hq: Some(hq),
        ..defaults.nlm
    };

    Nl4dParams {
        nlm,
        noise_map,
        ..defaults
    }
}

fn denoise_clip(params: Nl4dParams, frames: &[Vec<f32>]) -> ClipRun {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, WIDTH, HEIGHT).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let mut curve_seen = false;
    for frame in frames {
        denoiser.push_frame(frame);
        let output = denoiser.denoise().expect("denoise failed");
        curve_seen |= denoiser.front_for_test().current_noise_curve().is_some();

        if let Some(output_frame) = output {
            outputs.push(output_frame);
        }
    }

    denoiser
        .flush(|frame| {
            let output_frame = frame.to_vec();
            outputs.push(output_frame);
        })
        .expect("flush failed");

    assert_eq!(outputs.len(), frames.len(), "expected one output per input frame");

    ClipRun { outputs, curve_seen }
}

/// The standard deviation of what the denoiser removed, over every frame.
fn residual_std(inputs: &[Vec<f32>], outputs: &[Vec<f32>]) -> f64 {
    let residuals: Vec<f64> = inputs
        .iter()
        .zip(outputs)
        .flat_map(|(input, output)| input.iter().zip(output))
        .map(|(&noisy, &clean)| noisy as f64 - clean as f64)
        .collect();

    let count = residuals.len() as f64;
    let mean = residuals.iter().sum::<f64>() / count;
    let variance = residuals
        .iter()
        .map(|residual| (residual - mean).powi(2))
        .sum::<f64>()
        / count;
    variance.sqrt()
}

#[test]
fn the_noise_map_changes_the_output_on_brightness_dependent_noise() {
    let frames = ramp_clip(1);

    let map_on_params = ramp_params(ChannelMode::Luma, true, 1.0);
    let map_on = denoise_clip(map_on_params, &frames);
    assert!(map_on.curve_seen, "the ramp clip should form a noise curve");

    let map_off_params = ramp_params(ChannelMode::Luma, false, 1.0);
    let map_off = denoise_clip(map_off_params, &frames);
    assert!(
        !map_off.curve_seen,
        "a map-off denoiser should never build a curve"
    );

    assert_ne!(map_on.outputs, map_off.outputs);
}

#[test]
fn the_noise_map_off_matches_a_curveless_run() {
    let ratios = [1.5f32; NOISE_CURVE_BINS];
    let curveless = noise_curve_upload(None, true);
    let map_off = noise_curve_upload(Some(ratios), false);

    assert_eq!(curveless, ([0.0f32; NOISE_CURVE_BINS], 0));
    assert_eq!(map_off, curveless);
}

#[test]
fn the_noise_map_on_uploads_the_curve() {
    let ratios = [1.5f32; NOISE_CURVE_BINS];
    let upload = noise_curve_upload(Some(ratios), true);

    assert_eq!(upload, (ratios, 1));
}

#[test]
fn sigma_scale_still_scales_strength_with_the_noise_map_on() {
    let frames = ramp_clip(1);

    let mut removed = Vec::new();
    for noise_map in [true, false] {
        // A lower threshold keeps both arms off the point where all the noise is removed.
        let mut full_params = ramp_params(ChannelMode::Luma, noise_map, 1.0);
        full_params.lambda_ht = 2.0;
        full_params.flat_boost = 1.0;
        full_params.shadow_soften = 1.0;
        let full = denoise_clip(full_params, &frames);
        assert_eq!(full.curve_seen, noise_map);

        let mut scaled_params = ramp_params(ChannelMode::Luma, noise_map, 0.8);
        scaled_params.lambda_ht = 2.0;
        scaled_params.flat_boost = 1.0;
        scaled_params.shadow_soften = 1.0;
        let scaled = denoise_clip(scaled_params, &frames);

        let full_std = residual_std(&frames, &full.outputs);
        let scaled_std = residual_std(&frames, &scaled.outputs);
        assert!(
            scaled_std < full_std,
            "noise_map={noise_map}: sigma_scale 0.8 should remove less, got {scaled_std} vs {full_std}"
        );

        removed.push(full_std - scaled_std);
    }

    let ratio = removed[0] / removed[1];
    assert!(
        (0.5..=2.0).contains(&ratio),
        "sigma_scale's effect with the map on should be within 2x of it with the map off, \
         got on={} off={} ratio={ratio}",
        removed[0],
        removed[1]
    );
}

#[test]
fn a_chroma_denoiser_without_a_flat_boost_never_builds_a_curve() {
    let frames = ramp_clip(2);

    let mut map_on_params = ramp_params(ChannelMode::Chroma, true, 1.0);
    map_on_params.chroma_flat_boost = 1.0;
    let map_on = denoise_clip(map_on_params, &frames);

    let mut map_off_params = ramp_params(ChannelMode::Chroma, false, 1.0);
    map_off_params.chroma_flat_boost = 1.0;
    let map_off = denoise_clip(map_off_params, &frames);

    assert!(!map_on.curve_seen, "a chroma denoiser should never build a curve");
    assert_eq!(map_on.outputs, map_off.outputs);
}
