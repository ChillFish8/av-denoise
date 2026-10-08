use super::helpers::{R, SIGMA, make_client, static_clip_params, textured_base};
use crate::bench_api::HostIo;
use crate::collab::supports_f16_search;
use crate::nl4d::Nl4dDenoiser;
use crate::nlmeans::ChannelMode;
use crate::nlmeans::tests::helpers::noisy_field_over;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const RADIUS: u32 = 2;
const FRAME_COUNT: usize = 9;

#[test]
fn the_f16_search_follows_the_device_query() {
    let client = make_client();
    let params = static_clip_params(RADIUS);
    let denoiser = Nl4dDenoiser::<R>::new(&client, params, WIDTH, HEIGHT).expect("construction failed");

    let supported = supports_f16_search(&client);
    let has_search_ring = denoiser.front_for_test().search_ring().is_some();
    assert_eq!(denoiser.uses_f16_search(), supported);
    assert_eq!(
        has_search_ring, supported,
        "the search ring is allocated only for the f16 search"
    );
}

#[test]
fn the_f16_search_denoises_a_static_clip_like_the_f32_search() {
    let client = make_client();
    if !supports_f16_search(&client) {
        eprintln!("skipping: this device has no f16 search");
        return;
    }

    let base = textured_base(WIDTH, HEIGHT);

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let channel_count = channels.count() as usize;
        let clean = interleave_copies(&base, channel_count);
        let noisy_frames = static_noisy_frames(&base, channel_count);

        let f16_outputs = denoise_clip(channels, &noisy_frames, true);
        let f32_outputs = denoise_clip(channels, &noisy_frames, false);
        assert_eq!(f16_outputs.len(), FRAME_COUNT, "{channels:?}: f16 frame count");
        assert_eq!(f32_outputs.len(), FRAME_COUNT, "{channels:?}: f32 frame count");

        let f16_flat = f16_outputs.concat();
        let f32_flat = f32_outputs.concat();
        let clean_flat = clean.repeat(FRAME_COUNT);

        let search_gap = mean_abs_difference(&f16_flat, &f32_flat);
        let f16_error = rms_difference(&f16_flat, &clean_flat);
        let f32_error = rms_difference(&f32_flat, &clean_flat);
        let error_ratio = f16_error / f32_error;
        eprintln!(
            "{channels:?}: mean abs gap {search_gap:.3e}, rms error f16 {f16_error:.6e} f32 {f32_error:.6e} \
             (ratio {error_ratio:.5})"
        );

        assert!(
            search_gap < 1.0e-3,
            "{channels:?}: the f16 and f32 searches differ by {search_gap:.3e} on average"
        );
        assert!(
            f16_error <= f32_error * 1.02,
            "{channels:?}: the f16 search's error {f16_error:.6e} exceeds the f32 search's \
             {f32_error:.6e} by more than 2%"
        );
    }
}

/// Denoises `frames`, with the f16 search left as built when `f16_search` is set and forced off
/// otherwise.
fn denoise_clip(channels: ChannelMode, frames: &[Vec<f32>], f16_search: bool) -> Vec<Vec<f32>> {
    let client = make_client();
    let mut params = static_clip_params(RADIUS);
    params.nlm.channels = channels;

    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, WIDTH, HEIGHT).expect("construction failed");
    if !f16_search {
        denoiser.force_f32_search_for_test();
    }

    let mut outputs = Vec::new();
    for frame in frames {
        denoiser.push_frame(frame);
        if let Some(output) = denoiser.denoise().expect("denoise failed") {
            outputs.push(output);
        }
    }

    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    outputs
}

/// Interleaved frames of `base` in every channel, with independent noise per frame and channel.
fn static_noisy_frames(base: &[f32], channel_count: usize) -> Vec<Vec<f32>> {
    let mut frames = Vec::with_capacity(FRAME_COUNT);

    for index in 0..FRAME_COUNT {
        let mut frame = vec![0.0f32; base.len() * channel_count];

        for channel in 0..channel_count {
            let seed = (index * 3 + channel) as u32;
            let noisy = noisy_field_over(base, WIDTH, HEIGHT, SIGMA, seed);
            for (pixel, value) in noisy.iter().enumerate() {
                frame[pixel * channel_count + channel] = *value;
            }
        }

        frames.push(frame);
    }

    frames
}

fn interleave_copies(plane: &[f32], channel_count: usize) -> Vec<f32> {
    plane
        .iter()
        .flat_map(|value| std::iter::repeat_n(*value, channel_count))
        .collect()
}

fn mean_abs_difference(left: &[f32], right: &[f32]) -> f64 {
    let total: f64 = left
        .iter()
        .zip(right)
        .map(|(&left_value, &right_value)| (left_value as f64 - right_value as f64).abs())
        .sum();
    total / left.len() as f64
}

fn rms_difference(left: &[f32], right: &[f32]) -> f64 {
    let total: f64 = left
        .iter()
        .zip(right)
        .map(|(&left_value, &right_value)| (left_value as f64 - right_value as f64).powi(2))
        .sum();
    let mean = total / left.len() as f64;
    mean.sqrt()
}
