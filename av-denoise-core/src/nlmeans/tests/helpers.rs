use cubecl::prelude::*;
use cubecl::server::Handle;

pub(super) use crate::engine::align::StorageAlign;
use crate::engine::{DevicePlane, IngestTarget, SampleFormat, ingest};
#[cfg(feature = "vulkan")]
use crate::nlmeans::{
    ChannelMode,
    DenoisingMode,
    NlmDenoiser,
    NlmeansAlgorithm,
    NlmeansOptions,
    resolve_params,
};
use crate::noise::tests::read_temporal_stats_slot;
use crate::noise::{
    NOISE_CHANNELS,
    QuarterSettings,
    TemporalStatsCtx,
    run_temporal_noise_stats,
    temporal_noise_reading,
    temporal_stats_buf_bytes,
};
pub(crate) use crate::tune::tests::fixtures::{noisy_copy, noisy_field_over, seeded_unit_gaussian};

/// The sigma of the texture [panning_frames] slides.
const PAN_TEXTURE_SIGMA: f32 = 24.0 / 255.0;

pub(crate) fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

/// Buffer-binding alignment the test runtime reports, which `NlmDenoiser` lays its slots out against.
pub(super) fn test_align() -> StorageAlign {
    let client = make_client();
    StorageAlign::from_client(&client)
}

pub(super) fn make_uniform_frame(width: u32, height: u32, channels: u32, value: f32) -> Vec<f32> {
    vec![value; (width * height * channels) as usize]
}

/// Creates a frame with a square of noise so NLMeans has matching noisy patches to work with.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
pub(super) fn make_frame_with_noisy_region(
    width: u32,
    height: u32,
    channels: u32,
    base: f32,
    centre_x: u32,
    centre_y: u32,
    radius: u32,
    noise_value: f32,
) -> Vec<f32> {
    let mut frame = vec![base; (width * height * channels) as usize];

    for dy in 0..=radius * 2 {
        for dx in 0..=radius * 2 {
            let x = centre_x + dx - radius;
            let y = centre_y + dy - radius;

            if x < width && y < height {
                for channel in 0..channels {
                    frame[((y * width + x) * channels + channel) as usize] = noise_value;
                }
            }
        }
    }

    frame
}

/// Flat base value plus deterministic pseudo-Gaussian noise, densely packed as `pixels * channels`.
///
/// Each sample sums four hash-derived uniforms between -0.5 and 0.5 (Irwin-Hall) and rescales to
/// the per-channel standard deviation. `sigmas` is indexed per channel and wraps if shorter than
/// `channels`.
pub(super) fn make_noisy_gaussian_frame(
    width: u32,
    height: u32,
    channels: u32,
    base: f32,
    sigmas: &[f32],
) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height * channels) as usize];
    // Sum of 4 independent Uniform(-0.5, 0.5) samples has variance 4/12 = 1/3.
    let unit_std = (1.0f32 / 3.0f32).sqrt();

    for idx in 0..(width * height * channels) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }

        let channel = (idx % channels) as usize;
        let sigma = sigmas[channel % sigmas.len()];
        frame[idx as usize] = (base + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }

    frame
}

/// Builds a frame of grain correlated between horizontal neighbours by a `[0.25, 0.5, 0.25]` blur.
///
/// The blur gives a lag-1 correlation of two thirds for any input distribution and scales the
/// variance by 0.375, the sum of the taps squared.
pub(super) fn correlated_noisy_frame(
    width: u32,
    height: u32,
    base: f32,
    sigma_pre: f32,
    seed: u32,
) -> Vec<f32> {
    correlated_noisy_frame_with_tap(width, height, base, sigma_pre, seed, 0.25)
}

/// Builds grain correlated by a horizontal `[tap, 1 - 2 * tap, tap]` blur, clamped at the edges.
///
/// For taps `(tap, 1 - 2 * tap, tap)` over unit-variance white noise, the lag-1 correlation along x
/// is `2 * tap * (1 - 2 * tap) / (2 * tap^2 + (1 - 2 * tap)^2)`. A tap of `0.125` gives a
/// correlation of about `0.316`.
pub(super) fn correlated_noisy_frame_with_tap(
    width: u32,
    height: u32,
    base: f32,
    sigma_pre: f32,
    seed: u32,
    tap: f32,
) -> Vec<f32> {
    let centre_tap = 1.0 - 2.0 * tap;

    let mut raw = vec![0.0f32; (width * height) as usize];
    for idx in 0..(width * height) {
        let sample = seeded_unit_gaussian(idx, seed);
        raw[idx as usize] = sample * sigma_pre;
    }

    let mut out = vec![0.0f32; raw.len()];
    for y in 0..height {
        for x in 0..width {
            let left_x = x.saturating_sub(1);
            let right_x = (x + 1).min(width - 1);
            let left = raw[(y * width + left_x) as usize];
            let centre = raw[(y * width + x) as usize];
            let right = raw[(y * width + right_x) as usize];
            let blurred = tap * left + centre_tap * centre + tap * right;
            out[(y * width + x) as usize] = (base + blurred).clamp(0.0, 1.0);
        }
    }

    out
}

/// A deterministic frame with spatial structure at more than one scale.
///
/// Two out-of-phase sine waves plus a finer third one give NLMeans patches with varying content, so
/// the weights spread the way they do on real footage instead of all being maximal.
pub(super) fn make_textured_frame(width: u32, height: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let norm_x = x as f32 / width as f32;
            let norm_y = y as f32 / height as f32;
            let value = 0.5
                + 0.2
                    * (norm_x * 8.0 * std::f32::consts::PI).sin()
                    * (norm_y * 6.0 * std::f32::consts::PI).cos()
                + 0.1 * (norm_x * 20.0 * std::f32::consts::PI).sin();
            frame[(y * width + x) as usize] = value.clamp(0.05, 0.95);
        }
    }

    frame
}

/// Horizontal luma gradient from `low` to `high` inclusive, repeated down every row.
pub(super) fn make_gradient_frame(width: u32, height: u32, low: f32, high: f32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let fraction = x as f32 / (width - 1).max(1) as f32;
            frame[(y * width + x) as usize] = low + (high - low) * fraction;
        }
    }

    frame
}

/// Builds a Luma [NlmDenoiser] at the given temporal radius with default options.
#[cfg(feature = "vulkan")]
pub(super) fn test_denoiser(radius: u32, width: u32, height: u32) -> NlmDenoiser {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius },
        ..NlmeansOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Fast(options);
    let params = resolve_params(&algorithm, ChannelMode::Luma);
    let client = make_client();

    NlmDenoiser::new(&client, params, width, height)
}

/// A frame that ramps from `0.2` to `0.8` and shifts a little with `frame_index`.
///
/// Each `frame_index` gives distinct content.
#[cfg(feature = "vulkan")]
pub(super) fn ramp_frame(width: u32, height: u32, frame_index: usize) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let fraction = (x as f32 + y as f32 * width as f32) / (width * height) as f32;
            frame[(y * width + x) as usize] =
                (0.2 + 0.6 * fraction + frame_index as f32 * 0.01).clamp(0.0, 1.0);
        }
    }

    frame
}

/// Splits an interleaved host frame into one uploaded plane per channel.
pub(crate) fn upload_planes(client: &Client, frame: &[f32], channels: usize) -> Vec<Handle> {
    (0..channels)
        .map(|channel| {
            let plane: Vec<f32> = frame.iter().skip(channel).step_by(channels).copied().collect();
            let bytes = f32::as_bytes(&plane);
            client.create_from_slice(bytes)
        })
        .collect()
}

pub(crate) fn read_interleaved(client: &Client, planes: &[Handle]) -> Vec<f32> {
    let channels: Vec<Vec<f32>> = planes
        .iter()
        .map(|handle| {
            let bytes = client.read_one(handle.clone()).expect("read plane");
            f32::from_bytes(&bytes).to_vec()
        })
        .collect();

    let pixels = channels[0].len();
    let mut interleaved = Vec::with_capacity(pixels * channels.len());
    for pixel in 0..pixels {
        for channel in &channels {
            interleaved.push(channel[pixel]);
        }
    }

    interleaved
}

/// The f32 samples the real ingest kernel produces for one plane of `u8` codes.
pub(crate) fn normalise_with_ingest(client: &Client, codes: &[u8], width: u32, height: u32) -> Vec<f32> {
    let pixels = width * height;
    let input = client.create_from_slice(codes);
    let planes = [DevicePlane::new(&input, width, height)];
    let placeholder = client.create_from_slice(&[0u8; 4]);
    let scratch = client.empty(pixels as usize * 4);
    let target = IngestTarget {
        ring: &scratch,
        ring_len: pixels as usize,
        offset: 0,
        pixels,
        channels: 1,
    };

    ingest(client, &planes, SampleFormat::U8, &placeholder, target);

    let bytes = client.read_one(scratch).expect("read ingest");
    f32::from_bytes(&bytes).to_vec()
}

/// `count` frames of one seeded fine texture sliding `shift` pixels per frame, each with fresh
/// white noise of `sigma`.
///
/// Pixel x of frame t shows texture column `x + shift * t`, so each frame is its predecessor moved
/// `shift` pixels to the left.
pub(crate) fn panning_frames(
    width: u32,
    height: u32,
    shift: u32,
    count: usize,
    sigma: f32,
    seed: u32,
) -> Vec<Vec<f32>> {
    let texture_width = width + shift * count as u32;
    // The blur scales the variance by 0.375, as [correlated_noisy_frame] documents.
    let texture_sigma_pre = PAN_TEXTURE_SIGMA / 0.375f32.sqrt();
    let texture = correlated_noisy_frame(texture_width, height, 0.5, texture_sigma_pre, seed);

    let mut frames = Vec::with_capacity(count);
    for index in 0..count {
        let start_x = shift * index as u32;
        let mut clean = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let row_start = (row * texture_width + start_x) as usize;
            let row_end = row_start + width as usize;
            clean.extend_from_slice(&texture[row_start..row_end]);
        }

        let noise_seed = seed * 1000 + index as u32;
        let frame = noisy_field_over(&clean, width, height, sigma, noise_seed);
        frames.push(frame);
    }

    frames
}

/// Interleaves equal-length planes into one frame, pixel by pixel.
pub(crate) fn interleave_planes(planes: &[&[f32]]) -> Vec<f32> {
    let pixels = planes[0].len();
    let mut frame = Vec::with_capacity(pixels * planes.len());
    for pixel in 0..pixels {
        for plane in planes {
            frame.push(plane[pixel]);
        }
    }

    frame
}

/// The host reading's sigma over the uncompensated records of `new` against `previous`.
///
/// Both frames are interleaved with `channels` lanes per pixel. It is `None` when the reading has
/// no sample.
pub(crate) fn raw_reading_sigma(
    previous: &[f32],
    new: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Option<[f32; NOISE_CHANNELS as usize]> {
    let client = make_client();
    let frame_count = 2u32;
    let align = StorageAlign::from_client(&client);

    let mut ring = previous.to_vec();
    ring.extend_from_slice(new);
    let ring_bytes = f32::as_bytes(&ring);
    let input_buf = client.create_from_slice(ring_bytes);

    let stats_bytes = temporal_stats_buf_bytes(width, height, channels, frame_count, align);
    let stats_buf = client.empty(stats_bytes);

    let ctx = TemporalStatsCtx {
        width,
        height,
        channels,
        frame_count,
        slot_new: 1,
        slot_prev: 0,
        input_buf: &input_buf,
        stats_buf: &stats_buf,
        align,
        motion: None,
    };
    run_temporal_noise_stats(&client, &ctx, false).expect("temporal noise stats dispatch failed");

    let records = read_temporal_stats_slot(
        &client,
        &stats_buf,
        width,
        height,
        channels,
        frame_count,
        1,
        align,
    )
    .expect("the records read back");
    let settings = QuarterSettings::default();
    let reading = temporal_noise_reading(&records, channels, width, height, false, settings);

    reading.sample.map(|sample| sample.sigma)
}
