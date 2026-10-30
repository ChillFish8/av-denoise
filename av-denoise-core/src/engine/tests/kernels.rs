use cubecl::prelude::*;

use crate::engine::{DevicePlane, EgressSource, IngestTarget, SampleFormat, egress, ingest};

fn client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

/// Sample codes where the first three pixels are always 0, `max` and `max - 1`.
fn codes(pixels: usize, channel: u16, max: u16) -> Vec<u16> {
    let modulus = max as u32 + 1;
    let mut codes: Vec<u16> = (0..pixels)
        .map(|pixel| ((pixel as u32 * 37 + channel as u32 * 11) % modulus) as u16)
        .collect();

    let extremes = [0, max, max - 1];
    for (code, extreme) in codes.iter_mut().zip(extremes) {
        *code = extreme;
    }

    codes
}

fn encode(codes: &[u16], format: SampleFormat) -> Vec<u8> {
    let mut bytes: Vec<u8> = match format {
        SampleFormat::U8 => codes.iter().map(|&code| code as u8).collect(),
        SampleFormat::U16 { .. } => codes.iter().flat_map(|code| code.to_le_bytes()).collect(),
        SampleFormat::F32 => unreachable!("encode is only for word formats"),
    };
    let padded = bytes.len().div_ceil(4) * 4;
    bytes.resize(padded, 0);

    bytes
}

/// The GPU divide is not correctly rounded, so a sample can differ from the host by a few units in
/// the last place.
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = expected.abs() * f32::EPSILON * 4.0;
        let difference = (actual - expected).abs();
        assert!(
            difference <= tolerance,
            "sample {index}: got {actual}, expected {expected}"
        );
    }
}

fn run_ingest(width: u32, height: u32, channels: u32, format: SampleFormat) -> (Vec<f32>, Vec<f32>) {
    let client = client();
    let pixels = (width * height) as usize;
    let max = format.max_value();

    let channel_codes: Vec<Vec<u16>> = (0..channels as u16)
        .map(|channel| codes(pixels, channel, max as u16))
        .collect();
    let handles: Vec<_> = channel_codes
        .iter()
        .map(|codes| {
            let bytes = encode(codes, format);
            client.create_from_slice(&bytes)
        })
        .collect();
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();

    // Two slots, writing into the second, so the offset is exercised.
    let frame_len = pixels * channels as usize;
    let ring_values = vec![-1.0f32; frame_len * 2];
    let ring_bytes = f32::as_bytes(&ring_values);
    let ring = client.create_from_slice(ring_bytes);
    let placeholder = client.empty(4);
    let target = IngestTarget {
        ring: &ring,
        ring_len: frame_len * 2,
        offset: frame_len as u32,
        pixels: pixels as u32,
        channels,
    };

    ingest(&client, &planes, format, &placeholder, target);

    let bytes = client.read_one(ring).expect("read ring");
    let actual = f32::from_bytes(&bytes)[frame_len..].to_vec();

    let mut expected = vec![0.0f32; frame_len];
    for pixel in 0..pixels {
        for channel in 0..channels as usize {
            let code = channel_codes[channel][pixel];
            expected[pixel * channels as usize + channel] = code as f32 / max;
        }
    }

    (actual, expected)
}

#[test]
fn ingest_u8_luma_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(5, 3, 1, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u8_odd_sized_chroma_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(3, 3, 2, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_at_depth_16_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(4, 4, 1, SampleFormat::U16 { depth: 16 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u8_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 2, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 2, SampleFormat::U16 { depth: 12 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_at_depth_16_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 1, SampleFormat::U16 { depth: 16 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_f32_copies_samples_across_many_cubes() {
    run_f32_ingest(300, 3);
}

#[test]
fn ingest_f32_copies_samples() {
    run_f32_ingest(4, 3);
}

fn run_f32_ingest(width: u32, height: u32) {
    let client = client();
    let pixels = (width * height) as usize;
    let channel_values: Vec<Vec<f32>> = (0..2)
        .map(|channel| {
            let mut values: Vec<f32> = (0..pixels)
                .map(|pixel| (pixel + channel * 100) as f32 * 0.001)
                .collect();
            values[..3].copy_from_slice(&[0.0, 1.0, 1.0 - f32::EPSILON]);

            values
        })
        .collect();
    let handles: Vec<_> = channel_values
        .iter()
        .map(|values| {
            let bytes = f32::as_bytes(values);
            client.create_from_slice(bytes)
        })
        .collect();
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();
    let ring_values = vec![-1.0f32; pixels * 2];
    let ring_bytes = f32::as_bytes(&ring_values);
    let ring = client.create_from_slice(ring_bytes);
    let placeholder = client.empty(4);
    let target = IngestTarget {
        ring: &ring,
        ring_len: pixels * 2,
        offset: 0,
        pixels: pixels as u32,
        channels: 2,
    };

    ingest(&client, &planes, SampleFormat::F32, &placeholder, target);

    let bytes = client.read_one(ring).expect("read ring");
    let actual = f32::from_bytes(&bytes);

    for pixel in 0..pixels {
        for channel in 0..2 {
            assert_eq!(actual[pixel * 2 + channel], channel_values[channel][pixel]);
        }
    }
}

/// An interleaved frame whose first pixels hit below zero, zero, one, above one and the top two codes.
fn egress_frame(pixels: usize, channels: u32, format: SampleFormat) -> Vec<f32> {
    let max = format.max_value();
    let extremes = [
        -0.5,
        0.0,
        1.0,
        1.5,
        (max - 0.4) / max,
        (max - 1.0) / max,
        (max - 0.6) / max,
    ];

    let mut frame = vec![0.0f32; pixels * channels as usize];
    for pixel in 0..pixels {
        for lane in 0..channels as usize {
            let index = pixel * channels as usize + lane;
            let value = if pixel < extremes.len() {
                extremes[(pixel + lane) % extremes.len()]
            } else {
                ((index * 7919) % 1000) as f32 / 900.0 - 0.05
            };
            frame[index] = value;
        }
    }

    frame
}

fn bytes_per_sample(format: SampleFormat) -> usize {
    match format {
        SampleFormat::U8 => 1,
        _ => 2,
    }
}

/// The host quantisation, a clamp then a round half up.
fn quantise_planes(frame: &[f32], channels: u32, format: SampleFormat) -> Vec<Vec<u8>> {
    let pixels = frame.len() / channels as usize;
    let max = format.max_value();

    (0..channels as usize)
        .map(|channel| {
            let mut plane = Vec::new();
            for pixel in 0..pixels {
                let value = frame[pixel * channels as usize + channel].clamp(0.0, 1.0);
                let code = (value * max + 0.5) as u32;
                match format {
                    SampleFormat::U8 => plane.push(code as u8),
                    _ => plane.extend_from_slice(&(code as u16).to_le_bytes()),
                }
            }

            plane
        })
        .collect()
}

/// Runs `egress` and returns each plane's bytes up to its last pixel, ignoring the final word's padding.
fn run_egress(frame: &[f32], width: u32, height: u32, channels: u32, format: SampleFormat) -> Vec<Vec<u8>> {
    let client = client();
    let pixels = (width * height) as usize;
    let frame_bytes = f32::as_bytes(frame);
    let frame_handle = client.create_from_slice(frame_bytes);
    let plane_bytes = format.plane_bytes(pixels as u64) as usize;
    let outputs: Vec<_> = (0..channels)
        .map(|_| {
            let sentinel = vec![0xAAu8; plane_bytes];
            client.create_from_slice(&sentinel)
        })
        .collect();
    let planes: Vec<_> = outputs
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();
    let placeholder = client.empty(4);
    let source = EgressSource {
        frame: &frame_handle,
        pixels: pixels as u32,
        channels,
    };

    egress(&client, source, &planes, format, &placeholder);

    let length = pixels * bytes_per_sample(format);
    outputs
        .into_iter()
        .map(|handle| {
            let bytes = client.read_one(handle).expect("read plane");
            bytes[..length].to_vec()
        })
        .collect()
}

fn assert_egress_matches_oracle(width: u32, height: u32, channels: u32, format: SampleFormat) {
    let pixels = (width * height) as usize;
    let frame = egress_frame(pixels, channels, format);
    let actual = run_egress(&frame, width, height, channels, format);
    let expected = quantise_planes(&frame, channels, format);
    assert_eq!(actual, expected);
}

#[test]
fn egress_u8_luma_matches_the_pack_oracle() {
    assert_egress_matches_oracle(5, 3, 1, SampleFormat::U8);
}

#[test]
fn egress_u8_odd_sized_chroma_matches_the_pack_oracle() {
    assert_egress_matches_oracle(3, 3, 2, SampleFormat::U8);
}

#[test]
fn egress_u16_at_depth_16_hits_the_top_code() {
    assert_egress_matches_oracle(4, 4, 1, SampleFormat::U16 { depth: 16 });
}

#[test]
fn egress_u8_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 2, SampleFormat::U8);
}

#[test]
fn egress_u8_luma_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 1, SampleFormat::U8);
}

#[test]
fn egress_u16_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 2, SampleFormat::U16 { depth: 10 });
}

#[test]
fn egress_u16_at_depth_16_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 2, SampleFormat::U16 { depth: 16 });
}

#[test]
fn egress_f32_writes_unclamped_samples() {
    assert_f32_egress_copies(3, 2, 2);
}

#[test]
fn egress_f32_across_many_cubes_copies_samples() {
    assert_f32_egress_copies(300, 9, 2);
}

fn assert_f32_egress_copies(width: u32, height: u32, channels: u32) {
    let client = client();
    let pixels = (width * height) as usize;
    let frame = egress_frame(pixels, channels, SampleFormat::F32);
    let frame_bytes = f32::as_bytes(&frame);
    let frame_handle = client.create_from_slice(frame_bytes);
    let outputs: Vec<_> = (0..channels).map(|_| client.empty(pixels * 4)).collect();
    let planes: Vec<_> = outputs
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();
    let placeholder = client.empty(4);
    let source = EgressSource {
        frame: &frame_handle,
        pixels: pixels as u32,
        channels,
    };

    egress(&client, source, &planes, SampleFormat::F32, &placeholder);

    for (channel, handle) in outputs.into_iter().enumerate() {
        let bytes = client.read_one(handle).expect("read plane");
        let values = f32::from_bytes(&bytes);
        for pixel in 0..pixels {
            assert_eq!(values[pixel], frame[pixel * channels as usize + channel]);
        }
    }
}
