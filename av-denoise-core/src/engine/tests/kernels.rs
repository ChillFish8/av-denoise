use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::engine::{DevicePlane, EgressSource, IngestTarget, SampleFormat, egress, ingest};
use crate::nlmeans::kernels::gpu_pack_wire;
use crate::nlmeans::{BLOCK_1D, Depth, MAX_GRID_1D};

type R = WgpuRuntime;

fn client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
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

/// The GPU divide is not correctly rounded, so a sample can differ from the host by a few units in the last place.
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

fn run_ingest(
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    format: SampleFormat,
) -> (Vec<f32>, Vec<f32>) {
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
    let frame_len = pixels * stored_ch as usize;
    let ring_values = vec![-1.0f32; frame_len * 2];
    let ring = client.create_from_slice(f32::as_bytes(&ring_values));
    let placeholder = client.empty(4);
    let target = IngestTarget {
        ring: &ring,
        ring_len: frame_len * 2,
        offset: frame_len as u32,
        pixels: pixels as u32,
        channels,
        stored_ch,
    };
    ingest(&client, &planes, format, &placeholder, target);

    let bytes = client.read_one(ring).expect("read ring");
    let actual = f32::from_bytes(&bytes)[frame_len..].to_vec();

    let mut expected = vec![0.0f32; frame_len];
    for pixel in 0..pixels {
        for channel in 0..channels as usize {
            let code = channel_codes[channel][pixel];
            expected[pixel * stored_ch as usize + channel] = code as f32 / max;
        }
    }

    (actual, expected)
}

#[test]
fn ingest_u8_luma_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(5, 3, 1, 1, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u8_odd_sized_chroma_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(3, 3, 2, 2, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_yuv_pads_the_fourth_lane_with_zero() {
    let (actual, expected) = run_ingest(7, 5, 3, 4, SampleFormat::U16 { depth: 10 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_at_depth_16_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(4, 4, 1, 1, SampleFormat::U16 { depth: 16 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u8_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 3, 4, SampleFormat::U8);
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 2, 2, SampleFormat::U16 { depth: 12 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_u16_at_depth_16_across_many_cubes_matches_the_host_oracle() {
    let (actual, expected) = run_ingest(300, 3, 1, 1, SampleFormat::U16 { depth: 16 });
    assert_close(&actual, &expected);
}

#[test]
fn ingest_f32_copies_samples_across_many_cubes() {
    run_f32_ingest(300, 3);
}

#[test]
fn ingest_f32_copies_samples_and_zeroes_padding() {
    run_f32_ingest(4, 3);
}

fn run_f32_ingest(width: u32, height: u32) {
    let client = client();
    let pixels = (width * height) as usize;
    let channel_values: Vec<Vec<f32>> = (0..3)
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
        .map(|values| client.create_from_slice(f32::as_bytes(values)))
        .collect();
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();
    let ring_values = vec![-1.0f32; pixels * 4];
    let ring = client.create_from_slice(f32::as_bytes(&ring_values));
    let placeholder = client.empty(4);
    let target = IngestTarget {
        ring: &ring,
        ring_len: pixels * 4,
        offset: 0,
        pixels: pixels as u32,
        channels: 3,
        stored_ch: 4,
    };

    ingest(&client, &planes, SampleFormat::F32, &placeholder, target);

    let bytes = client.read_one(ring).expect("read ring");
    let actual = f32::from_bytes(&bytes);

    for pixel in 0..pixels {
        for channel in 0..3 {
            assert_eq!(actual[pixel * 4 + channel], channel_values[channel][pixel]);
        }

        assert_eq!(actual[pixel * 4 + 3], 0.0);
    }
}

/// An interleaved frame whose first pixels hit below zero, zero, one, above one and the top two codes.
///
/// The padding lane holds a value no plane should ever receive.
fn egress_frame(pixels: usize, channels: u32, stored_ch: u32, format: SampleFormat) -> Vec<f32> {
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

    let mut frame = vec![0.0f32; pixels * stored_ch as usize];
    for pixel in 0..pixels {
        for lane in 0..stored_ch as usize {
            let index = pixel * stored_ch as usize + lane;
            let value = if lane >= channels as usize {
                7.0
            } else if pixel < extremes.len() {
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

/// The host quantisation, which is `gpu_pack_wire`'s exact arithmetic.
fn quantise_planes(frame: &[f32], channels: u32, stored_ch: u32, format: SampleFormat) -> Vec<Vec<u8>> {
    let pixels = frame.len() / stored_ch as usize;
    let max = format.max_value();

    (0..channels as usize)
        .map(|channel| {
            let mut plane = Vec::new();
            for pixel in 0..pixels {
                let value = frame[pixel * stored_ch as usize + channel].clamp(0.0, 1.0);
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
fn run_egress(
    frame: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    format: SampleFormat,
) -> Vec<Vec<u8>> {
    let client = client();
    let pixels = (width * height) as usize;
    let frame_handle = client.create_from_slice(f32::as_bytes(frame));
    let plane_bytes = format.plane_bytes(pixels as u64) as usize;
    let outputs: Vec<_> = (0..channels)
        .map(|_| client.create_from_slice(&vec![0xAAu8; plane_bytes]))
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
        stored_ch,
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

fn assert_egress_matches_oracle(
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    format: SampleFormat,
) {
    let pixels = (width * height) as usize;
    let frame = egress_frame(pixels, channels, stored_ch, format);
    let actual = run_egress(&frame, width, height, channels, stored_ch, format);
    let expected = quantise_planes(&frame, channels, stored_ch, format);
    assert_eq!(actual, expected);
}

#[test]
fn egress_u8_luma_matches_the_pack_oracle() {
    assert_egress_matches_oracle(5, 3, 1, 1, SampleFormat::U8);
}

#[test]
fn egress_u8_odd_sized_chroma_matches_the_pack_oracle() {
    assert_egress_matches_oracle(3, 3, 2, 2, SampleFormat::U8);
}

#[test]
fn egress_u16_yuv_skips_the_padding_lane() {
    assert_egress_matches_oracle(7, 5, 3, 4, SampleFormat::U16 { depth: 12 });
}

#[test]
fn egress_u16_at_depth_16_hits_the_top_code() {
    assert_egress_matches_oracle(4, 4, 1, 1, SampleFormat::U16 { depth: 16 });
}

#[test]
fn egress_u8_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 3, 4, SampleFormat::U8);
}

#[test]
fn egress_u8_luma_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 1, 1, SampleFormat::U8);
}

#[test]
fn egress_u16_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 2, 2, SampleFormat::U16 { depth: 10 });
}

#[test]
fn egress_u16_at_depth_16_across_many_cubes_matches_the_pack_oracle() {
    assert_egress_matches_oracle(300, 9, 3, 4, SampleFormat::U16 { depth: 16 });
}

#[test]
fn egress_f32_writes_unclamped_samples() {
    assert_f32_egress_copies(3, 2, 2, 2);
}

#[test]
fn egress_f32_skips_the_padding_lane() {
    assert_f32_egress_copies(7, 5, 3, 4);
}

#[test]
fn egress_f32_across_many_cubes_copies_samples() {
    assert_f32_egress_copies(300, 9, 3, 4);
}

fn assert_f32_egress_copies(width: u32, height: u32, channels: u32, stored_ch: u32) {
    let client = client();
    let pixels = (width * height) as usize;
    let frame = egress_frame(pixels, channels, stored_ch, SampleFormat::F32);
    let frame_handle = client.create_from_slice(f32::as_bytes(&frame));
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
        stored_ch,
    };

    egress(&client, source, &planes, SampleFormat::F32, &placeholder);

    for (channel, handle) in outputs.into_iter().enumerate() {
        let bytes = client.read_one(handle).expect("read plane");
        let values = f32::from_bytes(&bytes);
        for pixel in 0..pixels {
            assert_eq!(values[pixel], frame[pixel * stored_ch as usize + channel]);
        }
    }
}

#[test]
fn egress_matches_gpu_pack_wire_for_chroma_bit_for_bit() {
    assert_matches_pack_wire(7, 5, 2, 2, SampleFormat::U8, Depth::Eight);
    assert_matches_pack_wire(7, 5, 2, 2, SampleFormat::U16 { depth: 10 }, Depth::Ten);
}

#[test]
fn egress_matches_gpu_pack_wire_for_chroma_across_many_cubes() {
    assert_matches_pack_wire(300, 9, 2, 2, SampleFormat::U8, Depth::Eight);
    assert_matches_pack_wire(300, 9, 2, 2, SampleFormat::U16 { depth: 12 }, Depth::Twelve);
}

#[test]
fn egress_matches_gpu_pack_wire_for_yuv_bit_for_bit() {
    assert_matches_pack_wire(7, 5, 3, 4, SampleFormat::U8, Depth::Eight);
    assert_matches_pack_wire(7, 5, 3, 4, SampleFormat::U16 { depth: 10 }, Depth::Ten);
}

#[test]
fn egress_matches_gpu_pack_wire_for_yuv_across_many_cubes() {
    assert_matches_pack_wire(300, 9, 3, 4, SampleFormat::U8, Depth::Eight);
    assert_matches_pack_wire(300, 9, 3, 4, SampleFormat::U16 { depth: 12 }, Depth::Twelve);
}

/// Launches `gpu_pack_wire` the way `pack_wire` does and splits its bytes back into planes.
fn run_pack_wire(frame: &[f32], pixels: usize, channels: u32, stored_ch: u32, depth: Depth) -> Vec<Vec<u8>> {
    let client = client();
    let pack = depth.wire_pack();
    let samples = pixels as u32 * channels;
    let words = samples.div_ceil(pack.samples_per_word());
    let split_planes = channels == 2;
    let outer = if split_planes { pixels as u32 } else { channels };
    let groups = words.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = groups * BLOCK_1D;
    let src = client.create_from_slice(f32::as_bytes(frame));
    let dst = client.empty(words as usize * 4);

    unsafe {
        gpu_pack_wire::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(groups),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(src, pixels * stored_ch as usize),
            ArrayArg::from_raw_parts(dst.clone(), words as usize),
            pack.max(),
            pixels as u32,
            channels,
            stored_ch,
            outer,
            split_planes,
            pack.samples_per_word(),
            words,
            total_threads,
        );
    }

    let bytes_each = depth.bytes_per_sample();
    let wire = client.read_one(dst).expect("read wire");
    let wire = &wire[..samples as usize * bytes_each];

    if split_planes {
        // Each plane is one contiguous half.
        return wire
            .chunks_exact(pixels * bytes_each)
            .map(<[u8]>::to_vec)
            .collect();
    }

    // A fused frame stays interleaved, one pixel after another.
    let mut planes = vec![Vec::new(); channels as usize];
    for pixel in wire.chunks_exact(channels as usize * bytes_each) {
        for (channel, sample) in pixel.chunks_exact(bytes_each).enumerate() {
            planes[channel].extend_from_slice(sample);
        }
    }

    planes
}

fn assert_matches_pack_wire(
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    format: SampleFormat,
    depth: Depth,
) {
    let pixels = (width * height) as usize;
    let frame = egress_frame(pixels, channels, stored_ch, format);
    let egressed = run_egress(&frame, width, height, channels, stored_ch, format);
    let packed = run_pack_wire(&frame, pixels, channels, stored_ch, depth);
    assert_eq!(egressed, packed);
}
