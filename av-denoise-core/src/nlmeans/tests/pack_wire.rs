use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::Depth;
use crate::nlmeans::kernels::gpu_pack_wire;

const BLOCK: u32 = 256;

/// Runs the kernel and returns the wire bytes it produced, trimmed to the
/// plane's exact byte length.
fn pack(
    src: &[f32],
    pixels: u32,
    channels: u32,
    stored_ch: u32,
    depth: Depth,
    split_planes: bool,
) -> Vec<u8> {
    let samples_per_word = 4 / depth.bytes_per_sample() as u32;
    let words = (pixels * channels).div_ceil(samples_per_word);
    let grid = words.div_ceil(BLOCK).max(1);

    pack_with_grid(src, pixels, channels, stored_ch, depth, split_planes, grid, BLOCK)
}

/// The same, with the launch geometry chosen by the caller so a test can
/// force a grid smaller than the word count.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the kernel's own argument list, plus the launch geometry"
)]
fn pack_with_grid(
    src: &[f32],
    pixels: u32,
    channels: u32,
    stored_ch: u32,
    depth: Depth,
    split_planes: bool,
    grid: u32,
    block: u32,
) -> Vec<u8> {
    let client = make_client();

    let samples = pixels * channels;
    let bytes = samples as usize * depth.bytes_per_sample();
    let samples_per_word = 4 / depth.bytes_per_sample() as u32;
    let words = samples.div_ceil(samples_per_word);
    let outer = if split_planes { pixels } else { channels };

    let total_threads = grid * block;

    let src_buf = client.create_from_slice(f32::as_bytes(src));
    let dst_buf = client.empty(words as usize * size_of::<u32>());

    unsafe {
        gpu_pack_wire::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(block),
            ArrayArg::from_raw_parts(src_buf, src.len()),
            ArrayArg::from_raw_parts(dst_buf.clone(), words as usize),
            depth.max_value(),
            pixels,
            channels,
            stored_ch,
            outer,
            split_planes,
            samples_per_word,
            words,
            total_threads,
        );
    }

    let out = client.read_one(dst_buf).expect("pack readback failed");
    out[..bytes].to_vec()
}

/// The only check that can catch a silently-dead kernel, since a kernel
/// compared against itself compares zeros to zeros.
#[test]
fn packed_bytes_match_the_host_converter_at_every_depth() {
    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let pixels = 64u32;
        // A ramp over the full range, including both ends and values that
        // land either side of a rounding boundary.
        let src: Vec<f32> = (0..pixels).map(|i| i as f32 / (pixels - 1) as f32).collect();

        let got = pack(&src, pixels, 1, 1, depth, false);
        let want = crate::frame::f32_to_plane(&src, depth);

        assert_eq!(
            got, want,
            "depth {depth:?} must match the host converter byte for byte"
        );
    }
}

#[test]
fn padding_lanes_are_skipped() {
    let pixels = 16u32;
    let channels = 3u32;
    let stored_ch = 4u32;

    // Every padding lane holds 1.0, which would be visible as 0xFF if it
    // ever reached the output.
    let mut src = vec![1.0f32; (pixels * stored_ch) as usize];
    let wanted: Vec<f32> = (0..pixels * channels).map(|i| i as f32 / 255.0).collect();
    for p in 0..pixels as usize {
        for c in 0..channels as usize {
            src[p * stored_ch as usize + c] = wanted[p * channels as usize + c];
        }
    }

    let got = pack(&src, pixels, channels, stored_ch, Depth::Eight, false);
    let want = crate::frame::f32_to_plane(&wanted, Depth::Eight);

    assert_eq!(got, want);
}

#[test]
fn a_sample_count_that_is_not_a_whole_number_of_words_writes_its_tail() {
    // 13 samples at 8-bit is three whole words plus one byte.
    let pixels = 13u32;
    let src: Vec<f32> = (0..pixels).map(|i| i as f32 / (pixels - 1) as f32).collect();

    let got = pack(&src, pixels, 1, 1, Depth::Eight, false);
    let want = crate::frame::f32_to_plane(&src, Depth::Eight);

    assert_eq!(got.len(), 13);
    assert_eq!(got, want);
}

#[test]
fn split_planes_writes_each_channel_as_one_region() {
    let pixels = 32u32;
    let channels = 2u32;

    let src: Vec<f32> = (0..pixels * channels).map(|i| i as f32 / 255.0).collect();

    let got = pack(&src, pixels, channels, channels, Depth::Eight, true);

    let (u, v) = crate::frame::unpack_uv_from_f32(&src, pixels as usize, Depth::Eight);
    let want: Vec<u8> = u.into_iter().chain(v).collect();

    assert_eq!(got, want);
}

/// Forces a grid far smaller than the word count so every thread runs the
/// strided loop several times. Without this the loop body runs at most
/// once and `word += total_threads` is never exercised.
#[test]
fn the_strided_loop_covers_every_word_when_the_grid_is_smaller_than_the_frame() {
    let pixels = 512u32;
    let src: Vec<f32> = (0..pixels).map(|i| (i % 256) as f32 / 255.0).collect();

    // 128 words against 64 threads, so each thread walks two words.
    let got = pack_with_grid(&src, pixels, 1, 1, Depth::Eight, false, 1, 64);
    let want = crate::frame::f32_to_plane(&src, Depth::Eight);

    assert_eq!(got, want);
}

#[test]
fn split_planes_at_ten_bit_writes_each_channel_as_one_region() {
    let pixels = 32u32;
    let channels = 2u32;

    let src: Vec<f32> = (0..pixels * channels)
        .map(|i| i as f32 / (pixels * channels - 1) as f32)
        .collect();

    let got = pack(&src, pixels, channels, channels, Depth::Ten, true);

    let (u, v) = crate::frame::unpack_uv_from_f32(&src, pixels as usize, Depth::Ten);
    let want: Vec<u8> = u.into_iter().chain(v).collect();

    assert_eq!(got, want);
}

#[test]
fn padding_lanes_are_skipped_at_ten_bit() {
    let pixels = 16u32;
    let channels = 3u32;
    let stored_ch = 4u32;

    let mut src = vec![1.0f32; (pixels * stored_ch) as usize];
    let wanted: Vec<f32> = (0..pixels * channels)
        .map(|i| i as f32 / (pixels * channels - 1) as f32)
        .collect();
    for p in 0..pixels as usize {
        for c in 0..channels as usize {
            src[p * stored_ch as usize + c] = wanted[p * channels as usize + c];
        }
    }

    let got = pack(&src, pixels, channels, stored_ch, Depth::Ten, false);
    let want = crate::frame::f32_to_plane(&wanted, Depth::Ten);

    assert_eq!(got, want);
}
