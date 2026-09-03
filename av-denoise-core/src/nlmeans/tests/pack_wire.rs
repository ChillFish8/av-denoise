use cubecl::prelude::*;

use super::helpers::{R, make_client};
#[cfg(feature = "vulkan")]
use super::helpers::{ramp_frame, test_denoiser};
use crate::Depth;
use crate::nlmeans::kernels::gpu_pack_wire;
#[cfg(feature = "vulkan")]
use crate::{
    ChannelMode,
    Denoiser,
    DenoiserOptions,
    DenoisingMode,
    OutputFormat,
    accelerate::Accelerator,
    device::Device,
};

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

/// Builds a top-level [`Denoiser`] at temporal radius 1 over `mode`,
/// collecting frames in `format`.
#[cfg(feature = "vulkan")]
fn denoiser(mode: ChannelMode, format: OutputFormat, w: u32, h: u32) -> Denoiser {
    let opts = DenoiserOptions::builder()
        .channel_mode(mode)
        .mode(DenoisingMode::Temporal { radius: 1 })
        .output_format(format)
        .build();

    Denoiser::create(&[Accelerator::Vulkan], &Device::Default, w, h, opts)
        .expect("denoiser construction failed")
}

/// Pushes `count` deterministic frames into `d`, one per index.
#[cfg(feature = "vulkan")]
fn push_ramp(d: &mut Denoiser, w: u32, h: u32, count: usize) {
    for i in 0..count {
        d.push_frame(&ramp_frame(w, h, i)).expect("push failed");
    }
}

/// The differential test the pack kernel lives or dies on. A kernel that
/// silently compiled to nothing returns zeros, which `f32_to_plane` of a
/// real frame never does.
#[cfg(feature = "vulkan")]
#[test]
fn wire_mode_output_matches_the_f32_path() {
    let (w, h) = (16u32, 16u32);

    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let mut f32_side = test_denoiser(1, w, h);
        let mut wire_side = denoiser(ChannelMode::Luma, OutputFormat::Wire { depth }, w, h);

        push_ramp(&mut f32_side, w, h, 3);
        push_ramp(&mut wire_side, w, h, 3);

        let want = f32_side
            .recv_frame()
            .expect("f32 recv failed")
            .expect("a frame is ready")
            .into_f32()
            .expect("an f32 denoiser returns f32");

        let got = wire_side
            .recv_frame()
            .expect("wire recv failed")
            .expect("a frame is ready")
            .into_wire()
            .expect("a wire denoiser returns wire bytes");

        assert_eq!(got, crate::frame::f32_to_plane(&want, depth), "depth {depth:?}");
    }
}

/// A chroma pair goes out as U's whole region followed by V's, not
/// interleaved, so it matches what a planar consumer writes.
#[cfg(feature = "vulkan")]
#[test]
fn wire_mode_chroma_matches_unpack_uv_from_f32() {
    let (w, h) = (16u32, 16u32);
    let depth = Depth::Ten;

    let mut f32_side = denoiser(ChannelMode::Chroma, OutputFormat::F32, w, h);
    let mut wire_side = denoiser(ChannelMode::Chroma, OutputFormat::Wire { depth }, w, h);

    // A chroma frame holds two channels per pixel, which `ramp_frame`
    // covers by producing twice as many values.
    push_ramp(&mut f32_side, w, h * 2, 3);
    push_ramp(&mut wire_side, w, h * 2, 3);

    let want = f32_side
        .recv_frame()
        .expect("f32 recv failed")
        .expect("a frame is ready")
        .into_f32()
        .expect("an f32 denoiser returns f32");

    let got = wire_side
        .recv_frame()
        .expect("wire recv failed")
        .expect("a frame is ready")
        .into_wire()
        .expect("a wire denoiser returns wire bytes");

    let (u, v) = crate::frame::unpack_uv_from_f32(&want, (w * h) as usize, depth);
    let expected: Vec<u8> = u.into_iter().chain(v).collect();

    assert_eq!(got, expected);
}

/// 13x3 luma is 39 samples, which is nine 8-bit words plus three bytes,
/// so the kernel's last word is a partial one.
#[cfg(feature = "vulkan")]
#[test]
fn wire_mode_handles_a_plane_that_is_not_a_whole_number_of_words() {
    let (w, h) = (13u32, 3u32);
    let depth = Depth::Eight;

    let mut f32_side = test_denoiser(1, w, h);
    let mut wire_side = denoiser(ChannelMode::Luma, OutputFormat::Wire { depth }, w, h);

    push_ramp(&mut f32_side, w, h, 3);
    push_ramp(&mut wire_side, w, h, 3);

    let want = f32_side
        .recv_frame()
        .expect("f32 recv failed")
        .expect("a frame is ready")
        .into_f32()
        .expect("an f32 denoiser returns f32");

    let got = wire_side
        .recv_frame()
        .expect("wire recv failed")
        .expect("a frame is ready")
        .into_wire()
        .expect("a wire denoiser returns wire bytes");

    assert_eq!(got.len(), 39);
    assert_eq!(got, crate::frame::f32_to_plane(&want, depth));
}

/// A wire buffer handed out while its own readback is still reading it
/// would let one frame's bytes appear in another's.
///
/// Both pushes land before either drain, so two readbacks are in flight
/// at once and the second occupies the slot the first has not released.
/// Draining after every push would leave one readback live at a time,
/// where a wrong slot index cannot be observed at all.
#[cfg(feature = "vulkan")]
#[test]
fn reusing_the_wire_slots_keeps_every_frame_distinct() {
    let (w, h) = (16u32, 16u32);
    let depth = Depth::Eight;

    let mut f32_side = denoiser(ChannelMode::Luma, OutputFormat::F32, w, h);
    let mut wire_side = denoiser(ChannelMode::Luma, OutputFormat::Wire { depth }, w, h);

    let mut got = Vec::new();
    let mut want = Vec::new();

    for pair in 0..3 {
        for k in 0..2 {
            let frame = ramp_frame(w, h, pair * 2 + k);
            f32_side.push_frame(&frame).expect("f32 push failed");
            wire_side.push_frame(&frame).expect("wire push failed");
        }

        while let Some(out) = f32_side.recv_frame().expect("f32 recv failed") {
            want.push(crate::frame::f32_to_plane(
                &out.into_f32().expect("f32 denoiser returns f32"),
                depth,
            ));
        }

        while let Some(out) = wire_side.recv_frame().expect("wire recv failed") {
            got.push(out.into_wire().expect("wire denoiser returns wire bytes"));
        }
    }

    assert_eq!(got.len(), 5, "six pushes at radius 1 emit five frames");
    assert_eq!(got, want);
}

#[cfg(feature = "vulkan")]
#[test]
fn an_f32_denoiser_allocates_no_wire_buffers() {
    let d = denoiser(ChannelMode::Luma, OutputFormat::F32, 16, 16);
    assert!(d.wire_outputs_for_test().is_none());
}

#[cfg(feature = "vulkan")]
#[test]
fn try_recv_frame_in_wire_mode_returns_none_when_nothing_is_in_flight() {
    let mut d = denoiser(
        ChannelMode::Luma,
        OutputFormat::Wire { depth: Depth::Eight },
        64,
        64,
    );
    assert_eq!(d.try_recv_frame().unwrap(), None);
}

/// Wire mode must not quietly become blocking, and the bytes it polls
/// out must be the ones the blocking path returns.
#[cfg(feature = "vulkan")]
#[test]
fn try_wait_still_reports_not_ready_without_blocking_in_wire_mode() {
    // A poll count is the wrong proxy for the wall-clock interval this
    // test needs to cover (cold pipeline compile plus dispatch plus
    // readback), since a faster CPU makes each poll cheaper and so
    // needs *more* of them for the same GPU latency. A deadline covers
    // both a slow GPU and a fast CPU the same way.
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

    let (w, h) = (64u32, 64u32);
    let format = OutputFormat::Wire { depth: Depth::Eight };

    // Two pushes at radius 1 prime the window and submit one denoise,
    // leaving exactly one readback in flight.
    let mut polled = denoiser(ChannelMode::Luma, format, w, h);
    push_ramp(&mut polled, w, h, 2);

    let start = std::time::Instant::now();
    let mut got = None;
    let mut polls = 0;
    while start.elapsed() < DEADLINE {
        polls += 1;
        if let Some(frame) = polled.try_recv_frame().unwrap() {
            got = Some(frame);
            break;
        }
    }
    let got = got.unwrap_or_else(|| panic!("readback never landed within {DEADLINE:?} ({polls} polls)"));

    let mut blocking = denoiser(ChannelMode::Luma, format, w, h);
    push_ramp(&mut blocking, w, h, 2);
    let expected = blocking
        .recv_frame()
        .unwrap()
        .expect("blocking denoiser should have a frame ready");

    assert!(matches!(got, crate::FrameOutput::Wire(_)));
    assert_eq!(got, expected);
}
