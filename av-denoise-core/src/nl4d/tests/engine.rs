use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::noisy_frames;
use crate::bench_api::HostIo;
use crate::engine::{DevicePlane, EdgePadding, Engine, Geometry, SampleFormat};
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;
use crate::nl4d::{Nl4d, Nl4dDenoiser, Nl4dOptions, resolve_params};
use crate::nlmeans::ChannelMode;
use crate::nlmeans::tests::helpers::{
    R,
    make_client,
    normalise_with_ingest,
    read_interleaved,
    upload_planes,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

fn geometry(channels: ChannelMode) -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        channels,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
    }
}

fn options(radius: u32, grain_export: bool) -> Nl4dOptions {
    Nl4dOptions {
        temporal_radius: radius,
        grain_export,
        ..Nl4dOptions::default()
    }
}

fn build_engine(client: &ComputeClient<R>, radius: u32, channels: ChannelMode) -> Nl4d<R> {
    let options = options(radius, false);
    let geometry = geometry(channels);

    Nl4d::new(client, options, geometry).expect("build")
}

fn emit(engine: &mut Nl4d<R>, client: &ComputeClient<R>, channel_count: usize) -> Vec<f32> {
    let pixels = (WIDTH * HEIGHT) as usize;
    let outputs: Vec<Handle> = (0..channel_count).map(|_| client.empty(pixels * 4)).collect();
    let planes: Vec<_> = outputs
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect();

    engine.emit_into(&planes).expect("emit");

    read_interleaved(client, &outputs)
}

/// Pushes one frame and returns how many frames the engine reports ready.
fn push_frame(engine: &mut Nl4d<R>, client: &ComputeClient<R>, frame: &[f32], channel_count: usize) -> usize {
    let handles = upload_planes(client, frame, channel_count);
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect();

    engine.push(&planes).expect("push")
}

fn push_context_frame(engine: &mut Nl4d<R>, client: &ComputeClient<R>, frame: &[f32], channel_count: usize) {
    let handles = upload_planes(client, frame, channel_count);
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect();

    engine.push_context(&planes).expect("push_context");
}

fn push_and_emit(
    engine: &mut Nl4d<R>,
    client: &ComputeClient<R>,
    frame: &[f32],
    channel_count: usize,
    outputs: &mut Vec<Vec<f32>>,
) {
    let ready = push_frame(engine, client, frame, channel_count);
    for _ in 0..ready {
        let output = emit(engine, client, channel_count);
        outputs.push(output);
    }
}

fn finish_and_emit(
    engine: &mut Nl4d<R>,
    client: &ComputeClient<R>,
    channel_count: usize,
    outputs: &mut Vec<Vec<f32>>,
) {
    let tail = engine.finish().expect("finish");
    for _ in 0..tail {
        let output = emit(engine, client, channel_count);
        outputs.push(output);
    }
}

/// Pushes every frame through `engine`, then the tail, returning each emitted frame.
fn drive(
    engine: &mut Nl4d<R>,
    client: &ComputeClient<R>,
    frames: &[Vec<f32>],
    channel_count: usize,
) -> Vec<Vec<f32>> {
    let mut outputs = Vec::new();

    for frame in frames {
        push_and_emit(engine, client, frame, channel_count, &mut outputs);
    }

    finish_and_emit(engine, client, channel_count, &mut outputs);

    outputs
}

/// Pushes each input, then the tail, emitting every frame as `u8` planes.
fn drive_u8(engine: &mut Nl4d<R>, client: &ComputeClient<R>, inputs: &[Handle]) -> Vec<Vec<u8>> {
    let pixels = (WIDTH * HEIGHT) as usize;
    let mut outputs = Vec::new();

    for input in inputs {
        let planes = [DevicePlane::new(input, WIDTH, HEIGHT)];
        let ready = engine.push(&planes).expect("push");
        emit_u8(engine, client, ready, pixels, &mut outputs);
    }

    let tail = engine.finish().expect("finish");
    emit_u8(engine, client, tail, pixels, &mut outputs);

    outputs
}

/// Emits `frame_count` frames as `u8` planes into `outputs`.
fn emit_u8(
    engine: &mut Nl4d<R>,
    client: &ComputeClient<R>,
    frame_count: usize,
    pixels: usize,
    outputs: &mut Vec<Vec<u8>>,
) {
    for _ in 0..frame_count {
        let output = client.empty(pixels);
        let planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];
        engine.emit_into(&planes).expect("emit");

        let bytes = client.read_one(output).expect("read");
        outputs.push(bytes.to_vec());
    }
}

struct Run {
    frames: Vec<Vec<f32>>,
    grain: Vec<GrainChunk>,
}

/// Runs the engine, pushing the first `context` frames with `push_context`.
fn run_engine(options: Nl4dOptions, channels: ChannelMode, frames: &[Vec<f32>], context: usize) -> Run {
    let client = make_client();
    let channel_count = channels.count() as usize;
    let geometry = geometry(channels);
    let mut engine = Nl4d::new(&client, options, geometry).expect("build");
    let mut outputs = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        if index < context {
            push_context_frame(&mut engine, &client, frame, channel_count);
            continue;
        }

        push_and_emit(&mut engine, &client, frame, channel_count, &mut outputs);
    }

    finish_and_emit(&mut engine, &client, channel_count, &mut outputs);

    // Drained through the trait object, the way the host layer reaches it.
    let dyn_engine: &mut dyn Engine = &mut engine;
    let grain = dyn_engine.drain_grain_chunks().expect("grain");

    Run {
        frames: outputs,
        grain,
    }
}

/// Runs the `Nl4dDenoiser` oracle, marking a continuation and skipping submits for the first
/// `context` frames.
fn run_oracle(options: Nl4dOptions, channels: ChannelMode, frames: &[Vec<f32>], context: usize) -> Run {
    let client = make_client();
    let params = resolve_params(&options, channels).expect("resolve");
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("build");
    let mut outputs = Vec::new();

    if context > 0 {
        denoiser.mark_continuation();
    }

    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);

        if index < context {
            continue;
        }

        let output = denoiser.denoise().expect("denoise");
        if let Some(samples) = output {
            outputs.push(samples);
        }
    }

    let collect = |output: &[f32]| {
        let samples = output.to_vec();
        outputs.push(samples);
    };
    denoiser.flush(collect).expect("flush");

    let grain = denoiser.drain_grain_chunks().expect("grain");

    Run {
        frames: outputs,
        grain,
    }
}

#[test]
fn luma_stream_matches_the_oracle_including_the_tail() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let actual_options = options(2, false);
    let actual = run_engine(actual_options, ChannelMode::Luma, &frames, 0);
    let expected_options = options(2, false);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(actual.frames.len(), 10);
    assert_eq!(actual.frames, expected.frames);
}

#[test]
fn chroma_stream_matches_the_oracle() {
    let frames = noisy_frames(WIDTH, HEIGHT, 2, 8);
    let actual_options = options(1, false);
    let actual = run_engine(actual_options, ChannelMode::Chroma, &frames, 0);
    let expected_options = options(1, false);
    let expected = run_oracle(expected_options, ChannelMode::Chroma, &frames, 0);
    assert_eq!(actual.frames.len(), 8);
    assert_eq!(actual.frames, expected.frames);
}

#[test]
fn yuv_stream_matches_the_oracle() {
    let frames = noisy_frames(WIDTH, HEIGHT, 3, 8);
    let actual_options = options(1, false);
    let actual = run_engine(actual_options, ChannelMode::Yuv, &frames, 0);
    let expected_options = options(1, false);
    let expected = run_oracle(expected_options, ChannelMode::Yuv, &frames, 0);
    assert_eq!(actual.frames, expected.frames);
}

#[test]
fn a_short_stream_matches_the_oracle() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 3);
    let actual_options = options(2, false);
    let actual = run_engine(actual_options, ChannelMode::Luma, &frames, 0);
    let expected_options = options(2, false);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(actual.frames.len(), 3);
    assert_eq!(actual.frames, expected.frames);
}

#[test]
fn a_context_led_stream_matches_the_oracle_continuation() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 12);
    let actual_options = options(2, false);
    let actual = run_engine(actual_options, ChannelMode::Luma, &frames, 4);
    let expected_options = options(2, false);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 4);
    assert_eq!(actual.frames, expected.frames);
}

#[test]
fn grain_chunks_match_the_oracle() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let actual_options = options(2, true);
    let actual = run_engine(actual_options, ChannelMode::Luma, &frames, 0);
    let expected_options = options(2, true);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 0);
    assert!(!actual.grain.is_empty());
    assert_eq!(actual.grain, expected.grain);
}

#[test]
fn a_second_stream_after_finish_matches_a_fresh_engine() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);

    drive(&mut engine, &client, &frames, 1);
    let second = drive(&mut engine, &client, &frames, 1);

    let fresh_options = options(2, false);
    let fresh = run_engine(fresh_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(second, fresh.frames);
}

#[test]
fn a_context_led_second_stream_matches_a_fresh_engine() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 12);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);

    drive(&mut engine, &client, &frames, 1);

    let mut second = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        if index < 4 {
            push_context_frame(&mut engine, &client, frame, 1);
            continue;
        }

        push_and_emit(&mut engine, &client, frame, 1, &mut second);
    }

    finish_and_emit(&mut engine, &client, 1, &mut second);

    let fresh_options = options(2, false);
    let fresh = run_engine(fresh_options, ChannelMode::Luma, &frames, 4);
    assert_eq!(second, fresh.frames);
}

/// Pushes frames until one is ready, leaving it unemitted, and returns the ready count and frames pushed.
fn engine_with_a_ready_frame(client: &ComputeClient<R>, frames: &[Vec<f32>]) -> (Nl4d<R>, usize, usize) {
    let mut engine = build_engine(client, 1, ChannelMode::Luma);
    let mut pushed = 0;

    loop {
        let ready = push_frame(&mut engine, client, &frames[pushed], 1);
        pushed += 1;

        if ready == 1 {
            return (engine, ready, pushed);
        }
    }
}

#[test]
fn push_before_emitting_returns_outputs_pending_and_keeps_the_frame() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let client = make_client();
    let (mut engine, _ready, pushed) = engine_with_a_ready_frame(&client, &frames);

    let handles = upload_planes(&client, &frames[pushed], 1);
    let planes = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];
    let refused = engine.push(&planes);
    assert!(matches!(refused, Err(Error::OutputsPending)));

    let output = emit(&mut engine, &client, 1);
    let expected_options = options(1, false);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(output, expected.frames[0]);
}

#[test]
fn finish_while_a_frame_is_ready_returns_outputs_pending() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let client = make_client();
    let (mut engine, _ready, _pushed) = engine_with_a_ready_frame(&client, &frames);

    let finished = engine.finish();
    assert!(matches!(finished, Err(Error::OutputsPending)));
}

fn engine_owing_a_tail(client: &ComputeClient<R>) -> (Nl4d<R>, usize) {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 6);
    let mut engine = build_engine(client, 1, ChannelMode::Luma);
    let mut outputs = Vec::new();

    for frame in &frames {
        push_and_emit(&mut engine, client, frame, 1, &mut outputs);
    }

    let tail = engine.finish().expect("finish");
    assert!(tail > 0);

    (engine, tail)
}

#[test]
fn push_while_a_tail_is_owed_returns_outputs_pending() {
    let client = make_client();
    let (mut engine, _tail) = engine_owing_a_tail(&client);
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 1);
    let handles = upload_planes(&client, &frames[0], 1);
    let planes = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];

    let pushed = engine.push(&planes);
    assert!(matches!(pushed, Err(Error::OutputsPending)));
}

#[test]
fn finish_while_a_tail_is_owed_returns_outputs_pending() {
    let client = make_client();
    let (mut engine, _tail) = engine_owing_a_tail(&client);

    let finished = engine.finish();
    assert!(matches!(finished, Err(Error::OutputsPending)));
}

#[test]
fn emit_after_the_last_tail_frame_returns_nothing_to_emit() {
    let client = make_client();
    let (mut engine, tail) = engine_owing_a_tail(&client);

    for _ in 0..tail {
        emit(&mut engine, &client, 1);
    }

    let output = client.empty((WIDTH * HEIGHT * 4) as usize);
    let planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];
    let emitted = engine.emit_into(&planes);
    assert!(matches!(emitted, Err(Error::NothingToEmit)));
}

#[test]
fn misuse_errors_do_not_poison() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 4);
    let client = make_client();
    let mut engine = build_engine(&client, 1, ChannelMode::Luma);
    let output = client.empty((WIDTH * HEIGHT * 4) as usize);
    let planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];

    let nothing = engine.emit_into(&planes);
    assert!(matches!(nothing, Err(Error::NothingToEmit)));

    let wrong_count = engine.push(&[]);
    assert!(matches!(wrong_count, Err(Error::PlaneMismatch(_))));

    let mut ready = 0;
    for frame in &frames {
        ready = push_frame(&mut engine, &client, frame, 1);
        if ready == 1 {
            break;
        }
    }

    assert_eq!(ready, 1);
}

#[test]
fn a_plane_mismatch_mid_stream_changes_no_state() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);
    let mut outputs = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        push_and_emit(&mut engine, &client, frame, 1, &mut outputs);

        if index == 5 {
            let wrong_count = engine.push(&[]);
            assert!(matches!(wrong_count, Err(Error::PlaneMismatch(_))));

            let wrong_context = engine.push_context(&[]);
            assert!(matches!(wrong_context, Err(Error::PlaneMismatch(_))));
        }
    }

    finish_and_emit(&mut engine, &client, 1, &mut outputs);

    let expected_options = options(2, false);
    let expected = run_oracle(expected_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(outputs, expected.frames);
}

#[test]
fn a_gpu_failure_poisons_until_reset() {
    let client = make_client();
    let mut engine = build_engine(&client, 1, ChannelMode::Luma);

    let failure = engine.fail_through_guard_for_test();
    assert!(matches!(failure, Err(Error::Gpu(_))));

    let frames = noisy_frames(WIDTH, HEIGHT, 1, 1);
    let handles = upload_planes(&client, &frames[0], 1);
    let input = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];
    let pushed = engine.push(&input);
    assert!(matches!(pushed, Err(Error::NeedsReset)));

    engine.reset();
    let pushed_after_reset = engine.push(&input);
    assert!(pushed_after_reset.is_ok());
}

#[test]
fn reset_mid_stream_with_a_ready_frame_matches_a_fresh_engine() {
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 10);
    let client = make_client();
    let (mut engine, ready, _pushed) = engine_with_a_ready_frame(&client, &frames);
    assert_eq!(ready, 1);

    engine.reset();
    let second = drive(&mut engine, &client, &frames, 1);

    let fresh_options = options(1, false);
    let fresh = run_engine(fresh_options, ChannelMode::Luma, &frames, 0);
    assert_eq!(second, fresh.frames);
}

#[test]
fn u8_input_matches_f32_input_from_the_ingest_kernel() {
    let client = make_client();
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 4);
    let codes: Vec<Vec<u8>> = frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&value| (value.clamp(0.0, 1.0) * 255.0) as u8)
                .collect()
        })
        .collect();
    let u8_inputs: Vec<Handle> = codes
        .iter()
        .map(|frame| client.create_from_slice(frame))
        .collect();
    let f32_inputs: Vec<Handle> = codes
        .iter()
        .map(|frame| {
            let normalised = normalise_with_ingest(&client, frame, WIDTH, HEIGHT);
            let bytes = f32::as_bytes(&normalised);
            client.create_from_slice(bytes)
        })
        .collect();

    let u8_geometry = Geometry {
        input: SampleFormat::U8,
        output: SampleFormat::U8,
        width: WIDTH,
        height: HEIGHT,
        channels: ChannelMode::Luma,
    };
    let f32_geometry = Geometry {
        input: SampleFormat::F32,
        ..u8_geometry
    };
    let u8_options = options(1, false);
    let f32_options = options(1, false);
    let mut u8_engine = Nl4d::new(&client, u8_options, u8_geometry).expect("build u8");
    let mut f32_engine = Nl4d::new(&client, f32_options, f32_geometry).expect("build f32");

    let u8_frames = drive_u8(&mut u8_engine, &client, &u8_inputs);
    let f32_frames = drive_u8(&mut f32_engine, &client, &f32_inputs);
    assert_eq!(u8_frames.len(), 4);
    assert_eq!(u8_frames, f32_frames);
}

#[test]
fn window_span_doubles_the_radius_with_shifted_edges() {
    let client = make_client();
    let engine = build_engine(&client, 2, ChannelMode::Luma);
    let span = engine.window_span();
    assert_eq!((span.behind, span.ahead), (4, 4));
    assert_eq!(span.edges, EdgePadding::Shifted);

    let held = engine.max_held_frames();
    assert_eq!(held, 4);
}

#[test]
fn frames_smaller_than_a_patch_are_invalid_geometry() {
    let client = make_client();
    let geometry = Geometry {
        width: 4,
        height: HEIGHT,
        ..geometry(ChannelMode::Luma)
    };

    let engine_options = options(1, false);
    let built = Nl4d::new(&client, engine_options, geometry);
    assert!(matches!(built, Err(Error::InvalidGeometry(_))));
}
