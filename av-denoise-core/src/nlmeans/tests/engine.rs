use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, make_client, make_noisy_gaussian_frame};
use crate::denoiser::FrameOutput;
use crate::engine::{DevicePlane, Engine, Geometry, SampleFormat};
use crate::error::Error;
use crate::nlmeans::{
    ChannelMode,
    DenoisingMode,
    NlmDenoiser,
    Nlmeans,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    resolve_params,
};

const WIDTH: u32 = 48;
const HEIGHT: u32 = 40;

fn geometry(channels: ChannelMode) -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        channels,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
    }
}

fn hq(radius: u32) -> NlmeansAlgorithm {
    let mode = match radius {
        0 => DenoisingMode::Spacial,
        radius => DenoisingMode::Temporal { radius },
    };
    let nlm = NlmeansOptions {
        mode,
        ..NlmeansOptions::default()
    };

    NlmeansAlgorithm::Hq(NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    })
}

fn frames(count: usize, channels: u32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| {
            let base = 0.3 + index as f32 * 0.01;
            make_noisy_gaussian_frame(WIDTH, HEIGHT, channels, base, &[0.03])
        })
        .collect()
}

/// Splits an interleaved host frame into one uploaded plane per channel.
fn upload_planes(client: &ComputeClient<R>, frame: &[f32], channels: usize) -> Vec<Handle> {
    (0..channels)
        .map(|channel| {
            let plane: Vec<f32> = frame.iter().skip(channel).step_by(channels).copied().collect();
            let bytes = f32::as_bytes(&plane);
            client.create_from_slice(bytes)
        })
        .collect()
}

fn read_interleaved(client: &ComputeClient<R>, planes: &[Handle]) -> Vec<f32> {
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

fn emit(engine: &mut Nlmeans<R>, client: &ComputeClient<R>, channels: usize) -> Vec<f32> {
    let pixels = (WIDTH * HEIGHT) as usize;
    let outputs: Vec<Handle> = (0..channels).map(|_| client.empty(pixels * 4)).collect();
    let planes: Vec<_> = outputs
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect();

    engine.emit_into(&planes).expect("emit");

    read_interleaved(client, &outputs)
}

/// Pushes one frame and returns how many frames the engine reports ready.
fn push_frame(engine: &mut Nlmeans<R>, client: &ComputeClient<R>, frame: &[f32], count: usize) -> usize {
    let handles = upload_planes(client, frame, count);
    let planes: Vec<_> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect();

    engine.push(&planes).expect("push")
}

/// Pushes one frame and emits whatever becomes ready into `outputs`.
fn push_and_emit(
    engine: &mut Nlmeans<R>,
    client: &ComputeClient<R>,
    frame: &[f32],
    count: usize,
    outputs: &mut Vec<Vec<f32>>,
) {
    let ready = push_frame(engine, client, frame, count);
    for _ in 0..ready {
        let output = emit(engine, client, count);
        outputs.push(output);
    }
}

/// Finishes the stream and emits every tail frame into `outputs`.
fn finish_and_emit(
    engine: &mut Nlmeans<R>,
    client: &ComputeClient<R>,
    count: usize,
    outputs: &mut Vec<Vec<f32>>,
) {
    let tail = engine.finish().expect("finish");
    for _ in 0..tail {
        let output = emit(engine, client, count);
        outputs.push(output);
    }
}

/// Pushes every frame through `engine`, then the tail, returning each emitted frame.
fn drive(
    engine: &mut Nlmeans<R>,
    client: &ComputeClient<R>,
    frames: &[Vec<f32>],
    count: usize,
) -> Vec<Vec<f32>> {
    let mut outputs = Vec::new();

    for frame in frames {
        push_and_emit(engine, client, frame, count, &mut outputs);
    }

    finish_and_emit(engine, client, count, &mut outputs);

    outputs
}

fn build_engine(client: &ComputeClient<R>, radius: u32, channels: ChannelMode) -> Nlmeans<R> {
    let algorithm = hq(radius);
    let geometry = geometry(channels);

    Nlmeans::new(client, algorithm, geometry).expect("build")
}

fn run_engine(radius: u32, channels: ChannelMode, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let client = make_client();
    let count = channels.count() as usize;
    let mut engine = build_engine(&client, radius, channels);

    drive(&mut engine, &client, frames, count)
}

fn run_oracle(radius: u32, channels: ChannelMode, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let client = make_client();
    let algorithm = hq(radius);
    let params = resolve_params(&algorithm, channels);
    let mut denoiser = NlmDenoiser::new(&client, params, WIDTH, HEIGHT);
    let mut outputs = Vec::new();

    for frame in frames {
        denoiser.push_frame(frame);

        let output = denoiser.denoise().expect("denoise");
        if let Some(output) = output {
            outputs.push(output.as_f32().expect("f32").to_vec());
        }
    }

    let collect = |output: &FrameOutput| {
        let samples = output.as_f32().expect("f32").to_vec();
        outputs.push(samples);
    };
    denoiser.flush(collect).expect("flush");

    outputs
}

#[test]
fn spatial_luma_matches_the_oracle() {
    let frames = frames(4, 1);
    let actual = run_engine(0, ChannelMode::Luma, &frames);
    let expected = run_oracle(0, ChannelMode::Luma, &frames);
    assert_eq!(actual, expected);
}

#[test]
fn temporal_chroma_matches_the_oracle_including_the_tail() {
    let frames = frames(8, 2);
    let actual = run_engine(2, ChannelMode::Chroma, &frames);
    let expected = run_oracle(2, ChannelMode::Chroma, &frames);
    assert_eq!(actual.len(), 8);
    assert_eq!(actual, expected);
}

#[test]
fn temporal_yuv_matches_the_oracle() {
    let frames = frames(6, 3);
    let actual = run_engine(1, ChannelMode::Yuv, &frames);
    let expected = run_oracle(1, ChannelMode::Yuv, &frames);
    assert_eq!(actual, expected);
}

#[test]
fn a_stream_shorter_than_the_window_matches_the_oracle() {
    let frames = frames(2, 1);
    let actual = run_engine(2, ChannelMode::Luma, &frames);
    let expected = run_oracle(2, ChannelMode::Luma, &frames);
    assert_eq!(actual.len(), 2);
    assert_eq!(actual, expected);
}

#[test]
fn a_second_stream_after_finish_matches_a_fresh_engine() {
    let frames = frames(6, 1);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);

    drive(&mut engine, &client, &frames, 1);
    let second = drive(&mut engine, &client, &frames, 1);

    let fresh = run_engine(2, ChannelMode::Luma, &frames);
    assert_eq!(second, fresh);
}

#[test]
fn push_context_then_one_push_matches_the_streaming_centre() {
    let radius = 2u32;
    let context = 2 * radius as usize;
    let window = context + 1;
    let frames = frames(window, 1);
    let client = make_client();
    let mut engine = build_engine(&client, radius, ChannelMode::Luma);

    for frame in &frames[..context] {
        let handles = upload_planes(&client, frame, 1);
        let planes = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];
        engine.push_context(&planes).expect("push context");
    }

    let ready = push_frame(&mut engine, &client, &frames[context], 1);
    assert_eq!(ready, 1);

    let actual = emit(&mut engine, &client, 1);
    let expected = run_oracle(radius, ChannelMode::Luma, &frames);
    assert_eq!(actual, expected[radius as usize]);
}

#[test]
fn push_before_emitting_returns_outputs_pending_and_keeps_the_frame() {
    let frames = frames(3, 1);
    let client = make_client();
    let mut engine = build_engine(&client, 0, ChannelMode::Luma);
    let handles = upload_planes(&client, &frames[0], 1);
    let planes = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];

    let ready = engine.push(&planes).expect("push");
    assert_eq!(ready, 1);

    let second = engine.push(&planes);
    assert!(matches!(second, Err(Error::OutputsPending)));

    let output = emit(&mut engine, &client, 1);
    let expected = run_oracle(0, ChannelMode::Luma, &frames[..1]);
    assert_eq!(output, expected[0]);
}

#[test]
fn finish_while_a_frame_is_ready_returns_outputs_pending() {
    let frames = frames(1, 1);
    let client = make_client();
    let mut engine = build_engine(&client, 0, ChannelMode::Luma);

    let ready = push_frame(&mut engine, &client, &frames[0], 1);
    assert_eq!(ready, 1);

    let finished = engine.finish();
    assert!(matches!(finished, Err(Error::OutputsPending)));
}

/// Pushes a stream and finishes it, leaving the whole tail owed.
fn engine_owing_a_tail(client: &ComputeClient<R>) -> (Nlmeans<R>, usize) {
    let frames = frames(4, 1);
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
    let frame = frames(1, 1);
    let handles = upload_planes(&client, &frame[0], 1);
    let planes = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];

    let pushed = engine.push(&planes);
    assert!(matches!(pushed, Err(Error::OutputsPending)));
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
    let client = make_client();
    let mut engine = build_engine(&client, 0, ChannelMode::Luma);
    let output = client.empty((WIDTH * HEIGHT * 4) as usize);
    let planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];

    let nothing = engine.emit_into(&planes);
    assert!(matches!(nothing, Err(Error::NothingToEmit)));

    let wrong_count = engine.push(&[]);
    assert!(matches!(wrong_count, Err(Error::PlaneMismatch(_))));

    let frame = frames(1, 1);
    let ready = push_frame(&mut engine, &client, &frame[0], 1);
    assert_eq!(ready, 1);
}

#[test]
fn a_plane_mismatch_mid_stream_changes_no_state() {
    let frames = frames(6, 1);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);
    let mut outputs = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        push_and_emit(&mut engine, &client, frame, 1, &mut outputs);

        if index == 3 {
            let wrong_count = engine.push(&[]);
            assert!(matches!(wrong_count, Err(Error::PlaneMismatch(_))));
        }
    }

    finish_and_emit(&mut engine, &client, 1, &mut outputs);

    let expected = run_oracle(2, ChannelMode::Luma, &frames);
    assert_eq!(outputs, expected);
}

#[test]
fn a_gpu_failure_poisons_until_reset() {
    let client = make_client();
    let mut engine = build_engine(&client, 0, ChannelMode::Luma);

    let failure = engine.fail_through_guard_for_test();
    assert!(matches!(failure, Err(Error::Gpu(_))));

    let frame = frames(1, 1);
    let handles = upload_planes(&client, &frame[0], 1);
    let input = [DevicePlane::new(&handles[0], WIDTH, HEIGHT)];
    let pushed = engine.push(&input);
    assert!(matches!(pushed, Err(Error::NeedsReset)));

    engine.reset();
    let ready = engine.push(&input).expect("push after reset");
    assert_eq!(ready, 1);
}

#[test]
fn reset_mid_stream_with_a_ready_frame_matches_a_fresh_engine() {
    let frames = frames(6, 1);
    let client = make_client();
    let mut engine = build_engine(&client, 2, ChannelMode::Luma);

    let mut ready = 0;
    for frame in &frames[..3] {
        ready = push_frame(&mut engine, &client, frame, 1);
    }
    assert_eq!(ready, 1);

    engine.reset();
    let second = drive(&mut engine, &client, &frames, 1);

    let fresh = run_engine(2, ChannelMode::Luma, &frames);
    assert_eq!(second, fresh);
}

#[test]
fn u8_planes_match_a_quantised_oracle() {
    let client = make_client();
    let geometry = Geometry {
        width: WIDTH,
        height: HEIGHT,
        channels: ChannelMode::Luma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let algorithm = hq(0);
    let mut engine = Nlmeans::new(&client, algorithm, geometry).expect("build");
    let codes: Vec<u8> = (0..WIDTH * HEIGHT).map(|index| (index % 251) as u8).collect();
    let input = client.create_from_slice(&codes);
    let output = client.empty((WIDTH * HEIGHT) as usize);
    let input_planes = [DevicePlane::new(&input, WIDTH, HEIGHT)];
    let output_planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];

    let ready = engine.push(&input_planes).expect("push");
    assert_eq!(ready, 1);

    engine.emit_into(&output_planes).expect("emit");
    let actual = client.read_one(output).expect("read");

    let normalised: Vec<f32> = codes.iter().map(|&code| code as f32 / 255.0).collect();
    let oracle = run_oracle(0, ChannelMode::Luma, &[normalised]);
    let expected: Vec<u8> = oracle[0]
        .iter()
        .map(|&value| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    assert_eq!(actual.to_vec(), expected);
}

#[test]
fn window_span_and_held_frames_follow_the_radius() {
    let client = make_client();
    let engine = build_engine(&client, 3, ChannelMode::Luma);
    let span = engine.window_span();
    assert_eq!((span.behind, span.ahead), (3, 3));

    let held = engine.max_held_frames();
    assert_eq!(held, 3);
}
