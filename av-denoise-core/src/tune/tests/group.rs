use std::path::Path;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{DevicePlane, Engine, EngineSpec, Geometry, SampleFormat};
use crate::error::Error;
use crate::nl4d::Nl4dOptions;
use crate::nlmeans::{ChannelMode, DenoisingMode, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::{
    BudgetRequest,
    SearchMode,
    TuneOutcome,
    TuneRequest,
    TunedGroup,
    TuningConfig,
    TuningFrame,
    TuningMode,
    build_engine,
    tune_frames,
    tuning_frame_count,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

/// One frame's planes, one list per engine.
type FrameHandles = Vec<Vec<Handle>>;

/// The read-back planes of one emitted frame, every engine's planes in group order.
type FrameBytes = Vec<Vec<u8>>;

fn geometry() -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        channels: ChannelMode::Luma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    }
}

fn nlmeans_spec(mode: DenoisingMode) -> EngineSpec {
    let options = NlmeansOptions {
        mode,
        ..NlmeansOptions::default()
    };

    EngineSpec::Nlmeans {
        algorithm: NlmeansAlgorithm::Fast(options),
        geometry: geometry(),
    }
}

fn spatial() -> Vec<EngineSpec> {
    let spec = nlmeans_spec(DenoisingMode::Spacial);
    vec![spec]
}

fn temporal() -> Vec<EngineSpec> {
    let spec = nlmeans_spec(DenoisingMode::Temporal { radius: 1 });
    vec![spec]
}

fn nl4d() -> Vec<EngineSpec> {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let spec = EngineSpec::Nl4d {
        options,
        geometry: geometry(),
    };

    vec![spec]
}

fn auto_config(store: &Path, budget: BudgetRequest) -> TuningConfig {
    TuningConfig {
        mode: TuningMode::Auto,
        store: Some(store.to_path_buf()),
        budget,
        ..TuningConfig::default()
    }
}

fn deterministic_config(store: &Path) -> TuningConfig {
    TuningConfig {
        deterministic: true,
        ..auto_config(store, BudgetRequest::Full)
    }
}

/// A ramp plus hashed grain, different for each `frame`.
fn plane_bytes(frame: u32) -> Vec<u8> {
    let pixels = WIDTH * HEIGHT;

    (0..pixels)
        .map(|index| {
            let ramp = (index % WIDTH) * 3;
            let grain = index.wrapping_mul(2_654_435_761).wrapping_add(frame * 97) >> 28;
            let value = 40 + ramp + grain;
            value.min(255) as u8
        })
        .collect()
}

fn upload_frame(client: &Client, specs: &[EngineSpec], frame: u32) -> FrameHandles {
    let bytes = plane_bytes(frame);

    specs
        .iter()
        .map(|spec| {
            let channels = spec.geometry().channels.count();
            (0..channels).map(|_| client.create_from_slice(&bytes)).collect()
        })
        .collect()
}

fn upload_clip(client: &Client, specs: &[EngineSpec], count: usize) -> Vec<FrameHandles> {
    (0..count as u32)
        .map(|frame| upload_frame(client, specs, frame))
        .collect()
}

fn device_planes(handles: &[Handle]) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect()
}

fn engine_planes(frame: &[Vec<Handle>]) -> Vec<Vec<DevicePlane<'_>>> {
    frame.iter().map(|handles| device_planes(handles)).collect()
}

fn push(group: &mut TunedGroup, frame: &[Vec<Handle>]) -> Result<usize, Error> {
    let lists = engine_planes(frame);
    let planes: Vec<&[DevicePlane<'_>]> = lists.iter().map(Vec::as_slice).collect();

    group.push(&planes)
}

fn push_context(group: &mut TunedGroup, frame: &[Vec<Handle>]) -> Result<(), Error> {
    let lists = engine_planes(frame);
    let planes: Vec<&[DevicePlane<'_>]> = lists.iter().map(Vec::as_slice).collect();

    group.push_context(&planes)
}

fn output_handles(client: &Client, specs: &[EngineSpec]) -> FrameHandles {
    let geometry = geometry();
    let bytes = geometry.output.plane_bytes(geometry.pixels()) as usize;

    specs
        .iter()
        .map(|spec| {
            let channels = spec.geometry().channels.count();
            (0..channels).map(|_| client.empty(bytes)).collect()
        })
        .collect()
}

fn read_frame(client: &Client, frame: &FrameHandles) -> FrameBytes {
    frame
        .iter()
        .flatten()
        .map(|handle| {
            let read = client.read_one(handle.clone()).expect("the plane reads");
            read.to_vec()
        })
        .collect()
}

/// Emits `count` frames from `group` and reads each back.
fn emit(group: &mut TunedGroup, client: &Client, specs: &[EngineSpec], count: usize) -> Vec<FrameBytes> {
    let mut emitted = Vec::with_capacity(count);

    for _ in 0..count {
        let handles = output_handles(client, specs);
        let lists = engine_planes(&handles);
        let planes: Vec<&[DevicePlane<'_>]> = lists.iter().map(Vec::as_slice).collect();
        group.emit_into(&planes).expect("the frame emits");

        let read = read_frame(client, &handles);
        emitted.push(read);
    }

    emitted
}

/// Emits `count` frames from a lone luma `engine` and reads each back.
fn emit_engine(engine: &mut dyn Engine, client: &Client, count: usize) -> Vec<FrameBytes> {
    let luma_only = spatial();
    let mut emitted = Vec::with_capacity(count);

    for _ in 0..count {
        let handles = output_handles(client, &luma_only);
        let planes = device_planes(&handles[0]);
        engine.emit_into(&planes).expect("the frame emits");

        let read = read_frame(client, &handles);
        emitted.push(read);
    }

    emitted
}

/// Every frame a lone `spec` engine running its baselines gives for `clip`, tail included.
fn baseline_output(
    device: &cubecl::Device,
    client: &Client,
    spec: EngineSpec,
    clip: &[FrameHandles],
) -> Vec<FrameBytes> {
    let off = TuningConfig {
        mode: TuningMode::Off,
        ..TuningConfig::default()
    };
    let mut baseline = build_engine(device, client, spec, &off).expect("the baseline builds");
    let mut output = Vec::new();

    for frame in clip {
        let planes = device_planes(&frame[0]);
        let ready = baseline.push(&planes).expect("the baseline push succeeds");
        let emitted = emit_engine(baseline.as_mut(), client, ready);
        output.extend(emitted);
    }

    let tail = baseline.finish().expect("the baseline finishes");
    let emitted = emit_engine(baseline.as_mut(), client, tail);
    output.extend(emitted);

    output
}

fn assert_frames_match(emitted: &[FrameBytes], expected: &[FrameBytes]) {
    assert_eq!(emitted.len(), expected.len());

    let pairs = emitted.iter().zip(expected);
    for (index, (emitted_frame, expected_frame)) in pairs.enumerate() {
        assert_eq!(emitted_frame, expected_frame, "frame {index}");
    }
}

fn assert_nothing_to_emit(group: &mut TunedGroup, client: &Client, specs: &[EngineSpec]) {
    let handles = output_handles(client, specs);
    let lists = engine_planes(&handles);
    let planes: Vec<&[DevicePlane<'_>]> = lists.iter().map(Vec::as_slice).collect();
    let emitted = group.emit_into(&planes);

    assert!(matches!(emitted, Err(Error::NothingToEmit)), "{emitted:?}");
}

#[test]
fn a_complete_store_builds_settled_with_no_buffering() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let tuning_frames: Vec<TuningFrame> = clip.iter().cloned().map(TuningFrame::new).collect();
    let request = TuneRequest {
        force: false,
        mode: SearchMode::Families,
    };

    let report = tune_frames(&device, &client, &specs, &tuning_frames, &config, &request);
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");
    let ready = push(&mut group, &clip[0]).expect("the push succeeds");

    assert_eq!(ready, 1);
    assert!(group.tune_report().is_none());
}

#[test]
fn an_empty_store_buffers_until_the_tuning_frame_count() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");

    for frame in &clip[..frame_count - 1] {
        let ready = push(&mut group, frame).expect("the push succeeds");
        assert_eq!(ready, 0);
    }

    let last = clip.last().expect("the clip has frames");
    let ready = push(&mut group, last).expect("the push succeeds");

    assert_eq!(ready, frame_count);
    assert!(group.tune_report().is_some());

    let emitted = emit(&mut group, &client, &specs, ready);
    assert_eq!(emitted.len(), frame_count);
    assert_nothing_to_emit(&mut group, &client, &specs);
}

#[test]
fn deterministic_group_output_matches_a_baseline_engine_bit_for_bit() {
    let store = tempfile::tempdir().unwrap();
    let config = deterministic_config(store.path());
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = temporal();
    let frame_count = tuning_frame_count(&specs) + 3;
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");
    let mut group_output = Vec::new();

    for frame in &clip {
        let ready = push(&mut group, frame).expect("the group push succeeds");
        let emitted = emit(&mut group, &client, &specs, ready);
        group_output.extend(emitted);
    }

    let tail = group.finish().expect("the group finishes");
    let emitted = emit(&mut group, &client, &specs, tail);
    group_output.extend(emitted);

    let report = group.tune_report().expect("the group tuned");
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let expected = baseline_output(&device, &client, specs[0], &clip);
    assert_eq!(expected.len(), frame_count);
    assert_frames_match(&group_output, &expected);
}

#[test]
fn finishing_before_the_count_tunes_and_emits_every_frame_once() {
    let store = tempfile::tempdir().unwrap();
    let config = deterministic_config(store.path());
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = temporal();
    let clip = upload_clip(&client, &specs, 3);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");

    for frame in &clip {
        let ready = push(&mut group, frame).expect("the push succeeds");
        assert_eq!(ready, 0);
    }

    let ready = group.finish().expect("the group finishes");

    assert_eq!(ready, 3);
    assert!(group.tune_report().is_some());

    let emitted = emit(&mut group, &client, &specs, ready);
    assert_nothing_to_emit(&mut group, &client, &specs);

    let expected = baseline_output(&device, &client, specs[0], &clip);
    assert_eq!(expected.len(), 3);
    assert_frames_match(&emitted, &expected);
}

#[test]
fn finishing_an_empty_stream_keeps_buffering() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");

    let tail = group.finish().expect("the group finishes");

    assert_eq!(tail, 0);
    assert!(group.tune_report().is_none());

    for frame in &clip[..frame_count - 1] {
        let ready = push(&mut group, frame).expect("the push succeeds");
        assert_eq!(ready, 0);
    }

    let last = clip.last().expect("the clip has frames");
    let ready = push(&mut group, last).expect("the push succeeds");

    assert_eq!(ready, frame_count);
    assert!(group.tune_report().is_some());
}

#[test]
fn reset_while_replaying_restarts_the_stream_cleanly() {
    let store = tempfile::tempdir().unwrap();
    let config = deterministic_config(store.path());
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = temporal();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");
    let mut ready = 0;

    for frame in &clip {
        ready += push(&mut group, frame).expect("the push succeeds");
    }

    assert!(ready > 2, "the replay queued {ready} frames");
    emit(&mut group, &client, &specs, 2);

    group.reset();

    let mut emitted = Vec::new();
    for frame in &clip {
        let ready = push(&mut group, frame).expect("the push after the reset succeeds");
        let frames = emit(&mut group, &client, &specs, ready);
        emitted.extend(frames);
    }

    let tail = group.finish().expect("the group finishes");
    let frames = emit(&mut group, &client, &specs, tail);
    emitted.extend(frames);
    assert_nothing_to_emit(&mut group, &client, &specs);

    let expected = baseline_output(&device, &client, specs[0], &clip);
    assert_eq!(expected.len(), frame_count);
    assert_frames_match(&emitted, &expected);
}

#[test]
fn context_frames_while_buffering_produce_no_output() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Seconds(2.0));
    let off = TuningConfig {
        mode: TuningMode::Off,
        ..TuningConfig::default()
    };
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nl4d();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");

    push(&mut group, &clip[0]).expect("the push succeeds");
    let late_context = push_context(&mut group, &clip[1]);
    assert!(
        matches!(late_context, Err(Error::ContextAfterPush)),
        "{late_context:?}"
    );

    group.reset();

    let mut group_emitted = 0;
    for frame in &clip[..2] {
        push_context(&mut group, frame).expect("the context push succeeds");
    }

    for frame in &clip[2..] {
        let ready = push(&mut group, frame).expect("the push succeeds");
        let emitted = emit(&mut group, &client, &specs, ready);
        group_emitted += emitted.len();
    }

    let tail = group.finish().expect("the group finishes");
    let emitted = emit(&mut group, &client, &specs, tail);
    group_emitted += emitted.len();

    let mut settled = build_engine(&device, &client, specs[0], &off).expect("the engine builds");
    let mut settled_emitted = 0;
    for frame in &clip[..2] {
        let planes = device_planes(&frame[0]);
        settled.push_context(&planes).expect("the context push succeeds");
    }

    for frame in &clip[2..] {
        let planes = device_planes(&frame[0]);
        let ready = settled.push(&planes).expect("the push succeeds");
        let emitted = emit_engine(settled.as_mut(), &client, ready);
        settled_emitted += emitted.len();
    }

    let tail = settled.finish().expect("the engine finishes");
    let emitted = emit_engine(settled.as_mut(), &client, tail);
    settled_emitted += emitted.len();

    assert!(group.tune_report().is_some());
    assert_eq!(group_emitted, settled_emitted);

    push(&mut group, &clip[0]).expect("the push succeeds");
    let late_context = push_context(&mut group, &clip[1]);
    assert!(
        matches!(late_context, Err(Error::ContextAfterPush)),
        "{late_context:?}"
    );
}

#[test]
fn reset_while_buffering_starts_over() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");

    for frame in &clip[..2] {
        let ready = push(&mut group, frame).expect("the push succeeds");
        assert_eq!(ready, 0);
    }

    group.reset();

    for frame in &clip[..frame_count - 1] {
        let ready = push(&mut group, frame).expect("the push succeeds");
        assert_eq!(ready, 0);
    }

    assert!(group.tune_report().is_none());
}

#[test]
fn an_unwritable_store_still_streams() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("not-a-directory");
    std::fs::write(&store, b"a file").unwrap();
    let config = auto_config(&store, BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut group = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");
    let mut ready = 0;

    for frame in &clip {
        ready += push(&mut group, frame).expect("the push succeeds");
    }

    assert_eq!(ready, frame_count);

    let report = group.tune_report().expect("the group tuned");
    assert!(!report.saved, "{report:?}");

    let emitted = emit(&mut group, &client, &specs, ready);
    assert_eq!(emitted.len(), frame_count);
}

#[test]
fn a_second_group_reuses_the_first_groups_result() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = spatial();
    let frame_count = tuning_frame_count(&specs);
    let clip = upload_clip(&client, &specs, frame_count);
    let mut first =
        TunedGroup::new(&device, &client, specs.clone(), config.clone()).expect("the group builds");
    let mut ready = 0;

    for frame in &clip {
        ready += push(&mut first, frame).expect("the push succeeds");
    }

    emit(&mut first, &client, &specs, ready);

    let report = first.tune_report().expect("the first group tuned");
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let mut second = TunedGroup::new(&device, &client, specs.clone(), config).expect("the group builds");
    let ready = push(&mut second, &clip[0]).expect("the push succeeds");

    assert_eq!(ready, 1);
    assert!(second.tune_report().is_none());
}
