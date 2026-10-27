#![cfg(feature = "vulkan")]

use std::io::{Cursor, Read};

use super::{SCENE_CLIP_SIZE, SharedBuffer, frame_count, multi_scene_clip, temporal_opts};
use crate::pipeline::source::open_y4m;
use crate::pipeline::stage::frame_permits;
use crate::pipeline::{RunSettings, run_with};

fn run_over(
    bytes: Vec<u8>,
    workers: usize,
    budget: u64,
    shared_device: bool,
) -> Result<Vec<u8>, anyhow::Error> {
    let output = SharedBuffer::default();
    let writer = output.clone();
    let options = temporal_opts();
    let opener = move || {
        let cursor = Cursor::new(bytes);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };
    let settings = RunSettings {
        workers,
        frame_budget_bytes: budget,
        visible: false,
        grain_table: None,
        tune_budget: None,
        timing_report: false,
        shared_device,
    };

    run_with(&options, opener, settings, writer)?;

    let written = output.0.lock().expect("buffer lock").clone();

    Ok(written)
}

struct ClosedPipe;

impl std::io::Write for ClosedPipe {
    fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_pipe_round_trips_every_frame_and_its_colour_range() {
    let clip = multi_scene_clip(30);
    let output = run_over(clip, 2, 1 << 30, false).expect("the run succeeds");
    let frames = frame_count(&output);

    assert_eq!(frames, 30);
    assert!(output.windows(17).any(|window| window == b"XCOLORRANGE=LIMIT"));
}

#[test]
fn own_devices_write_the_same_frames_as_one_worker_and_a_shared_device() {
    let clip = multi_scene_clip(40);
    let one_worker_clip = clip.clone();
    let own_devices_clip = clip.clone();

    let one_worker = run_over(one_worker_clip, 1, 1 << 30, false).expect("the one-worker run succeeds");
    let own_devices = run_over(own_devices_clip, 2, 1 << 30, false).expect("the own-device run succeeds");
    let shared_device = run_over(clip, 2, 1 << 30, true).expect("the shared-device run succeeds");
    let frames = frame_count(&one_worker);

    assert_eq!(frames, 40);
    assert!(own_devices == one_worker, "own devices changed the output");
    assert!(shared_device == one_worker, "a shared device changed the output");
}

#[test]
fn a_one_frame_pipe_round_trips() {
    let clip = multi_scene_clip(1);
    let output = run_over(clip, 2, 1 << 30, false).expect("the run succeeds");
    let frames = frame_count(&output);

    assert_eq!(frames, 1);
}

#[test]
fn an_empty_pipe_is_rejected() {
    let clip = multi_scene_clip(0);
    let err = run_over(clip, 1, 1 << 30, false).expect_err("no frames is an error");

    assert!(err.to_string().contains("no decodable frames"), "got {err}");
}

/// Runs at the smallest budget `checked_frame_permits` accepts, so the floor decides frames in flight.
#[test]
fn a_run_at_the_permit_floor_finishes() {
    let frame_bytes = SCENE_CLIP_SIZE * SCENE_CLIP_SIZE * 3 / 2;
    let floor = frame_permits(0, frame_bytes, 2, 1);
    let budget = floor as u64 * frame_bytes as u64;

    let clip = multi_scene_clip(40);
    let output = run_over(clip, 2, budget, false).expect("the floor must not deadlock");
    let frames = frame_count(&output);

    assert_eq!(frames, 40);
}

#[test]
fn a_failed_write_is_reported_over_the_worker_disconnect() {
    let clip = multi_scene_clip(30);
    let options = temporal_opts();
    let opener = move || {
        let cursor = Cursor::new(clip);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };
    let settings = RunSettings {
        workers: 2,
        frame_budget_bytes: 1 << 30,
        visible: false,
        grain_table: None,
        tune_budget: None,
        timing_report: false,
        shared_device: false,
    };

    let err = run_with(&options, opener, settings, ClosedPipe).expect_err("the write fails");

    assert!(err.to_string().contains("broken pipe"), "got {err}");
}
