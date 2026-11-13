use std::ops::RangeInclusive;

use super::synthetic::{FRESH_GRAIN, FROZEN_GRAIN, Record, record};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{CHUNK_FRAMES, FRAME_RECORD_LEN, LAG_RECORD_LEN, PERSISTENCE_LAGS};
use crate::nl4d::grain::modes::GrainMode::{Fresh, Frozen};
use crate::nl4d::grain::modes::{
    GrainMode,
    StaticGrain,
    absorb_short_runs,
    grain_runs,
    held_frames,
    snap_to_cuts,
};
use crate::nl4d::grain::report::frame_rows;
use crate::nl4d::grain::segment::SceneGrain;
use crate::nl4d::grain::table::FitOptions;

/// A declaration of the one range `start..=end`.
fn declared(start: u64, end: u64) -> Vec<RangeInclusive<u64>> {
    let range = start..=end;

    vec![range]
}

/// Runs of `(held, frames)` laid end to end.
fn marks(parts: &[(bool, usize)]) -> Vec<bool> {
    let mut marked = Vec::new();
    for &(held, frames) in parts {
        marked.extend(std::iter::repeat_n(held, frames));
    }

    marked
}

fn unreadable() -> Record {
    [0.0; FRAME_RECORD_LEN]
}

fn repeated(grain: (f64, f64), frames: usize) -> Vec<Record> {
    let (lag_one, lag_eight) = grain;
    let single = record(lag_one, lag_eight);

    vec![single; frames]
}

fn joined(parts: &[Vec<Record>]) -> Vec<Record> {
    parts.concat()
}

fn scene(first_frame: u64, records: &[Record]) -> SceneGrain {
    let chunks = records.chunks(CHUNK_FRAMES as usize).map(chunk_of).collect();

    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks: Vec::new(),
    }
}

fn chunk_of(records: &[Record]) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = records.len() as u32;
    chunk.frame_records = records.iter().flatten().copied().collect();

    chunk
}

/// Back-to-back scenes, one per entry of `scene_records`, starting at frame 0.
fn scenes(scene_records: &[Vec<Record>]) -> Vec<SceneGrain> {
    let mut first_frame = 0;
    let mut built = Vec::new();
    for records in scene_records {
        let next = scene(first_frame, records);
        built.push(next);
        first_frame += records.len() as u64;
    }

    built
}

fn runs_of(scenes: &[SceneGrain], options: &FitOptions) -> Vec<(u64, u64, GrainMode)> {
    let runs = grain_runs(scenes, options).expect("the declarations should be valid");

    runs.iter()
        .map(|run| (run.first_frame, run.last_frame, run.mode))
        .collect()
}

fn detected(scenes: &[SceneGrain]) -> Vec<(u64, u64, GrainMode)> {
    let options = FitOptions::default();

    runs_of(scenes, &options)
}

fn single_scene(records: &[Record]) -> Vec<SceneGrain> {
    let only = scene(0, records);

    vec![only]
}

/// The first frame of the one frozen run after a fresh run, checking the runs are just those two.
fn frozen_start(runs: &[(u64, u64, GrainMode)]) -> u64 {
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert_eq!(runs[0].2, Fresh);
    assert_eq!(runs[1].2, Frozen);
    assert_eq!(runs[0].1 + 1, runs[1].0);

    runs[1].0
}

#[test]
fn steady_high_persistence_is_one_frozen_run() {
    let records = repeated(FROZEN_GRAIN, 200);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 199, Frozen)]);
}

#[test]
fn low_persistence_is_fresh() {
    let records = repeated(FRESH_GRAIN, 200);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 199, Fresh)]);
}

#[test]
fn slow_grain_is_fresh() {
    let records = repeated((0.67, 0.31), 200);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 199, Fresh)]);
}

#[test]
fn a_ratio_of_exactly_the_threshold_is_frozen() {
    let records = repeated((0.50, 0.30), 200);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 199, Frozen)]);
}

#[test]
fn a_short_dip_does_not_split_a_frozen_run() {
    let records = joined(&[
        repeated(FROZEN_GRAIN, 100),
        repeated((0.01, 0.01), 10),
        repeated(FROZEN_GRAIN, 100),
    ]);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 209, Frozen)]);
}

#[test]
fn unreadable_frames_take_the_nearest_reading() {
    let records = joined(&[
        repeated(FROZEN_GRAIN, 100),
        vec![unreadable(); 80],
        repeated(FROZEN_GRAIN, 100),
    ]);
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 279, Frozen)]);
}

#[test]
fn fresh_then_frozen_changes_near_the_boundary() {
    let records = joined(&[repeated(FRESH_GRAIN, 200), repeated(FROZEN_GRAIN, 200)]);
    let clip = single_scene(&records);

    let runs = detected(&clip);
    let start = frozen_start(&runs);

    assert!((176..=224).contains(&start), "{runs:?}");
    assert_eq!(runs[1].1, 399);
}

#[test]
fn a_change_snaps_to_a_scene_cut_within_24_frames() {
    let second_scene = joined(&[repeated(FRESH_GRAIN, 10), repeated(FROZEN_GRAIN, 300)]);
    let clip = scenes(&[repeated(FRESH_GRAIN, 210), second_scene]);

    assert_eq!(detected(&clip), vec![(0, 209, Fresh), (210, 519, Frozen)]);
}

#[test]
fn a_change_far_from_any_cut_stays() {
    let records = joined(&[repeated(FRESH_GRAIN, 220), repeated(FROZEN_GRAIN, 300)]);
    let clip = single_scene(&records);

    let runs = detected(&clip);
    let start = frozen_start(&runs);

    assert!((196..=244).contains(&start), "{runs:?}");
    assert_eq!(runs[1].1, 519);
}

#[test]
fn a_frozen_run_crosses_scene_cuts() {
    let scene_records = vec![repeated(FROZEN_GRAIN, 100); 3];
    let clip = scenes(&scene_records);

    assert_eq!(detected(&clip), vec![(0, 299, Frozen)]);
}

#[test]
fn frozen_fresh_frozen_gives_three_runs() {
    let records = joined(&[
        repeated(FROZEN_GRAIN, 300),
        repeated(FRESH_GRAIN, 300),
        repeated(FROZEN_GRAIN, 300),
    ]);
    let clip = single_scene(&records);

    let runs = detected(&clip);
    let modes: Vec<GrainMode> = runs.iter().map(|run| run.2).collect();

    assert_eq!(modes, vec![Frozen, Fresh, Frozen]);
}

#[test]
fn no_readings_at_all_is_fresh() {
    let records = vec![unreadable(); 100];
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 99, Fresh)]);
}

#[test]
fn a_held_run_without_lag_eight_readings_is_frozen() {
    let (lag_one, lag_eight) = FROZEN_GRAIN;
    let mut held = record(lag_one, lag_eight);
    let lag_eight_start = (PERSISTENCE_LAGS - 1) * LAG_RECORD_LEN;
    held[lag_eight_start..].fill(0.0);
    let records = vec![held; 200];
    let clip = single_scene(&records);

    assert_eq!(detected(&clip), vec![(0, 199, Frozen)]);
}

#[test]
fn a_clip_shorter_than_the_window_still_answers() {
    let frozen_records = repeated(FROZEN_GRAIN, 10);
    let frozen_clip = single_scene(&frozen_records);
    let fresh_records = repeated(FRESH_GRAIN, 10);
    let fresh_clip = single_scene(&fresh_records);

    assert_eq!(detected(&frozen_clip), vec![(0, 9, Frozen)]);
    assert_eq!(detected(&fresh_clip), vec![(0, 9, Fresh)]);
}

#[test]
fn off_leaves_everything_fresh() {
    let records = repeated(FROZEN_GRAIN, 200);
    let clip = single_scene(&records);
    let options = FitOptions::default().with_static_grain(StaticGrain::Off);

    assert_eq!(runs_of(&clip, &options), vec![(0, 199, Fresh)]);
}

#[test]
fn all_freezes_everything() {
    let records = repeated(FRESH_GRAIN, 200);
    let clip = single_scene(&records);
    let options = FitOptions::default().with_static_grain(StaticGrain::All);

    assert_eq!(runs_of(&clip, &options), vec![(0, 199, Frozen)]);
}

#[test]
fn declared_frames_are_frozen_under_off() {
    let records = repeated(FRESH_GRAIN, 200);
    let clip = single_scene(&records);
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::Off)
        .with_static_frames(declared(40, 59));

    let expected = vec![(0, 39, Fresh), (40, 59, Frozen), (60, 199, Fresh)];
    assert_eq!(runs_of(&clip, &options), expected);
}

#[test]
fn declared_scenes_are_frozen() {
    let scene_records = vec![repeated(FRESH_GRAIN, 50); 3];
    let clip = scenes(&scene_records);
    let options = FitOptions::default().with_static_scenes(declared(1, 1));

    let expected = vec![(0, 49, Fresh), (50, 99, Frozen), (100, 149, Fresh)];
    assert_eq!(runs_of(&clip, &options), expected);
}

#[test]
fn overlapping_declarations_merge_into_one_run() {
    let clip = scenes(&[
        repeated(FRESH_GRAIN, 51),
        repeated(FRESH_GRAIN, 49),
        repeated(FRESH_GRAIN, 50),
    ]);
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::Off)
        .with_static_frames(vec![20..=40, 10..=30, 41..=50])
        .with_static_scenes(declared(1, 1));

    let expected = vec![(0, 9, Fresh), (10, 99, Frozen), (100, 149, Fresh)];
    assert_eq!(runs_of(&clip, &options), expected);
}

#[test]
fn a_declaration_touching_a_detected_run_joins_it() {
    let records = joined(&[repeated(FRESH_GRAIN, 100), repeated(FROZEN_GRAIN, 200)]);
    let clip = single_scene(&records);
    let options = FitOptions::default().with_static_frames(declared(60, 130));

    assert_eq!(runs_of(&clip, &options), vec![(0, 59, Fresh), (60, 299, Frozen)]);
}

#[test]
fn a_frame_range_past_the_end_is_refused() {
    let records = repeated(FRESH_GRAIN, 100);
    let clip = single_scene(&records);
    let options = FitOptions::default().with_static_frames(declared(90, 100));

    let error = grain_runs(&clip, &options).expect_err("the range should be refused");

    let message = error.to_string();
    assert!(message.contains("past the last frame 99"), "{message}");
}

#[test]
fn a_scene_range_past_the_end_is_refused() {
    let scene_records = vec![repeated(FRESH_GRAIN, 50); 2];
    let clip = scenes(&scene_records);
    let options = FitOptions::default().with_static_scenes(declared(2, 2));

    let error = grain_runs(&clip, &options).expect_err("the range should be refused");

    let message = error.to_string();
    assert!(message.contains("past the last scene 1"), "{message}");
}

#[test]
fn a_range_ending_before_it_starts_is_refused() {
    let records = repeated(FRESH_GRAIN, 100);
    let clip = single_scene(&records);
    let options = FitOptions::default().with_static_frames(declared(50, 40));

    let error = grain_runs(&clip, &options).expect_err("the range should be refused");

    let message = error.to_string();
    assert!(message.contains("ends before it starts"), "{message}");
}

#[test]
fn a_declaration_under_all_is_still_checked() {
    let records = repeated(FRESH_GRAIN, 100);
    let clip = single_scene(&records);
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::All)
        .with_static_frames(declared(90, 100));

    let error = grain_runs(&clip, &options).expect_err("the range should be refused");

    let message = error.to_string();
    assert!(message.contains("past the last frame 99"), "{message}");
}

#[test]
fn an_empty_clip_has_no_runs() {
    let options = FitOptions::default();

    assert_eq!(runs_of(&[], &options), vec![]);
}

#[test]
fn a_declaration_on_an_empty_clip_is_refused() {
    let options = FitOptions::default().with_static_frames(declared(0, 0));

    let error = grain_runs(&[], &options).expect_err("the range should be refused");

    let message = error.to_string();
    assert!(message.contains("a clip with no frames"), "{message}");
}

#[test]
fn an_unread_gap_takes_the_nearer_answer_on_each_side() {
    let smoothed = [Some(0.70), None, None, None, None, Some(0.05)];

    let held = held_frames(&smoothed);

    assert_eq!(held, [true, true, true, false, false, false]);
}

#[test]
fn an_unread_frame_equally_far_from_both_answers_takes_the_earlier() {
    let fresh_first = [Some(0.05), None, None, None, Some(0.70)];
    let held_first = [Some(0.70), None, None, None, Some(0.05)];

    let from_fresh = held_frames(&fresh_first);
    let from_held = held_frames(&held_first);

    assert_eq!(from_fresh, [false, false, false, true, true]);
    assert_eq!(from_held, [true, true, true, false, false]);
}

#[test]
fn a_short_held_run_between_fresh_runs_turns_fresh() {
    let mut held = marks(&[(false, 30), (true, 10), (false, 30)]);

    absorb_short_runs(&mut held);

    assert_eq!(held, marks(&[(false, 70)]));
}

#[test]
fn short_fresh_runs_turn_held_before_short_held_runs_turn_fresh() {
    let mut held = marks(&[(true, 30), (false, 10), (true, 10), (false, 30)]);

    absorb_short_runs(&mut held);

    assert_eq!(held, marks(&[(true, 50), (false, 30)]));
}

#[test]
fn a_snap_that_would_empty_a_run_is_skipped() {
    let records = vec![unreadable(); 30];
    let clip = scenes(&[records.clone(), records]);
    let rows = frame_rows(&clip);
    let mut frozen = marks(&[(false, 10), (true, 10), (false, 40)]);

    snap_to_cuts(&rows, &mut frozen);

    assert_eq!(frozen, marks(&[(false, 10), (true, 20), (false, 30)]));
}
