#![cfg(feature = "vulkan")]

use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use av_denoise::tune::{BudgetRequest, TuneOutcome, TuneRequest, is_tuned};
use av_denoise::{PlaneOptions, TuningMode, install_compilation_cache_at};
use av_denoise_core::tune::{TUNE_UNKNOWN, tuning_budget};

use super::{SharedBuffer, frame_count, multi_scene_clip, scene_clip_layout, temporal_opts, y4m_reader};
use crate::cli::InputSource;
use crate::pipeline::convert::SourcePixel;
use crate::pipeline::decode::FrameMsg;
use crate::pipeline::source::open_y4m;
use crate::pipeline::tune::{run_budget, tune_before_workers, worker_options};
use crate::pipeline::{RunSettings, budget_request, run_with, tune_input};

fn run_clip(bytes: Vec<u8>, tuning: TuningMode, tune_budget: Option<f64>) -> Vec<u8> {
    let output = SharedBuffer::default();
    let writer = output.clone();
    let mut options = temporal_opts();
    options.tuning = tuning;
    let opener = move || {
        let cursor = Cursor::new(bytes);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };
    let settings = RunSettings {
        workers: 2,
        frame_budget_bytes: 1 << 30,
        visible: false,
        grain_table: None,
        tune_budget,
        timing_report: false,
    };

    run_with(&options, opener, settings, writer).expect("the run succeeds");

    output.0.lock().expect("buffer lock").clone()
}

fn tuned_options() -> PlaneOptions {
    let mut options = temporal_opts();
    options.tuning = TuningMode::Auto;
    options
}

fn clip_file(dir: &Path, frames: usize) -> InputSource {
    let path = dir.join("clip.y4m");
    let clip = multi_scene_clip(frames);
    std::fs::write(&path, clip).expect("the clip is written");

    InputSource::File(path)
}

/// Counts the tuning results saved under the kernel cache at `cache_root`.
fn stored_results(cache_root: &Path) -> usize {
    let mut count = 0;
    let hash_dirs = std::fs::read_dir(cache_root).expect("the cache root is readable");

    for hash_dir in hash_dirs {
        let hash_dir = hash_dir.expect("the cache entry is readable");
        let tune_dir = hash_dir.path().join("tune");
        let Ok(entries) = std::fs::read_dir(&tune_dir) else {
            continue;
        };

        let results = entries.filter_map(Result::ok).filter(is_json).count();
        count += results;
    }

    count
}

fn is_json(entry: &std::fs::DirEntry) -> bool {
    let path = entry.path();

    path.extension().is_some_and(|extension| extension == "json")
}

#[test]
fn a_pipe_without_a_length_asks_for_the_unknown_length_budget() {
    let clip = multi_scene_clip(3);
    let reader = y4m_reader(clip);
    let opened = open_y4m(reader).expect("the clip opens");

    let frame_time = Duration::from_millis(1);

    let request = run_budget(None, &opened.info);
    let budget = tuning_budget(request, frame_time).expect("an unknown length is affordable");

    assert_eq!(request, BudgetRequest::Automatic { frames: None });
    assert_eq!(budget.total, TUNE_UNKNOWN);
}

#[test]
fn a_set_budget_overrides_the_automatic_one() {
    let clip = multi_scene_clip(3);
    let reader = y4m_reader(clip);
    let opened = open_y4m(reader).expect("the clip opens");

    let request = run_budget(Some(12.5), &opened.info);

    assert_eq!(request, BudgetRequest::Seconds(12.5));
}

#[test]
fn workers_never_tune() {
    let options = tuned_options();

    let workers = worker_options(&options);

    assert_eq!(workers.tuning, TuningMode::Cached);
}

#[test]
fn no_set_budget_keeps_the_fallback() {
    let request = budget_request(None, BudgetRequest::Full);

    assert_eq!(request, BudgetRequest::Full);
}

/// A piped clip shorter than the tuning frame count loops its frames for tuning and still writes
/// every frame once.
#[test]
fn a_three_frame_pipe_tunes_on_the_unknown_length_budget_and_round_trips() {
    let cache = tempfile::tempdir().expect("the temporary directory is created");
    install_compilation_cache_at(cache.path()).expect("the cache installs");
    let clip = multi_scene_clip(3);

    let tuned = run_clip(clip.clone(), TuningMode::Auto, None);
    let stored = stored_results(cache.path());
    let untuned = run_clip(clip, TuningMode::Off, None);
    let options = tuned_options();
    let layout = scene_clip_layout();
    let tuned_now = is_tuned(&options, layout);

    assert_eq!(stored, 1, "the tuning result is saved");
    assert!(tuned_now, "tuning completed");
    assert_eq!(frame_count(&tuned), 3);
    // The default algorithm only has bit-identical candidates, so tuning cannot change the output.
    assert_eq!(tuned, untuned);
}

/// The tuning frames are re-fed ahead of the rest of a longer clip, in order.
#[test]
fn a_run_without_a_store_tunes_and_writes_the_untuned_frames() {
    let clip = multi_scene_clip(30);

    let tuned = run_clip(clip.clone(), TuningMode::Auto, Some(30.0));
    let untuned = run_clip(clip, TuningMode::Off, None);
    let options = tuned_options();
    let layout = scene_clip_layout();
    let tuned_now = is_tuned(&options, layout);

    assert!(
        av_denoise::compilation_cache_dir().is_none(),
        "this test runs without a kernel cache"
    );
    assert!(tuned_now, "the workers' process remembers the tuned result");
    assert_eq!(frame_count(&tuned), 30);
    assert_eq!(tuned, untuned);
}

#[test]
fn cached_tuning_stores_nothing_and_writes_the_untuned_frames() {
    let cache = tempfile::tempdir().expect("the temporary directory is created");
    install_compilation_cache_at(cache.path()).expect("the cache installs");
    let clip = multi_scene_clip(30);

    let cached = run_clip(clip.clone(), TuningMode::Cached, None);
    let stored = stored_results(cache.path());
    let untuned = run_clip(clip, TuningMode::Off, None);
    let options = tuned_options();
    let layout = scene_clip_layout();
    let tuned_now = is_tuned(&options, layout);

    assert_eq!(stored, 0);
    assert!(!tuned_now);
    assert_eq!(frame_count(&cached), 30);
    assert_eq!(cached, untuned);
}

#[test]
fn tuning_an_input_with_tuning_off_decodes_it_and_skips() {
    let dir = tempfile::tempdir().expect("the temporary directory is created");
    let input = clip_file(dir.path(), 3);
    let options = temporal_opts();
    let request = TuneRequest {
        budget: BudgetRequest::Full,
        force: false,
    };

    let report = tune_input(&options, &input, &request).expect("the input decodes");

    let reason = "tuning is off".to_string();
    assert_eq!(report.outcome, TuneOutcome::Skipped(reason));
}

#[test]
fn tuning_an_input_without_frames_is_an_error() {
    let dir = tempfile::tempdir().expect("the temporary directory is created");
    let input = clip_file(dir.path(), 0);
    let options = temporal_opts();
    let request = TuneRequest {
        budget: BudgetRequest::Full,
        force: false,
    };

    let result = tune_input(&options, &input, &request);

    assert!(result.is_err());
}

#[test]
fn a_decode_error_among_the_first_frames_is_kept_in_order_and_skips_tuning() {
    let cache = tempfile::tempdir().expect("the temporary directory is created");
    install_compilation_cache_at(cache.path()).expect("the cache installs");
    let clip = multi_scene_clip(4);
    let reader = y4m_reader(clip);
    let mut opened = open_y4m(reader).expect("the clip opens");
    let (frames_tx, frames_rx) = crossbeam_channel::unbounded::<FrameMsg>();

    for index in 0..4 {
        if index == 2 {
            let decode_error = anyhow::anyhow!("corrupt packet");
            frames_tx.send(Err(decode_error)).expect("the receiver is alive");
        }

        let frame = opened
            .decoder
            .read_video_frame::<u8>()
            .expect("the clip has 4 frames");
        let shared = Arc::new(frame);
        let decoded = u8::into_decoded(shared);
        frames_tx.send(Ok(decoded)).expect("the receiver is alive");
    }

    drop(frames_tx);

    let options = tuned_options();
    let layout = scene_clip_layout();

    let prefix = tune_before_workers(&options, &opened.info, &frames_rx, 64, None);
    let left_over = frames_rx.iter().count();
    let stored = stored_results(cache.path());
    let tuned_now = is_tuned(&options, layout);

    let kinds: Vec<bool> = prefix.iter().map(|message| message.is_ok()).collect();
    assert_eq!(
        kinds,
        [true, true, false],
        "the error ends the prefix in stream order"
    );
    assert_eq!(left_over, 2, "frames after the error stay in the channel");
    assert_eq!(stored, 0);
    assert!(!tuned_now);
}
