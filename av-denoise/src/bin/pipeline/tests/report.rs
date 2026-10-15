#[cfg(feature = "vulkan")]
use std::io::{Cursor, Read};
use std::time::Duration;

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
use av_denoise::accelerate::Accelerator;
use av_denoise::{PlanarTiming, PlaneTiming};

#[cfg(feature = "vulkan")]
use super::{SharedBuffer, multi_scene_clip, temporal_opts};
use crate::pipeline::report::{RunSummary, WorkerTiming, busy_share, busy_streams, format_report};
#[cfg(feature = "vulkan")]
use crate::pipeline::source::open_y4m;
#[cfg(feature = "vulkan")]
use crate::pipeline::{RunSettings, run_with};

fn plane(blocked_ms: u64, submit_ms: u64, busy_ms: Option<u64>, frames: u64) -> PlaneTiming {
    PlaneTiming {
        blocked: Duration::from_millis(blocked_ms),
        submit: Duration::from_millis(submit_ms),
        device_busy: busy_ms.map(Duration::from_millis),
        frames,
    }
}

fn summary(luma_busy: Option<u64>) -> RunSummary {
    let planes = PlanarTiming {
        luma: Some(plane(100, 400, luma_busy, 50)),
        chroma: Some(plane(50, 200, Some(250), 50)),
        yuv: None,
    };
    let worker = WorkerTiming {
        wall: Duration::from_secs(1),
        setup: Duration::from_millis(100),
        idle: Duration::from_millis(100),
        planes,
        accelerator: None,
    };

    RunSummary {
        wall: Duration::from_secs(2),
        frames: 50,
        workers: vec![worker],
    }
}

#[test]
fn the_report_gives_per_frame_host_time_and_busy_share() {
    let measured = summary(Some(1000));
    let report = format_report(&measured);

    assert!(report.contains("50 frames in 2.00 s (25.00 fps)"), "{report}");
    assert!(report.contains("luma"), "{report}");
    assert!(report.contains("2.00 ms"), "luma blocked per frame: {report}");
    assert!(report.contains("8.00 ms"), "luma submit per frame: {report}");
    assert!(report.contains("50.0%"), "luma busy share: {report}");
    assert!(report.contains("12.5%"), "chroma busy share: {report}");
}

#[test]
fn a_plane_without_device_time_reports_not_available() {
    let measured = summary(None);
    let report = format_report(&measured);
    let luma_line = report
        .lines()
        .find(|line| line.trim_start().starts_with("luma"))
        .expect("luma row");

    assert!(luma_line.contains("n/a"), "{report}");
}

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "rocm"))]
fn two_workers_on(accelerator: Option<Accelerator>) -> RunSummary {
    let mut measured = summary(Some(1000));
    measured.workers[0].accelerator = accelerator;

    let second = measured.workers[0].clone();
    measured.workers.push(second);
    measured
}

#[test]
fn a_single_worker_heading_is_singular() {
    let measured = summary(None);
    let report = format_report(&measured);

    assert!(report.starts_with("Timing report, 1 worker, "), "{report}");
}

#[test]
fn busy_share_is_the_mean_over_every_stream() {
    let busy = Some(Duration::from_millis(3000));
    let wall = Duration::from_secs(2);

    let one_stream = busy_share(busy, wall, 1);
    let two_streams = busy_share(busy, wall, 2);

    assert_eq!(one_stream, "150.0%");
    assert_eq!(two_streams, "75.0%");
}

#[test]
fn workers_on_a_shared_queue_count_as_one_stream() {
    let streams = busy_streams(true, 3);

    assert_eq!(streams, 1);
}

#[test]
fn workers_on_separate_streams_count_one_stream_each() {
    let streams = busy_streams(false, 3);

    assert_eq!(streams, 3);
}

#[cfg(feature = "vulkan")]
#[test]
fn workers_on_a_shared_queue_share_one_busy_figure() {
    let measured = two_workers_on(Some(Accelerator::Vulkan));
    let report = format_report(&measured);

    assert!(report.starts_with("Timing report, 2 workers, "), "{report}");
    assert!(report.contains("100.0%"), "luma busy over one queue: {report}");
    assert!(report.contains("25.0%"), "chroma busy over one queue: {report}");
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[test]
fn workers_on_separate_streams_report_the_mean_busy_share() {
    #[cfg(feature = "rocm")]
    let accelerator = Accelerator::Rocm;
    #[cfg(not(feature = "rocm"))]
    let accelerator = Accelerator::Cuda;

    let measured = two_workers_on(Some(accelerator));
    let report = format_report(&measured);

    assert!(report.contains("50.0%"), "luma busy per stream: {report}");
    assert!(report.contains("12.5%"), "chroma busy per stream: {report}");
    assert!(report.contains("the mean over the workers' streams"), "{report}");
}

#[test]
fn unaccounted_time_is_what_the_parts_leave() {
    let measured = summary(None);
    let worker = &measured.workers[0];

    assert_eq!(worker.denoising(), Duration::from_millis(750));
    assert_eq!(worker.unaccounted(), Duration::from_millis(50));
}

#[cfg(feature = "vulkan")]
fn run_clip(bytes: Vec<u8>, workers: usize, timing_report: bool) -> (Vec<u8>, RunSummary) {
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
        frame_budget_bytes: 1 << 30,
        visible: false,
        grain_table: None,
        tune_budget: None,
        timing_report,
    };

    let summary = run_with(&options, opener, settings, writer).expect("the run succeeds");
    let written = output.0.lock().expect("buffer lock").clone();

    (written, summary)
}

#[cfg(feature = "vulkan")]
#[test]
fn every_workers_parts_add_up_to_its_wall_time() {
    let clip = multi_scene_clip(30);
    let (_, summary) = run_clip(clip, 1, true);

    assert_eq!(summary.frames, 30);

    let worker = &summary.workers[0];
    let margin = worker.wall / 10 + Duration::from_millis(50);
    assert!(worker.unaccounted() <= margin, "{worker:?}");

    let covered = worker.setup + worker.idle + worker.denoising();
    assert!(covered <= worker.wall + Duration::from_millis(5), "{worker:?}");
}

#[cfg(feature = "vulkan")]
#[test]
fn the_timing_report_leaves_the_output_unchanged() {
    let clip = multi_scene_clip(30);
    let plain_clip = clip.clone();
    let (plain, _) = run_clip(plain_clip, 2, false);
    let (reported, _) = run_clip(clip, 2, true);

    assert_eq!(reported, plain);
}
