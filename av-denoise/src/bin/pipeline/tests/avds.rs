#![cfg(feature = "vulkan")]

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use av_denoise::avds::{AvdsFrame, AvdsMeta, PlanesMeasured, read_avds};
use av_denoise::{
    Algorithm,
    ChannelIntent,
    ChromaLayout,
    DenoisingMode,
    FitOptions,
    Nl4dOptions,
    PlaneOptions,
    SceneGrain,
    TuningMode,
    build_table,
    frame_rows,
};

use super::{SharedBuffer, grainy_clip, multi_scene_clip, pattern, y4m_decoder, y4m_reader};
use crate::cli::{
    Av1FgsArgs,
    GrainPlane,
    ReportArgs,
    SynthesizeArgs,
    SynthesizeKind,
    report_text,
    run_synthesize,
};
use crate::pipeline::convert::SourcePixel;
use crate::pipeline::decode::FrameMsg;
use crate::pipeline::source::open_y4m;
use crate::pipeline::worker::{WorkerSettings, spawn_workers};
use crate::pipeline::{RunSettings, dispatch, run_with};

const CLIP_FRAMES: usize = 60;
/// The frame rate of every test clip.
const FRAME_RATE: (u64, u64) = (25, 1);
/// Mid-grey luma and neutral chroma.
const MID_CODE: i16 = 128;
/// Codes either way of [MID_CODE] the grain reaches.
const GRAIN_REACH: u8 = 4;

fn nl4d_opts(intent: ChannelIntent, grain_data: bool) -> PlaneOptions {
    let algorithm = Nl4dOptions {
        grain_data,
        ..Nl4dOptions::default()
    };

    PlaneOptions {
        accelerators: vec![av_denoise::testing::test_accelerator()],
        device: av_denoise::testing::test_device(),
        intent,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::Nl4d(algorithm),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

/// Runs the pipeline over `clip`, writing the grain data file to `avds`, and returns the output.
fn run(
    clip: Vec<u8>,
    workers: usize,
    intent: ChannelIntent,
    avds: Option<PathBuf>,
) -> Result<Vec<u8>, anyhow::Error> {
    let grain_data = avds.is_some();
    let options = nl4d_opts(intent, grain_data);
    let opener = move || {
        let cursor = Cursor::new(clip);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };
    let settings = RunSettings {
        workers,
        frame_budget_bytes: 1 << 30,
        visible: false,
        avds,
        tune_budget: None,
        timing_report: false,
        shared_device: false,
    };

    let output = SharedBuffer::default();
    let writer = output.clone();
    run_with(&options, opener, settings, writer)?;

    let written = output.0.lock().expect("buffer lock").clone();

    Ok(written)
}

/// Denoises `clip` on one worker and returns every scene's grain as the worker sent it.
fn measured_scenes(clip: Vec<u8>, intent: ChannelIntent) -> Vec<SceneGrain> {
    let options = nl4d_opts(intent, true);
    let reader = y4m_reader(clip);
    let mut opened = open_y4m(reader).expect("the clip opens");
    let (frames_tx, frames_rx) = crossbeam_channel::unbounded::<FrameMsg>();

    let decoder = y4m_decoder(&mut opened);
    while let Ok(frame) = decoder.read_video_frame::<u8>() {
        let shared = Arc::new(frame);
        let decoded = u8::into_decoded(shared);
        frames_tx.send(Ok(decoded)).expect("the receiver is alive");
    }

    drop(frames_tx);

    let (grain_tx, grain_rx) = crossbeam_channel::unbounded::<SceneGrain>();
    let settings = WorkerSettings::for_run(1, false, false);
    let layout = opened.info.layout;
    let (job_tx, handles, output_rx) = spawn_workers(&options, layout, 1, settings, Some(&grain_tx));
    drop(grain_tx);

    let frames = frames_rx.iter();
    dispatch::<u8>(frames, &opened.info, &job_tx).expect("every frame is staged");
    drop(job_tx);

    for handle in handles {
        let joined = handle.join().expect("the worker does not panic");
        joined.expect("the worker succeeds");
    }

    drop(output_rx);

    grain_rx.iter().collect()
}

fn read_file(path: &Path) -> (AvdsMeta, Vec<AvdsFrame>) {
    let file = File::open(path).expect("the grain data file exists");

    read_avds(file).expect("the grain data file reads")
}

/// Fresh grain of up to [GRAIN_REACH] codes around [MID_CODE], `len` samples long.
fn grain_plane(seed: u32, len: usize) -> Vec<u8> {
    let samples = pattern(seed, len);
    let span = 2 * GRAIN_REACH + 1;

    samples
        .iter()
        .map(|&sample| {
            let offset = (sample % span) as i16 - GRAIN_REACH as i16;
            (MID_CODE + offset) as u8
        })
        .collect()
}

/// A 320x192 8-bit 4:2:0 clip of one flat scene under fresh grain on every plane and frame.
fn chroma_grainy_clip(frames: u32) -> Vec<u8> {
    let width = 320;
    let height = 192;
    let luma_len = width * height;
    let chroma_len = luma_len / 4;
    let frame_rate = y4m::Ratio::new(25, 1);
    let mut bytes = Vec::new();
    let mut encoder = y4m::encode(width, height, frame_rate)
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("header should write");

    for index in 0..frames {
        let luma_plane = grain_plane(3 * index + 101, luma_len);
        let cb_plane = grain_plane(3 * index + 102, chroma_len);
        let cr_plane = grain_plane(3 * index + 103, chroma_len);
        let frame = y4m::Frame::new([&luma_plane, &cb_plane, &cr_plane], None);

        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

fn synthesize_report(input: PathBuf, output: PathBuf) {
    let report = ReportArgs { input, output };
    let kind = SynthesizeKind::Report(report);
    let args = SynthesizeArgs { kind };

    run_synthesize(&args).expect("the report is synthesized");
}

#[test]
fn a_run_with_export_avds_writes_one_row_per_frame() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let clip = grainy_clip(CLIP_FRAMES);

    run(clip, 3, ChannelIntent::LumaChroma, Some(path.clone())).expect("the run succeeds");

    let (meta, rows) = read_file(&path);
    let frames: Vec<u64> = rows.iter().map(|row| row.frame).collect();
    let in_order: Vec<u64> = (0..CLIP_FRAMES as u64).collect();
    let scene_starts = rows.iter().filter(|row| row.first_of_scene).count();
    let temporary = dir.path().join("run.avds.tmp");

    assert_eq!(frames, in_order);
    assert_eq!(meta.frames, CLIP_FRAMES as u64);
    assert_eq!(meta.frame_rate, FRAME_RATE);
    assert_eq!((meta.width, meta.height, meta.bit_depth), (192, 128, 8));
    assert_eq!(meta.subsampling, ChromaLayout::Yuv420);
    assert_eq!(meta.planes_measured, PlanesMeasured::LumaChroma);
    assert!(meta.writer.starts_with("av-denoise "), "got {}", meta.writer);
    assert!(scene_starts >= 2, "expected several scenes, got {scene_starts}");
    assert!(rows.iter().all(|row| row.table.is_some()));
    assert!(rows.iter().all(|row| row.persistence.is_some()));
    assert!(rows.iter().all(|row| row.chroma_persistence.is_some()));
    assert!(!temporary.exists());
}

#[test]
fn a_luma_only_run_leaves_the_chroma_columns_null() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let clip = grainy_clip(30);

    run(clip, 1, ChannelIntent::Luma, Some(path.clone())).expect("the run succeeds");

    let (meta, rows) = read_file(&path);

    assert_eq!(rows.len(), 30);
    assert_eq!(meta.planes_measured, PlanesMeasured::Luma);
    assert!(rows.iter().all(|row| row.table.is_some()));
    assert!(rows.iter().all(|row| row.chroma_persistence.is_none()));
    assert!(rows.iter().all(|row| row.chroma_noise.is_none()));
    assert!(rows.iter().all(|row| row.chroma_colour.is_none()));
}

#[test]
fn export_avds_does_not_change_the_output() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let plain_clip = grainy_clip(40);
    let export_clip = grainy_clip(40);

    let without = run(plain_clip, 1, ChannelIntent::LumaChroma, None).expect("plain run");
    let with = run(export_clip, 1, ChannelIntent::LumaChroma, Some(path)).expect("export run");

    assert_eq!(without, with);
}

#[test]
fn synthesize_av1_fgs_matches_the_measured_table() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let synthesized_path = dir.path().join("synthesized.tbl");
    let run_clip = grainy_clip(CLIP_FRAMES);
    let measured_clip = grainy_clip(CLIP_FRAMES);

    run(run_clip, 3, ChannelIntent::LumaChroma, Some(path.clone())).expect("the run succeeds");
    let measured = measured_scenes(measured_clip, ChannelIntent::LumaChroma);

    let av1_fgs = Av1FgsArgs {
        input: path,
        output: synthesized_path.clone(),
        planes: vec![GrainPlane::Luma],
        fine_noise: 0.0,
    };
    let kind = SynthesizeKind::Av1Fgs(av1_fgs);
    let args = SynthesizeArgs { kind };
    run_synthesize(&args).expect("the table is synthesized");

    let options = FitOptions::default();
    let expected = build_table(&measured, FRAME_RATE, &options);
    let synthesized = std::fs::read_to_string(&synthesized_path).expect("the synthesized table exists");
    let entries = expected.matches("\nE ").count();

    assert!(entries >= 2, "expected an entry per scene, got {expected}");
    assert_eq!(synthesized, expected);
}

#[test]
fn synthesize_report_matches_the_measured_report() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let synthesized_path = dir.path().join("synthesized.jsonl");
    let run_clip = grainy_clip(CLIP_FRAMES);
    let measured_clip = grainy_clip(CLIP_FRAMES);

    run(run_clip, 3, ChannelIntent::LumaChroma, Some(path.clone())).expect("the run succeeds");
    let measured = measured_scenes(measured_clip, ChannelIntent::LumaChroma);

    synthesize_report(path, synthesized_path.clone());

    let measured_rows = frame_rows(&measured);
    let expected = report_text(&measured_rows).expect("the rows encode");
    let synthesized = std::fs::read_to_string(&synthesized_path).expect("the synthesized report exists");

    assert_eq!(synthesized.lines().count(), CLIP_FRAMES);
    assert_eq!(synthesized, expected);
}

#[test]
fn synthesize_report_of_a_luma_only_run_has_null_chroma() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let synthesized_path = dir.path().join("synthesized.jsonl");
    let clip = chroma_grainy_clip(12);

    run(clip, 1, ChannelIntent::Luma, Some(path.clone())).expect("the run succeeds");
    synthesize_report(path, synthesized_path.clone());

    let synthesized = std::fs::read_to_string(&synthesized_path).expect("the synthesized report exists");
    let rows: Vec<serde_json::Value> = synthesized
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is JSON"))
        .collect();

    assert_eq!(rows.len(), 12);
    assert!(rows.iter().any(|row| row["strength"].is_number()));
    assert!(rows.iter().all(|row| row["strength_cb"].is_null()));
    assert!(rows.iter().all(|row| row["strength_cr"].is_null()));
}

#[test]
fn synthesize_report_of_a_run_denoising_both_has_chroma_strength() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let synthesized_path = dir.path().join("synthesized.jsonl");
    let clip = chroma_grainy_clip(12);

    run(clip, 1, ChannelIntent::LumaChroma, Some(path.clone())).expect("the run succeeds");
    synthesize_report(path, synthesized_path.clone());

    let synthesized = std::fs::read_to_string(&synthesized_path).expect("the synthesized report exists");
    let rows: Vec<serde_json::Value> = synthesized
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is JSON"))
        .collect();

    assert_eq!(rows.len(), 12);
    assert!(rows.iter().all(|row| row["strength_cb"].is_number()));
    assert!(rows.iter().all(|row| row["strength_cr"].is_number()));
}

#[test]
fn a_failed_run_leaves_no_avds_file() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("run.avds");
    let clip = multi_scene_clip(0);

    let result = run(clip, 1, ChannelIntent::LumaChroma, Some(path.clone()));
    let temporary = dir.path().join("run.avds.tmp");

    assert!(result.is_err());
    assert!(!path.exists());
    assert!(!temporary.exists());
}

#[test]
fn an_unwritable_avds_path_fails_before_decoding() {
    let path = PathBuf::from("/nonexistent-directory/run.avds");
    let clip = multi_scene_clip(10);

    let error = run(clip, 1, ChannelIntent::LumaChroma, Some(path)).expect_err("cannot write there");

    assert!(error.to_string().contains("grain data file"), "got {error}");
}
