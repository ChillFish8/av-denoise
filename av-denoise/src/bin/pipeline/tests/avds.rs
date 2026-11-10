#![cfg(feature = "vulkan")]

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use av_denoise::avds::{AvdsFrame, AvdsMeta, PlanesMeasured, read_avds, scenes_from_rows};
use av_denoise::{
    Algorithm,
    ChannelIntent,
    ChromaLayout,
    DenoisingMode,
    FitOptions,
    Nl4dOptions,
    PlaneOptions,
    TuningMode,
    build_table,
};

use super::{SharedBuffer, TestDir, grainy_clip, multi_scene_clip};
use crate::cli::{Av1FgsArgs, GrainPlane, ReportArgs, SynthesizeArgs, SynthesizeKind, run_synthesize};
use crate::pipeline::source::open_y4m;
use crate::pipeline::{RunSettings, run_with};

const DIR_PREFIX: &str = "avd-avds";
const CLIP_FRAMES: usize = 60;

fn nl4d_opts(intent: ChannelIntent) -> PlaneOptions {
    let algorithm = Nl4dOptions {
        grain_data: true,
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

/// Where a run writes its grain outputs.
struct Outputs {
    avds: PathBuf,
    table: Option<PathBuf>,
    report: Option<PathBuf>,
}

fn run(clip: Vec<u8>, workers: usize, intent: ChannelIntent, outputs: Outputs) -> Result<(), anyhow::Error> {
    let options = nl4d_opts(intent);
    let opener = move || {
        let cursor = Cursor::new(clip);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };
    let settings = RunSettings {
        workers,
        frame_budget_bytes: 1 << 30,
        visible: false,
        grain_table: outputs.table,
        grain_report: outputs.report,
        avds: Some(outputs.avds),
        tune_budget: None,
        timing_report: false,
        shared_device: false,
    };

    let output = SharedBuffer::default();
    run_with(&options, opener, settings, output)?;

    Ok(())
}

fn read_file(path: &Path) -> (AvdsMeta, Vec<AvdsFrame>) {
    let file = File::open(path).expect("the grain data file exists");

    read_avds(file).expect("the grain data file reads")
}

#[test]
fn a_run_with_export_avds_writes_one_row_per_frame() {
    let dir = TestDir::new(DIR_PREFIX, "rows");
    let path = dir.path().join("run.avds");
    let clip = grainy_clip(CLIP_FRAMES);
    let outputs = Outputs {
        avds: path.clone(),
        table: None,
        report: None,
    };

    run(clip, 3, ChannelIntent::LumaChroma, outputs).expect("the run succeeds");

    let (meta, rows) = read_file(&path);
    let frames: Vec<u64> = rows.iter().map(|row| row.frame).collect();
    let in_order: Vec<u64> = (0..CLIP_FRAMES as u64).collect();
    let scene_starts = rows.iter().filter(|row| row.first_of_scene).count();
    let temporary = dir.path().join("run.avds.tmp");

    assert_eq!(frames, in_order);
    assert_eq!(meta.frames, CLIP_FRAMES as u64);
    assert_eq!(meta.frame_rate, (25, 1));
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
    let dir = TestDir::new(DIR_PREFIX, "luma-only");
    let path = dir.path().join("run.avds");
    let clip = grainy_clip(30);
    let outputs = Outputs {
        avds: path.clone(),
        table: None,
        report: None,
    };

    run(clip, 1, ChannelIntent::Luma, outputs).expect("the run succeeds");

    let (meta, rows) = read_file(&path);

    assert_eq!(rows.len(), 30);
    assert_eq!(meta.planes_measured, PlanesMeasured::Luma);
    assert!(rows.iter().all(|row| row.table.is_some()));
    assert!(rows.iter().all(|row| row.chroma_persistence.is_none()));
    assert!(rows.iter().all(|row| row.chroma_noise.is_none()));
    assert!(rows.iter().all(|row| row.chroma_colour.is_none()));
}

#[test]
fn the_file_rebuilds_the_runs_grain_table() {
    let dir = TestDir::new(DIR_PREFIX, "table");
    let path = dir.path().join("run.avds");
    let table_path = dir.path().join("grain.tbl");
    let clip = grainy_clip(CLIP_FRAMES);
    let outputs = Outputs {
        avds: path.clone(),
        table: Some(table_path.clone()),
        report: None,
    };

    run(clip, 2, ChannelIntent::LumaChroma, outputs).expect("the run succeeds");

    let (meta, rows) = read_file(&path);
    let scenes = scenes_from_rows(&rows);
    let options = FitOptions::default();
    let rebuilt = build_table(&scenes, meta.frame_rate, &options);
    let written = std::fs::read_to_string(&table_path).expect("the table exists");

    assert!(written.contains("\nE "), "no entry was fitted, got {written}");
    assert_eq!(rebuilt, written);
}

#[test]
fn synthesize_av1_fgs_matches_the_runs_table() {
    let dir = TestDir::new(DIR_PREFIX, "synthesize-table");
    let path = dir.path().join("run.avds");
    let table_path = dir.path().join("grain.tbl");
    let synthesized_path = dir.path().join("synthesized.tbl");
    let clip = grainy_clip(CLIP_FRAMES);
    let outputs = Outputs {
        avds: path.clone(),
        table: Some(table_path.clone()),
        report: None,
    };

    run(clip, 2, ChannelIntent::LumaChroma, outputs).expect("the run succeeds");

    let av1_fgs = Av1FgsArgs {
        input: path,
        output: synthesized_path.clone(),
        planes: vec![GrainPlane::Luma],
        fine_noise: 0.0,
    };
    let kind = SynthesizeKind::Av1Fgs(av1_fgs);
    let args = SynthesizeArgs { kind };
    run_synthesize(&args).expect("the table is synthesized");

    let written = std::fs::read(&table_path).expect("the run's table exists");
    let synthesized = std::fs::read(&synthesized_path).expect("the synthesized table exists");
    let text = String::from_utf8_lossy(&written);

    assert!(text.contains("\nE "), "no entry was fitted, got {text}");
    assert_eq!(synthesized, written);
}

#[test]
fn synthesize_report_matches_the_runs_report() {
    let dir = TestDir::new(DIR_PREFIX, "synthesize-report");
    let path = dir.path().join("run.avds");
    let report_path = dir.path().join("report.jsonl");
    let synthesized_path = dir.path().join("synthesized.jsonl");
    let clip = grainy_clip(CLIP_FRAMES);
    let outputs = Outputs {
        avds: path.clone(),
        table: None,
        report: Some(report_path.clone()),
    };

    run(clip, 3, ChannelIntent::LumaChroma, outputs).expect("the run succeeds");

    let report = ReportArgs {
        input: path,
        output: synthesized_path.clone(),
    };
    let kind = SynthesizeKind::Report(report);
    let args = SynthesizeArgs { kind };
    run_synthesize(&args).expect("the report is synthesized");

    let written = std::fs::read(&report_path).expect("the run's report exists");
    let synthesized = std::fs::read(&synthesized_path).expect("the synthesized report exists");
    let lines = written.iter().filter(|&&byte| byte == b'\n').count();

    assert_eq!(lines, CLIP_FRAMES);
    assert_eq!(synthesized, written);
}

#[test]
fn synthesize_report_of_a_luma_only_run_has_null_chroma() {
    let dir = TestDir::new(DIR_PREFIX, "synthesize-luma-only");
    let path = dir.path().join("run.avds");
    let report_path = dir.path().join("report.jsonl");
    let synthesized_path = dir.path().join("synthesized.jsonl");
    let clip = grainy_clip(30);
    let outputs = Outputs {
        avds: path.clone(),
        table: None,
        report: Some(report_path.clone()),
    };

    run(clip, 1, ChannelIntent::Luma, outputs).expect("the run succeeds");

    let report = ReportArgs {
        input: path,
        output: synthesized_path.clone(),
    };
    let kind = SynthesizeKind::Report(report);
    let args = SynthesizeArgs { kind };
    run_synthesize(&args).expect("the report is synthesized");

    let written = std::fs::read_to_string(&report_path).expect("the run's report exists");
    let synthesized = std::fs::read_to_string(&synthesized_path).expect("the synthesized report exists");
    let null_chroma = synthesized
        .lines()
        .all(|line| line.contains("\"strength_cb\":null") && line.contains("\"strength_cr\":null"));

    assert_eq!(synthesized, written);
    assert!(null_chroma, "got {synthesized}");
}

#[test]
fn a_failed_run_leaves_no_avds_file() {
    let dir = TestDir::new(DIR_PREFIX, "failed");
    let path = dir.path().join("run.avds");
    let clip = multi_scene_clip(0);
    let outputs = Outputs {
        avds: path.clone(),
        table: None,
        report: None,
    };

    let result = run(clip, 1, ChannelIntent::LumaChroma, outputs);
    let temporary = dir.path().join("run.avds.tmp");

    assert!(result.is_err());
    assert!(!path.exists());
    assert!(!temporary.exists());
}

#[test]
fn an_unwritable_avds_path_fails_before_decoding() {
    let path = PathBuf::from("/nonexistent-directory/run.avds");
    let clip = multi_scene_clip(10);
    let outputs = Outputs {
        avds: path,
        table: None,
        report: None,
    };

    let error = run(clip, 1, ChannelIntent::LumaChroma, outputs).expect_err("cannot write there");

    assert!(error.to_string().contains("grain data file"), "got {error}");
}
