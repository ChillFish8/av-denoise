use std::path::PathBuf;

use av_denoise::{GrainChunk, SceneGrain};

use super::TestDir;
use crate::pipeline::grain_report::ReportPath;

/// The persistence sums each frame record holds, 3 for each of the 8 lags.
const FRAME_RECORD_LEN: usize = 24;

const DIR_PREFIX: &str = "avd-report";

fn scene(first_frame: u64, frames: u32) -> SceneGrain {
    let record_len = frames as usize * FRAME_RECORD_LEN;
    let chunk = GrainChunk {
        frames,
        frame_records: vec![0.0; record_len],
        ..GrainChunk::empty()
    };

    SceneGrain {
        first_frame,
        chunks: vec![chunk],
        chroma_chunks: Vec::new(),
    }
}

#[test]
fn the_report_has_one_line_per_frame_in_order() {
    let dir = TestDir::new(DIR_PREFIX, "lines");
    let path = dir.path().join("report.jsonl");
    let later = scene(3, 2);
    let earlier = scene(0, 3);
    let scenes = [later, earlier];

    let report = ReportPath::create(&path).expect("the report is created");
    report.write(&scenes).expect("the report is written");

    let text = std::fs::read_to_string(&path).expect("the report exists");
    let lines: Vec<&str> = text.lines().collect();

    assert!(text.ends_with('\n'));
    assert_eq!(lines.len(), 5);

    for (index, line) in lines.iter().enumerate() {
        let value: serde_json::Value = serde_json::from_str(line).expect("each line is JSON");
        let object = value.as_object().expect("each line is an object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let persistence = object["persistence"].as_array().expect("persistence is a list");

        let expected_keys = [
            "cells",
            "frame",
            "persistence",
            "scene",
            "strength",
            "strength_cb",
            "strength_cr",
        ];
        assert_eq!(keys, expected_keys);
        assert_eq!(object["frame"], index as u64);
        assert_eq!(persistence.len(), 8);
    }
}

#[test]
fn a_missing_parent_folder_fails_at_create() {
    let path = PathBuf::from("/nonexistent-directory/report.jsonl");

    let result = ReportPath::create(&path);
    let error = result.err().expect("cannot create a file there");

    assert!(error.to_string().contains("grain report"), "got {error}");
}

#[test]
fn dropping_an_unwritten_report_removes_its_temporary_file() {
    let dir = TestDir::new(DIR_PREFIX, "dropped");
    let path = dir.path().join("report.jsonl");
    let temporary = dir.path().join("report.jsonl.tmp");

    let report = ReportPath::create(&path).expect("the report is created");
    assert!(temporary.exists());

    drop(report);

    assert!(!temporary.exists());
    assert!(!path.exists());
}

#[test]
fn a_written_report_leaves_no_temporary_file() {
    let dir = TestDir::new(DIR_PREFIX, "written");
    let path = dir.path().join("report.jsonl");
    let temporary = dir.path().join("report.jsonl.tmp");
    let scenes = [scene(0, 2)];

    let report = ReportPath::create(&path).expect("the report is created");
    report.write(&scenes).expect("the report is written");

    assert!(path.exists());
    assert!(!temporary.exists());
}

#[cfg(feature = "vulkan")]
mod runs {
    use std::io::{Cursor, Read};
    use std::path::Path;

    use av_denoise::{Algorithm, ChannelIntent, DenoisingMode, Nl4dOptions, PlaneOptions, TuningMode};

    use super::DIR_PREFIX;
    use crate::pipeline::source::open_y4m;
    use crate::pipeline::tests::{SharedBuffer, TestDir, pattern};
    use crate::pipeline::{RunSettings, run_with};

    const CLIP_WIDTH: usize = 320;
    const CLIP_HEIGHT: usize = 192;
    const CLIP_FRAMES: usize = 12;
    /// Luma samples per chroma sample in 4:2:0.
    const LUMA_PER_CHROMA: usize = 4;
    /// Mid-grey luma and neutral chroma.
    const MID_CODE: i16 = 128;
    /// Codes either way of [MID_CODE] the grain reaches.
    const GRAIN_REACH: u8 = 4;

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

    /// An 8-bit 4:2:0 clip of one flat scene under fresh grain on every plane and frame.
    fn chroma_grainy_clip() -> Vec<u8> {
        let luma_len = CLIP_WIDTH * CLIP_HEIGHT;
        let chroma_len = luma_len / LUMA_PER_CHROMA;
        let frame_rate = y4m::Ratio::new(25, 1);
        let mut bytes = Vec::new();
        let mut encoder = y4m::encode(CLIP_WIDTH, CLIP_HEIGHT, frame_rate)
            .with_colorspace(y4m::Colorspace::C420)
            .write_header(&mut bytes)
            .expect("header should write");

        for index in 0..CLIP_FRAMES as u32 {
            let luma_plane = grain_plane(3 * index + 101, luma_len);
            let cb_plane = grain_plane(3 * index + 102, chroma_len);
            let cr_plane = grain_plane(3 * index + 103, chroma_len);
            let frame = y4m::Frame::new([&luma_plane, &cb_plane, &cr_plane], None);

            encoder.write_frame(&frame).expect("frame should write");
        }

        bytes
    }

    fn report_options(intent: ChannelIntent) -> PlaneOptions {
        let algorithm = Nl4dOptions {
            grain_report: true,
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

    /// Runs the clip with the report on and returns each line's JSON object.
    fn report_rows(intent: ChannelIntent, report: &Path) -> Vec<serde_json::Value> {
        let options = report_options(intent);
        let clip = chroma_grainy_clip();
        let opener = move || {
            let cursor = Cursor::new(clip);
            let reader: Box<dyn Read> = Box::new(cursor);
            open_y4m(reader)
        };
        let settings = RunSettings {
            workers: 1,
            frame_budget_bytes: 1 << 30,
            visible: false,
            grain_table: None,
            grain_report: Some(report.to_path_buf()),
            tune_budget: None,
            timing_report: false,
            shared_device: false,
        };

        let output = SharedBuffer::default();
        run_with(&options, opener, settings, output).expect("the run succeeds");

        let text = std::fs::read_to_string(report).expect("the report exists");
        text.lines()
            .map(|line| serde_json::from_str(line).expect("each line is JSON"))
            .collect()
    }

    #[test]
    fn a_luma_only_run_reports_no_chroma_strength() {
        let dir = TestDir::new(DIR_PREFIX, "luma-only");
        let path = dir.path().join("report.jsonl");

        let rows = report_rows(ChannelIntent::Luma, &path);

        assert_eq!(rows.len(), CLIP_FRAMES);
        assert!(rows.iter().any(|row| row["strength"].is_number()));
        assert!(rows.iter().all(|row| row["strength_cb"].is_null()));
        assert!(rows.iter().all(|row| row["strength_cr"].is_null()));
    }

    #[test]
    fn a_run_denoising_both_reports_chroma_strength() {
        let dir = TestDir::new(DIR_PREFIX, "both");
        let path = dir.path().join("report.jsonl");

        let rows = report_rows(ChannelIntent::LumaChroma, &path);

        assert_eq!(rows.len(), CLIP_FRAMES);
        assert!(rows.iter().all(|row| row["strength_cb"].is_number()));
        assert!(rows.iter().all(|row| row["strength_cr"].is_number()));
    }
}
