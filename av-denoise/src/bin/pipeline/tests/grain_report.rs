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

        assert_eq!(keys, ["cells", "frame", "persistence", "scene", "strength"]);
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
