#![cfg(feature = "vulkan")]

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use av_denoise::accelerate::Accelerator;
use av_denoise::{Algorithm, ChannelIntent, DenoisingMode, Device, Nl4dOptions, PlaneOptions};

use super::{SharedBuffer, grainy_clip, multi_scene_clip};
use crate::pipeline::run_with;
use crate::pipeline::source::open_y4m;

/// A directory unique to one test, removed when it drops.
struct TestDir(PathBuf);

impl TestDir {
    fn new(test_name: &str) -> Self {
        let name = format!("avd-grain-{}-{test_name}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).expect("the test directory is created");

        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn nl4d_opts(grain_export: bool) -> PlaneOptions {
    let algorithm = Nl4dOptions {
        grain_export,
        ..Nl4dOptions::default()
    };

    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::Nl4d(algorithm),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

fn run(bytes: Vec<u8>, workers: usize, table: Option<PathBuf>) -> Result<Vec<u8>, anyhow::Error> {
    let output = SharedBuffer::default();
    let writer = output.clone();
    let grain_export = table.is_some();
    let options = nl4d_opts(grain_export);
    let opener = move || {
        let cursor = Cursor::new(bytes);
        let reader: Box<dyn Read> = Box::new(cursor);
        open_y4m(reader)
    };

    run_with(&options, opener, workers, 1 << 30, false, writer, table)?;

    let written = output.0.lock().expect("buffer lock").clone();

    Ok(written)
}

#[test]
fn a_run_with_the_flag_writes_a_table() {
    let dir = TestDir::new("writes");
    let path = dir.path().join("grain.tbl");
    let clip = grainy_clip(40);

    run(clip, 2, Some(path.clone())).expect("the run succeeds");

    let text = std::fs::read_to_string(&path).expect("the table exists");
    let temporary = dir.path().join("grain.tbl.tmp");

    assert!(text.starts_with("filmgrn1\n"));
    assert!(text.contains("\nE "), "no entry was fitted, got {text}");
    assert!(!temporary.exists());
}

#[test]
fn the_flag_does_not_change_the_output() {
    let dir = TestDir::new("output");
    let path = dir.path().join("grain.tbl");
    let plain_clip = grainy_clip(40);
    let export_clip = grainy_clip(40);

    let without = run(plain_clip, 1, None).expect("plain run");
    let with = run(export_clip, 1, Some(path)).expect("export run");

    assert_eq!(without, with);
}

#[test]
fn worker_count_does_not_change_the_table() {
    let dir = TestDir::new("workers");
    let single = dir.path().join("single.tbl");
    let multi = dir.path().join("multi.tbl");
    let single_clip = grainy_clip(60);
    let multi_clip = grainy_clip(60);

    run(single_clip, 1, Some(single.clone())).expect("single worker");
    run(multi_clip, 3, Some(multi.clone())).expect("three workers");

    let single_text = std::fs::read_to_string(single).expect("single table");
    let multi_text = std::fs::read_to_string(multi).expect("multi table");
    let entries = single_text.matches("\nE ").count();

    assert!(entries >= 2, "expected an entry per scene, got {single_text}");
    assert_eq!(single_text, multi_text);
}

#[test]
fn failed_run_leaves_no_table() {
    let dir = TestDir::new("failed");
    let path = dir.path().join("grain.tbl");
    let clip = multi_scene_clip(0);

    let result = run(clip, 1, Some(path.clone()));
    let temporary = dir.path().join("grain.tbl.tmp");

    assert!(result.is_err());
    assert!(!path.exists());
    assert!(!temporary.exists());
}

#[test]
fn an_unwritable_path_fails_before_decoding() {
    let path = PathBuf::from("/nonexistent-directory/grain.tbl");
    let clip = multi_scene_clip(10);

    let err = run(clip, 1, Some(path)).expect_err("cannot write there");

    assert!(err.to_string().contains("grain table"), "got {err}");
}
