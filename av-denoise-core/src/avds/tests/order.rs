use super::{Measured, file_of, sample_meta, scene, scenes_of};
use crate::avds::{AvdsSceneWriter, AvdsWriter, read_avds, rows_from_scenes};

#[test]
fn rows_written_in_order_across_out_of_order_scenes() {
    let scenes = scenes_of(&[10, 30, 5], Measured::ALL);
    let expected = rows_from_scenes(&scenes);
    let meta = sample_meta(0);
    let writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");
    let mut ordered = AvdsSceneWriter::new(writer);

    let [first, second, third] = <[_; 3]>::try_from(scenes).expect("three scenes");

    ordered.push(third).expect("the last scene is held");

    assert_eq!(ordered.held_scenes(), 1);
    assert_eq!(ordered.frames_written(), 0);

    ordered.push(first).expect("the first scene is written");

    assert_eq!(ordered.held_scenes(), 1);
    assert_eq!(ordered.frames_written(), 10);

    ordered.push(second).expect("the middle scene is written");

    assert_eq!(ordered.held_scenes(), 0);
    assert_eq!(ordered.frames_written(), 45);

    let bytes = ordered.finish().expect("the file finishes");
    let file = file_of(&bytes);
    let (read_meta, rows) = read_avds(file).expect("the file reads");
    let frames: Vec<u64> = rows.iter().map(|row| row.frame).collect();
    let in_order: Vec<u64> = (0..45).collect();

    assert_eq!(read_meta.frames, 45);
    assert_eq!(frames, in_order);
    assert_eq!(rows, expected);
}

#[test]
fn writing_rows_out_of_order_is_refused() {
    let scenes = scenes_of(&[3, 3], Measured::TABLE_ONLY);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(0);
    let mut writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");

    let late_start = writer.write_rows(&rows[3..]);

    assert!(late_start.is_err(), "rows must start at frame 0");

    writer.write_rows(&rows[..3]).expect("the first scene is written");
    let repeated = writer.write_rows(&rows[..3]);
    let skipped = writer.write_rows(&rows[4..]);

    assert!(repeated.is_err(), "rows already written are refused");
    assert!(skipped.is_err(), "a gap is refused");
}

#[test]
fn a_scene_never_reached_fails_the_finish() {
    let scenes = scenes_of(&[3, 3], Measured::TABLE_ONLY);
    let meta = sample_meta(0);
    let writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");
    let mut ordered = AvdsSceneWriter::new(writer);
    let [_, second] = <[_; 2]>::try_from(scenes).expect("two scenes");

    ordered.push(second).expect("the scene is held");
    let error = ordered.finish().expect_err("frames 0 to 2 are missing");

    assert!(error.to_string().contains("frames 0 to 2"), "got {error}");
}

#[test]
fn a_scene_given_twice_is_refused() {
    let scenes = scenes_of(&[3], Measured::TABLE_ONLY);
    let meta = sample_meta(0);
    let writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");
    let mut ordered = AvdsSceneWriter::new(writer);
    let first = scenes[0].clone();
    let repeat = scenes[0].clone();

    ordered.push(first).expect("the scene is written");
    let error = ordered.push(repeat).expect_err("the scene was already given");

    assert!(error.to_string().contains("frame 0"), "got {error}");
}

#[test]
fn an_overlapping_scene_is_refused() {
    let scenes = scenes_of(&[10, 30], Measured::TABLE_ONLY);
    let meta = sample_meta(0);
    let writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");
    let mut ordered = AvdsSceneWriter::new(writer);
    let [first, second] = <[_; 2]>::try_from(scenes).expect("two scenes");

    ordered.push(second).expect("the scene is held");
    let inside_held = scene(20, 5, Measured::TABLE_ONLY);
    let held_error = ordered.push(inside_held).expect_err("it overlaps the held scene");

    assert!(
        held_error.to_string().contains("held scene at frame 10"),
        "got {held_error}"
    );

    let reaching_held = scene(5, 6, Measured::TABLE_ONLY);
    let reach_error = ordered
        .push(reaching_held)
        .expect_err("it runs into the held scene");

    assert!(
        reach_error.to_string().contains("held scene at frame 10"),
        "got {reach_error}"
    );

    ordered.push(first).expect("the first scene is written");
    let inside_written = scene(5, 3, Measured::TABLE_ONLY);
    let written_error = ordered
        .push(inside_written)
        .expect_err("it overlaps written frames");

    assert!(
        written_error.to_string().contains("already written"),
        "got {written_error}"
    );
    assert_eq!(ordered.held_scenes(), 0);
}

#[test]
fn a_gap_of_one_frame_is_named_as_one_frame() {
    let meta = sample_meta(0);
    let writer = AvdsWriter::new(Vec::new(), &meta).expect("the writer starts");
    let mut ordered = AvdsSceneWriter::new(writer);
    let late = scene(1, 3, Measured::TABLE_ONLY);

    ordered.push(late).expect("the scene is held");
    let error = ordered.finish().expect_err("frame 0 is missing");

    assert!(error.to_string().contains("covers frame 0,"), "got {error}");
}
