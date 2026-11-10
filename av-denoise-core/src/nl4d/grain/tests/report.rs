use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{CHROMA_RECORD_LEN, FRAME_RECORD_LEN, LAG_RECORD_LEN, PERSISTENCE_LAGS};
use crate::nl4d::grain::report::{FrameGrain, frame_rows};
use crate::nl4d::grain::segment::SceneGrain;

fn chunk_with<const LEN: usize>(records: &[[f64; LEN]]) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = records.len() as u32;
    chunk.frame_records = records.iter().flatten().copied().collect();

    chunk
}

fn scene_of(first_frame: u64, chunks: Vec<GrainChunk>) -> SceneGrain {
    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks: Vec::new(),
    }
}

fn chroma_scene(luma_chunks: Vec<GrainChunk>, chroma_chunks: Vec<GrainChunk>) -> SceneGrain {
    SceneGrain {
        first_frame: 0,
        chunks: luma_chunks,
        chroma_chunks,
    }
}

/// A chroma record with `cells` cells in each plane, Cb at `cb_codes` and Cr at `cr_codes` std.
fn chroma_record(cells: f64, cb_codes: f64, cr_codes: f64) -> [f64; CHROMA_RECORD_LEN] {
    let cb_variance = (cb_codes / 255.0) * (cb_codes / 255.0);
    let cr_variance = (cr_codes / 255.0) * (cr_codes / 255.0);

    [cells, cells * cb_variance, 0.0, cells, cells * cr_variance, 0.0]
}

fn scene_with(first_frame: u64, records: &[[f64; FRAME_RECORD_LEN]]) -> SceneGrain {
    let chunk = chunk_with(records);

    scene_of(first_frame, vec![chunk])
}

/// Fills `(lag, cells, auto, cross)` entries and leaves every other lag zero.
fn record_with(lags: &[(usize, f64, f64, f64)]) -> [f64; FRAME_RECORD_LEN] {
    let mut record = [0.0; FRAME_RECORD_LEN];
    for &(lag, cells, auto_sum, cross_sum) in lags {
        let start = (lag - 1) * LAG_RECORD_LEN;
        record[start] = cells;
        record[start + 1] = auto_sum;
        record[start + 2] = cross_sum;
    }

    record
}

const VARIANCE: f64 = (2.0 / 255.0) * (2.0 / 255.0);

fn empty_record() -> [f64; FRAME_RECORD_LEN] {
    [0.0; FRAME_RECORD_LEN]
}

fn lag_one_record(cells: f64, cross_share: f64) -> [f64; FRAME_RECORD_LEN] {
    let auto_sum = cells * VARIANCE;

    record_with(&[(1, cells, auto_sum, auto_sum * cross_share)])
}

fn three_frame_records() -> [[f64; FRAME_RECORD_LEN]; 3] {
    [
        empty_record(),
        lag_one_record(1000.0, 0.2),
        lag_one_record(1000.0, 0.4),
    ]
}

fn assert_near(reading: Option<f64>, expected: f64) {
    let value = reading.expect("the reading should be present");
    assert!((value - expected).abs() < 1e-9, "{value} is not near {expected}");
}

fn expected_null_row(frame: u64) -> FrameGrain {
    FrameGrain {
        frame,
        scene: 0,
        cells: 0,
        strength: None,
        strength_cb: None,
        strength_cr: None,
        persistence: [None; PERSISTENCE_LAGS],
    }
}

#[test]
fn a_row_pools_the_pair_behind_and_the_pair_ahead() {
    let records = three_frame_records();
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[1].cells, 2000);
    assert_near(rows[1].strength, 2.0);
    assert_near(rows[1].persistence[0], 0.3);
}

#[test]
fn the_first_frame_of_a_scene_reads_from_the_frames_ahead() {
    let records = three_frame_records();
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[0].cells, 1000);
    assert_near(rows[0].persistence[0], 0.2);
}

#[test]
fn the_last_frame_of_a_scene_reads_from_the_frames_behind() {
    let records = three_frame_records();
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[2].cells, 1000);
    assert_near(rows[2].persistence[0], 0.4);
}

#[test]
fn a_far_lag_reads_the_record_that_many_frames_ahead() {
    let mut records = [empty_record(); 9];
    records[8] = record_with(&[(8, 800.0, 800.0 * VARIANCE, 80.0 * VARIANCE)]);
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_near(rows[0].persistence[7], 0.1);
    assert_near(rows[8].persistence[7], 0.1);
    for row in &rows[1..8] {
        assert_eq!(row.persistence[7], None);
    }
}

#[test]
fn a_window_never_reaches_into_the_next_scene() {
    let first_records = [empty_record(), lag_one_record(1000.0, 0.5)];
    let second_records = [lag_one_record(1000.0, 0.5), empty_record()];
    let first = scene_with(0, &first_records);
    let second = scene_with(2, &second_records);

    let rows = frame_rows(&[first, second]);

    assert_eq!(rows[1].cells, 1000);
    assert_eq!(rows[2].cells, 1000);
    assert_eq!(rows[3].cells, 0);
}

#[test]
fn a_scene_shorter_than_nine_frames_never_reads_the_long_lag() {
    let mut records = [lag_one_record(1000.0, 0.5); 5];
    records[0] = empty_record();
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|row| row.persistence[7].is_none()));
    assert!(rows.iter().all(|row| row.persistence[0].is_some()));
}

#[test]
fn a_single_frame_scene_gives_one_null_row() {
    let scene = scene_with(7, &[empty_record()]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows, vec![expected_null_row(7)]);
}

#[test]
fn too_few_cells_over_both_sides_gives_a_null_reading_with_its_count() {
    let records = [
        empty_record(),
        lag_one_record(99.0, 0.5),
        lag_one_record(100.0, 0.5),
    ];
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[1].cells, 199);
    assert_eq!(rows[1].strength, None);
    assert_eq!(rows[1].persistence[0], None);
}

#[test]
fn a_negative_cross_sum_reads_as_zero_persistence() {
    let record = record_with(&[(1, 1000.0, 1.0, -0.5)]);
    let scene = scene_with(0, &[record]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[0].persistence[0], Some(0.0));
}

#[test]
fn a_cross_sum_above_the_auto_sum_reads_as_one() {
    let record = record_with(&[(1, 1000.0, 1.0, 1.5)]);
    let scene = scene_with(0, &[record]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows[0].persistence[0], Some(1.0));
}

#[test]
fn rows_follow_frame_order_across_unsorted_scenes() {
    let later = scene_with(48, &[empty_record(); 2]);
    let earlier = scene_with(0, &[empty_record(); 48]);

    let rows = frame_rows(&[later, earlier]);

    let frames: Vec<u64> = rows.iter().map(|row| row.frame).collect();
    let expected: Vec<u64> = (0..50).collect();
    assert_eq!(frames, expected);
    assert!(rows[..48].iter().all(|row| row.scene == 0));
    assert!(rows[48..].iter().all(|row| row.scene == 1));
}

#[test]
fn a_window_spans_chunks_of_one_scene() {
    let first = chunk_with(&[empty_record(); 24]);
    let second_records = [lag_one_record(1000.0, 0.5), empty_record(), empty_record()];
    let second = chunk_with(&second_records);
    let scene = scene_of(0, vec![first, second]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows.len(), 27);
    assert_near(rows[23].persistence[0], 0.5);
    assert_near(rows[24].persistence[0], 0.5);
}

#[test]
fn missing_records_give_null_rows() {
    let mut chunk = GrainChunk::empty();
    chunk.frames = 3;
    let scene = scene_of(0, vec![chunk]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.cells == 0 && row.strength.is_none()));
}

#[test]
fn merging_chunks_appends_their_frame_records() {
    let mut first = chunk_with(&[lag_one_record(300.0, 0.5)]);
    let second = chunk_with(&[lag_one_record(400.0, 0.5), lag_one_record(500.0, 0.5)]);

    first.merge(&second);

    assert_eq!(first.frame_records.len(), 3 * FRAME_RECORD_LEN);
    assert_eq!(first.frame_records[0], 300.0);
    assert_eq!(first.frame_records[FRAME_RECORD_LEN], 400.0);
    assert_eq!(first.frame_records[2 * FRAME_RECORD_LEN], 500.0);
}

#[test]
fn a_row_serialises_with_the_report_keys() {
    let third_frame = FrameGrain {
        frame: 2,
        scene: 0,
        cells: 500,
        strength: Some(1.5),
        strength_cb: Some(0.75),
        strength_cr: Some(1.25),
        persistence: [Some(0.5), Some(0.25), None, None, None, None, None, None],
    };
    let null_row = FrameGrain {
        frame: 0,
        cells: 0,
        strength: None,
        strength_cb: None,
        strength_cr: None,
        persistence: [None; PERSISTENCE_LAGS],
        ..third_frame
    };

    let json = serde_json::to_string(&third_frame).unwrap();
    let null_json = serde_json::to_string(&null_row).unwrap();

    let expected = concat!(
        r#"{"frame":2,"scene":0,"cells":500,"strength":1.5,"strength_cb":0.75,"strength_cr":1.25,"#,
        r#""persistence":[0.5,0.25,null,null,null,null,null,null]}"#,
    );
    let null_expected = concat!(
        r#"{"frame":0,"scene":0,"cells":0,"strength":null,"strength_cb":null,"strength_cr":null,"#,
        r#""persistence":[null,null,null,null,null,null,null,null]}"#,
    );
    assert_eq!(json, expected);
    assert_eq!(null_json, null_expected);
}

#[test]
fn chroma_strength_pools_the_pair_behind_and_ahead() {
    let luma_chunk = chunk_with(&[empty_record(); 3]);
    let chroma_records = [
        [0.0; CHROMA_RECORD_LEN],
        chroma_record(150.0, 2.0, 4.0),
        chroma_record(150.0, 2.0, 4.0),
    ];
    let chroma_chunk = chunk_with(&chroma_records);
    let scene = chroma_scene(vec![luma_chunk], vec![chroma_chunk]);

    let rows = frame_rows(&[scene]);

    assert_near(rows[1].strength_cb, 2.0);
    assert_near(rows[1].strength_cr, 4.0);
    assert_eq!(rows[0].strength_cb, None);
    assert_eq!(rows[2].strength_cr, None);
}

#[test]
fn a_scene_without_chroma_records_has_no_chroma_strength() {
    let records = [lag_one_record(1000.0, 0.5); 3];
    let scene = scene_with(0, &records);

    let rows = frame_rows(&[scene]);

    assert!(rows.iter().all(|row| row.strength.is_some()));
    assert!(rows.iter().all(|row| row.strength_cb.is_none()));
    assert!(rows.iter().all(|row| row.strength_cr.is_none()));
}

#[test]
fn chroma_rows_ignore_luma_chunk_boundaries() {
    let first_luma = chunk_with(&[empty_record(); 24]);
    let second_luma = chunk_with(&[empty_record(); 3]);
    let mut first_chroma_records = [[0.0; CHROMA_RECORD_LEN]; 10];
    let mut second_chroma_records = [[0.0; CHROMA_RECORD_LEN]; 17];
    first_chroma_records[9] = chroma_record(150.0, 2.0, 4.0);
    second_chroma_records[0] = chroma_record(150.0, 2.0, 4.0);
    second_chroma_records[14] = chroma_record(150.0, 3.0, 1.0);
    second_chroma_records[15] = chroma_record(150.0, 3.0, 1.0);
    let first_chroma = chunk_with(&first_chroma_records);
    let second_chroma = chunk_with(&second_chroma_records);
    let scene = chroma_scene(vec![first_luma, second_luma], vec![first_chroma, second_chroma]);

    let rows = frame_rows(&[scene]);

    assert_eq!(rows.len(), 27);
    assert_near(rows[9].strength_cb, 2.0);
    assert_near(rows[9].strength_cr, 4.0);
    assert_near(rows[24].strength_cb, 3.0);
    assert_near(rows[24].strength_cr, 1.0);
}
