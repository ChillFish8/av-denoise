use super::{Measured, sample_meta, scenes_of, written_file};
use crate::avds::{
    AV1_CHROMA_TABLE_GROUPS,
    AV1_TABLE_GROUPS,
    AvdsFrame,
    AvdsMeta,
    FitOptions,
    PlanesMeasured,
    REPORT_GROUPS,
    av1_table,
    chroma_unavailable,
    read_avds,
    read_avds_groups,
    report_rows,
    rows_from_scenes,
};
use crate::{ChromaLayout, SceneGrain, build_table, frame_rows};

const LUMA_ONLY: Measured = Measured {
    persistence: true,
    noise: true,
    colour: true,
    chroma: false,
};

/// The scenes written to a file and read back.
fn read_back(scenes: &[SceneGrain]) -> (AvdsMeta, Vec<AvdsFrame>) {
    let rows = rows_from_scenes(scenes);
    let meta = sample_meta(rows.len() as u64);
    let file = written_file(&meta, &rows);

    read_avds(file).expect("the file reads")
}

#[test]
fn av1_fgs_from_the_file_matches_the_direct_table() {
    let scenes = scenes_of(&[30, 24, 7], Measured::ALL);
    let (meta, rows) = read_back(&scenes);
    let options = FitOptions::default();

    let table = av1_table(&meta, rows, &options).expect("the table builds");
    let direct = build_table(&scenes, meta.frame_rate, &options);

    assert_eq!(table, direct);
}

#[test]
fn report_from_the_file_matches_the_direct_report() {
    let scenes = scenes_of(&[30, 24, 7], Measured::ALL);
    let (_, rows) = read_back(&scenes);

    let report = report_rows(rows);
    let direct = frame_rows(&scenes);

    assert_eq!(report.len(), 61);
    assert_eq!(report, direct);
}

#[test]
fn outputs_from_their_groups_match_the_direct_outputs() {
    let scenes = scenes_of(&[30, 24, 7], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let options = FitOptions::with_fine_noise(0.5);

    let table_file = written_file(&meta, &rows);
    let (_, table_rows) = read_avds_groups(table_file, &AV1_TABLE_GROUPS).expect("the file reads");
    let report_file = written_file(&meta, &rows);
    let (_, report_rows_read) = read_avds_groups(report_file, &REPORT_GROUPS).expect("the file reads");

    let table = av1_table(&meta, table_rows, &options).expect("the table builds");
    let report = report_rows(report_rows_read);
    let direct_table = build_table(&scenes, meta.frame_rate, &options);
    let direct_report = frame_rows(&scenes);

    assert_eq!(table, direct_table);
    assert_eq!(report, direct_report);
}

#[test]
fn a_luma_only_file_reports_null_chroma() {
    let scenes = scenes_of(&[12, 3], LUMA_ONLY);
    let (_, rows) = read_back(&scenes);

    let report = report_rows(rows);
    let direct = frame_rows(&scenes);

    assert_eq!(report.len(), 15);
    assert_eq!(report, direct);
    assert!(report.iter().all(|row| row.strength_cb.is_none()));
    assert!(report.iter().all(|row| row.strength_cr.is_none()));
}

#[test]
fn short_scenes_synthesize_like_the_direct_path() {
    let scenes = scenes_of(&[1, 5, 1], Measured::ALL);
    let (meta, rows) = read_back(&scenes);
    let options = FitOptions::default();

    let table = av1_table(&meta, rows.clone(), &options).expect("the table builds");
    let direct_table = build_table(&scenes, meta.frame_rate, &options);
    let report = report_rows(rows);
    let direct_report = frame_rows(&scenes);

    assert_eq!(table, direct_table);
    assert_eq!(report, direct_report);
}

#[test]
fn a_zero_frame_rate_is_refused() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let mut meta = sample_meta(rows.len() as u64);
    meta.frame_rate = (24, 0);
    let options = FitOptions::default();

    let error = av1_table(&meta, rows, &options).expect_err("a zero denominator is refused");

    assert!(error.to_string().contains("24/0"), "got {error}");
}

#[test]
fn a_fine_noise_share_outside_zero_to_one_is_refused() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);

    for share in [-0.1, 1.5, f64::NAN] {
        let options = FitOptions::with_fine_noise(share);

        let error = av1_table(&meta, rows.clone(), &options).expect_err("the share is refused");

        assert!(error.to_string().contains("not between 0 and 1"), "got {error}");
    }
}

#[test]
fn rows_without_luma_tables_are_refused() {
    let rows: Vec<AvdsFrame> = (0..3)
        .map(|frame| AvdsFrame::unmeasured(frame, 0, frame == 0))
        .collect();
    let meta = sample_meta(rows.len() as u64);
    let options = FitOptions::default();

    let error = av1_table(&meta, rows, &options).expect_err("rows without tables are refused");

    assert!(error.to_string().contains("no luma table"), "got {error}");
}

#[test]
fn an_empty_file_gives_an_empty_table() {
    let meta = sample_meta(0);
    let options = FitOptions::default();

    let table = av1_table(&meta, Vec::new(), &options).expect("the table builds");

    assert_eq!(table, "filmgrn1\n");
}

#[test]
fn av1_fgs_with_chroma_from_the_file_matches_the_direct_table() {
    let scenes = scenes_of(&[30, 24, 7], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let options = FitOptions::default().with_chroma(true);
    let file = written_file(&meta, &rows);
    let (_, chroma_rows) = read_avds_groups(file, &AV1_CHROMA_TABLE_GROUPS).expect("the file reads");

    let table = av1_table(&meta, chroma_rows, &options).expect("the table builds");
    let direct = build_table(&scenes, meta.frame_rate, &options);

    assert_eq!(table, direct);
}

#[test]
fn chroma_is_unavailable_for_a_luma_only_file() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let mut meta = sample_meta(rows.len() as u64);
    meta.planes_measured = PlanesMeasured::Luma;

    let reason = chroma_unavailable(&meta, &rows);

    assert!(reason.is_some());
}

#[test]
fn chroma_is_unavailable_for_a_422_file() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let mut meta = sample_meta(rows.len() as u64);
    meta.subsampling = ChromaLayout::Yuv422;

    let reason = chroma_unavailable(&meta, &rows);

    assert!(reason.is_some());
}

#[test]
fn chroma_is_unavailable_without_chroma_table_rows() {
    let scenes = scenes_of(&[4], LUMA_ONLY);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);

    let reason = chroma_unavailable(&meta, &rows);

    assert!(reason.is_some());
}

#[test]
fn chroma_is_unavailable_without_chroma_colour_rows() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let mut rows = rows_from_scenes(&scenes);
    for row in &mut rows {
        row.chroma_colour = None;
    }

    let meta = sample_meta(rows.len() as u64);

    let reason = chroma_unavailable(&meta, &rows);

    assert_eq!(
        reason.as_deref(),
        Some("the file holds no chroma colour measurements")
    );
}

#[test]
fn chroma_is_available_for_a_measured_420_file() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);

    let reason = chroma_unavailable(&meta, &rows);

    assert_eq!(reason, None);
}

#[test]
fn av1_table_refuses_chroma_it_cannot_write() {
    let scenes = scenes_of(&[4], LUMA_ONLY);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let options = FitOptions::default().with_chroma(true);

    let error = av1_table(&meta, rows, &options).expect_err("chroma cannot be written");

    assert!(
        error.to_string().contains("chroma grain cannot be written"),
        "got {error}"
    );
}
