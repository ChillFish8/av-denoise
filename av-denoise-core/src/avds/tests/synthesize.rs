use super::{Measured, sample_meta, scenes_of, written_file};
use crate::avds::{
    AV1_CHROMA_TABLE_GROUPS,
    AV1_TABLE_GROUPS,
    AvdsFrame,
    AvdsGroup,
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
use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    BUCKETS_PER_GROUP,
    CHUNK_FRAMES,
    FRAME_TABLE_LEN,
    HIST_LEN,
    LAG_COUNT,
    STD_BUCKETS,
};
use crate::{ChromaLayout, GrainChunk, SceneGrain, build_table, frame_rows};

const LUMA_ONLY: Measured = Measured {
    persistence: true,
    noise: true,
    colour: true,
    chroma: false,
};

/// The std bucket of every block in [grain_scene], about 0.59 codes.
const GRAIN_BUCKET: usize = 24;
/// The luma bins of every block in [grain_scene].
const GRAIN_BINS: std::ops::Range<usize> = 4..12;
const GRAIN_BLOCKS_PER_BIN: f32 = 300.0;
/// The scaling lines of every plane.
const ALL_PLANES: [&str; 3] = ["sY", "sCb", "sCr"];

/// One frame's tables of white grain for `planes` planes, every block at [GRAIN_BUCKET].
fn grain_tables(planes: usize) -> Vec<f32> {
    let pixels = GRAIN_BLOCKS_PER_BIN * GRAIN_BINS.len() as f32 * 64.0;
    let group = GRAIN_BUCKET / BUCKETS_PER_GROUP;
    let record_start = 2 * HIST_LEN + group * AUTOCOV_LEN;
    let mut values = vec![0.0; planes * FRAME_TABLE_LEN];

    for plane in 0..planes {
        let start = plane * FRAME_TABLE_LEN;
        for bin in GRAIN_BINS {
            let slot = bin * STD_BUCKETS + GRAIN_BUCKET;
            values[start + slot] = GRAIN_BLOCKS_PER_BIN;
            values[start + HIST_LEN + slot] = GRAIN_BLOCKS_PER_BIN;
        }

        values[start + record_start] = 1e-5 * pixels;
        values[start + record_start + LAG_COUNT] = pixels;
    }

    values
}

/// A one-chunk scene of white grain on luma and both chroma planes.
fn grain_scene() -> SceneGrain {
    let frames = CHUNK_FRAMES as usize;
    let luma_tables = grain_tables(1);
    let mut luma = GrainChunk::empty();
    luma.frames = CHUNK_FRAMES;
    luma.frame_tables = luma_tables.repeat(frames);

    let chroma_tables = grain_tables(2);
    let mut chroma = GrainChunk::empty();
    chroma.frames = CHUNK_FRAMES;
    chroma.frame_tables = chroma_tables.repeat(frames);

    SceneGrain {
        first_frame: 0,
        chunks: vec![luma],
        chroma_chunks: vec![chroma],
    }
}

/// The options with the bit depth of `meta`, as [av1_table] fits.
fn options_for(meta: &AvdsMeta, options: &FitOptions) -> FitOptions {
    let options = options.clone();

    options.with_bit_depth(meta.bit_depth)
}

/// Each scaling point over two to its entry's scaling shift, for every entry and plane in turn.
///
/// Only the scaling lines named in `planes`, such as `sY` or `sCb`, are read.
fn effective_points(table: &str, planes: &[&str]) -> Vec<f64> {
    let mut points = Vec::new();
    let mut scale = 0.0;
    for line in table.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some(&name) = words.first() else {
            continue;
        };

        if name == "p" {
            let shift: u32 = words[4].parse().expect("a scaling shift");
            scale = (1u32 << shift) as f64;
            continue;
        }

        if !planes.contains(&name) {
            continue;
        }

        let values = words.iter().skip(3).step_by(2);
        for value in values {
            let value: f64 = value.parse().expect("a scaling value");
            points.push(value / scale);
        }
    }

    points
}

/// Zeroes the one-frame counts of both planes in every row's chroma table.
fn clear_chroma_frame_counts(rows: &mut [AvdsFrame]) {
    for row in rows {
        let Some(table) = row.chroma_table.as_mut() else {
            continue;
        };

        let (plane_tables, _) = table.as_chunks_mut::<FRAME_TABLE_LEN>();
        for plane_table in plane_tables {
            plane_table[HIST_LEN..2 * HIST_LEN].fill(0.0);
        }
    }
}

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
    let direct_options = options_for(&meta, &options);

    let table = av1_table(&meta, rows, &options).expect("the table builds");
    let direct = build_table(&scenes, meta.frame_rate, &direct_options);

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
    let direct_options = options_for(&meta, &options);

    let table_file = written_file(&meta, &rows);
    let (_, table_rows) = read_avds_groups(table_file, &AV1_TABLE_GROUPS).expect("the file reads");
    let report_file = written_file(&meta, &rows);
    let (_, report_rows_read) = read_avds_groups(report_file, &REPORT_GROUPS).expect("the file reads");

    let table = av1_table(&meta, table_rows, &options).expect("the table builds");
    let report = report_rows(report_rows_read);
    let direct_table = build_table(&scenes, meta.frame_rate, &direct_options);
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
    let direct_options = options_for(&meta, &options);

    let table = av1_table(&meta, rows.clone(), &options).expect("the table builds");
    let direct_table = build_table(&scenes, meta.frame_rate, &direct_options);
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
fn a_bad_bit_depth_is_refused() {
    let scenes = vec![grain_scene()];
    let rows = rows_from_scenes(&scenes);
    let options = FitOptions::default();

    for depth in [0, 17] {
        let mut meta = sample_meta(rows.len() as u64);
        meta.bit_depth = depth;

        let error = av1_table(&meta, rows.clone(), &options).expect_err("the depth is refused");
        let message = error.to_string();

        assert!(message.contains("not between 1 and 16"), "got {message}");
    }
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
    let scenes = vec![grain_scene()];
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let options = FitOptions::default().with_chroma(true);
    let direct_options = options_for(&meta, &options);
    let file = written_file(&meta, &rows);
    let (_, chroma_rows) = read_avds_groups(file, &AV1_CHROMA_TABLE_GROUPS).expect("the file reads");

    let table = av1_table(&meta, chroma_rows, &options).expect("the table builds");
    let direct = build_table(&scenes, meta.frame_rate, &direct_options);

    assert_eq!(
        AV1_CHROMA_TABLE_GROUPS,
        [AvdsGroup::Table, AvdsGroup::ChromaTable]
    );
    let cb_points = effective_points(&table, &["sCb"]);

    assert!(cb_points.iter().any(|&point| point > 0.0), "{table}");
    assert_eq!(table, direct);
}

#[test]
fn av1_table_takes_rounding_from_the_bit_depth() {
    let scenes = vec![grain_scene()];
    let rows = rows_from_scenes(&scenes);
    let mut eight_bit = sample_meta(rows.len() as u64);
    eight_bit.bit_depth = 8;
    let mut ten_bit = sample_meta(rows.len() as u64);
    ten_bit.bit_depth = 10;
    let options = FitOptions::default().with_chroma(true);

    let eight_bit_table = av1_table(&eight_bit, rows.clone(), &options).expect("the table builds");
    let ten_bit_table = av1_table(&ten_bit, rows, &options).expect("the table builds");
    let eight_bit_points = effective_points(&eight_bit_table, &ALL_PLANES);
    let ten_bit_points = effective_points(&ten_bit_table, &ALL_PLANES);

    assert_ne!(eight_bit_table, ten_bit_table);
    assert_eq!(eight_bit_points.len(), ten_bit_points.len());
    assert!(eight_bit_points.iter().any(|&point| point > 0.0));
    for (&eight, &ten) in eight_bit_points.iter().zip(&ten_bit_points) {
        assert!(ten >= eight, "{ten} vs {eight}");
    }
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
fn chroma_is_unavailable_without_one_frame_counts() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let mut rows = rows_from_scenes(&scenes);
    clear_chroma_frame_counts(&mut rows);
    let meta = sample_meta(rows.len() as u64);

    let reason = chroma_unavailable(&meta, &rows);

    assert_eq!(
        reason.as_deref(),
        Some("the file holds no one-frame chroma measurements")
    );
}

#[test]
fn chroma_is_available_without_chroma_colour_rows() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let mut rows = rows_from_scenes(&scenes);
    for row in &mut rows {
        row.chroma_colour = None;
    }

    let meta = sample_meta(rows.len() as u64);

    let reason = chroma_unavailable(&meta, &rows);

    assert_eq!(reason, None);
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
