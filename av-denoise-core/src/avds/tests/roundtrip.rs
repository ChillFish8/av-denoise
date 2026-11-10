use std::fs::File;
use std::sync::Arc;

use arrow_array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Encoding;
use parquet::file::metadata::KeyValue;
use parquet::file::reader::{FileReader, SerializedFileReader};

use super::{Measured, file_of, sample_meta, scenes_of, written_file};
use crate::avds::meta::{FRAMES_KEY, VERSION_KEY, metadata_pairs};
use crate::avds::schema::{arrow_schema, columns, writer_properties};
use crate::avds::write::record_batch;
use crate::avds::{AvdsFrame, AvdsGroup, read_avds, read_avds_groups, rows_from_scenes, scenes_from_rows};

/// A file of `batch` with `pairs` as its metadata, written without the grain data writer.
fn raw_file(batch: &RecordBatch, pairs: Vec<KeyValue>) -> File {
    let properties = writer_properties(&columns(), pairs);
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(properties)).expect("writer");
    writer.write(batch).expect("rows are written");
    writer.close().expect("the file closes");

    file_of(&bytes)
}

fn sample_batch(rows: &[AvdsFrame]) -> RecordBatch {
    let columns = columns();
    let schema = arrow_schema(&columns);
    let schema = Arc::new(schema);

    record_batch(&schema, &columns, rows).expect("the batch builds")
}

/// The metadata pairs of a file with `frames` rows.
fn pairs_with_frames(frames: u64) -> Vec<KeyValue> {
    let meta = sample_meta(frames);
    let mut pairs = metadata_pairs(&meta);
    let key = FRAMES_KEY.to_owned();
    let value = frames.to_string();
    let frames_pair = KeyValue::new(key, value);
    pairs.push(frames_pair);

    pairs
}

#[test]
fn rows_round_trip_through_a_file() {
    let scenes = scenes_of(&[30, 5, 1], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let file = written_file(&meta, &rows);

    let (read_meta, read_rows) = read_avds(file).expect("the file reads");

    assert_eq!(rows.len(), 36);
    assert_eq!(read_meta, meta);
    assert_eq!(read_rows, rows);
}

#[test]
fn a_group_read_holds_only_those_groups() {
    let scenes = scenes_of(&[30, 5], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let file = written_file(&meta, &rows);
    let groups = [AvdsGroup::Persistence, AvdsGroup::ChromaColour];

    let (read_meta, read_rows) = read_avds_groups(file, &groups).expect("the file reads");

    assert_eq!(read_meta, meta);
    assert_eq!(read_rows.len(), rows.len());
    for (read_row, row) in read_rows.iter().zip(&rows) {
        let mut expected = AvdsFrame::unmeasured(row.frame, row.scene, row.first_of_scene);
        expected.persistence = row.persistence.clone();
        expected.chroma_colour = row.chroma_colour.clone();

        assert_eq!(read_row, &expected);
    }
}

#[test]
fn frames_metadata_counts_the_rows_written() {
    let scenes = scenes_of(&[10], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(0);
    let file = written_file(&meta, &rows);

    let (read_meta, _) = read_avds(file).expect("the file reads");

    assert_eq!(read_meta.frames, 10);
}

#[test]
fn unmeasured_groups_read_back_as_null() {
    let scenes = scenes_of(&[30], Measured::TABLE_ONLY);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let file = written_file(&meta, &rows);
    let raw = file.try_clone().expect("the file clones");

    let (_, read_rows) = read_avds(file).expect("the file reads");

    for row in &read_rows {
        assert!(row.table.is_some());
        assert_eq!(row.persistence, None);
        assert_eq!(row.chroma_persistence, None);
        assert_eq!(row.noise, None);
        assert_eq!(row.chroma_noise, None);
        assert_eq!(row.colour, None);
        assert_eq!(row.chroma_colour, None);
    }

    let mut reader = ParquetRecordBatchReaderBuilder::try_new(raw)
        .expect("the file opens")
        .build()
        .expect("the reader builds");
    let batch = reader.next().expect("a batch").expect("the batch reads");
    for name in [
        "persistence_cells",
        "persistence_cb_cells",
        "noise_rho",
        "chroma_noise_sigma",
        "colour_luma_log_grain",
        "colour_cr_fit_header",
    ] {
        let column = batch.column_by_name(name).expect("the column exists");

        assert_eq!(column.null_count(), 30, "{name} is null on every row");
    }

    let table = batch
        .column_by_name("table_source_hist")
        .expect("the column exists");

    assert_eq!(table.null_count(), 0);
}

#[test]
fn rows_regroup_into_the_exports_chunks() {
    let scenes = scenes_of(&[1, 23, 24, 25, 60], Measured::ALL);
    let rows = rows_from_scenes(&scenes);

    let regrouped = scenes_from_rows(rows);

    let chunk_frames: Vec<u32> = regrouped[3].chunks.iter().map(|chunk| chunk.frames).collect();

    assert_eq!(chunk_frames, [24, 1]);
    assert_eq!(regrouped, scenes);
}

#[test]
fn table_only_scenes_regroup_without_chroma_chunks() {
    let scenes = scenes_of(&[5, 40], Measured::TABLE_ONLY);
    let rows = rows_from_scenes(&scenes);

    let regrouped = scenes_from_rows(rows);

    assert!(regrouped.iter().all(|scene| scene.chroma_chunks.is_empty()));
    assert_eq!(regrouped, scenes);
}

#[test]
fn rows_number_scenes_in_frame_order() {
    let mut scenes = scenes_of(&[3, 2], Measured::TABLE_ONLY);
    scenes.reverse();

    let rows = rows_from_scenes(&scenes);

    let frames: Vec<u64> = rows.iter().map(|row| row.frame).collect();
    let scene_indices: Vec<u64> = rows.iter().map(|row| row.scene).collect();
    let firsts: Vec<bool> = rows.iter().map(|row| row.first_of_scene).collect();

    assert_eq!(frames, [0, 1, 2, 3, 4]);
    assert_eq!(scene_indices, [0, 0, 0, 1, 1]);
    assert_eq!(firsts, [true, false, false, true, false]);
}

#[test]
fn an_unknown_major_version_is_refused() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let batch = sample_batch(&rows);
    let mut pairs = pairs_with_frames(4);
    for pair in &mut pairs {
        if pair.key == VERSION_KEY {
            pair.value = Some("2.0".to_owned());
        }
    }

    let file = raw_file(&batch, pairs);
    let error = read_avds(file).expect_err("version 2 is refused");

    assert!(error.to_string().contains("version 2.0"), "got {error}");
}

#[test]
fn a_missing_column_is_refused_with_its_name() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let batch = sample_batch(&rows);
    let schema = batch.schema();
    let dropped = schema.index_of("noise_rho").expect("the column exists");
    let kept: Vec<usize> = (0..batch.num_columns())
        .filter(|&index| index != dropped)
        .collect();
    let projected = batch.project(&kept).expect("the batch projects");

    let pairs = pairs_with_frames(4);
    let file = raw_file(&projected, pairs);
    let error = read_avds(file).expect_err("a missing column is refused");

    assert!(error.to_string().contains("noise_rho"), "got {error}");
}

#[test]
fn a_row_count_that_differs_from_frames_is_refused() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let batch = sample_batch(&rows);

    let pairs = pairs_with_frames(5);
    let file = raw_file(&batch, pairs);
    let error = read_avds(file).expect_err("a short file is refused");

    assert!(error.to_string().contains("4 rows"), "got {error}");
}

#[test]
fn out_of_order_frames_are_refused() {
    let scenes = scenes_of(&[4], Measured::ALL);
    let mut rows = rows_from_scenes(&scenes);
    rows[1].frame = 2;
    rows[2].frame = 1;
    let batch = sample_batch(&rows);

    let pairs = pairs_with_frames(4);
    let file = raw_file(&batch, pairs);
    let error = read_avds(file).expect_err("out of order frames are refused");

    assert!(error.to_string().contains("frame 2 is at row 1"), "got {error}");
}

#[test]
#[should_panic(expected = "frame_noise holds 29 values")]
fn a_wrong_length_field_panics_with_its_name() {
    let mut scenes = scenes_of(&[1], Measured::ALL);
    scenes[0].chunks[0].frame_noise.pop();

    rows_from_scenes(&scenes);
}

#[test]
#[should_panic(expected = "has 3 chroma and 4 luma frames")]
fn chroma_chunks_covering_other_frames_panic() {
    let mut scenes = scenes_of(&[4], Measured::TABLE_ONLY);
    let chroma_scene = scenes_of(&[3], Measured::ALL);
    scenes[0].chroma_chunks = chroma_scene[0].chroma_chunks.clone();

    rows_from_scenes(&scenes);
}

#[test]
fn integer_and_float_columns_use_their_encodings() {
    let scenes = scenes_of(&[30], Measured::ALL);
    let rows = rows_from_scenes(&scenes);
    let meta = sample_meta(rows.len() as u64);
    let file = written_file(&meta, &rows);

    let reader = SerializedFileReader::new(file).expect("the file opens");
    let row_group = reader.metadata().row_group(0);

    let mut checked = 0;
    for column in row_group.columns() {
        let path = column.column_path().string();
        let mask = column
            .page_encoding_stats_mask()
            .expect("data page encodings are kept");
        let expected = match path.as_str() {
            "frame" | "scene" => Encoding::DELTA_BINARY_PACKED,
            "first_of_scene" => Encoding::RLE,
            "table_source_hist.list.item" | "table_kept_hist.list.item" => Encoding::RLE_DICTIONARY,
            _ => Encoding::BYTE_STREAM_SPLIT,
        };

        assert!(
            mask.is_only(expected),
            "{path} uses {mask:?}, expected {expected}"
        );
        checked += 1;
    }

    assert_eq!(checked, 3 + columns().len());
}
