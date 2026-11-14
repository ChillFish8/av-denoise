use std::io::Write;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder};
use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type, UInt32Type};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;

use super::meta::{FRAMES_KEY, metadata_pairs};
use super::schema::{AvdsGroup, Column, Value, arrow_schema, columns, list_item, writer_properties};
use super::{AvdsFrame, AvdsMeta};

/// Writes a grain data file row by row, in frame order.
pub struct AvdsWriter<W: Write + Send> {
    writer: ArrowWriter<W>,
    schema: Arc<Schema>,
    columns: Vec<Column>,
    /// The frame the next row must hold, which is also the rows written so far.
    next_frame: u64,
}

impl<W: Write + Send> AvdsWriter<W> {
    /// Starts a file in `writer` with the metadata of `meta`.
    pub fn new(writer: W, meta: &AvdsMeta) -> Result<Self, anyhow::Error> {
        let pairs = metadata_pairs(meta);
        let columns = columns();
        let schema = arrow_schema(&columns);
        let schema = Arc::new(schema);
        let properties = writer_properties(&columns, pairs);
        let writer_schema = schema.clone();
        let writer = ArrowWriter::try_new(writer, writer_schema, Some(properties))
            .context("unable to start the grain data file")?;

        Ok(Self {
            writer,
            schema,
            columns,
            next_frame: 0,
        })
    }

    /// Writes the next rows, which must follow the last rows written in frame order.
    ///
    /// The first row of a file is frame 0.
    pub fn write_rows(&mut self, rows: &[AvdsFrame]) -> Result<(), anyhow::Error> {
        for (offset, row) in rows.iter().enumerate() {
            let expected = self.next_frame + offset as u64;
            let got = row.frame;
            if got != expected {
                anyhow::bail!("grain data rows must be in frame order, got frame {got} for frame {expected}");
            }

            check_lengths(row)?;
        }

        let batch = record_batch(&self.schema, &self.columns, rows)?;
        self.writer
            .write(&batch)
            .context("unable to write rows to the grain data file")?;
        self.next_frame += rows.len() as u64;

        Ok(())
    }

    pub fn frames_written(&self) -> u64 {
        self.next_frame
    }

    /// Writes the footer, with `frames` set to the rows written, and returns the inner writer.
    pub fn finish(mut self) -> Result<W, anyhow::Error> {
        let key = FRAMES_KEY.to_owned();
        let value = self.next_frame.to_string();
        let frames = KeyValue::new(key, value);
        self.writer.append_key_value_metadata(frames);

        let inner = self
            .writer
            .into_inner()
            .context("unable to finish the grain data file")?;

        Ok(inner)
    }
}

/// Writes a whole grain data file of `rows`, which start at frame 0 and are in frame order.
pub fn write_avds<W: Write + Send>(
    writer: W,
    meta: &AvdsMeta,
    rows: &[AvdsFrame],
) -> Result<(), anyhow::Error> {
    let mut avds = AvdsWriter::new(writer, meta)?;
    avds.write_rows(rows)?;
    avds.finish()?;

    Ok(())
}

/// Refuses a row whose measured groups hold the wrong number of values.
fn check_lengths(row: &AvdsFrame) -> Result<(), anyhow::Error> {
    for group in AvdsGroup::ALL {
        let float_len = row.floats(group).map(<[f32]>::len);
        let double_len = row.doubles(group).map(<[f64]>::len);
        let Some(len) = float_len.or(double_len) else {
            continue;
        };

        let expected = group.len();
        if len != expected {
            let name = group.name();
            let frame = row.frame;
            anyhow::bail!("frame {frame}'s {name} holds {len} values instead of {expected}");
        }
    }

    Ok(())
}

/// The rows as one batch of the file's columns.
pub(super) fn record_batch(
    schema: &Arc<Schema>,
    columns: &[Column],
    rows: &[AvdsFrame],
) -> Result<RecordBatch, anyhow::Error> {
    let frames = rows.iter().map(|row| row.frame);
    let scenes = rows.iter().map(|row| row.scene);
    let firsts: Vec<bool> = rows.iter().map(|row| row.first_of_scene).collect();
    let frame_array = UInt64Array::from_iter_values(frames);
    let scene_array = UInt64Array::from_iter_values(scenes);
    let first_array = BooleanArray::from(firsts);
    let frame_array: ArrayRef = Arc::new(frame_array);
    let scene_array: ArrayRef = Arc::new(scene_array);
    let first_array: ArrayRef = Arc::new(first_array);

    let mut arrays = vec![frame_array, scene_array, first_array];
    for column in columns {
        let array = column_array(column, rows);
        arrays.push(array);
    }

    let batch_schema = schema.clone();
    let batch = RecordBatch::try_new(batch_schema, arrays).context("unable to build grain data rows")?;

    Ok(batch)
}

fn column_array(column: &Column, rows: &[AvdsFrame]) -> ArrayRef {
    let group = column.group;
    match column.value {
        Value::Count => {
            primitive_array::<f32, UInt32Type>(column, rows, |row| row.floats(group), |count| count as u32)
        },
        Value::Float => {
            primitive_array::<f32, Float32Type>(column, rows, |row| row.floats(group), |value| value)
        },
        Value::Double => {
            primitive_array::<f64, Float64Type>(column, rows, |row| row.doubles(group), |value| value)
        },
    }
}

/// One column over `rows`, null where a row lacks the column's group.
///
/// A null list slot still holds zeros, since a list column's values are never null.
fn primitive_array<S: Copy, T: ArrowPrimitiveType>(
    column: &Column,
    rows: &[AvdsFrame],
    values_of: impl Fn(&AvdsFrame) -> Option<&[S]>,
    convert: impl Fn(S) -> T::Native,
) -> ArrayRef {
    let len = column.indices.len();
    let item_builder = PrimitiveBuilder::<T>::with_capacity(rows.len() * len);

    if !column.list {
        let index = column.indices[0];
        let mut builder = item_builder;
        for row in rows {
            let value = values_of(row).map(|values| values[index]);
            let converted = value.map(&convert);
            builder.append_option(converted);
        }

        let array = builder.finish();
        return Arc::new(array);
    }

    let item = list_item(T::DATA_TYPE);
    let mut builder =
        FixedSizeListBuilder::with_capacity(item_builder, len as i32, rows.len()).with_field(item);
    for row in rows {
        let Some(values) = values_of(row) else {
            builder.values().append_value_n(T::Native::default(), len);
            builder.append(false);
            continue;
        };

        for &index in &column.indices {
            let value = convert(values[index]);
            builder.values().append_value(value);
        }

        builder.append(true);
    }

    let array = builder.finish();

    Arc::new(array)
}
