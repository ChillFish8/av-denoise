use anyhow::Context;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray, RecordBatch};
use arrow_schema::{DataType, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::ChunkReader;

use super::meta::meta_from_pairs;
use super::schema::{Column, FIRST_OF_SCENE_COLUMN, FRAME_COLUMN, Group, SCENE_COLUMN, Value, columns};
use super::{AvdsFrame, AvdsMeta};

/// Rows read per batch.
const BATCH_FRAMES: usize = 1024;

/// Reads a grain data file's metadata and every row.
///
/// Fails on a major version this crate does not know, a missing column, a row count that differs
/// from the metadata's `frames`, or rows whose frames do not run from 0 in order.
pub fn read_avds<R: ChunkReader + 'static>(reader: R) -> Result<(AvdsMeta, Vec<AvdsFrame>), anyhow::Error> {
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(reader).context("unable to open the grain data file")?;

    let file_metadata = builder.metadata().file_metadata();
    let pairs = file_metadata.key_value_metadata().map_or(&[][..], Vec::as_slice);
    let meta = meta_from_pairs(pairs)?;

    let row_count = file_metadata.num_rows() as u64;
    if row_count != meta.frames {
        let frames = meta.frames;
        anyhow::bail!("the grain data file holds {row_count} rows but its metadata says {frames} frames");
    }

    let columns = columns();
    let schema = builder.schema();
    check_columns(schema, &columns)?;

    let batches = builder
        .with_batch_size(BATCH_FRAMES)
        .build()
        .context("unable to read the grain data file")?;

    let mut rows = Vec::new();
    for batch in batches {
        let batch = batch.context("unable to read rows of the grain data file")?;
        let batch_rows = rows_of(&batch, &columns)?;
        for row in batch_rows {
            let expected = rows.len() as u64;
            let frame = row.frame;
            if frame != expected {
                anyhow::bail!("grain data rows are out of frame order, frame {frame} is at row {expected}");
            }

            rows.push(row);
        }
    }

    Ok((meta, rows))
}

/// Refuses a file that lacks a column, or holds one of another type.
fn check_columns(schema: &Schema, columns: &[Column]) -> Result<(), anyhow::Error> {
    let identity = [
        (FRAME_COLUMN, DataType::UInt64),
        (SCENE_COLUMN, DataType::UInt64),
        (FIRST_OF_SCENE_COLUMN, DataType::Boolean),
    ];
    for (name, expected) in identity {
        let actual = column_type(schema, name)?;
        if actual != &expected {
            anyhow::bail!("the grain data file's {name} column is {actual} instead of {expected}");
        }
    }

    for column in columns {
        let actual = column_type(schema, &column.name)?;
        if !type_matches(actual, column) {
            let name = &column.name;
            let expected = column.data_type();
            anyhow::bail!("the grain data file's {name} column is {actual} instead of {expected}");
        }
    }

    Ok(())
}

fn column_type<'a>(schema: &'a Schema, name: &str) -> Result<&'a DataType, anyhow::Error> {
    let field = schema.field_with_name(name).ok();
    let field = field.with_context(|| format!("the grain data file has no {name} column"))?;

    Ok(field.data_type())
}

/// Whether `actual` holds the column's values, whatever its list items are named.
fn type_matches(actual: &DataType, column: &Column) -> bool {
    let item_type = column.item_type();
    if !column.list {
        return actual == &item_type;
    }

    let DataType::FixedSizeList(item, size) = actual else {
        return false;
    };

    *size as usize == column.indices.len() && item.data_type() == &item_type
}

fn rows_of(batch: &RecordBatch, columns: &[Column]) -> Result<Vec<AvdsFrame>, anyhow::Error> {
    let frames = identity_column::<UInt64Type>(batch, FRAME_COLUMN)?;
    let scenes = identity_column::<UInt64Type>(batch, SCENE_COLUMN)?;
    let firsts = batch
        .column_by_name(FIRST_OF_SCENE_COLUMN)
        .and_then(|array| array.as_boolean_opt())
        .with_context(|| format!("the grain data file has no {FIRST_OF_SCENE_COLUMN} column"))?;
    if firsts.null_count() > 0 {
        anyhow::bail!("the grain data file's {FIRST_OF_SCENE_COLUMN} column holds nulls");
    }

    let mut rows = Vec::with_capacity(batch.num_rows());
    for index in 0..batch.num_rows() {
        let frame = frames.value(index);
        let scene = scenes.value(index);
        let first_of_scene = firsts.value(index);
        let row = AvdsFrame::unmeasured(frame, scene, first_of_scene);
        rows.push(row);
    }

    let mut seen_groups: Vec<Group> = Vec::new();
    for column in columns {
        let array = batch
            .column_by_name(&column.name)
            .with_context(|| format!("the grain data file has no {} column", column.name))?;
        let group = column.group;
        let first_of_group = !seen_groups.contains(&group);
        seen_groups.push(group);

        let read = ColumnRead {
            array,
            column,
            first_of_group,
        };
        match column.value {
            Value::Count => {
                read.scatter::<UInt32Type, f32>(&mut rows, |row| row.floats_mut(group), |count| count as f32)?
            },
            Value::Float => {
                read.scatter::<Float32Type, f32>(&mut rows, |row| row.floats_mut(group), |value| value)?
            },
            Value::Double => {
                read.scatter::<Float64Type, f64>(&mut rows, |row| row.doubles_mut(group), |value| value)?
            },
        }
    }

    Ok(rows)
}

fn identity_column<'a, T: ArrowPrimitiveType>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a PrimitiveArray<T>, anyhow::Error> {
    let array = batch
        .column_by_name(name)
        .and_then(|array| array.as_primitive_opt::<T>())
        .with_context(|| format!("the grain data file has no {name} column"))?;
    if array.null_count() > 0 {
        anyhow::bail!("the grain data file's {name} column holds nulls");
    }

    Ok(array)
}

/// One column of a batch being read into rows.
struct ColumnRead<'a> {
    array: &'a ArrayRef,
    column: &'a Column,
    /// Whether this is the first column of its group, which decides where the group is present.
    first_of_group: bool,
}

impl ColumnRead<'_> {
    /// Copies the column's values into each row's group, refusing nulls that differ within a group.
    fn scatter<T: ArrowPrimitiveType, S: Copy + Default>(
        &self,
        rows: &mut [AvdsFrame],
        slot_of: impl Fn(&mut AvdsFrame) -> Option<&mut Option<Vec<S>>>,
        convert: impl Fn(T::Native) -> S,
    ) -> Result<(), anyhow::Error> {
        let view = self.view::<T>()?;
        let group = self.column.group;
        let group_len = group.len();

        for (index, row) in rows.iter_mut().enumerate() {
            let frame = row.frame;
            let values = view.row(index);
            let slot = slot_of(row).expect("the column's group stores its value type");

            if self.first_of_group && values.is_some() {
                let zeroed = vec![S::default(); group_len];
                *slot = Some(zeroed);
            }

            if values.is_some() != slot.is_some() {
                let name = group.name();
                anyhow::bail!("frame {frame} has some {name} columns null and others not");
            }

            let (Some(values), Some(target)) = (values, slot.as_mut()) else {
                continue;
            };

            for (&target_index, &value) in self.column.indices.iter().zip(values) {
                target[target_index] = convert(value);
            }
        }

        Ok(())
    }

    fn view<T: ArrowPrimitiveType>(&self) -> Result<ColumnView<'_, T>, anyhow::Error> {
        let name = &self.column.name;
        if !self.column.list {
            let single = self
                .array
                .as_primitive_opt::<T>()
                .with_context(|| format!("the grain data file's {name} column has the wrong type"))?;

            return Ok(ColumnView::Single(single));
        }

        let list = self
            .array
            .as_fixed_size_list_opt()
            .with_context(|| format!("the grain data file's {name} column is not a list"))?;
        let items = list
            .values()
            .as_primitive_opt::<T>()
            .with_context(|| format!("the grain data file's {name} column holds the wrong type"))?;
        let len = self.column.indices.len();

        Ok(ColumnView::List { list, items, len })
    }
}

/// A column's values for each row.
enum ColumnView<'a, T: ArrowPrimitiveType> {
    Single(&'a PrimitiveArray<T>),
    List {
        list: &'a FixedSizeListArray,
        items: &'a PrimitiveArray<T>,
        /// Values in each row's list.
        len: usize,
    },
}

impl<T: ArrowPrimitiveType> ColumnView<'_, T> {
    /// Row `index`'s values, `None` when the row is null.
    fn row(&self, index: usize) -> Option<&[T::Native]> {
        match self {
            ColumnView::Single(single) => {
                let valid = single.is_valid(index);
                valid.then(|| &single.values()[index..index + 1])
            },
            ColumnView::List { list, items, len } => {
                if !list.is_valid(index) {
                    return None;
                }

                let start = list.value_offset_at(index);
                let values = &items.values()[start..start + *len];

                Some(values)
            },
        }
    }
}
