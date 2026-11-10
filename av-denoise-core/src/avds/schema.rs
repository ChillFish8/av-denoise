use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{WriterProperties, WriterPropertiesBuilder};
use parquet::schema::types::ColumnPath;

use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    CHROMA_LAGS,
    CHROMA_PLANES,
    CHROMA_RECORD_LEN,
    FRAME_RECORD_LEN,
    FRAME_TABLE_LEN,
    HIST_LEN,
    LAG_COUNT,
    LAG_RECORD_LEN,
    PERSISTENCE_LAGS,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::snapshot::colour_frame_len;
use crate::noise::colour::{COLOUR_CELLS, COLOUR_FIT_PLANE_WORDS};
use crate::noise::{
    NOISE_CHANNELS,
    NOISE_CURVE_BINS,
    READING_ACCEPTED,
    READING_CURVE_VALID,
    READING_IMMERKAER_LOW,
    READING_POPULATED_BINS,
    READING_QUARTER_MEDIAN,
    READING_RATIOS,
    READING_RHO,
    READING_SAMPLE_VALID_BASE,
    READING_SIGMA,
    READING_SIGMA_LOW,
    READING_STATIC_FRACTION,
    READING_WORDS,
};

pub(super) const FRAME_COLUMN: &str = "frame";
pub(super) const SCENE_COLUMN: &str = "scene";
pub(super) const FIRST_OF_SCENE_COLUMN: &str = "first_of_scene";

/// The name of a list column's values inside the Parquet schema.
const LIST_ITEM: &str = "item";
/// Frames per row group.
const ROW_GROUP_FRAMES: usize = 1024;
const ZSTD_LEVEL: i32 = 3;

/// A group of columns that is measured or not as a whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Group {
    Table,
    Persistence,
    ChromaPersistence,
    Noise,
    ChromaNoise,
    Colour,
    ChromaColour,
}

/// How a column stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Value {
    /// Whole `f32` counts stored as `u32`.
    Count,
    Float,
    Double,
}

/// One measurement column of the file.
#[derive(Debug, Clone)]
pub(super) struct Column {
    pub name: String,
    pub group: Group,
    pub value: Value,
    /// The positions in the group's values of each of the column's values, in order.
    pub indices: Vec<usize>,
    /// Whether the column is a fixed-size list rather than a single number.
    pub list: bool,
}

impl Group {
    pub(super) const ALL: [Group; 7] = [
        Group::Table,
        Group::Persistence,
        Group::ChromaPersistence,
        Group::Noise,
        Group::ChromaNoise,
        Group::Colour,
        Group::ChromaColour,
    ];

    /// Values in one frame of the group.
    pub(super) fn len(self) -> usize {
        match self {
            Group::Table => FRAME_TABLE_LEN,
            Group::Persistence => FRAME_RECORD_LEN,
            Group::ChromaPersistence => CHROMA_RECORD_LEN,
            Group::Noise | Group::ChromaNoise => READING_WORDS as usize,
            Group::Colour => colour_frame_len(1),
            Group::ChromaColour => colour_frame_len(CHROMA_PLANES as u32),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Group::Table => "table statistics",
            Group::Persistence => "persistence",
            Group::ChromaPersistence => "chroma persistence",
            Group::Noise => "noise reading",
            Group::ChromaNoise => "chroma noise reading",
            Group::Colour => "colour fit",
            Group::ChromaColour => "chroma colour fit",
        }
    }
}

impl Column {
    fn new(name: impl Into<String>, group: Group, value: Value, indices: Vec<usize>) -> Self {
        let name = name.into();
        let list = indices.len() > 1;

        Self {
            name,
            group,
            value,
            indices,
            list,
        }
    }

    /// A column of `len` values in a row from `start` of the group's values.
    fn span(name: impl Into<String>, group: Group, value: Value, start: usize, len: usize) -> Self {
        let indices = (start..start + len).collect();

        Self::new(name, group, value, indices)
    }

    /// The Parquet path of the column's values, which sit under `list.item` in a list.
    pub(super) fn path(&self) -> ColumnPath {
        if !self.list {
            return ColumnPath::from(self.name.as_str());
        }

        let name = self.name.clone();
        let list = "list".to_owned();
        let item = LIST_ITEM.to_owned();
        let parts = vec![name, list, item];

        ColumnPath::new(parts)
    }

    pub(super) fn item_type(&self) -> DataType {
        match self.value {
            Value::Count => DataType::UInt32,
            Value::Float => DataType::Float32,
            Value::Double => DataType::Float64,
        }
    }

    pub(super) fn data_type(&self) -> DataType {
        let item_type = self.item_type();
        if !self.list {
            return item_type;
        }

        let item = list_item(item_type);
        let size = self.indices.len() as i32;

        DataType::FixedSizeList(item, size)
    }
}

/// The non-null field of a list column's values.
pub(super) fn list_item(item_type: DataType) -> Arc<Field> {
    let field = Field::new(LIST_ITEM, item_type, false);

    Arc::new(field)
}

/// Every measurement column, grouped and in file order.
pub(super) fn columns() -> Vec<Column> {
    let mut columns = table_columns();
    columns.extend(persistence_columns());
    columns.extend(chroma_persistence_columns());

    let noise = noise_columns(Group::Noise, "noise");
    columns.extend(noise);

    let chroma_noise = noise_columns(Group::ChromaNoise, "chroma_noise");
    columns.extend(chroma_noise);

    let colour = colour_columns(Group::Colour, &["luma"]);
    columns.extend(colour);

    let chroma_colour = colour_columns(Group::ChromaColour, &["cb", "cr"]);
    columns.extend(chroma_colour);

    columns
}

/// The two histograms, then each strength group's lag sums and pixel count.
fn table_columns() -> Vec<Column> {
    let grouped_start = 2 * HIST_LEN;
    let mut lag_sums = Vec::with_capacity(STRENGTH_GROUPS * LAG_COUNT);
    let mut pixels = Vec::with_capacity(STRENGTH_GROUPS);
    for group in 0..STRENGTH_GROUPS {
        let record_start = grouped_start + group * AUTOCOV_LEN;
        lag_sums.extend(record_start..record_start + LAG_COUNT);
        pixels.push(record_start + LAG_COUNT);
    }

    let source = Column::span("table_source_hist", Group::Table, Value::Count, 0, HIST_LEN);
    let kept = Column::span("table_kept_hist", Group::Table, Value::Count, HIST_LEN, HIST_LEN);
    let lag_sums = Column::new("table_lag_sums", Group::Table, Value::Float, lag_sums);
    let pixels = Column::new("table_pixels", Group::Table, Value::Float, pixels);

    vec![source, kept, lag_sums, pixels]
}

/// Each lag's cell count, auto sum and cross sum, as lists over the lags.
fn persistence_columns() -> Vec<Column> {
    let parts = ["cells", "auto", "cross"];

    let mut columns = Vec::with_capacity(parts.len());
    for (part_index, part) in parts.into_iter().enumerate() {
        let indices = (0..PERSISTENCE_LAGS)
            .map(|lag_index| lag_index * LAG_RECORD_LEN + part_index)
            .collect();
        let name = format!("persistence_{part}");
        let column = Column::new(name, Group::Persistence, Value::Double, indices);
        columns.push(column);
    }

    columns
}

/// The lag-1 cell count, auto sum and cross sum of Cb and of Cr.
fn chroma_persistence_columns() -> Vec<Column> {
    let planes = ["cb", "cr"];
    let parts = ["cells", "auto", "cross"];
    let plane_len = CHROMA_LAGS * LAG_RECORD_LEN;

    let mut columns = Vec::with_capacity(planes.len() * parts.len());
    for (plane_index, plane) in planes.into_iter().enumerate() {
        for (part_index, part) in parts.into_iter().enumerate() {
            let name = format!("persistence_{plane}_{part}");
            let start = plane_index * plane_len + part_index;
            let column = Column::span(name, Group::ChromaPersistence, Value::Double, start, 1);
            columns.push(column);
        }
    }

    columns
}

fn noise_columns(group: Group, prefix: &str) -> Vec<Column> {
    let channels = NOISE_CHANNELS as usize;
    let parts = [
        ("immerkaer_low", READING_IMMERKAER_LOW, channels),
        ("sample_valid", READING_SAMPLE_VALID_BASE, channels),
        ("sigma", READING_SIGMA, channels),
        ("sigma_low", READING_SIGMA_LOW, channels),
        ("rho", READING_RHO, 1),
        ("static_fraction", READING_STATIC_FRACTION, 1),
        ("accepted", READING_ACCEPTED, 1),
        ("curve_valid", READING_CURVE_VALID, 1),
        ("quarter_median", READING_QUARTER_MEDIAN, 1),
        ("populated_bins", READING_POPULATED_BINS, 1),
        ("curve_ratios", READING_RATIOS, NOISE_CURVE_BINS),
    ];

    let mut columns = Vec::with_capacity(parts.len());
    for (part, start, len) in parts {
        let name = format!("{prefix}_{part}");
        let column = Column::span(name, group, Value::Float, start as usize, len);
        columns.push(column);
    }

    columns
}

/// Each plane's log grain table, log gain table and fit header.
fn colour_columns(group: Group, planes: &[&str]) -> Vec<Column> {
    let cells = COLOUR_CELLS as usize;
    let header_len = COLOUR_FIT_PLANE_WORDS as usize;
    let plane_len = 2 * cells + header_len;

    let mut columns = Vec::with_capacity(3 * planes.len());
    for (plane_index, plane) in planes.iter().enumerate() {
        let plane_start = plane_index * plane_len;
        let parts = [
            ("log_grain", plane_start, cells),
            ("log_gain", plane_start + cells, cells),
            ("fit_header", plane_start + 2 * cells, header_len),
        ];
        for (part, start, len) in parts {
            let name = format!("colour_{plane}_{part}");
            let column = Column::span(name, group, Value::Float, start, len);
            columns.push(column);
        }
    }

    columns
}

/// The Arrow schema of the identity columns followed by `columns`.
pub(super) fn arrow_schema(columns: &[Column]) -> Schema {
    let frame = Field::new(FRAME_COLUMN, DataType::UInt64, false);
    let scene = Field::new(SCENE_COLUMN, DataType::UInt64, false);
    let first_of_scene = Field::new(FIRST_OF_SCENE_COLUMN, DataType::Boolean, false);

    let mut fields = vec![frame, scene, first_of_scene];

    for column in columns {
        let data_type = column.data_type();
        let field = Field::new(column.name.as_str(), data_type, true);
        fields.push(field);
    }

    Schema::new(fields)
}

/// Zstd compression and an explicit encoding on every column, with `metadata` in the footer.
///
/// Frame and scene indices are delta coded and the scene flag is run-length coded. Histogram
/// counts are dictionary coded, falling back to delta coding. Floats are byte-stream split.
pub(super) fn writer_properties(columns: &[Column], metadata: Vec<KeyValue>) -> WriterProperties {
    let level = ZstdLevel::try_new(ZSTD_LEVEL).expect("the zstd level is in range");
    let mut builder = WriterProperties::builder()
        .set_compression(Compression::ZSTD(level))
        .set_max_row_group_row_count(Some(ROW_GROUP_FRAMES))
        .set_key_value_metadata(Some(metadata));

    let frame_path = ColumnPath::from(FRAME_COLUMN);
    let scene_path = ColumnPath::from(SCENE_COLUMN);
    let first_path = ColumnPath::from(FIRST_OF_SCENE_COLUMN);
    builder = encoded(builder, frame_path, Encoding::DELTA_BINARY_PACKED);
    builder = encoded(builder, scene_path, Encoding::DELTA_BINARY_PACKED);
    builder = encoded(builder, first_path, Encoding::RLE);

    for column in columns {
        let path = column.path();
        builder = match column.value {
            Value::Count => builder
                .set_column_dictionary_enabled(path.clone(), true)
                .set_column_encoding(path, Encoding::DELTA_BINARY_PACKED),
            Value::Float | Value::Double => encoded(builder, path, Encoding::BYTE_STREAM_SPLIT),
        };
    }

    builder.build()
}

/// Sets `path` to `encoding` with no dictionary.
fn encoded(
    builder: WriterPropertiesBuilder,
    path: ColumnPath,
    encoding: Encoding,
) -> WriterPropertiesBuilder {
    builder
        .set_column_dictionary_enabled(path.clone(), false)
        .set_column_encoding(path, encoding)
}
