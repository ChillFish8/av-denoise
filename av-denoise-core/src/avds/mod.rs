//! The `.avds` grain data file
//!
//! A Parquet file with one row per output frame that holds every grain measurement a denoise run
//! made. Any Parquet reader can open it. Single numbers are their own columns and arrays are
//! fixed-size list columns. A group of columns is null on a frame where it was not measured.
//!
//! - [AvdsWriter] writes rows in frame order as they arrive.
//! - [AvdsSceneWriter] takes scenes in any order and writes them in frame order.
//! - [write_avds] writes a whole file at once and [read_avds] reads one back.
//! - [read_avds_groups] reads only some groups of columns.
//! - [av1_table] and [report_rows] build grain outputs from the rows of a file.

mod meta;
mod order;
mod read;
mod rows;
mod schema;
mod synthesize;
mod write;

#[cfg(test)]
mod tests;

pub use self::order::AvdsSceneWriter;
pub use self::read::{read_avds, read_avds_groups};
pub use self::rows::{rows_from_scenes, scenes_from_rows};
pub use self::schema::AvdsGroup;
pub use self::synthesize::{
    AV1_CHROMA_TABLE_GROUPS,
    AV1_TABLE_GROUPS,
    REPORT_GROUPS,
    av1_table,
    chroma_unavailable,
    report_rows,
};
pub use self::write::{AvdsWriter, write_avds};
use crate::{ChromaLayout, ChromaSiting};
pub use crate::{
    DEFAULT_FINE_NOISE,
    DEFAULT_STRENGTH,
    FitOptions,
    GrainMode,
    GrainRun,
    GrainTable,
    MAX_STRENGTH,
    StaticGrain,
};

/// The file format version a writer stamps on every file.
pub const AVDS_VERSION: &str = "1";

/// One frame of a grain data file.
#[derive(Debug, Clone, PartialEq)]
pub struct AvdsFrame {
    /// The output index of the frame.
    pub frame: u64,
    /// The position of the frame's scene, counting scenes in frame order.
    pub scene: u64,
    /// Whether the frame is the first of its scene.
    pub first_of_scene: bool,
    /// The luma table statistics, one frame of
    /// [GrainChunk::frame_tables](crate::GrainChunk::frame_tables), `None` when not measured.
    pub table: Option<Vec<f32>>,
    /// The chroma table statistics, one frame of chroma
    /// [GrainChunk::frame_tables](crate::GrainChunk::frame_tables), Cb then Cr, `None` when not
    /// measured.
    pub chroma_table: Option<Vec<f32>>,
    /// The luma persistence sums, one frame of
    /// [GrainChunk::frame_records](crate::GrainChunk::frame_records), `None` when not measured.
    pub persistence: Option<Vec<f64>>,
    /// The chroma persistence sums, 6 values, Cb then Cr, `None` when not measured.
    pub chroma_persistence: Option<Vec<f64>>,
    /// The luma pass's noise reading, 30 values, `None` when not measured.
    pub noise: Option<Vec<f32>>,
    /// The chroma pass's noise reading, 30 values, `None` when not measured.
    pub chroma_noise: Option<Vec<f32>>,
    /// The luma colour fit, 1974 values, `None` when not measured.
    pub colour: Option<Vec<f32>>,
    /// The chroma colour fit, 1974 values per plane, Cb then Cr, `None` when not measured.
    pub chroma_colour: Option<Vec<f32>>,
}

impl AvdsFrame {
    /// A row with every group unmeasured.
    fn unmeasured(frame: u64, scene: u64, first_of_scene: bool) -> Self {
        Self {
            frame,
            scene,
            first_of_scene,
            table: None,
            chroma_table: None,
            persistence: None,
            chroma_persistence: None,
            noise: None,
            chroma_noise: None,
            colour: None,
            chroma_colour: None,
        }
    }

    /// The values of a group stored as `f32`, `None` for a group stored as `f64`.
    fn floats(&self, group: AvdsGroup) -> Option<&[f32]> {
        let values = match group {
            AvdsGroup::Table => &self.table,
            AvdsGroup::ChromaTable => &self.chroma_table,
            AvdsGroup::Noise => &self.noise,
            AvdsGroup::ChromaNoise => &self.chroma_noise,
            AvdsGroup::Colour => &self.colour,
            AvdsGroup::ChromaColour => &self.chroma_colour,
            AvdsGroup::Persistence | AvdsGroup::ChromaPersistence => return None,
        };

        values.as_deref()
    }

    /// The values of a group stored as `f64`, `None` for a group stored as `f32`.
    fn doubles(&self, group: AvdsGroup) -> Option<&[f64]> {
        let values = match group {
            AvdsGroup::Persistence => &self.persistence,
            AvdsGroup::ChromaPersistence => &self.chroma_persistence,
            AvdsGroup::Table
            | AvdsGroup::ChromaTable
            | AvdsGroup::Noise
            | AvdsGroup::ChromaNoise
            | AvdsGroup::Colour
            | AvdsGroup::ChromaColour => {
                return None;
            },
        };

        values.as_deref()
    }

    fn floats_mut(&mut self, group: AvdsGroup) -> Option<&mut Option<Vec<f32>>> {
        match group {
            AvdsGroup::Table => Some(&mut self.table),
            AvdsGroup::ChromaTable => Some(&mut self.chroma_table),
            AvdsGroup::Noise => Some(&mut self.noise),
            AvdsGroup::ChromaNoise => Some(&mut self.chroma_noise),
            AvdsGroup::Colour => Some(&mut self.colour),
            AvdsGroup::ChromaColour => Some(&mut self.chroma_colour),
            AvdsGroup::Persistence | AvdsGroup::ChromaPersistence => None,
        }
    }

    fn doubles_mut(&mut self, group: AvdsGroup) -> Option<&mut Option<Vec<f64>>> {
        match group {
            AvdsGroup::Persistence => Some(&mut self.persistence),
            AvdsGroup::ChromaPersistence => Some(&mut self.chroma_persistence),
            AvdsGroup::Table
            | AvdsGroup::ChromaTable
            | AvdsGroup::Noise
            | AvdsGroup::ChromaNoise
            | AvdsGroup::Colour
            | AvdsGroup::ChromaColour => None,
        }
    }
}

/// What a grain data file says about the run.
#[derive(Debug, Clone, PartialEq)]
pub struct AvdsMeta {
    /// The program and version that wrote the file, for example `av-denoise 0.6.1`.
    pub writer: String,
    /// The output frame rate as a numerator and denominator.
    pub frame_rate: (u64, u64),
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32,
    pub subsampling: ChromaLayout,
    pub chroma_siting: ChromaSiting,
    /// Rows in the file.
    ///
    /// Writers count it from the rows they write, so its value is ignored when writing.
    pub frames: u64,
    pub planes_measured: PlanesMeasured,
}

/// The planes a run measured grain on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanesMeasured {
    Luma,
    LumaChroma,
}

impl PlanesMeasured {
    /// The name the file's metadata uses.
    pub fn name(self) -> &'static str {
        match self {
            PlanesMeasured::Luma => "luma",
            PlanesMeasured::LumaChroma => "luma,chroma",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "luma" => Some(PlanesMeasured::Luma),
            "luma,chroma" => Some(PlanesMeasured::LumaChroma),
            _ => None,
        }
    }
}
