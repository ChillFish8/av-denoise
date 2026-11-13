mod chroma;
mod chunk;
pub(crate) mod consts;
mod export;
mod fit;
mod gaussian;
mod modes;
mod persistence;
mod repeat;
mod report;
mod seeds;
mod segment;
pub(crate) mod snapshot;
mod table;
mod template;

#[cfg(test)]
mod tests;

pub use self::chunk::{GrainChunk, GrainChunks};
pub(crate) use self::export::{GrainExport, GrainGeometry, GrainParts};
pub use self::modes::{GrainMode, GrainRun, StaticGrain};
pub(crate) use self::persistence::MeasuredPlanes;
pub use self::report::{FrameGrain, frame_rows};
pub use self::segment::SceneGrain;
pub(crate) use self::snapshot::PassSources;
pub use self::table::{
    DEFAULT_FINE_NOISE,
    DEFAULT_STRENGTH,
    FitOptions,
    GrainTable,
    MAX_STRENGTH,
    build_table,
};
