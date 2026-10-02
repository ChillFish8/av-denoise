#![expect(dead_code, reason = "consumed by the grain export tasks that follow")]

mod chunk;
mod consts;
mod fit;
mod gaussian;
mod segment;
mod table;
mod template;

#[cfg(test)]
mod tests;

pub use self::chunk::GrainChunk;
pub use self::segment::SceneGrain;
pub use self::table::build_table;
