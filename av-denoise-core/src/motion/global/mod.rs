mod compose;
mod model;

#[cfg(test)]
mod tests;

pub use self::compose::{AFFINE_WORDS, NO_PAIR, compose_gaps, run_compose_models};
pub use self::model::{MODEL_WORDS, ModelRing};
