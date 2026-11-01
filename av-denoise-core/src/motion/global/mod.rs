mod compose;
mod fit;
mod model;

#[cfg(test)]
mod tests;

pub use self::compose::{AFFINE_WORDS, NO_PAIR, compose_gaps, run_compose_models};
pub use self::fit::{
    FIT_BAND_WORDS,
    FIT_INLIER_CAP,
    FIT_LANES,
    FIT_SCALES,
    FIT_STAGE_GATE,
    FIT_STAGE_POLISH,
    FIT_STAGE_SCORE,
    FIT_STAGE_SOLVE,
    FIT_STAGE_WEIGH,
    FIT_STATE_WORDS,
    FIT_THREADS,
    FitScratch,
    MAX_ROLL_TAN,
};
#[cfg_attr(not(test), expect(unused_imports, reason = "only the tests run a fit so far"))]
pub(crate) use self::fit::{FitField, run_model_fit};
pub use self::model::{MODEL_WORDS, ModelRing};
