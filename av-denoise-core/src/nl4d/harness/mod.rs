//! Synthetic clips with known motion, and the scores that compare a
//! motion field against them.
//!
//! This exists for the `mc_accuracy` bench. It is not a stable
//! interface.

#![doc(hidden)]

mod score;
mod synth;

pub use score::{KindScore, Score, score};
pub use synth::{Clip, MotionClass, Still, synthesise};
