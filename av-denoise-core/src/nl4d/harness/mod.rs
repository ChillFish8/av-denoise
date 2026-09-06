//! Synthetic clips with known motion, and the scores that compare a
//! motion field against them.
//!
//! This exists for the `mc_accuracy` bench. It is not a stable
//! interface.

#![doc(hidden)]

mod score;
mod synth;

pub use score::{score, KindScore, Score};
pub use synth::{synthesise, Clip, MotionClass, Still};
