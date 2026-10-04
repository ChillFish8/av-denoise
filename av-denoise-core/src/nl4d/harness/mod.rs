// Shared with the mc_accuracy bench, not a stable interface.
#![doc(hidden)]

mod score;
mod synth;

pub use self::score::{KindScore, Score, score};
pub use self::synth::{Clip, MotionClass, Still, synthesise};
