//! GPU kernels that belong to nl4d alone.

mod regularise;

pub use regularise::{REGULARISE_CANDIDATES, nl4d_mv_regularise};
