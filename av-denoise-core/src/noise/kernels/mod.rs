mod colour;
mod colour_fit;
mod curve;
mod derive;
mod history;
mod sample;
mod select;
mod stats;
mod window;

pub use self::colour::{
    noise_colour_accumulate,
    noise_colour_clear,
    noise_colour_levels,
    noise_colour_means,
    noise_colour_window,
};
pub(crate) use self::colour_fit::{colour_gain, colour_gain_pair};
pub use self::colour_fit::{
    noise_colour_fit_base,
    noise_colour_fit_ceilings,
    noise_colour_fit_gains,
    noise_colour_fit_kernel,
    noise_colour_fit_table,
};
pub use self::curve::noise_curve_bins;
pub use self::history::noise_record;
pub use self::sample::{noise_block_accept, noise_block_derive, noise_sample_finish};
#[cfg(test)]
pub(crate) use self::select::{key_value, order_key};
pub use self::select::{noise_select_histogram, noise_select_resolve};
pub use self::stats::{
    nlm_noise_partial,
    nlm_noise_reduce,
    nlm_temporal_noise_stats,
    nlm_temporal_stats_zero,
};
pub use self::window::noise_window_fold;
