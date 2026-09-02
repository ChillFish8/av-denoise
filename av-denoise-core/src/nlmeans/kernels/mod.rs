//! The GPU kernels the denoiser launches.
//!
//! Everything here is cubecl code that runs on the device. The host side
//! decides which kernels to launch and with what arguments, which is the
//! `dispatch` module's job.
//!
//! A denoise pass measures how alike patches are, turns those distances
//! into weights, accumulates the weighted pixels, and then normalises
//! the result.
//!
//! `fused` does the first three steps in one kernel, which is the fast
//! path for small patches. `separable` splits the distance step into
//! horizontal and vertical passes so the cost stays linear as patches
//! grow. `accumulate` holds the shared accumulation and normalisation
//! steps.
//!
//! `bilateral` is the prefilter, `noise` measures the noise level,
//! [`motion`] tracks movement between frames, `memory` holds the copy,
//! zero, and wire-packing utilities, and `helpers` holds the small
//! pieces the kernels share.

mod accumulate;
mod bilateral;
mod fused;
pub(crate) mod helpers;
mod memory;
pub mod motion;
mod noise;
mod separable;

pub use accumulate::{nlm_accumulate, nlm_finish};
pub use bilateral::nlm_bilateral;
pub use fused::{
    nlm_dist_2d_weight,
    nlm_dist_2d_weight_ref,
    nlm_fused_pair_accumulate_window,
    nlm_fused_pair_accumulate_window_ref,
    nlm_fused_single_window,
    nlm_fused_single_window_ref,
};
pub use memory::{gpu_copy, gpu_pack_wire, gpu_zero_buffers};
pub use noise::{nlm_noise_partial, nlm_noise_reduce, nlm_temporal_noise_stats, nlm_temporal_stats_zero};
pub use separable::{
    nlm_distance,
    nlm_distance_pair,
    nlm_distance_pair_ref,
    nlm_distance_ref,
    nlm_horizontal_sum,
    nlm_horizontal_sum_pair,
    nlm_vertical_weight,
    nlm_vweight_pair_accumulate,
};
