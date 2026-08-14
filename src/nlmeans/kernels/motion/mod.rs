//! The GPU kernels that track motion between frames.
//!
//! Temporal denoising averages a pixel with the same position in nearby
//! frames. When the picture moves, that position holds different content
//! in each frame, which blurs moving edges.
//!
//! These kernels work out where each block of pixels moved to, then
//! shift the neighbouring frames back into line before the denoising
//! weights are computed.
//!
//! `downscale` builds the image pyramid, `block_match` searches each
//! level for the best match, `chain` joins adjacent-frame results into
//! one longer motion vector, and `warp` shifts a frame by the field that
//! comes out.

mod block_match;
mod chain;
mod downscale;
mod warp;

pub use block_match::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};
pub use chain::{nlm_mc_chain_compose, nlm_mc_pair_zero};
pub use downscale::{nlm_mc_downscale, nlm_mc_extract_luma};
pub use warp::nlm_mc_warp;
