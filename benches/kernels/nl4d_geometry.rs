//! The search geometry `nl4d` ships with, in one place.
//!
//! Every value here comes from `Nl4dParams::default()`. The grouping
//! bench and the hard-threshold bench both build their frame ring and
//! member buffers from these, so a single set of constants describes
//! what the denoiser actually runs. Two copies of the shipped geometry
//! is how one of them goes stale.

/// `Nl4dParams::default().temporal_radius`.
pub const RADIUS: u32 = 2;
/// `Nl4dParams::default().refine`.
pub const REFINE: u32 = 2;
/// `Nl4dParams::default().spatial_radius`.
pub const SPATIAL_RADIUS: u32 = 9;
/// `collab::MAX_K`, the group size the filter runs at.
pub const K_MAX: u32 = 8;
/// `Nl4dParams::default().lambda_ht`.
pub const LAMBDA_HT: f32 = 5.3;
/// `Nl4dParams::default().confidence_variance`, the `use_member_sigma`
/// flag `collab_filter_ht` compiles against.
pub const CONFIDENCE_VARIANCE: bool = true;

/// The motion field's block stride. Held at `collab::PATCH_SIZE` so a
/// block boundary lines up with a patch boundary.
pub const BLK_STEP: u32 = 8;
/// The library's own default motion block side length
/// (`MotionCompensationMode::Mvtools`'s `blksize`), distinct from
/// [`BLK_STEP`] above.
pub const BLKSIZE: u32 = 16;
/// `thsad(BLKSIZE, 1.0)` in normalised SAD units (block_area *
/// THSAD_PIXEL, see `crate::nlmeans::motion::thsad`), hand-computed
/// here since that function is crate-private.
pub const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;

/// Frames in the ring a pass reads.
pub const N_FRAMES: u32 = 2 * RADIUS + 1;
/// The physical ring slot a pass is centred on.
pub const CENTRE_SLOT: u32 = RADIUS;

/// The centre slot is skipped, and physical slots run `0..N_FRAMES`, so
/// `NEIGHBOUR_SLOTS[t]` for the neighbour at temporal offset `k` is
/// `k + RADIUS`, laid out in the same `neighbour_idx_for_k` order
/// `crate::nlmeans::motion::chain` uses: ascending k on the negative
/// side first, then ascending k on the positive side.
pub const NEIGHBOUR_SLOTS: [u32; (2 * RADIUS) as usize] = [0, 1, 3, 4];

/// Sigma the hard-threshold bench filters at.
pub const SIGMA: f32 = 0.02;
