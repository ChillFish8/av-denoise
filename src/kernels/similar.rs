use cubecl::cube;
use cubecl::prelude::*;

#[cube]
/// Find the blocks most similar to the reference block at (ref_r, ref_c).
///
/// Similarity is the normalised squared Euclidean distance between
/// 2-D DCT representations:
///
/// d(P, Q) = ‖ T_{2D}(P) − T_{2D}(Q) ‖² / k²
///
/// Only blocks whose distance is ≤ tau_match are kept.
/// At most n_max blocks are returned (the n_max−1 most similar plus the
/// reference block, which is always included with d=0).
///
/// The group is then padded to the next power of 2 by repeating the last
/// (least similar) block.  This padding is required because the WHT only
/// accepts power-of-2 lengths.  Padded entries are excluded from the final
/// aggregation step.
pub(crate) fn find_similar_blocks() {}
