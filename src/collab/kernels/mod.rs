//! GPU kernels for the collaborative filter core.
//!
//! Collaborative filtering starts by grouping similar patches together,
//! then does its filtering work in a transform domain rather than on raw
//! pixels. `group_temporal` finds, for each reference patch, the patches
//! that match it best in a window around it and in a motion-guided
//! window on each neighbour frame in a ring, so a group carries members
//! whose grain is independent of the reference frame's own. `group`
//! holds the position helpers that search shares with the kernels
//! downstream of it. `transforms` holds the primitives that move a
//! patch, or a stack of similar patches, in and out of the transform
//! domain. An 8-point DCT runs along each of a patch's two spatial axes,
//! and a Haar transform runs along the stack axis that groups similar
//! patches together. `filter_ht` runs a group through those transforms,
//! shrinks the coefficients with a hard threshold, and writes back the
//! filtered reference patch, the shrinkage step collaborative filtering
//! is built around. `aggregate` then blends every reference's filtered
//! patch back onto one frame plane, weighted by how much its group
//! agreed on it, turning the overlapping per-reference estimates into a
//! single output value per pixel.

pub mod aggregate;
pub mod filter_ht;
pub mod group;
pub mod group_temporal;
pub mod transforms;
