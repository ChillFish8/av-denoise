//! GPU kernels for the collaborative filter core.
//!
//! Collaborative filtering starts by grouping similar patches together,
//! then does its filtering work in a transform domain rather than on raw
//! pixels. `group` finds, for each reference patch, the patches in a
//! window around it that match it well enough to join its group.
//! `transforms` holds the primitives that move a patch, or a stack of
//! similar patches, in and out of the transform domain. An 8-point DCT
//! runs along each of a patch's two spatial axes, and a Haar transform
//! runs along the stack axis that groups similar patches together.
//! `filter_ht` runs a group through those transforms, shrinks the
//! coefficients with a hard threshold, and writes back the filtered
//! reference patch, the shrinkage step collaborative filtering is built
//! around. `filter_wiener` runs a second such pass, steering a softer
//! Wiener shrinkage with a pilot estimate instead of thresholding hard,
//! which preserves detail a hard threshold alone would lose.
//! `aggregate` then blends every reference's filtered patch back
//! onto one frame plane, weighted by how much its group agreed on it,
//! turning the overlapping per-reference estimates into a single output
//! value per pixel.

pub mod aggregate;
pub mod filter_ht;
pub mod filter_wiener;
pub mod group;
pub mod transforms;
