//! Integration tests for the `nlmeans` module. Split by feature so
//! each file holds a coherent slice of behaviour:
//! - [`helpers`] — shared fixtures (`make_client`, frame builders).
//! - [`spatial`] — single-frame fused path: passthrough, strength,
//!   self-weight, symmetry, edge clamping.
//! - [`temporal`] — multi-frame ring behaviour: window priming, flush,
//!   frame-count parity.
//! - [`separable`] — high-`patch_radius` separable path.
//! - [`prefilter`] — external clip and bilateral prefilter paths.
//! - [`motion_compensation`] — MC end-to-end smoke tests.
//! - [`validation`] — `NlmParams::validate` accept/reject matrix.
//! - [`util`] — public helper roundtrips and numerical-edge regression.

mod helpers;

mod motion_compensation;
mod prefilter;
mod separable;
mod spatial;
mod temporal;
mod util;
mod validation;
