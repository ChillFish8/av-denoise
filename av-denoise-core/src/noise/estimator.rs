/// The smallest folded sigma, 0.1 in 8-bit terms.
///
/// An estimate near zero would send the derived strength to infinity.
pub(crate) const SIGMA_FLOOR: f32 = 0.1 / 255.0;
