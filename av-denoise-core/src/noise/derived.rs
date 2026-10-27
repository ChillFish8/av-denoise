#[cfg(test)]
use super::snapshot::NoiseSnapshot;
#[cfg(test)]
use super::state::StrengthMapKind;
#[cfg(test)]
use super::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};
#[cfg(test)]
use crate::collab::kernels::aggregate::weight_scale;
use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;

/// The per-channel smoothed sigma, 3 words.
///
/// It is the temporal-only chain, or the unboosted low chain until that has a sample. Channels
/// past the count read 0.
pub(crate) const DERIVED_SIGMA: u32 = 0;
/// `sigma0 * sigma0 * g_max * g_max`, or the floor when that is not finite or not above it.
pub(crate) const DERIVED_WEIGHT_SCALE: u32 = 3;
/// 1.0 when the current curve exists and the consumer applies it.
pub(crate) const DERIVED_CURVE_VALID: u32 = 4;
/// One of the `STRENGTH_MAP_*` modes as `f32`.
pub(crate) const DERIVED_MAP_MODE: u32 = 5;
/// The motion-compensation SAD floor, prefilter fraction included, 0 without motion compensation.
pub(crate) const DERIVED_SAD_FLOOR: u32 = 6;
/// The confidence-only pass's SAD floor, 0 without that pass.
pub(crate) const DERIVED_CONFIDENCE_SAD_FLOOR: u32 = 7;
pub(crate) const DERIVED_H2_INV_NORM: u32 = 8;
pub(crate) const DERIVED_PILOT_H2_INV_NORM: u32 = 9;
/// The main pass's noise offset, 0 under the NLM pilot.
pub(crate) const DERIVED_NOISE_OFFSET: u32 = 10;
pub(crate) const DERIVED_INPUT_NOISE_OFFSET: u32 = 11;
/// The smoothed grain correlation, 0 while unset.
pub(crate) const DERIVED_RHO: u32 = 12;
/// The `f32` words of the derived buffer.
pub(crate) const DERIVED_WORDS: u32 = 16;

/// The values every noise consumer reads, as the derived buffer holds them.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DerivedValues {
    pub sigma: [f32; 3],
    pub weight_scale: f32,
    pub curve_valid: bool,
    pub map_mode: u32,
    pub sad_noise_floor: f32,
    pub confidence_sad_noise_floor: f32,
    pub h2_inv_norm: f32,
    pub pilot_h2_inv_norm: f32,
    pub noise_offset: f32,
    pub input_noise_offset: f32,
    pub rho: f32,
}

impl DerivedValues {
    /// The buffer words, with flags as 0.0 or 1.0, the map mode as an exact `f32` and the spare
    /// words at 0.0.
    pub fn to_words(&self) -> [f32; DERIVED_WORDS as usize] {
        let mut words = [0.0f32; DERIVED_WORDS as usize];
        let sigma_start = DERIVED_SIGMA as usize;

        words[sigma_start..sigma_start + 3].copy_from_slice(&self.sigma);
        words[DERIVED_WEIGHT_SCALE as usize] = self.weight_scale;
        words[DERIVED_CURVE_VALID as usize] = if self.curve_valid { 1.0 } else { 0.0 };
        words[DERIVED_MAP_MODE as usize] = self.map_mode as f32;
        words[DERIVED_SAD_FLOOR as usize] = self.sad_noise_floor;
        words[DERIVED_CONFIDENCE_SAD_FLOOR as usize] = self.confidence_sad_noise_floor;
        words[DERIVED_H2_INV_NORM as usize] = self.h2_inv_norm;
        words[DERIVED_PILOT_H2_INV_NORM as usize] = self.pilot_h2_inv_norm;
        words[DERIVED_NOISE_OFFSET as usize] = self.noise_offset;
        words[DERIVED_INPUT_NOISE_OFFSET as usize] = self.input_noise_offset;
        words[DERIVED_RHO as usize] = self.rho;

        words
    }

    pub fn from_words(words: &[f32]) -> Self {
        let word_count = words.len();
        assert_eq!(word_count, DERIVED_WORDS as usize, "the derived buffer's words");

        let sigma_start = DERIVED_SIGMA as usize;
        let mut sigma = [0.0f32; 3];
        sigma.copy_from_slice(&words[sigma_start..sigma_start + 3]);

        Self {
            sigma,
            weight_scale: words[DERIVED_WEIGHT_SCALE as usize],
            curve_valid: words[DERIVED_CURVE_VALID as usize] != 0.0,
            map_mode: words[DERIVED_MAP_MODE as usize] as u32,
            sad_noise_floor: words[DERIVED_SAD_FLOOR as usize],
            confidence_sad_noise_floor: words[DERIVED_CONFIDENCE_SAD_FLOOR as usize],
            h2_inv_norm: words[DERIVED_H2_INV_NORM as usize],
            pilot_h2_inv_norm: words[DERIVED_PILOT_H2_INV_NORM as usize],
            noise_offset: words[DERIVED_NOISE_OFFSET as usize],
            input_noise_offset: words[DERIVED_INPUT_NOISE_OFFSET as usize],
            rho: words[DERIVED_RHO as usize],
        }
    }
}

/// What the consuming denoiser makes of the noise.
///
/// The default applies no curve, has no profile gain and clamps the weight scale at
/// [RECIPROCAL_FLOOR](crate::collab::kernels::transforms::RECIPROCAL_FLOOR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseConsumer {
    pub curve_applies: bool,
    /// The largest DCT profile gain, 0.0 when nothing reads the weight scale.
    pub weight_g_max: f32,
    pub weight_floor: f32,
}

impl Default for NoiseConsumer {
    fn default() -> Self {
        Self {
            curve_applies: false,
            weight_g_max: 0.0,
            weight_floor: RECIPROCAL_FLOOR,
        }
    }
}

/// The host-computed factors the fold turns smoothed sigmas into derived values with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct DerivedCoefficients {
    /// `h2 = h2_per_inverse_variance * count / sum(sigma^2)`, or `None` for a fixed strength.
    pub h2_per_inverse_variance: Option<f32>,
    pub h2_fixed: f32,
    /// `1 / strength_scale^2` under the NLM pilot, else 1.0.
    pub pilot_h2_factor: f32,
    /// `offset = offset_per_variance * sum(low^2)`, 0.0 with the noise floor off.
    pub offset_per_variance: f32,
    /// Whether the main pass subtracts the offset, false under `NlmSpatial`.
    pub main_offset: bool,
    pub sad_per_sigma: f32,
    pub confidence_sad_per_sigma: f32,
}

#[cfg(test)]
impl DerivedCoefficients {
    /// The derived values the fold writes for `snapshot`'s chains, rho, curve and classes, in the
    /// fold's own arithmetic.
    pub(crate) fn derive_host(
        &self,
        snapshot: &NoiseSnapshot,
        channels: usize,
        consumer: NoiseConsumer,
        map: StrengthMapKind,
    ) -> DerivedValues {
        let unboosted = snapshot.low_unboosted.unwrap_or_default();
        let sigma = snapshot.temporal_only.unwrap_or(unboosted);
        let median = snapshot.median.unwrap_or_default();
        let low = snapshot.low.unwrap_or_default();

        let floor = consumer.weight_floor;
        assert_eq!(
            floor, RECIPROCAL_FLOOR,
            "the weight scale clamps at the reciprocal floor"
        );

        let profile = [consumer.weight_g_max; 8];
        let weight = weight_scale(sigma[0], &profile);

        let curve_valid = snapshot.curve.is_some() && consumer.curve_applies;
        let map_mode = match map {
            StrengthMapKind::Luma(_) if curve_valid => STRENGTH_MAP_LUMA,
            StrengthMapKind::Chroma { .. } if snapshot.classes_present => STRENGTH_MAP_ALL,
            _ => STRENGTH_MAP_OFF,
        };

        let mut median_sum = 0.0f32;
        let mut low_sum = 0.0f32;
        for channel in 0..channels {
            median_sum += median[channel] * median[channel];
            low_sum += low[channel] * low[channel];
        }

        let h2_inv_norm = match self.h2_per_inverse_variance {
            Some(per_inverse_variance) => per_inverse_variance * channels as f32 / median_sum,
            None => self.h2_fixed,
        };
        let input_noise_offset = self.offset_per_variance * low_sum;
        let noise_offset = if self.main_offset { input_noise_offset } else { 0.0 };

        DerivedValues {
            sigma,
            weight_scale: weight,
            curve_valid,
            map_mode,
            sad_noise_floor: self.sad_per_sigma * median[0],
            confidence_sad_noise_floor: self.confidence_sad_per_sigma * median[0],
            h2_inv_norm,
            pilot_h2_inv_norm: h2_inv_norm * self.pilot_h2_factor,
            noise_offset,
            input_noise_offset,
            rho: snapshot.rho.unwrap_or(0.0),
        }
    }
}
