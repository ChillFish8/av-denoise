use super::correlation::correlation_factor;
use super::curve::NoiseCurve;
use super::derived::DerivedValues;
use super::estimator::{EMA_ALPHA, NoiseEstimator};
use super::snapshot::NoiseSnapshot;
use super::strength_map::QuarterClasses;
use super::temporal::{TemporalNoiseReading, TemporalNoiseSample};

/// The host noise estimate of one stream.
///
/// It holds four smoothed chains, the smoothed correlation, and the curve and classes of the last
/// lender with a sample.
#[derive(Debug, Default)]
pub(crate) struct HostNoise {
    /// The median chain, which drives the strength.
    pub(crate) median: NoiseEstimator,
    /// The low chain, which reads lower quartiles and drives the noise offsets.
    pub(crate) low: NoiseEstimator,
    /// The low chain without the correlation boost.
    ///
    /// A consumer that squares sigma into a shrinkage threshold pays for an over-read twice, so
    /// it reads this instead.
    pub(crate) low_unboosted: NoiseEstimator,
    /// The temporal median alone, with no spatial maximum and no correlation boost.
    ///
    /// It only updates on a fold with a temporal sample.
    pub(crate) temporal_only: NoiseEstimator,
    /// The smoothed grain correlation between neighbouring pixels.
    ///
    /// The first temporal sample sets it directly so the opening frames are not under-corrected.
    /// `None` reads as white noise.
    pub(crate) rho: Option<f32>,
    pub(crate) curve: Option<NoiseCurve>,
    /// The quarter classes, present exactly when `curve` is.
    pub(crate) classes: Option<QuarterClasses>,
}

/// The smoothed chains a fold hands back for the strength and offset formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HostFold {
    pub(crate) median: [f32; 3],
    pub(crate) low: [f32; 3],
}

impl HostNoise {
    /// Folds one slot's readings into every chain.
    ///
    /// `immerkaer` holds the per-channel Immerkær sigmas from the slot's abs sums. Each chain takes
    /// the larger of an Immerkær reading and a temporal reading, so the temporal side can raise an
    /// estimate but an unreliable shot cannot lower it. `sigma_scale` applies after combining and
    /// before smoothing.
    pub(crate) fn fold(
        &mut self,
        immerkaer: [f32; 3],
        temporal: Option<TemporalNoiseSample>,
        immerkaer_low: [f32; 3],
        channels: usize,
        sigma_scale: f32,
    ) -> HostFold {
        let mut raw = immerkaer;
        let mut raw_low = immerkaer_low;
        let mut raw_low_unboosted = immerkaer_low;
        let mut raw_temporal_only: Option<[f32; 3]> = None;

        if let Some(sample) = temporal {
            let factor = correlation_factor(sample.rho);
            for c in 0..channels {
                raw[c] = raw[c].max(sample.sigma[c] * factor);
                raw_low[c] = raw_low[c].max(sample.sigma_low[c] * factor);
                raw_low_unboosted[c] = raw_low_unboosted[c].max(sample.sigma_low[c]);
            }

            raw_temporal_only = Some(sample.sigma);

            let rho = match self.rho {
                None => sample.rho,
                Some(previous) => EMA_ALPHA * sample.rho + (1.0 - EMA_ALPHA) * previous,
            };
            self.rho = Some(rho);
        }

        for c in 0..channels {
            raw[c] *= sigma_scale;
            raw_low[c] *= sigma_scale;
            raw_low_unboosted[c] *= sigma_scale;
        }

        if let Some(raw_temporal) = raw_temporal_only.as_mut() {
            for sigma in raw_temporal.iter_mut().take(channels) {
                *sigma *= sigma_scale;
            }
        }

        let updated = self.median.update(&raw[..channels]);
        let mut median = [0.0f32; 3];
        median[..channels].copy_from_slice(updated);

        let updated_low = self.low.update(&raw_low[..channels]);
        let mut low = [0.0f32; 3];
        low[..channels].copy_from_slice(updated_low);

        self.low_unboosted.update(&raw_low_unboosted[..channels]);

        if let Some(raw_temporal) = raw_temporal_only {
            self.temporal_only.update(&raw_temporal[..channels]);
        }

        HostFold { median, low }
    }

    /// Takes the lender's curve and classes when the lender has a sample.
    pub(crate) fn adopt(&mut self, lender: &TemporalNoiseReading) {
        if lender.sample.is_none() {
            return;
        }

        self.curve = lender.curve;
        self.classes = lender.classes.clone();
    }

    pub(crate) fn reset(&mut self) {
        self.median.reset();
        self.low.reset();
        self.low_unboosted.reset();
        self.temporal_only.reset();
        self.rho = None;
        self.curve = None;
        self.classes = None;
    }

    /// The median chain's per-channel sigma, zero before the first fold.
    pub(crate) fn sigmas(&self, channels: usize) -> [f32; 3] {
        chain_sigmas(&self.median, channels)
    }

    /// The unboosted low chain's per-channel sigma, zero before the first fold.
    pub(crate) fn sigmas_low_unboosted(&self, channels: usize) -> [f32; 3] {
        chain_sigmas(&self.low_unboosted, channels)
    }

    /// The temporal chain's per-channel sigma.
    ///
    /// Falls back to [Self::sigmas_low_unboosted] until a temporal sample lands.
    pub(crate) fn sigmas_temporal_only(&self, channels: usize) -> [f32; 3] {
        if self.temporal_only.current().is_none() {
            return self.sigmas_low_unboosted(channels);
        }

        chain_sigmas(&self.temporal_only, channels)
    }

    /// The state as a GPU snapshot holds it, with `derived` as its derived values.
    pub(crate) fn snapshot(&self, channels: usize, derived: DerivedValues) -> NoiseSnapshot {
        let curve = self.curve.map(|curve| (curve.ratios, curve.sigma_quarter_median));

        NoiseSnapshot {
            median: chain_snapshot(&self.median, channels),
            low: chain_snapshot(&self.low, channels),
            low_unboosted: chain_snapshot(&self.low_unboosted, channels),
            temporal_only: chain_snapshot(&self.temporal_only, channels),
            rho: self.rho,
            curve,
            classes_present: self.classes.is_some(),
            derived,
        }
    }
}

/// The chain's sigmas with channels past `channels` at 0, `None` before its first update.
fn chain_snapshot(chain: &NoiseEstimator, channels: usize) -> Option<[f32; 3]> {
    chain.current()?;

    let sigmas = chain_sigmas(chain, channels);
    Some(sigmas)
}

fn chain_sigmas(chain: &NoiseEstimator, channels: usize) -> [f32; 3] {
    let mut sigmas = [0.0f32; 3];
    if let Some(smoothed) = chain.current() {
        sigmas[..channels].copy_from_slice(&smoothed[..channels]);
    }

    sigmas
}
