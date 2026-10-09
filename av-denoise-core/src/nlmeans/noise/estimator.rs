/// How much weight the newest frame's estimate carries when smoothing.
pub(in crate::nlmeans) const EMA_ALPHA: f32 = 0.2;
/// The smallest smoothed sigma, 0.1 in 8-bit terms.
///
/// An estimate near zero would send the derived strength to infinity.
pub(super) const SIGMA_FLOOR: f32 = 0.1 / 255.0;

/// The smoothed per-channel noise level for one stream.
///
/// Smoothing stops a single busy frame from spiking the strength, and the floor keeps near-clean
/// content on a usable normalisation factor.
#[derive(Debug, Default)]
pub(in crate::nlmeans) struct NoiseEstimator {
    ema: Option<Vec<f32>>,
}

impl NoiseEstimator {
    /// Folds per-channel sigmas into the running estimate and returns the smoothed result.
    ///
    /// The first call takes the sample outright. Every element is floored at [SIGMA_FLOOR].
    pub(in crate::nlmeans) fn update(&mut self, sigmas: &[f32]) -> &[f32] {
        match &mut self.ema {
            Some(ema) => {
                for (smoothed, &sample) in ema.iter_mut().zip(sigmas.iter()) {
                    *smoothed = (EMA_ALPHA * sample + (1.0 - EMA_ALPHA) * *smoothed).max(SIGMA_FLOOR);
                }
            },
            None => {
                let floored: Vec<f32> = sigmas.iter().map(|&sigma| sigma.max(SIGMA_FLOOR)).collect();
                self.ema = Some(floored);
            },
        }

        self.ema.as_deref().unwrap()
    }

    pub(in crate::nlmeans) fn reset(&mut self) {
        self.ema = None;
    }

    pub(in crate::nlmeans) fn current(&self) -> Option<&[f32]> {
        self.ema.as_deref()
    }
}
