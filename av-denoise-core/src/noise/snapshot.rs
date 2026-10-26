use anyhow::anyhow;
use cubecl::prelude::*;

use super::NOISE_CURVE_BINS;
use super::derived::DerivedValues;
use super::fold::{
    STATE_CLASSES_PRESENT,
    STATE_CURVE_VALID,
    STATE_LOW,
    STATE_LOW_UNBOOSTED,
    STATE_MEDIAN,
    STATE_QUARTER_MEDIAN,
    STATE_RATIOS,
    STATE_RHO,
    STATE_RHO_VALID,
    STATE_TEMPORAL_ONLY,
};
use super::state::NoiseState;

/// A readback of one noise state's chains, curve and derived values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NoiseSnapshot {
    pub median: Option<[f32; 3]>,
    pub low: Option<[f32; 3]>,
    pub low_unboosted: Option<[f32; 3]>,
    pub temporal_only: Option<[f32; 3]>,
    pub rho: Option<f32>,
    /// The curve's ratios and the quarter median they are relative to.
    pub curve: Option<([f32; NOISE_CURVE_BINS], f32)>,
    pub classes_present: bool,
    pub derived: DerivedValues,
}

impl NoiseState {
    /// Reads the stream state and the derived values back, waiting for every queued fold.
    ///
    /// A state that holds fixed values has no chains, curve or classes.
    pub(crate) fn snapshot(&self) -> Result<NoiseSnapshot, anyhow::Error> {
        let derived_handle = self.derived().clone();
        let derived_bytes = self
            .client
            .read_one(derived_handle)
            .map_err(|error| anyhow!("the derived values read back failed: {error:?}"))?;
        let derived_words = f32::from_bytes(&derived_bytes);
        let derived = DerivedValues::from_words(derived_words);

        let Some(measurement) = self.measurement.as_ref() else {
            let snapshot = NoiseSnapshot {
                median: None,
                low: None,
                low_unboosted: None,
                temporal_only: None,
                rho: None,
                curve: None,
                classes_present: false,
                derived,
            };
            return Ok(snapshot);
        };

        let state_handle = measurement.fold_state.clone();
        let state_bytes = self
            .client
            .read_one(state_handle)
            .map_err(|error| anyhow!("the noise state read back failed: {error:?}"))?;
        let state = f32::from_bytes(&state_bytes);

        let curve_valid = state[STATE_CURVE_VALID as usize] != 0.0;
        let ratios_start = STATE_RATIOS as usize;
        let ratios_end = ratios_start + NOISE_CURVE_BINS;
        let curve = curve_valid.then(|| {
            let mut ratios = [0.0f32; NOISE_CURVE_BINS];
            ratios.copy_from_slice(&state[ratios_start..ratios_end]);

            (ratios, state[STATE_QUARTER_MEDIAN as usize])
        });

        let rho_valid = state[STATE_RHO_VALID as usize] != 0.0;
        let rho = rho_valid.then_some(state[STATE_RHO as usize]);

        let snapshot = NoiseSnapshot {
            median: chain(state, STATE_MEDIAN),
            low: chain(state, STATE_LOW),
            low_unboosted: chain(state, STATE_LOW_UNBOOSTED),
            temporal_only: chain(state, STATE_TEMPORAL_ONLY),
            rho,
            curve,
            classes_present: state[STATE_CLASSES_PRESENT as usize] != 0.0,
            derived,
        };

        Ok(snapshot)
    }
}

/// The three values of the chain starting at `start`, `None` while its flag is clear.
fn chain(state: &[f32], start: u32) -> Option<[f32; 3]> {
    let start = start as usize;
    let valid = state[start] != 0.0;

    valid.then(|| [state[start + 1], state[start + 2], state[start + 3]])
}
