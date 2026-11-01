use cubecl::prelude::*;

use super::profile_key::profile_key;
use super::selector::EngineKernels;
use crate::engine::{DevicePlane, Engine, EngineSpec, PassSpec, SampleFormat};
use crate::error::Error;

/// What one engine of a tuning set is built from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TuneUnit(pub(crate) EngineSpec);

impl TuneUnit {
    pub(crate) fn frames(specs: &[EngineSpec]) -> Vec<TuneUnit> {
        specs.iter().copied().map(TuneUnit).collect()
    }

    /// The passes the engine runs, luma before chroma.
    pub(crate) fn passes(&self) -> Vec<PassSpec> {
        let (luma, chroma) = self.0.passes();
        luma.into_iter().chain(chroma).collect()
    }

    pub(crate) fn is_nl4d(&self) -> bool {
        matches!(self.0, EngineSpec::Nl4d { .. })
    }

    /// How many planes each push and emit takes.
    pub(crate) fn plane_count(&self) -> usize {
        self.0.geometry().chroma.plane_count()
    }

    /// The width and height of plane `index`.
    pub(crate) fn plane_dims(&self, index: usize) -> (u32, u32) {
        self.0.geometry().plane_dims(index)
    }

    /// The bytes plane `index` takes in `format`.
    pub(crate) fn plane_bytes(&self, index: usize, format: SampleFormat) -> usize {
        let (width, height) = self.plane_dims(index);
        let pixels = u64::from(width) * u64::from(height);

        format.plane_bytes(pixels) as usize
    }

    pub(crate) fn input(&self) -> SampleFormat {
        self.0.geometry().input
    }

    pub(crate) fn output(&self) -> SampleFormat {
        self.0.geometry().output
    }

    pub(crate) fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        let geometry = self.0.geometry();
        geometry.check_planes(planes, format)
    }

    /// Builds the engine on `client`, picking each pass's launch site candidates from `kernels`.
    pub(crate) fn build(
        &self,
        client: &Client,
        kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<Box<dyn Engine>, Error> {
        self.0.build(client, kernels)
    }
}

/// The profile key of every pass of `units`, in unit order and luma before chroma.
pub(crate) fn unit_profile_keys(units: &[TuneUnit]) -> Vec<String> {
    units
        .iter()
        .flat_map(TuneUnit::passes)
        .map(|pass| profile_key(&pass))
        .collect()
}
