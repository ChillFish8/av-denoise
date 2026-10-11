use std::fmt::Debug;

use cubecl::prelude::*;

use super::device::DeviceLimits;
use super::fidelity::{CompareResult, Fidelity};
use super::key::SiteKey;

/// One way to run a site, with the stable name the store and the benches use.
pub trait SiteCandidate: Copy + Eq + Debug + Send + Sync + 'static {
    /// The id of the site this candidate belongs to.
    const SITE: &'static str;
    /// Every candidate, baseline first.
    const ALL: &'static [Self];

    fn name(self) -> &'static str;

    fn fidelity(self) -> Fidelity;

    /// The candidate that matches the original kernel and launch shape exactly.
    fn baseline() -> Self {
        Self::ALL[0]
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|candidate| candidate.name() == name)
    }

    fn is_baseline(self) -> bool {
        self == Self::baseline()
    }
}

/// The cube shape a candidate launches with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchShape {
    pub x: u32,
    pub y: u32,
}

impl LaunchShape {
    /// Threads per cube.
    pub fn units(self) -> u32 {
        self.x * self.y
    }
}

/// One tunable launch point, holding the buffers and values its kernels read.
///
/// `launch` writes only to the outputs it is handed.
pub trait VariantSite<R: Runtime>: Clone + Send {
    type Candidate: SiteCandidate;
    type Outputs: Clone + Send;

    /// Whether every candidate takes a few short samples before full timing.
    const PRESCREEN: bool = false;

    fn client(&self) -> &ComputeClient<R>;

    fn key(&self) -> SiteKey;

    fn shape(&self, candidate: Self::Candidate) -> LaunchShape;

    /// The bytes of every shared-memory buffer the candidate's kernel allocates.
    fn shared_bytes(&self, candidate: Self::Candidate) -> usize;

    /// The narrowest plane the candidate's kernel works on.
    fn min_plane_size(&self, _candidate: Self::Candidate) -> u32 {
        1
    }

    fn launch(&self, candidate: Self::Candidate, outputs: &Self::Outputs);

    /// The pipeline's own output buffers.
    fn real_outputs(&self) -> Self::Outputs;

    /// Freshly allocated outputs shaped like the real ones, zeroed on the GPU.
    fn scratch_outputs(&self) -> Self::Outputs;

    /// Zeroes `outputs` on the GPU.
    fn clear_outputs(&self, outputs: &Self::Outputs);

    /// Reads both outputs back and checks the candidate's against the baseline's under `fidelity`.
    fn compare(
        &self,
        baseline: &Self::Outputs,
        candidate: &Self::Outputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error>;

    fn baseline(&self) -> Self::Candidate {
        Self::Candidate::baseline()
    }

    /// Whether the candidate's shared memory, thread count, cube shape and plane width fit `limits`.
    ///
    /// The baseline skips the plane width check, so it launches wherever the original kernel did.
    fn fits(&self, candidate: Self::Candidate, limits: &DeviceLimits) -> bool {
        let shape = self.shape(candidate);
        let shared_bytes = self.shared_bytes(candidate);
        let min_plane = self.min_plane_size(candidate);
        let (max_x, max_y, _) = limits.max_cube_dim;

        let shared_fits = shared_bytes <= limits.max_shared_bytes;
        let units_fit = shape.units() <= limits.max_units_per_cube;
        let dims_fit = shape.x <= max_x && shape.y <= max_y;
        let plane_fits = candidate.is_baseline() || limits.plane_size_min >= min_plane;

        shared_fits && units_fit && dims_fit && plane_fits
    }
}

/// A site whose baseline does not fit the device.
#[derive(Debug, thiserror::Error)]
#[error(
    "the {site} baseline does not fit this device. It needs {shared_bytes} bytes of shared memory and \
     {units} threads per cube"
)]
pub struct FitError {
    pub site: &'static str,
    pub shared_bytes: usize,
    pub units: u32,
}

/// Every candidate of `site` that fits `limits`, baseline first.
///
/// A baseline whose shared memory, thread count or cube shape does not fit is an error, as a launch
/// that cannot run is.
pub fn fit_candidates<R: Runtime, S: VariantSite<R>>(
    site: &S,
    limits: &DeviceLimits,
) -> Result<Vec<S::Candidate>, FitError> {
    let baseline = site.baseline();

    if !site.fits(baseline, limits) {
        let shape = site.shape(baseline);
        let shared_bytes = site.shared_bytes(baseline);

        return Err(FitError {
            site: S::Candidate::SITE,
            shared_bytes,
            units: shape.units(),
        });
    }

    let fitted = S::Candidate::ALL
        .iter()
        .copied()
        .filter(|&candidate| site.fits(candidate, limits))
        .collect();

    Ok(fitted)
}
