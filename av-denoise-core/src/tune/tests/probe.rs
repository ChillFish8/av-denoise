use std::sync::{Arc, Mutex};

use cubecl::prelude::*;

use super::runtime::TestRuntime;
use crate::tune::{CompareResult, DeviceLimits, Fidelity, LaunchShape, SiteCandidate, SiteKey, VariantSite};

/// A device with a 32 KB shared-memory cap and 16-wide planes, the Apple-class limits.
pub(crate) fn small_device_limits() -> DeviceLimits {
    DeviceLimits {
        plane_size_min: 16,
        plane_size_max: 16,
        max_shared_bytes: 32 * 1024,
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 64),
        compute_units: None,
        f16_buffer: false,
        f16_arithmetic: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbeCandidate {
    Small,
    Medium,
    Wide,
    Huge,
}

impl SiteCandidate for ProbeCandidate {
    const SITE: &'static str = "probe";
    const ALL: &'static [Self] = &[
        ProbeCandidate::Small,
        ProbeCandidate::Medium,
        ProbeCandidate::Wide,
        ProbeCandidate::Huge,
    ];

    fn name(self) -> &'static str {
        match self {
            ProbeCandidate::Small => "small",
            ProbeCandidate::Medium => "medium",
            ProbeCandidate::Wide => "wide",
            ProbeCandidate::Huge => "huge",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// A site that records which candidate each launch ran instead of touching the GPU.
#[derive(Clone)]
pub(crate) struct ProbeSite {
    pub(crate) client: ComputeClient<TestRuntime>,
    pub(crate) width: u32,
    pub(crate) baseline_shared: usize,
    pub(crate) baseline_plane: u32,
    pub(crate) launched: Arc<Mutex<Vec<ProbeCandidate>>>,
}

impl ProbeSite {
    pub(crate) fn new(client: &ComputeClient<TestRuntime>) -> Self {
        Self {
            client: client.clone(),
            width: 64,
            baseline_shared: 1024,
            baseline_plane: 1,
            launched: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn launches(&self) -> Vec<ProbeCandidate> {
        self.launched.lock().expect("probe lock").clone()
    }
}

impl VariantSite<TestRuntime> for ProbeSite {
    type Candidate = ProbeCandidate;
    type Outputs = ();

    fn client(&self) -> &ComputeClient<TestRuntime> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(ProbeCandidate::SITE, self.width, 64).build()
    }

    fn shape(&self, candidate: ProbeCandidate) -> LaunchShape {
        match candidate {
            ProbeCandidate::Wide => LaunchShape { x: 64, y: 16 },
            _ => LaunchShape { x: 32, y: 8 },
        }
    }

    fn shared_bytes(&self, candidate: ProbeCandidate) -> usize {
        match candidate {
            ProbeCandidate::Small => self.baseline_shared,
            ProbeCandidate::Medium => 24 * 1024,
            ProbeCandidate::Wide => 1024,
            ProbeCandidate::Huge => 48 * 1024,
        }
    }

    fn min_plane_size(&self, candidate: ProbeCandidate) -> u32 {
        match candidate {
            ProbeCandidate::Small => self.baseline_plane,
            ProbeCandidate::Medium => 16,
            _ => 1,
        }
    }

    fn launch(&self, candidate: ProbeCandidate, _outputs: &()) {
        self.launched.lock().expect("probe lock").push(candidate);
    }

    fn real_outputs(&self) {}

    fn scratch_outputs(&self) {}

    fn clear_outputs(&self, _outputs: &()) {}

    fn compare(
        &self,
        _baseline: &(),
        _candidate: &(),
        _fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        Ok(CompareResult::Pass)
    }
}
