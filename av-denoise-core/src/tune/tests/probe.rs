use std::sync::{Arc, Mutex};
use std::time::Duration;

use cubecl::prelude::*;

use super::runtime::TestRuntime;
use crate::tune::{
    CompareResult,
    DeviceLimits,
    Families,
    FamilyId,
    Fidelity,
    FlatFamily,
    LaunchShape,
    Legality,
    SiteCandidate,
    SiteKey,
    VariantSite,
};

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
        int64_atomic_add: false,
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

    fn families() -> Families<Self> {
        Families::Flat(&[FlatFamily {
            id: FamilyId("all"),
            representative: ProbeCandidate::Small,
            members: Self::ALL,
        }])
    }
}

/// A site that records which candidate each launch ran instead of touching the GPU.
#[derive(Clone)]
pub(crate) struct ProbeSite {
    pub(crate) client: ComputeClient<TestRuntime>,
    pub(crate) width: u32,
    pub(crate) baseline_shared: usize,
    /// The narrowest plane the site's baseline works on.
    pub(crate) baseline_plane: u32,
    /// The baseline this site reports instead of the first listed candidate.
    pub(crate) site_baseline: Option<ProbeCandidate>,
    /// Candidates this site never allows.
    pub(crate) illegal: Vec<ProbeCandidate>,
    pub(crate) launched: Arc<Mutex<Vec<ProbeCandidate>>>,
    /// Whether allocating scratch outputs panics, which makes any tuning of this site fail.
    pub(crate) fail_scratch: bool,
    /// Candidates whose launch panics.
    pub(crate) panicking: Vec<ProbeCandidate>,
    /// How long each launch blocks the calling thread.
    pub(crate) launch_delay: Duration,
    /// Whether this is a copy made by [VariantSite::on_client].
    pub(crate) copy: bool,
    /// Whether every compile on a copy panics, as a helper whose device was lost would fail.
    pub(crate) copy_fails: bool,
    /// How long each compile on a copy blocks the calling thread.
    pub(crate) copy_delay: Duration,
    /// The candidates compiled on copies.
    pub(crate) copied: Arc<Mutex<Vec<ProbeCandidate>>>,
    /// The candidates compiled so far, kept across runs like the on-disk compilation cache.
    ///
    /// With it set, a launch of a candidate it lacks waits `cold_delay` and adds it.
    pub(crate) cache: Option<Arc<Mutex<Vec<ProbeCandidate>>>>,
    pub(crate) cold_delay: Duration,
}

impl ProbeSite {
    pub(crate) fn new(client: &ComputeClient<TestRuntime>) -> Self {
        Self {
            client: client.clone(),
            width: 64,
            baseline_shared: 1024,
            baseline_plane: 1,
            site_baseline: None,
            illegal: Vec::new(),
            launched: Arc::new(Mutex::new(Vec::new())),
            fail_scratch: false,
            panicking: Vec::new(),
            launch_delay: Duration::ZERO,
            copy: false,
            copy_fails: false,
            copy_delay: Duration::ZERO,
            copied: Arc::new(Mutex::new(Vec::new())),
            cache: None,
            cold_delay: Duration::ZERO,
        }
    }

    /// How long launching `candidate` blocks, which is `cold_delay` the first time a cache sees it.
    fn delay_for(&self, candidate: ProbeCandidate) -> Duration {
        let Some(cache) = &self.cache else {
            return self.launch_delay;
        };

        let mut cached = cache.lock().expect("probe lock");
        if cached.contains(&candidate) {
            return self.launch_delay;
        }

        cached.push(candidate);
        self.cold_delay
    }

    pub(crate) fn launches(&self) -> Vec<ProbeCandidate> {
        self.launched.lock().expect("probe lock").clone()
    }

    /// The candidates compiled on copies, sorted.
    pub(crate) fn copied(&self) -> Vec<ProbeCandidate> {
        let mut copied = self.copied.lock().expect("probe lock").clone();
        copied.sort_by_key(|candidate| candidate.name());
        copied
    }
}

impl VariantSite<TestRuntime> for ProbeSite {
    type Candidate = ProbeCandidate;
    type Outputs = ();

    const COPIES: bool = true;

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
        let baseline = VariantSite::baseline(self);
        if candidate == baseline {
            return self.baseline_plane;
        }

        match candidate {
            ProbeCandidate::Medium => 16,
            _ => 1,
        }
    }

    fn legal(&self, candidate: ProbeCandidate, _limits: &DeviceLimits, _legality: &Legality) -> bool {
        !self.illegal.contains(&candidate)
    }

    fn baseline(&self) -> ProbeCandidate {
        self.site_baseline.unwrap_or_else(ProbeCandidate::baseline)
    }

    fn launch(&self, candidate: ProbeCandidate, _outputs: &()) {
        let delay = self.delay_for(candidate);
        std::thread::sleep(delay);

        if self.panicking.contains(&candidate) {
            panic!("the {} probe launch panicked", candidate.name());
        }

        self.launched.lock().expect("probe lock").push(candidate);
    }

    fn real_outputs(&self) {}

    fn scratch_outputs(&self) {
        if self.fail_scratch {
            panic!("scratch allocation failed");
        }
    }

    fn clear_outputs(&self, _outputs: &()) {}

    fn on_client(&self, client: &ComputeClient<TestRuntime>) -> Option<Self> {
        Some(Self {
            client: client.clone(),
            copy: true,
            ..self.clone()
        })
    }

    fn compile_only(&self, candidate: ProbeCandidate, outputs: &()) {
        if !self.copy {
            self.launch(candidate, outputs);
            return;
        }

        std::thread::sleep(self.copy_delay);

        if self.copy_fails || self.panicking.contains(&candidate) {
            panic!("the {} probe compile panicked", candidate.name());
        }

        self.copied.lock().expect("probe lock").push(candidate);
    }

    fn compare(
        &self,
        _baseline: &(),
        _candidate: &(),
        _fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        Ok(CompareResult::Pass)
    }
}
