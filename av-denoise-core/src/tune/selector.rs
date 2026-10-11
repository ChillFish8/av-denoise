use cubecl::prelude::*;

use super::device::DeviceLimits;
use super::key::SiteKey;
use super::plan::{KernelPlan, PipelineProfile};
use super::site::{SiteCandidate, VariantSite, fit_candidates};

/// What an engine needs to pick a candidate at each of its launch sites.
#[derive(Debug, Clone, Default)]
pub struct EngineKernels {
    pub plan: KernelPlan,
    pub profile: PipelineProfile,
}

/// Picks and launches the planned candidate at each site.
#[derive(Clone)]
pub struct KernelSelector<R: Runtime> {
    client: ComputeClient<R>,
    limits: DeviceLimits,
    plan: KernelPlan,
    profile: PipelineProfile,
}

impl<R: Runtime> KernelSelector<R> {
    /// A selector that runs every site's baseline.
    pub fn baseline(client: &ComputeClient<R>) -> Self {
        let kernels = EngineKernels::default();
        Self::new(client, kernels)
    }

    pub fn new(client: &ComputeClient<R>, kernels: EngineKernels) -> Self {
        let limits = DeviceLimits::from_client(client);
        Self::with_limits(client, kernels, limits)
    }

    /// A selector that checks candidates against `limits` instead of the client's own.
    pub fn with_limits(client: &ComputeClient<R>, kernels: EngineKernels, limits: DeviceLimits) -> Self {
        Self {
            client: client.clone(),
            limits,
            plan: kernels.plan,
            profile: kernels.profile,
        }
    }

    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub fn limits(&self) -> &DeviceLimits {
        &self.limits
    }

    pub fn profile(&self) -> PipelineProfile {
        self.profile
    }

    /// Launches the planned candidate of `site` on its real outputs.
    ///
    /// A key with no plan entry, an unknown name or a candidate that does not fit runs the baseline.
    pub fn run<S: VariantSite<R>>(&self, site: &S) -> Result<(), anyhow::Error> {
        let fitted = fit_candidates(site, &self.limits)?;
        let key = site.key();
        let candidate = self.planned::<S>(&key, &fitted);
        let outputs = site.real_outputs();

        site.launch(candidate, &outputs);

        Ok(())
    }

    fn planned<S: VariantSite<R>>(&self, key: &SiteKey, allowed: &[S::Candidate]) -> S::Candidate {
        let named = self.plan.get(key);
        let parsed = named.and_then(S::Candidate::from_name);

        match parsed {
            Some(candidate) if allowed.contains(&candidate) => candidate,
            _ => S::Candidate::baseline(),
        }
    }
}
