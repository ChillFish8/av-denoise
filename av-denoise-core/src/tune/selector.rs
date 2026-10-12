use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;

use cubecl::prelude::*;

use super::device::DeviceLimits;
use super::fidelity::FidelityClass;
use super::gpu::profile_launches;
use super::key::SiteKey;
use super::ledger::{LedgerChoice, Observed, TuneLedger};
use super::plan::{KernelPlan, PipelineProfile};
use super::result::SiteRanking;
use super::session::{timed_count, tune_site};
use super::site::{SiteCandidate, VariantSite, fit_candidates};

/// What an engine needs to pick a candidate at each of its launch sites.
#[derive(Debug, Clone, Default)]
pub struct EngineKernels {
    pub plan: KernelPlan,
    pub profile: PipelineProfile,
    /// Shared tuning state, attached only while a tuning pass drives the engine.
    pub ledger: Option<Arc<TuneLedger>>,
}

/// Picks and launches the planned candidate at each site.
#[derive(Clone)]
pub struct KernelSelector<R: Runtime> {
    client: ComputeClient<R>,
    limits: DeviceLimits,
    plan: KernelPlan,
    profile: PipelineProfile,
    ledger: Option<Arc<TuneLedger>>,
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
            ledger: kernels.ledger,
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

    /// Launches the chosen candidate of `site` on its real outputs.
    ///
    /// A key with no choice, an unknown name or a candidate that does not fit runs the baseline.
    /// With a ledger attached, a census phase times the launch and a tune phase tunes the target key
    /// in place before its real launch.
    pub fn run<S: VariantSite<R>>(&self, site: &S) -> Result<(), anyhow::Error> {
        let fitted = fit_candidates(site, &self.limits)?;
        let outputs = site.real_outputs();

        if self.ledger.is_none() && self.plan.is_empty() {
            let baseline = S::Candidate::baseline();
            site.launch(baseline, &outputs);
            return Ok(());
        }

        let key = site.key();

        let Some(ledger) = self.ledger.as_ref() else {
            let named = self.plan.get(&key);
            let candidate = planned::<R, S>(&fitted, named);
            site.launch(candidate, &outputs);
            return Ok(());
        };

        let allowed = allowed_candidates::<R, S>(&fitted, ledger.deterministic());
        let timed = timed_count(site, &allowed);
        let observed = ledger.observe(&key, timed);

        match observed {
            Observed::Idle => {
                let candidate = self.ledger_planned::<S>(&key, &allowed, ledger);
                site.launch(candidate, &outputs);
            },
            Observed::Census => {
                let candidate = self.ledger_planned::<S>(&key, &allowed, ledger);
                self.census_launch(site, &outputs, &key, candidate, timed, ledger);
            },
            Observed::Tune { deadline } => {
                self.tune_key(site, &key, &allowed, deadline, ledger);
                let candidate = self.ledger_planned::<S>(&key, &allowed, ledger);
                site.launch(candidate, &outputs);
            },
        }

        Ok(())
    }

    /// Launches `candidate` once on the real outputs and adds its time to the census.
    ///
    /// When timing fails before the work ran, the launch runs untimed. When it fails after, the
    /// launch already happened and only a warning is logged.
    fn census_launch<S: VariantSite<R>>(
        &self,
        site: &S,
        outputs: &S::Outputs,
        key: &SiteKey,
        candidate: S::Candidate,
        candidates: usize,
        ledger: &TuneLedger,
    ) {
        let profiled = profile_launches(site, candidate, outputs, 1);

        match profiled {
            Ok(elapsed) => ledger.add_census(key, candidates, elapsed),
            Err(failure) if failure.work_ran => {
                tracing::warn!(error = %failure, %key, "timing a census launch failed after it ran");
            },
            Err(failure) => {
                tracing::warn!(error = %failure, %key, "timing a census launch failed, it runs untimed");
                site.launch(candidate, outputs);
            },
        }
    }

    /// Tunes `key` in place and records its ranking, or records a failure so its baseline runs.
    ///
    /// Tuning never fails the launch. An error or a panic inside it settles the target as failed.
    fn tune_key<S: VariantSite<R>>(
        &self,
        site: &S,
        key: &SiteKey,
        allowed: &[S::Candidate],
        deadline: Instant,
        ledger: &TuneLedger,
    ) {
        let guard = TargetGuard::new(ledger, key);
        let attempt = AssertUnwindSafe(|| tune_site(site, allowed, deadline));
        let outcome = catch_unwind(attempt);

        match outcome {
            Ok(Ok(ranking)) => {
                tracing::info!(%key, ranked = ranking.candidates.len(), "tuned a launch site");
                guard.record_ranking(ranking);
            },
            Ok(Err(error)) => {
                tracing::warn!(%key, %error, "tuning failed, the baseline runs for this key");
            },
            Err(_) => {
                tracing::warn!(%key, "tuning panicked, the baseline runs for this key");
            },
        }
    }

    /// The candidate the ledger names for `key`, falling back to the selector's own plan.
    ///
    /// A key whose tuning failed runs the baseline.
    fn ledger_planned<S: VariantSite<R>>(
        &self,
        key: &SiteKey,
        allowed: &[S::Candidate],
        ledger: &TuneLedger,
    ) -> S::Candidate {
        let choice = ledger.choice(key);
        let named = match &choice {
            LedgerChoice::Baseline => None,
            LedgerChoice::Named(name) => Some(name.as_str()),
            LedgerChoice::Unplanned => self.plan.get(key),
        };

        planned::<R, S>(allowed, named)
    }
}

/// Settles a claimed target as failed unless a ranking is recorded through it.
///
/// Dropping it while a panic unwinds settles the target as failed too.
struct TargetGuard<'a> {
    ledger: &'a TuneLedger,
    key: &'a SiteKey,
    ranked: bool,
}

impl<'a> TargetGuard<'a> {
    fn new(ledger: &'a TuneLedger, key: &'a SiteKey) -> Self {
        Self {
            ledger,
            key,
            ranked: false,
        }
    }

    fn record_ranking(mut self, ranking: SiteRanking) {
        let key = self.key.clone();
        self.ledger.record_ranking(key, ranking);
        self.ranked = true;
    }
}

impl Drop for TargetGuard<'_> {
    fn drop(&mut self) {
        if !self.ranked {
            let key = self.key.clone();
            self.ledger.record_failure(key);
        }
    }
}

/// The candidate `named` names, or the baseline when the name is missing, unknown or not in `allowed`.
fn planned<R: Runtime, S: VariantSite<R>>(allowed: &[S::Candidate], named: Option<&str>) -> S::Candidate {
    let parsed = named.and_then(S::Candidate::from_name);

    match parsed {
        Some(candidate) if allowed.contains(&candidate) => candidate,
        _ => S::Candidate::baseline(),
    }
}

/// The fitted candidates a run may use, only exact ones when `deterministic` is set.
fn allowed_candidates<R: Runtime, S: VariantSite<R>>(
    fitted: &[S::Candidate],
    deterministic: bool,
) -> Vec<S::Candidate> {
    fitted
        .iter()
        .copied()
        .filter(|candidate| !deterministic || candidate.fidelity().class() == FidelityClass::Exact)
        .collect()
}
