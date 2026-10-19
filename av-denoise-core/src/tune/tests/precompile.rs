use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::fixtures::{CollabSpec, collab_site};
use super::probe::{ProbeCandidate, ProbeSite};
use super::runtime::{TestRuntime, test_client, test_legality};
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::{bind, profile_launches, sync, zero_words};
use crate::tune::precompile::{fits_in_one_batch, precompile};
use crate::tune::sites::{CollabOutputs, CollabSite, CollabVariant};
use crate::tune::{
    CompareResult,
    DeviceLimits,
    Fidelity,
    LaunchShape,
    SiteCandidate,
    SiteKey,
    VariantSite,
    legal_candidates,
};

/// A compile allowance no test runs out of.
const NO_CEILING: Duration = Duration::MAX;

#[test]
fn precompiling_launches_each_candidate_once_in_order() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let allowed = ProbeCandidate::ALL;

    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), NO_CEILING, Duration::ZERO, &[]);
    let launches = site.launches();

    assert_eq!(precompiled.kept, vec![0, 1, 2]);
    assert!(precompiled.dropped.is_empty());
    assert!(precompiled.skipped.is_empty());
    assert_eq!(
        launches,
        vec![
            ProbeCandidate::Small,
            ProbeCandidate::Medium,
            ProbeCandidate::Wide
        ]
    );
}

#[test]
fn a_candidate_whose_launch_panics_is_dropped_with_a_reason() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.panicking = vec![ProbeCandidate::Medium];
    let allowed = ProbeCandidate::ALL;

    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), NO_CEILING, Duration::ZERO, &[]);
    let reason = precompiled
        .reason_for(1)
        .expect("the medium candidate was dropped");

    assert_eq!(precompiled.kept, vec![0, 2]);
    assert!(reason.contains("medium"), "{reason}");
}

#[test]
fn a_spent_compile_ceiling_compiles_only_the_baseline() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let allowed = ProbeCandidate::ALL;

    let precompiled = precompile(
        &site,
        allowed,
        &[0, 1, 2],
        &(),
        Duration::ZERO,
        Duration::ZERO,
        &[],
    );
    let launches = site.launches();

    assert_eq!(precompiled.kept, vec![0]);
    assert_eq!(precompiled.skipped, vec![1, 2]);
    assert!(precompiled.dropped.is_empty());
    assert_eq!(launches, vec![ProbeCandidate::Small]);
}

#[test]
fn a_later_stage_past_the_compile_ceiling_compiles_nothing() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let allowed = ProbeCandidate::ALL;

    let precompiled = precompile(&site, allowed, &[1, 2], &(), Duration::ZERO, Duration::ZERO, &[]);

    assert!(precompiled.kept.is_empty());
    assert_eq!(precompiled.skipped, vec![1, 2]);
    assert!(site.launches().is_empty());
}

/// Each launch takes 200 ms, so the ceiling runs out after the second.
#[test]
fn the_compile_ceiling_skips_the_candidates_left_once_it_runs_out() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.launch_delay = Duration::from_millis(200);
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(300);

    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), allowance, Duration::ZERO, &[]);

    assert_eq!(precompiled.kept, vec![0, 1]);
    assert_eq!(precompiled.skipped, vec![2]);
    assert!(precompiled.spent >= Duration::from_millis(400));
}

/// The batch spends 400 ms before the medium launch panics, and the one-by-one retry of the
/// baseline spends the rest of the ceiling.
#[test]
fn the_compile_ceiling_also_cuts_the_one_by_one_retry_short() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.launch_delay = Duration::from_millis(200);
    site.panicking = vec![ProbeCandidate::Medium];
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(500);

    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), allowance, Duration::ZERO, &[]);

    assert_eq!(precompiled.kept, vec![0]);
    assert_eq!(precompiled.skipped, vec![1, 2]);
    assert!(precompiled.dropped.is_empty());
}

#[test]
fn a_ceiling_that_covers_every_estimated_compile_compiles_in_one_batch() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_secs(10);
    let estimate = Duration::from_secs(2);

    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), allowance, estimate, &[]);

    assert_eq!(precompiled.kept, vec![0, 1, 2, 3]);
    assert!(!precompiled.one_by_one);
}

/// Each launch takes 200 ms against a 500 ms ceiling, so the third launch is the last to start.
#[test]
fn a_short_compile_ceiling_compiles_one_candidate_at_a_time_and_stops_within_one_compile() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    let launch_delay = Duration::from_millis(200);
    site.launch_delay = launch_delay;
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(500);
    let slack = Duration::from_millis(100);

    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), allowance, launch_delay, &[]);

    assert!(precompiled.one_by_one);
    assert_eq!(precompiled.kept, vec![0, 1, 2]);
    assert_eq!(precompiled.skipped, vec![3]);
    assert!(
        precompiled.spent < allowance + launch_delay + slack,
        "{:?}",
        precompiled.spent
    );
}

#[test]
fn a_stage_batches_only_while_the_ceiling_outlasts_every_estimated_compile() {
    let estimate = Duration::from_secs(2);

    assert!(fits_in_one_batch(Duration::from_secs(9), estimate, 4));
    assert!(!fits_in_one_batch(Duration::from_secs(8), estimate, 4));
    assert!(fits_in_one_batch(Duration::MAX, estimate, usize::MAX));
    assert!(!fits_in_one_batch(Duration::ZERO, Duration::ZERO, 4));
}

/// Asks for a shared-memory buffer of this many floats.
#[cube(launch_unchecked)]
fn shared_probe(output: &mut Array<f32>, #[comptime] floats: u32) {
    let mut scratch = SharedMemory::<f32>::new(floats as usize);
    let lane = UNIT_POS as usize;
    scratch[lane] = 1.0;
    sync_cube();
    output[lane] = scratch[lane];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SharedProbe {
    Fits,
    Oversized,
}

impl SiteCandidate for SharedProbe {
    const SITE: &'static str = "shared_probe";
    const ALL: &'static [Self] = &[SharedProbe::Fits, SharedProbe::Oversized];

    fn name(self) -> &'static str {
        match self {
            SharedProbe::Fits => "fits",
            SharedProbe::Oversized => "oversized",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// Floats in the shared probe's output and in its fitting candidate's shared buffer.
const SHARED_PROBE_LANES: u32 = 64;

/// A site whose second candidate asks for more shared memory than the device has.
///
/// It reports no shared memory, so the fit check lets the oversized candidate through to the compiler.
#[derive(Clone)]
pub(super) struct SharedSite {
    client: ComputeClient<TestRuntime>,
    oversized_floats: u32,
}

impl SharedSite {
    pub(super) fn new(client: &ComputeClient<TestRuntime>) -> Self {
        let limits = DeviceLimits::from_client(client);
        let device_floats = limits.max_shared_bytes / size_of::<f32>();
        let oversized_floats = (device_floats + 1024) as u32;

        Self {
            client: client.clone(),
            oversized_floats,
        }
    }
}

impl VariantSite<TestRuntime> for SharedSite {
    type Candidate = SharedProbe;
    type Outputs = Handle;

    fn client(&self) -> &ComputeClient<TestRuntime> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(SharedProbe::SITE, SHARED_PROBE_LANES, 1).build()
    }

    fn shape(&self, _candidate: SharedProbe) -> LaunchShape {
        LaunchShape {
            x: SHARED_PROBE_LANES,
            y: 1,
        }
    }

    fn shared_bytes(&self, _candidate: SharedProbe) -> usize {
        0
    }

    fn launch(&self, candidate: SharedProbe, outputs: &Handle) {
        let floats = match candidate {
            SharedProbe::Fits => SHARED_PROBE_LANES,
            SharedProbe::Oversized => self.oversized_floats,
        };
        let output = bind::<TestRuntime>(outputs, SHARED_PROBE_LANES as usize);
        let cube_count = CubeCount::new_single();
        let cube_dim = CubeDim::new_1d(SHARED_PROBE_LANES);

        // SAFETY: the output holds one float per lane, and each lane writes only its own.
        unsafe {
            shared_probe::launch_unchecked::<TestRuntime>(&self.client, cube_count, cube_dim, output, floats);
        }
    }

    fn real_outputs(&self) -> Handle {
        self.scratch_outputs()
    }

    fn scratch_outputs(&self) -> Handle {
        let bytes = SHARED_PROBE_LANES as usize * size_of::<f32>();
        let output = self.client.empty(bytes);
        zero_words(&self.client, &output, SHARED_PROBE_LANES as usize);

        output
    }

    fn clear_outputs(&self, outputs: &Handle) {
        zero_words(&self.client, outputs, SHARED_PROBE_LANES as usize);
    }

    fn compare(
        &self,
        _baseline: &Handle,
        _candidate: &Handle,
        _fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        Ok(CompareResult::Pass)
    }
}

#[test]
fn a_candidate_that_cannot_compile_is_dropped_with_a_reason() {
    let client = test_client();
    let site = SharedSite::new(&client);
    let outputs = site.scratch_outputs();

    let precompiled = precompile(
        &site,
        SharedProbe::ALL,
        &[0, 1],
        &outputs,
        NO_CEILING,
        Duration::ZERO,
        &[],
    );
    let reason = precompiled
        .reason_for(1)
        .expect("the oversized candidate was dropped");

    assert_eq!(precompiled.kept, vec![0]);
    assert!(!reason.is_empty(), "{reason}");
    sync(&client).expect("the device syncs after the failed compile");
}

/// Builds the client with the compilation cache off, so every first launch in this process compiles.
pub(super) fn uncached_client() -> ComputeClient<TestRuntime> {
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = None;

    let set = AssertUnwindSafe(|| CubeClRuntimeConfig::set(config));
    let installed = catch_unwind(set);
    assert!(
        installed.is_ok(),
        "this test needs a process of its own, run it with nextest"
    );

    test_client()
}

/// The wall time of `candidate`'s first timed launch, compile included.
pub(super) fn first_timed_launch(
    site: &CollabSite<TestRuntime, f32>,
    candidate: CollabVariant,
    outputs: &CollabOutputs,
) -> Duration {
    let started = Instant::now();
    profile_launches(site, candidate, outputs, 1).expect("the launch is timed");

    started.elapsed()
}

#[test]
fn a_precompiled_candidate_compiles_nothing_at_its_first_timed_launch() {
    let client = uncached_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Luma, 1);
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits::from_client(&client);
    let legality = test_legality(&client);
    let allowed = legal_candidates(&site, &limits, &legality).expect("the baseline fits");
    let outputs = site.scratch_outputs();
    let warmed = 1;
    let cold = 2;

    assert!(allowed.len() >= 3, "{} candidates", allowed.len());

    let precompiled = precompile(
        &site,
        &allowed,
        &[0, warmed],
        &outputs,
        NO_CEILING,
        Duration::ZERO,
        &[],
    );
    let warmed_first = first_timed_launch(&site, allowed[warmed], &outputs);
    let cold_first = first_timed_launch(&site, allowed[cold], &outputs);

    assert_eq!(precompiled.kept, vec![0, warmed]);
    assert!(
        warmed_first * 4 < cold_first,
        "pre-compiled first launch {warmed_first:?}, cold first launch {cold_first:?}"
    );
}
