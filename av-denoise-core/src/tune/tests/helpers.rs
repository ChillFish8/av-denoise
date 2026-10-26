use std::sync::Arc;
use std::time::Duration;

use cubecl::prelude::*;

use super::fixtures::{CollabSpec, collab_site};
use super::precompile::{SharedProbe, SharedSite, compile_limits, uncached_client};
use super::probe::{ProbeCandidate, ProbeSite};
use super::runtime::test_client;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::sync;
use crate::tune::precompile::precompile;
use crate::tune::{CompileHelpers, HelperFactory, SiteCandidate, TuneLedger, VariantSite};

/// A compile allowance no test runs out of.
const NO_CEILING: Duration = Duration::MAX;

/// Two helpers that are the test client itself, which is enough to drive the parallel path.
fn cloned_helpers(client: &Client) -> Vec<Client> {
    vec![client.clone(), client.clone()]
}

#[test]
fn helpers_compile_every_candidate_but_the_baseline_and_the_site_client_loads_each_in_order() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0, 1, 2]);
    assert!(precompiled.dropped.is_empty());
    assert!(precompiled.skipped.is_empty());
    assert_eq!(precompiled.helpers, 2);
    assert!(!precompiled.helpers_failed);
    assert_eq!(site.copied(), vec![ProbeCandidate::Medium, ProbeCandidate::Wide]);
    assert_eq!(
        site.launches(),
        vec![
            ProbeCandidate::Small,
            ProbeCandidate::Medium,
            ProbeCandidate::Wide
        ]
    );
}

#[test]
fn a_helper_past_the_ceiling_skips_what_it_had_not_started_and_the_rest_still_load() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.copy_delay = Duration::from_millis(100);
    let helpers = vec![client.clone()];
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(150);

    let limits = compile_limits(allowance, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), &limits, &helpers);

    assert_eq!(site.copied(), vec![ProbeCandidate::Medium, ProbeCandidate::Wide]);
    assert_eq!(precompiled.kept, vec![0, 1, 2]);
    assert_eq!(precompiled.skipped, vec![3]);
    assert!(precompiled.dropped.is_empty());
}

#[test]
fn retries_on_the_site_client_stop_at_the_ceiling() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.copy_fails = true;
    site.launch_delay = Duration::from_millis(100);
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(150);

    let limits = compile_limits(allowance, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0, 1]);
    assert_eq!(precompiled.skipped, vec![2, 3]);
    assert!(precompiled.dropped.is_empty());
}

#[test]
fn a_slow_load_past_the_ceiling_skips_the_helper_candidates_after_it() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.launch_delay = Duration::from_millis(1100);
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;
    let allowance = Duration::from_millis(50);

    let limits = compile_limits(allowance, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), &limits, &helpers);

    assert_eq!(site.copied().len(), 3);
    assert_eq!(precompiled.kept, vec![0, 1]);
    assert_eq!(precompiled.skipped, vec![2, 3]);
}

#[test]
fn a_candidate_that_fails_everywhere_is_dropped_with_the_site_clients_reason() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.panicking = vec![ProbeCandidate::Medium];
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), &limits, &helpers);
    let reason = precompiled
        .reason_for(1)
        .expect("the panicking candidate was dropped");

    assert_eq!(precompiled.kept, vec![0, 2]);
    assert!(reason.contains("launching medium panicked"), "{reason}");
}

#[test]
fn candidates_failing_helpers_could_not_compile_are_compiled_on_the_site_client() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.copy_fails = true;
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0, 1, 2, 3]);
    assert!(precompiled.dropped.is_empty());
    assert!(precompiled.skipped.is_empty());
    assert!(site.copied().is_empty());
    assert_eq!(site.launches().len(), 4);
    assert!(precompiled.helpers_failed);
}

#[test]
fn helpers_stay_in_use_when_the_site_client_fails_their_candidates_too() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.panicking = vec![ProbeCandidate::Medium, ProbeCandidate::Wide, ProbeCandidate::Huge];
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2, 3], &(), &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0]);
    assert_eq!(precompiled.dropped.len(), 3);
    assert!(!precompiled.helpers_failed);
}

#[test]
fn a_ledger_stops_handing_out_helpers_once_they_failed() {
    let client = test_client();
    let ledger = TuneLedger::new(false);
    let helper_client = client.clone();
    let factory: HelperFactory = Box::new(move || {
        let helpers = CompileHelpers::new(vec![helper_client]);
        Some(Arc::new(helpers))
    });
    ledger.set_compile_helpers(factory);

    let before = ledger.compile_helpers().len();
    ledger.fail_compile_helpers();
    let after = ledger.compile_helpers().len();

    assert_eq!(before, 1);
    assert_eq!(after, 0);
}

#[test]
fn with_the_compilation_cache_off_helpers_are_not_used() {
    let client = uncached_client();
    let site = ProbeSite::new(&client);
    let helpers = cloned_helpers(&client);
    let allowed = ProbeCandidate::ALL;

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, allowed, &[0, 1, 2], &(), &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0, 1, 2]);
    assert_eq!(precompiled.helpers, 0);
    assert!(site.copied().is_empty());
}

#[test]
fn collab_sizes_that_share_a_key_bucket_have_different_compile_identities() {
    let client = test_client();
    let narrow_spec = CollabSpec::new(61, 37, ChannelMode::Luma, 1);
    let wide_spec = CollabSpec::new(62, 37, ChannelMode::Luma, 1);
    let narrow = collab_site(&client, &narrow_spec);
    let wide = collab_site(&client, &wide_spec);
    let narrow_identity = narrow.compile_identity();
    let wide_identity = wide.compile_identity();

    assert_eq!(narrow.key(), wide.key());
    assert_ne!(narrow_identity, wide_identity);

    // ROCm can crash at exit after uploads that were never launched, and a sync avoids it.
    sync(&client).expect("the uploads finish");
}

#[test]
fn a_site_that_cannot_copy_itself_compiles_on_its_own_client_despite_helpers() {
    let client = test_client();
    let site = SharedSite::new(&client);
    let outputs = site.scratch_outputs();
    let helpers = cloned_helpers(&client);

    let limits = compile_limits(NO_CEILING, Duration::ZERO);
    let precompiled = precompile(&site, SharedProbe::ALL, &[0], &outputs, &limits, &helpers);

    assert_eq!(precompiled.kept, vec![0]);
    assert_eq!(precompiled.helpers, 0);
}

super::runtime::wgpu_only! {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    use super::precompile::first_timed_launch;
    use super::runtime::{test_adapter, test_legality};
    use crate::tune::{DeviceLimits, legal_candidates, vulkan_compile_helpers};

    /// Builds the test client on a fresh compilation cache of its own, so every kernel compiles.
    ///
    /// The directory is named by the process and the time, so no earlier run can have warmed it.
    fn private_cache_client() -> (Client, PathBuf) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH);
        let nanos = now.expect("time runs forward").as_nanos();
        let name = format!("av-denoise-helper-test-{}-{nanos}", std::process::id());
        let dir = std::env::temp_dir().join(name);
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = true;
        config.environment.path = CacheConfig::Directory(dir.clone());
        CubeClRuntimeConfig::set(config);

        let client = test_client();
        (client, dir)
    }

    fn two_helpers() -> CompileHelpers {
        let (instance, adapter) = test_adapter();
        let made = vulkan_compile_helpers(&instance, &adapter, 2);
        made.expect("the helpers' devices open")
    }

    /// Compiles one batch of a luma site `width` wide on the helpers, then checks that the site
    /// client loaded every candidate instead of compiling it, and that each matches the baseline.
    fn assert_helpers_compile_a_batch_the_site_client_loads(
        client: &Client,
        helpers: &CompileHelpers,
        width: u32,
    ) {
        let helper_clients = helpers.clients();
        let spec = CollabSpec::new(width, 37, ChannelMode::Luma, 1);
        let site = collab_site(client, &spec);
        let limits = DeviceLimits::from_client(client);
        let legality = test_legality(client);
        let legal = legal_candidates(&site, &limits, &legality);
        let allowed = legal.expect("the baseline fits");
        let indices = [0, 1, 2, 3];
        let cold = 4;
        let outputs = site.scratch_outputs();

        // The census compiles the baseline before any batch, so the batch's load time covers only
        // cache loads.
        site.launch(allowed[0], &outputs);
        sync(client).expect("the baseline compiles");

        let limits = compile_limits(NO_CEILING, Duration::ZERO);
        let precompiled = precompile(&site, &allowed, &indices, &outputs, &limits, helper_clients);

        assert_eq!(precompiled.kept, indices.to_vec());
        assert_eq!(precompiled.helpers, 2);

        let cold_first = first_timed_launch(&site, allowed[cold], &outputs);
        assert!(
            precompiled.load * 2 < cold_first,
            "width {width}: loading 4 candidates took {:?}, one cold compile {cold_first:?}",
            precompiled.load
        );

        let baseline_outputs = site.scratch_outputs();
        site.launch(allowed[0], &baseline_outputs);

        for &index in &indices[1..] {
            let candidate = allowed[index];
            let candidate_outputs = site.scratch_outputs();
            site.launch(candidate, &candidate_outputs);
            let compared = site
                .compare(&baseline_outputs, &candidate_outputs, candidate.fidelity())
                .expect("the outputs read back");

            assert!(compared.passed(), "{}: {compared:?}", candidate.name());
        }
    }

    #[test]
    fn helpers_compile_cold_candidates_that_the_site_client_then_loads_without_compiling() {
        let (client, cache_dir) = private_cache_client();
        let helpers = two_helpers();

        assert_helpers_compile_a_batch_the_site_client_loads(&client, &helpers, 61);

        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn two_exact_sizes_in_one_key_bucket_each_load_from_the_cache() {
        let (client, cache_dir) = private_cache_client();
        let helpers = two_helpers();

        assert_helpers_compile_a_batch_the_site_client_loads(&client, &helpers, 61);
        assert_helpers_compile_a_batch_the_site_client_loads(&client, &helpers, 62);

        let _ = std::fs::remove_dir_all(cache_dir);
    }
}
