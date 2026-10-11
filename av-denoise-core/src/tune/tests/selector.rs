use super::probe::{ProbeCandidate, ProbeSite, small_device_limits};
use super::runtime::test_client;
use crate::tune::{EngineKernels, KernelPlan, KernelSelector, SiteCandidate, VariantSite, fit_candidates};

fn selector_with(plan: KernelPlan) -> KernelSelector<super::runtime::TestRuntime> {
    let client = test_client();
    let kernels = EngineKernels {
        plan,
        ..EngineKernels::default()
    };
    let limits = small_device_limits();

    KernelSelector::with_limits(&client, kernels, limits)
}

#[test]
fn candidate_names_round_trip() {
    for &candidate in ProbeCandidate::ALL {
        let name = candidate.name();
        let parsed = ProbeCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(ProbeCandidate::baseline(), ProbeCandidate::Small);
}

#[test]
fn the_fit_check_drops_candidates_past_the_shared_memory_cap() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let limits = small_device_limits();

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    assert_eq!(
        fitted,
        vec![
            ProbeCandidate::Small,
            ProbeCandidate::Medium,
            ProbeCandidate::Wide
        ]
    );
}

#[test]
fn the_fit_check_counts_threads_cube_dims_and_plane_width() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let mut limits = small_device_limits();
    limits.max_units_per_cube = 1024;
    limits.max_cube_dim = (32, 1024, 64);
    limits.plane_size_min = 8;

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    // `Wide` is 64 threads across, past the 32 the device allows, and `Medium` needs 16-wide planes.
    assert_eq!(fitted, vec![ProbeCandidate::Small]);
}

#[test]
fn the_fit_check_drops_candidates_past_the_thread_cap() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let mut limits = small_device_limits();
    limits.max_units_per_cube = 512;

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    assert_eq!(fitted, vec![ProbeCandidate::Small, ProbeCandidate::Medium]);
}

#[test]
fn the_baseline_fits_planes_narrower_than_it_needs() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.baseline_plane = 8;
    let mut limits = small_device_limits();
    limits.plane_size_min = 4;

    let fitted = fit_candidates(&site, &limits).expect("the baseline skips the plane check");

    // `Medium` needs 16-wide planes and is not the baseline, so the 4-wide planes drop it.
    assert_eq!(fitted, vec![ProbeCandidate::Small, ProbeCandidate::Wide]);
}

#[test]
fn a_baseline_that_does_not_fit_is_an_error() {
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.baseline_shared = 40 * 1024;
    let limits = small_device_limits();

    let error = fit_candidates(&site, &limits).expect_err("the baseline is too large");

    assert_eq!(error.site, "probe");
    assert_eq!(error.shared_bytes, 40 * 1024);
}

#[test]
fn a_missing_plan_entry_runs_the_baseline() {
    let selector = selector_with(KernelPlan::baseline());
    let client = test_client();
    let site = ProbeSite::new(&client);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Small]);
}

#[test]
fn a_planned_candidate_runs() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let mut plan = KernelPlan::baseline();
    plan.insert(site.key(), "medium");
    let selector = selector_with(plan);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Medium]);
}

/// A stored winner can come from another resolution in the same bucket, or from a device with
/// more shared memory, and must never be launched where it does not fit.
#[test]
fn a_planned_candidate_that_no_longer_fits_runs_the_baseline() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let mut plan = KernelPlan::baseline();
    plan.insert(site.key(), "huge");
    let selector = selector_with(plan);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Small]);
}

#[test]
fn an_unknown_planned_name_runs_the_baseline() {
    let client = test_client();
    let site = ProbeSite::new(&client);
    let mut plan = KernelPlan::baseline();
    plan.insert(site.key(), "retired");
    let selector = selector_with(plan);

    selector.run(&site).expect("the run succeeds");

    assert_eq!(site.launches(), vec![ProbeCandidate::Small]);
}

#[test]
fn a_site_wide_plan_entry_runs_at_every_key() {
    let plan = KernelPlan::baseline().with_site("probe", "wide");
    let selector = selector_with(plan);
    let client = test_client();
    let mut narrow = ProbeSite::new(&client);
    narrow.width = 64;
    let mut wide = ProbeSite::new(&client);
    wide.width = 4096;

    selector.run(&narrow).expect("the run succeeds");
    selector.run(&wide).expect("the run succeeds");

    assert_eq!(narrow.launches(), vec![ProbeCandidate::Wide]);
    assert_eq!(wide.launches(), vec![ProbeCandidate::Wide]);
}

#[test]
fn a_baseline_that_does_not_fit_fails_the_run() {
    let selector = selector_with(KernelPlan::baseline());
    let client = test_client();
    let mut site = ProbeSite::new(&client);
    site.baseline_shared = 40 * 1024;

    let result = selector.run(&site);

    assert!(result.is_err());
    assert!(site.launches().is_empty());
}
