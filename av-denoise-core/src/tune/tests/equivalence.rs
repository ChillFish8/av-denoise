use super::runtime::TestRuntime;
use crate::tune::{DeviceLimits, SiteCandidate, VariantSite, fit_candidates};

/// Launches every candidate of `site` that fits this device into fresh scratch outputs, and checks
/// each against the baseline's under the candidate's own fidelity class.
///
/// Returns the baseline's outputs so the caller can check they hold real work, because a kernel that
/// silently writes nothing compares equal to another that writes nothing.
pub(crate) fn assert_candidates_match_baseline<S: VariantSite<TestRuntime>>(
    site: &S,
    label: &str,
) -> S::Outputs {
    let limits = DeviceLimits::from_client(site.client());
    let fitted = fit_candidates(site, &limits).expect("the baseline fits the test device");
    let baseline = site.baseline();
    let baseline_outputs = site.scratch_outputs();
    site.launch(baseline, &baseline_outputs);

    for candidate in fitted {
        if candidate.is_baseline() {
            continue;
        }

        let candidate_outputs = site.scratch_outputs();
        site.launch(candidate, &candidate_outputs);

        let fidelity = candidate.fidelity();
        let verdict = site
            .compare(&baseline_outputs, &candidate_outputs, fidelity)
            .expect("the outputs read back");
        assert!(
            verdict.passed(),
            "{label}: candidate {} differs from the baseline: {verdict:?}",
            candidate.name()
        );
    }

    baseline_outputs
}
