use super::runtime::{TestRuntime, test_legality};
use crate::tune::{DeviceLimits, Fidelity, FidelityClass, SiteCandidate, VariantSite, legal_candidates};

/// Launches every candidate of `site` that this device can run into fresh scratch outputs, and checks
/// each against the baseline's under the candidate's own fidelity class.
///
/// Returns the baseline's outputs so the caller can check they hold real work, because a kernel that
/// silently writes nothing compares equal to another that writes nothing.
pub(crate) fn assert_candidates_match_baseline<S: VariantSite<TestRuntime>>(
    site: &S,
    label: &str,
) -> S::Outputs {
    assert_listed_candidates_match_baseline(site, label, S::Candidate::ALL)
}

/// [assert_candidates_match_baseline] over only the fitted candidates in `listed`.
///
/// An exact integer candidate is checked bit for bit against the site's integer reference instead,
/// and the reference against the baseline under its own fidelity.
pub(crate) fn assert_listed_candidates_match_baseline<S: VariantSite<TestRuntime>>(
    site: &S,
    label: &str,
    listed: &[S::Candidate],
) -> S::Outputs {
    let client = site.client();
    let limits = DeviceLimits::from_client(client);
    let legality = test_legality(client);
    let fitted = legal_candidates(site, &limits, &legality).expect("the baseline fits the test device");
    let baseline = site.baseline();
    let baseline_outputs = site.scratch_outputs();
    site.launch(baseline, &baseline_outputs);

    let reference = site
        .integer_reference()
        .filter(|candidate| fitted.contains(candidate));
    let reference_outputs =
        reference.map(|candidate| reference_matches_baseline(site, label, candidate, &baseline_outputs));

    for candidate in fitted {
        let is_reference = Some(candidate) == reference;
        if candidate == baseline || is_reference || !listed.contains(&candidate) {
            continue;
        }

        let candidate_outputs = site.scratch_outputs();
        site.launch(candidate, &candidate_outputs);

        let exact_integer = candidate.fidelity().class() == FidelityClass::ExactInteger;
        let (expected, fidelity, against) = if exact_integer {
            let Some(expected) = reference_outputs.as_ref() else {
                panic!(
                    "{label}: candidate {} has no fitted integer reference",
                    candidate.name()
                );
            };

            (expected, Fidelity::Exact, "the integer reference")
        } else {
            (&baseline_outputs, candidate.fidelity(), "the baseline")
        };

        let verdict = site
            .compare(expected, &candidate_outputs, fidelity)
            .expect("the outputs read back");
        assert!(
            verdict.passed(),
            "{label}: candidate {} differs from {against}: {verdict:?}",
            candidate.name()
        );
    }

    baseline_outputs
}

/// Launches the integer reference, checks it against the baseline under its own fidelity, and returns
/// its outputs.
fn reference_matches_baseline<S: VariantSite<TestRuntime>>(
    site: &S,
    label: &str,
    reference: S::Candidate,
    baseline_outputs: &S::Outputs,
) -> S::Outputs {
    let reference_outputs = site.scratch_outputs();
    site.launch(reference, &reference_outputs);

    let fidelity = reference.fidelity();
    let verdict = site
        .compare(baseline_outputs, &reference_outputs, fidelity)
        .expect("the outputs read back");
    assert!(
        verdict.passed(),
        "{label}: integer reference {} differs from the baseline: {verdict:?}",
        reference.name()
    );

    reference_outputs
}
