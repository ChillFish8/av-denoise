use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::consts::{BASELINE_INDEX, MAX_LAUNCHES_PER_SAMPLE, REWARM, SAMPLE_TARGET, STAGE_KEEP};
use super::decide::{
    DecideLimits,
    SiteTiming,
    Staging,
    median,
    prescreen,
    sample_in_stages_with,
    sample_until_settled,
};
use super::fidelity::{CompareResult, Fidelity, FidelityClass};
use super::gpu::{GpuSampler, profile_launches, sync};
use super::precompile::{CompileAccount, Precompiled, precompile};
use super::result::{CandidateRecord, CompileProgress, SiteRanking};
use super::site::{SiteCandidate, VariantSite};

/// Launches queued between syncs while the baseline re-warms.
const REWARM_BATCH: u32 = 8;

/// Single launches timed to calibrate the batch size, of which the median counts.
const CALIBRATION_SAMPLES: usize = 3;

/// Tunes one site in place and ranks every candidate that passed its output check.
///
/// `allowed` lists the candidates that fit the device, baseline first. A key with only its baseline
/// is ranked without timing. Candidates run only into scratch output sets, so the pipeline's own
/// buffers stay untouched. A site with an allowed integer reference gets a third set for it. The
/// baseline re-warms first and the batch size is calibrated from it. A site that asks for one gets a
/// pre-screen. Every survivor's output is checked as [check_outputs] describes, and the rest are timed
/// round-robin until the stop rule ends the site or `deadline` caps it. A staged site runs these steps
/// once per stage, as [sample_in_stages_with](crate::tune::sample_in_stages_with) describes. Before
/// timing, each stage's candidates are compiled once, and the compile time goes to `compile` instead
/// of counting towards the caps. When the compile ceiling skips any candidate, the ranking is
/// partial, and it counts how many it skipped. A stage whose integer reference was skipped or failed
/// to compile drops its exact integer candidates before any of them is timed.
pub(crate) fn tune_site<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    deadline: Instant,
    compile: CompileAccount<'_>,
) -> Result<SiteRanking, anyhow::Error> {
    let identity = site.compile_identity();

    if allowed.len() == 1 {
        let ranking = untimed_baseline(allowed[BASELINE_INDEX], identity);
        return Ok(ranking);
    }

    let baseline_outputs = site.scratch_outputs();
    let candidate_outputs = site.scratch_outputs();
    let reference = integer_reference(site, allowed);

    rewarm(site, &candidate_outputs)?;
    let launches = calibrate(site, &candidate_outputs)?;

    let mut limits = DecideLimits::default();
    let remaining = deadline.saturating_duration_since(Instant::now());
    limits.site_cap = limits.site_cap.min(remaining);

    let tried = RefCell::new(StageCounts::default());
    let reference_compiled = Cell::new(false);
    let compile_stage = |indices: &[usize]| {
        let mut precompiled = compile_members(site, allowed, indices, &candidate_outputs, compile)?;

        {
            let mut counts = tried.borrow_mut();
            counts.asked.extend(indices);
            counts.skipped.extend(&precompiled.skipped);
        }

        let integers = keep_integers_with_reference(allowed, reference.as_ref(), &mut precompiled);
        reference_compiled.set(integers.reference_kept);
        tried.borrow_mut().skipped.extend(&integers.left_untimed);

        Ok(precompiled)
    };

    let checks = CheckOutputs {
        baseline: &baseline_outputs,
        candidate: &candidate_outputs,
        reference: reference.as_ref(),
    };

    if S::STAGES > 1 {
        let members = |stage: usize, seed: usize| stage_members(site, allowed, stage, seed);
        let staging = Staging {
            stages: S::STAGES,
            keep: STAGE_KEEP,
            prescreen: S::PRESCREEN,
            members: &members,
        };
        let check = |indices: &[usize]| {
            let reference_kept = reference_compiled.get();
            check_outputs(site, allowed, indices, &checks, reference_kept)
        };
        let mut sampler = GpuSampler::new(site, allowed, &candidate_outputs, launches);
        let timing = sample_in_stages_with(
            &mut sampler,
            &staging,
            BASELINE_INDEX,
            &limits,
            compile_stage,
            check,
        )?;

        let progress = tried.borrow().progress();
        let ranking = ranking_from(allowed, &timing, progress, identity);
        return Ok(ranking);
    }

    let indices: Vec<usize> = (0..allowed.len()).collect();
    let precompiled = compile_stage(&indices)?;
    let compiled = precompiled.kept;

    let screened = if S::PRESCREEN {
        let mut sampler = GpuSampler::new(site, allowed, &candidate_outputs, launches);
        prescreen(&mut sampler, &compiled, BASELINE_INDEX, &limits)?
    } else {
        compiled
    };

    let reference_kept = reference_compiled.get();
    let checked = check_outputs(site, allowed, &screened, &checks, reference_kept);
    let mut sampler = GpuSampler::new(site, allowed, &candidate_outputs, launches);
    let timing = sample_until_settled(&mut sampler, &checked, BASELINE_INDEX, &limits)?;

    let progress = tried.borrow().progress();
    let ranking = ranking_from(allowed, &timing, progress, identity);
    Ok(ranking)
}

/// The candidates a key's stages asked to compile, and those the compile ceiling skipped.
#[derive(Default)]
struct StageCounts {
    asked: BTreeSet<usize>,
    skipped: BTreeSet<usize>,
}

impl StageCounts {
    fn progress(&self) -> CompileProgress {
        CompileProgress {
            left: self.skipped.len(),
            total: self.asked.len(),
        }
    }
}

/// Compiles the candidates at `indices` within the pass's compile ceiling and records the time it took.
///
/// A compile of [REWARM](crate::tune::REWARM) or longer can let the clocks drop, so the baseline runs
/// again before timing resumes, and that re-warm counts as timing. A baseline that does not compile
/// fails the site.
pub(crate) fn compile_members<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &S::Outputs,
    compile: CompileAccount<'_>,
) -> Result<Precompiled, anyhow::Error> {
    let identity = site.compile_identity();
    let limits = compile.limits(&identity, allowed, indices);
    // Sites that cannot copy themselves never ask, so they never make helper devices.
    let helpers = if S::COPIES { compile.helpers::<R>() } else { &[] };
    let precompiled = precompile(site, allowed, indices, outputs, &limits, helpers);
    compile.record(&precompiled);

    if precompiled.helpers_failed {
        tracing::warn!(
            site = S::Candidate::SITE,
            key = %compile.key,
            "the compile helpers failed candidates the tuned client compiled, so later stages compile on \
             the tuned client"
        );
        compile.fail_helpers();
    }

    tracing::debug!(
        site = S::Candidate::SITE,
        key = %compile.key,
        kept = precompiled.kept.len(),
        dropped = precompiled.dropped.len(),
        skipped = precompiled.skipped.len(),
        reused = precompiled.reused,
        one_by_one = precompiled.one_by_one,
        helpers = precompiled.helpers,
        load = ?precompiled.load,
        spent = ?precompiled.spent,
        "compiled a stage's candidates"
    );

    if !precompiled.skipped.is_empty() {
        let skipped = candidate_names(allowed, &precompiled.skipped);
        tracing::info!(
            site = S::Candidate::SITE,
            key = %compile.key,
            ?skipped,
            "the compile ceiling is spent, so this key's remaining candidates are skipped and a later run \
             tunes it again"
        );
    }

    if let Some(reason) = precompiled.reason_for(BASELINE_INDEX) {
        anyhow::bail!("the baseline did not compile: {reason}");
    }

    if precompiled.spent >= REWARM {
        rewarm(site, outputs)?;
    }

    Ok(precompiled)
}

fn candidate_names<C: SiteCandidate>(allowed: &[C], indices: &[usize]) -> Vec<&'static str> {
    indices.iter().map(|&index| allowed[index].name()).collect()
}

/// The indices into `allowed` that stage `stage` times around `allowed[seed]`.
fn stage_members<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    stage: usize,
    seed: usize,
) -> Vec<usize> {
    let seed_candidate = allowed[seed];

    (0..allowed.len())
        .filter(|&index| site.in_stage(stage, seed_candidate, allowed[index]))
        .collect()
}

/// About how many of `allowed` tuning times.
///
/// An unstaged site times all of them. A staged site times the first stage around the baseline, and
/// each later stage times [STAGE_KEEP](crate::tune::STAGE_KEEP) seeds' members, each seed already timed.
pub(crate) fn timed_count<R: Runtime, S: VariantSite<R>>(site: &S, allowed: &[S::Candidate]) -> usize {
    if S::STAGES <= 1 {
        return allowed.len();
    }

    let first = stage_members(site, allowed, 0, 0);
    let mut count = first.len();

    for stage in 1..S::STAGES {
        let around = stage_members(site, allowed, stage, 0);
        let fresh_per_seed = around.len().saturating_sub(1);
        count += STAGE_KEEP * fresh_per_seed;
    }

    count.min(allowed.len())
}

/// Runs the baseline into `outputs` for [REWARM](crate::tune::REWARM), so an idle gap before the
/// site cannot drop the clocks.
pub(crate) fn rewarm<R: Runtime, S: VariantSite<R>>(
    site: &S,
    outputs: &S::Outputs,
) -> Result<(), anyhow::Error> {
    let client = site.client();
    let baseline = site.baseline();
    let started = Instant::now();

    while started.elapsed() < REWARM {
        for _ in 0..REWARM_BATCH {
            site.launch(baseline, outputs);
        }

        sync(client)?;
    }

    Ok(())
}

/// The launches per sample that make one sample last at least [SAMPLE_TARGET](crate::tune::SAMPLE_TARGET).
pub(crate) fn calibrate<R: Runtime, S: VariantSite<R>>(
    site: &S,
    outputs: &S::Outputs,
) -> Result<u32, anyhow::Error> {
    let baseline = site.baseline();
    let mut singles = Vec::with_capacity(CALIBRATION_SAMPLES);

    for _ in 0..CALIBRATION_SAMPLES {
        let single = profile_launches(site, baseline, outputs, 1)?;
        singles.push(single);
    }

    let single_median = median(&singles);
    let single = single_median.max(Duration::from_nanos(1));
    let needed = SAMPLE_TARGET.as_secs_f64() / single.as_secs_f64();
    let launches = needed.ceil() as u32;
    let clamped = launches.clamp(1, MAX_LAUNCHES_PER_SAMPLE);

    Ok(clamped)
}

/// A site's integer reference, as an index into its allowed candidates, with the outputs it runs into.
struct IntegerReference<O> {
    index: usize,
    outputs: O,
}

/// The scratch output sets [check_outputs] runs candidates into.
struct CheckOutputs<'a, O> {
    baseline: &'a O,
    candidate: &'a O,
    reference: Option<&'a IntegerReference<O>>,
}

/// The site's integer reference when it is allowed, with a scratch output set of its own.
fn integer_reference<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
) -> Option<IntegerReference<S::Outputs>> {
    let candidate = site.integer_reference()?;
    let index = allowed.iter().position(|&allowed| allowed == candidate)?;
    let outputs = site.scratch_outputs();

    Some(IntegerReference { index, outputs })
}

fn is_exact_integer<C: SiteCandidate>(candidate: C) -> bool {
    candidate.fidelity().class() == FidelityClass::ExactInteger
}

/// What [keep_integers_with_reference] decided for one stage.
#[derive(Default)]
struct IntegerStage {
    /// Whether the integer reference compiled and is among the stage's kept candidates.
    reference_kept: bool,
    /// Exact integer candidates removed because the compile ceiling skipped the reference.
    left_untimed: Vec<usize>,
}

/// Removes a stage's exact integer candidates from what it kept when its integer reference was skipped
/// or did not compile.
///
/// Only a skipped reference leaves candidates untimed for a later run. A reference that failed to
/// compile is a drop, which does not make the ranking partial.
fn keep_integers_with_reference<C: SiteCandidate, O>(
    allowed: &[C],
    reference: Option<&IntegerReference<O>>,
    precompiled: &mut Precompiled,
) -> IntegerStage {
    let Some(reference) = reference else {
        return IntegerStage::default();
    };

    let skipped = precompiled.skipped.contains(&reference.index);
    let dropped = precompiled.reason_for(reference.index).is_some();
    if !skipped && !dropped {
        let reference_kept = precompiled.kept.contains(&reference.index);
        return IntegerStage {
            reference_kept,
            ..IntegerStage::default()
        };
    }

    let removed: Vec<usize> = precompiled
        .kept
        .iter()
        .copied()
        .filter(|&index| is_exact_integer(allowed[index]))
        .collect();
    let names = candidate_names(allowed, &removed);
    precompiled
        .kept
        .retain(|&index| !is_exact_integer(allowed[index]));

    if skipped {
        tracing::info!(
            site = C::SITE,
            removed = ?names,
            "the compile ceiling skipped the integer reference, so its exact-integer candidates are \
             skipped this run"
        );
    } else {
        tracing::info!(
            site = C::SITE,
            removed = ?names,
            "the integer reference did not compile, so its exact-integer candidates are dropped this run"
        );
    }

    let left_untimed = if skipped { removed } else { Vec::new() };
    IntegerStage {
        reference_kept: false,
        left_untimed,
    }
}

/// The indices whose candidate's output passes its fidelity check, keeping the baseline when asked.
///
/// Each candidate runs once into cleared outputs, so the check sees a single launch. Most candidates
/// are checked against the baseline. When `reference_compiled` is set, the integer reference runs
/// once after the baseline and is checked against it under its own fidelity. Every other exact
/// integer candidate must then match the reference bit for bit, and is dropped when the reference
/// failed or did not run. A failed check or a failed readback drops the candidate with a warning.
fn check_outputs<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    outputs: &CheckOutputs<'_, S::Outputs>,
    reference_compiled: bool,
) -> Vec<usize> {
    let baseline = allowed[BASELINE_INDEX];
    site.clear_outputs(outputs.baseline);
    site.launch(baseline, outputs.baseline);

    let reference = outputs.reference.filter(|_| reference_compiled);
    let reference_passed = match reference {
        Some(reference) => check_reference(site, allowed, reference, outputs),
        None => false,
    };

    let mut passed = Vec::with_capacity(indices.len());

    for &index in indices {
        let is_reference = reference.is_some_and(|reference| reference.index == index);
        let candidate = allowed[index];

        let keep = if index == BASELINE_INDEX {
            true
        } else if is_reference {
            reference_passed
        } else if is_exact_integer(candidate) {
            check_integer_follower(site, candidate, reference, reference_passed, outputs)
        } else {
            site.clear_outputs(outputs.candidate);
            site.launch(candidate, outputs.candidate);

            let fidelity = candidate.fidelity();
            let verdict = site.compare(outputs.baseline, outputs.candidate, fidelity);
            passes(candidate, verdict, "the baseline")
        };

        if keep {
            passed.push(index);
        }
    }

    passed
}

/// Runs the integer reference into its own outputs and checks it against the baseline's.
fn check_reference<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    reference: &IntegerReference<S::Outputs>,
    outputs: &CheckOutputs<'_, S::Outputs>,
) -> bool {
    let candidate = allowed[reference.index];
    site.clear_outputs(&reference.outputs);
    site.launch(candidate, &reference.outputs);

    let fidelity = candidate.fidelity();
    let verdict = site.compare(outputs.baseline, &reference.outputs, fidelity);

    passes(candidate, verdict, "the baseline")
}

/// Whether an exact integer candidate other than the reference matches the reference bit for bit.
fn check_integer_follower<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    reference: Option<&IntegerReference<S::Outputs>>,
    reference_passed: bool,
    outputs: &CheckOutputs<'_, S::Outputs>,
) -> bool {
    let Some(reference) = reference else {
        tracing::warn!(
            site = S::Candidate::SITE,
            candidate = candidate.name(),
            "dropping an exact integer candidate because no integer reference ran to check it against"
        );
        return false;
    };

    if !reference_passed {
        tracing::warn!(
            site = S::Candidate::SITE,
            candidate = candidate.name(),
            "dropping an exact integer candidate because its integer reference failed its own check"
        );
        return false;
    }

    site.clear_outputs(outputs.candidate);
    site.launch(candidate, outputs.candidate);

    let verdict = site.compare(&reference.outputs, outputs.candidate, Fidelity::Exact);

    passes(candidate, verdict, "the integer reference")
}

/// Whether `verdict` passed, logging why a candidate is dropped when it did not.
fn passes<C: SiteCandidate>(
    candidate: C,
    verdict: Result<CompareResult, anyhow::Error>,
    against: &'static str,
) -> bool {
    match verdict {
        Ok(CompareResult::Pass) => true,
        Ok(CompareResult::Fail(reason)) => {
            tracing::warn!(
                site = C::SITE,
                candidate = candidate.name(),
                against,
                %reason,
                "dropping a candidate whose output differs from what it is checked against"
            );
            false
        },
        Err(error) => {
            tracing::warn!(
                site = C::SITE,
                candidate = candidate.name(),
                %error,
                "dropping a candidate whose output could not be checked"
            );
            false
        },
    }
}

/// The ranking of a key whose baseline is its only candidate, which needs no timing.
fn untimed_baseline<C: SiteCandidate>(baseline: C, identity: String) -> SiteRanking {
    let record = CandidateRecord {
        name: baseline.name().to_string(),
        fidelity: baseline.fidelity().class(),
        baseline: true,
        median_ns: 0,
        samples: 0,
    };

    SiteRanking {
        candidates: vec![record],
        partial: false,
        progress: CompileProgress::default(),
        identity,
    }
}

/// The timed candidates as records, fastest first.
///
/// The ranking is partial when `progress` says the compile ceiling left any candidate.
fn ranking_from<C: SiteCandidate>(
    allowed: &[C],
    timing: &SiteTiming,
    progress: CompileProgress,
    identity: String,
) -> SiteRanking {
    let mut candidates: Vec<CandidateRecord> = timing
        .timed
        .iter()
        .map(|stats| {
            let candidate = allowed[stats.index];
            let candidate_median = stats.samples.median();

            CandidateRecord {
                name: candidate.name().to_string(),
                fidelity: candidate.fidelity().class(),
                baseline: stats.index == 0,
                median_ns: candidate_median.as_nanos() as u64,
                samples: stats.samples.count() as u32,
            }
        })
        .collect();

    candidates.sort_by_key(|record| record.median_ns);

    SiteRanking {
        candidates,
        partial: progress.left > 0,
        progress,
        identity,
    }
}
