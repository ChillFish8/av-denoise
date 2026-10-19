use std::cell::Cell;
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
use super::fidelity::CompareResult;
use super::gpu::{GpuSampler, profile_launches, sync};
use super::precompile::{CompileAccount, Precompiled, precompile};
use super::result::{CandidateRecord, SiteRanking};
use super::site::{SiteCandidate, VariantSite};

/// Launches queued between syncs while the baseline re-warms.
const REWARM_BATCH: u32 = 8;

/// Single launches timed to calibrate the batch size, of which the median counts.
const CALIBRATION_SAMPLES: usize = 3;

/// Tunes one site in place and ranks every candidate that passed its output check.
///
/// `allowed` lists the candidates that fit the device, baseline first. A key with only its baseline
/// is ranked without timing. Candidates run only into two scratch output sets, so the pipeline's own
/// buffers stay untouched. The baseline re-warms first and the batch size is calibrated from it. A
/// site that asks for one gets a pre-screen. Every survivor's output is checked against the
/// baseline's, and the rest are timed round-robin until the stop rule ends the site or `deadline`
/// caps it. A staged site runs these steps once per stage, as
/// [sample_in_stages_with](crate::tune::sample_in_stages_with) describes. Before timing, each stage's
/// candidates are compiled once, and the compile time goes to `compile` instead of counting towards
/// the caps. When the compile ceiling skips any candidate, the ranking is partial.
pub(crate) fn tune_site<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    deadline: Instant,
    compile: CompileAccount<'_>,
) -> Result<SiteRanking, anyhow::Error> {
    if allowed.len() == 1 {
        let ranking = untimed_baseline(allowed[BASELINE_INDEX]);
        return Ok(ranking);
    }

    let baseline_outputs = site.scratch_outputs();
    let candidate_outputs = site.scratch_outputs();

    rewarm(site, &candidate_outputs)?;
    let launches = calibrate(site, &candidate_outputs)?;

    let mut limits = DecideLimits::default();
    let remaining = deadline.saturating_duration_since(Instant::now());
    limits.site_cap = limits.site_cap.min(remaining);

    let skipped_any = Cell::new(false);
    let compile_stage = |indices: &[usize]| {
        let precompiled = compile_members(site, allowed, indices, &candidate_outputs, compile)?;
        if !precompiled.skipped.is_empty() {
            skipped_any.set(true);
        }

        Ok(precompiled)
    };

    if S::STAGES > 1 {
        let members = |stage: usize, seed: usize| stage_members(site, allowed, stage, seed);
        let staging = Staging {
            stages: S::STAGES,
            keep: STAGE_KEEP,
            prescreen: S::PRESCREEN,
            members: &members,
        };
        let check =
            |indices: &[usize]| check_outputs(site, allowed, indices, &baseline_outputs, &candidate_outputs);
        let mut sampler = GpuSampler::new(site, allowed, &candidate_outputs, launches);
        let timing = sample_in_stages_with(
            &mut sampler,
            &staging,
            BASELINE_INDEX,
            &limits,
            compile_stage,
            check,
        )?;

        let ranking = ranking_from(allowed, &timing, skipped_any.get());
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

    let checked = check_outputs(site, allowed, &screened, &baseline_outputs, &candidate_outputs);
    let mut sampler = GpuSampler::new(site, allowed, &candidate_outputs, launches);
    let timing = sample_until_settled(&mut sampler, &checked, BASELINE_INDEX, &limits)?;

    let ranking = ranking_from(allowed, &timing, skipped_any.get());
    Ok(ranking)
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
    let allowance = compile.left();
    let estimate = compile.estimate();
    // Sites that cannot copy themselves never ask, so they never make helper devices.
    let helpers = if S::COPIES { compile.helpers::<R>() } else { &[] };
    let precompiled = precompile(site, allowed, indices, outputs, allowance, estimate, helpers);
    compile.record(&precompiled);

    if precompiled.helpers_failed {
        tracing::warn!(
            site = S::Candidate::SITE,
            key = %compile.key,
            "every compile helper failed, so later stages compile on the tuned client"
        );
        compile.fail_helpers();
    }

    tracing::debug!(
        site = S::Candidate::SITE,
        key = %compile.key,
        kept = precompiled.kept.len(),
        dropped = precompiled.dropped.len(),
        skipped = precompiled.skipped.len(),
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

/// The indices whose candidate's output passes its fidelity check, keeping the baseline when asked.
///
/// Each candidate runs once into cleared outputs, so the check sees a single launch. A failed check
/// or a failed readback drops the candidate with a warning.
fn check_outputs<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    indices: &[usize],
    baseline_outputs: &S::Outputs,
    candidate_outputs: &S::Outputs,
) -> Vec<usize> {
    let baseline = allowed[BASELINE_INDEX];
    site.clear_outputs(baseline_outputs);
    site.launch(baseline, baseline_outputs);

    let mut passed = Vec::with_capacity(indices.len());

    for &index in indices {
        if index == BASELINE_INDEX {
            passed.push(BASELINE_INDEX);
            continue;
        }

        let candidate = allowed[index];
        site.clear_outputs(candidate_outputs);
        site.launch(candidate, candidate_outputs);

        let fidelity = candidate.fidelity();
        let verdict = site.compare(baseline_outputs, candidate_outputs, fidelity);

        match verdict {
            Ok(CompareResult::Pass) => passed.push(index),
            Ok(CompareResult::Fail(reason)) => {
                tracing::warn!(
                    site = S::Candidate::SITE,
                    candidate = candidate.name(),
                    %reason,
                    "dropping a candidate whose output differs from the baseline"
                );
            },
            Err(error) => {
                tracing::warn!(
                    site = S::Candidate::SITE,
                    candidate = candidate.name(),
                    %error,
                    "dropping a candidate whose output could not be checked"
                );
            },
        }
    }

    passed
}

/// The ranking of a key whose baseline is its only candidate, which needs no timing.
fn untimed_baseline<C: SiteCandidate>(baseline: C) -> SiteRanking {
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
    }
}

/// The timed candidates as records, fastest first.
fn ranking_from<C: SiteCandidate>(allowed: &[C], timing: &SiteTiming, partial: bool) -> SiteRanking {
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
    SiteRanking { candidates, partial }
}
