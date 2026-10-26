use std::time::Duration;

use av_denoise_core::tune::{
    Basis,
    PROFILE_BLOCK_FRAMES,
    PROFILE_MIN_BLOCKS,
    PipelineProfile,
    PriorityTable,
    ProfileRecord,
    SearchMode,
};

use super::setup::r9700_profile;
use crate::tune::driver::{
    ProfileComparison,
    ProfilePhase,
    ProfileRun,
    ProfileTrial,
    alternative_profiles_for,
    time_profiles,
};

const PACKED_FIRST: PriorityTable = PriorityTable {
    sites: &[],
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::SplitSearch,
        PipelineProfile::F32Search,
    ],
};

const BASELINE_LAST: PriorityTable = PriorityTable {
    sites: &[],
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::F32Search,
        PipelineProfile::Baseline,
    ],
};

const ORDER: [PipelineProfile; 3] = [
    PipelineProfile::SplitSearch,
    PipelineProfile::PackedAtomics,
    PipelineProfile::F32Search,
];

/// Frame times in milliseconds, with the baseline at 100.
fn frame_millis(profile: PipelineProfile, split: u64, packed: u64, f32_search: u64) -> u64 {
    match profile {
        PipelineProfile::Baseline => 100,
        PipelineProfile::SplitSearch => split,
        PipelineProfile::PackedAtomics => packed,
        PipelineProfile::F32Search => f32_search,
    }
}

fn block_times(millis: u64) -> Vec<Duration> {
    let frame_time = Duration::from_millis(millis);
    let frames = PROFILE_MIN_BLOCKS * PROFILE_BLOCK_FRAMES;

    vec![frame_time; frames]
}

/// Times `order` with the frame times `millis` gives, returning the run and the profiles it tried.
fn run_profiles(
    order: &[PipelineProfile],
    mode: SearchMode,
    millis: impl Fn(PipelineProfile) -> u64,
) -> (ProfileRun<PipelineProfile>, Vec<PipelineProfile>) {
    run_with_faults(order, mode, millis, |_| None)
}

/// Like [run_profiles], but a profile `fault` answers for ends with that trial instead.
fn run_with_faults(
    order: &[PipelineProfile],
    mode: SearchMode,
    millis: impl Fn(PipelineProfile) -> u64,
    fault: impl Fn(PipelineProfile) -> Option<ProfileTrial<PipelineProfile>>,
) -> (ProfileRun<PipelineProfile>, Vec<PipelineProfile>) {
    let mut tried = Vec::new();

    let run = time_profiles(order, mode, |profile| {
        tried.push(profile);

        if let Some(trial) = fault(profile) {
            return trial;
        }

        let baseline_millis = millis(PipelineProfile::Baseline);
        let alternative_millis = millis(profile);
        let comparison = ProfileComparison {
            baseline_times: block_times(baseline_millis),
            alternative_times: block_times(alternative_millis),
            significant: Vec::new(),
            finished: true,
        };

        ProfileTrial::Timed {
            comparison,
            built: profile,
        }
    });

    (run, tried)
}

fn compared_records(run: &ProfileRun<PipelineProfile>) -> &[ProfileRecord] {
    match &run.phase {
        ProfilePhase::Compared { records, .. } => records,
        other => panic!("the profiles were not compared: {other:?}"),
    }
}

fn record_profiles(records: &[ProfileRecord]) -> Vec<PipelineProfile> {
    records.iter().map(|record| record.profile).collect()
}

fn winner_of(run: &ProfileRun<PipelineProfile>) -> Option<PipelineProfile> {
    run.winner.as_ref().map(|winner| winner.built)
}

#[test]
fn profiles_follow_the_device_table_order() {
    let device = r9700_profile();

    let packed_first = alternative_profiles_for(true, &device, &PACKED_FIRST);
    let baseline_last = alternative_profiles_for(true, &device, &BASELINE_LAST);

    let expected_packed_first = vec![
        PipelineProfile::PackedAtomics,
        PipelineProfile::SplitSearch,
        PipelineProfile::F32Search,
    ];
    assert_eq!(packed_first, expected_packed_first);
    assert_eq!(baseline_last, ORDER);
}

#[test]
fn profile_timing_stops_at_the_first_profile_that_does_not_win() {
    let (run, tried) = run_profiles(&ORDER, SearchMode::Families, |profile| {
        frame_millis(profile, 90, 89, 50)
    });

    let records = compared_records(&run);
    let recorded = record_profiles(records);

    let reached = [PipelineProfile::SplitSearch, PipelineProfile::PackedAtomics];
    assert_eq!(tried, reached);
    assert_eq!(
        recorded,
        [
            PipelineProfile::Baseline,
            PipelineProfile::SplitSearch,
            PipelineProfile::PackedAtomics,
        ]
    );

    let winner = winner_of(&run);
    assert_eq!(winner, Some(PipelineProfile::PackedAtomics));
}

#[test]
fn exhaustive_mode_times_every_legal_profile() {
    let (run, tried) = run_profiles(&ORDER, SearchMode::Exhaustive, |profile| {
        frame_millis(profile, 90, 89, 50)
    });

    let records = compared_records(&run);
    let recorded = record_profiles(records);

    assert_eq!(tried, ORDER);
    assert_eq!(recorded[1..], ORDER);

    let winner = winner_of(&run);
    assert_eq!(winner, Some(PipelineProfile::F32Search));
}

#[test]
fn the_baseline_profile_is_always_timed() {
    let device = r9700_profile();
    let order = alternative_profiles_for(true, &device, &BASELINE_LAST);

    let (run, tried) = run_profiles(&order, SearchMode::Families, |profile| {
        frame_millis(profile, 120, 50, 50)
    });

    let records = compared_records(&run);
    let baseline = records[0];
    let baseline_blocks = PROFILE_MIN_BLOCKS as u32;

    assert!(!order.contains(&PipelineProfile::Baseline));
    assert_eq!(tried, [PipelineProfile::SplitSearch]);
    assert_eq!(baseline.profile, PipelineProfile::Baseline);
    assert_eq!(baseline.blocks, baseline_blocks);
    assert_eq!(baseline.median_frame_ns, 100_000_000);

    let winner = winner_of(&run);
    assert_eq!(winner, None);
}

#[test]
fn a_profile_that_does_not_build_or_compare_is_not_a_loss() {
    let order = [PipelineProfile::SplitSearch, PipelineProfile::PackedAtomics];
    let millis = |profile| frame_millis(profile, 90, 90, 90);
    let reached = order.to_vec();

    let (unbuilt_run, unbuilt_tried) = run_with_faults(&order, SearchMode::Families, millis, |profile| {
        let unbuilt = profile == PipelineProfile::SplitSearch;
        unbuilt.then_some(ProfileTrial::Unbuilt)
    });
    let (failed_run, failed_tried) = run_with_faults(&order, SearchMode::Families, millis, |profile| {
        let failed = profile == PipelineProfile::SplitSearch;
        failed.then_some(ProfileTrial::Failed)
    });

    let unbuilt_records = compared_records(&unbuilt_run);
    let unbuilt_recorded = record_profiles(unbuilt_records);
    let unbuilt_winner = winner_of(&unbuilt_run);
    let failed_winner = winner_of(&failed_run);

    assert_eq!(unbuilt_tried, reached);
    assert_eq!(
        unbuilt_recorded,
        [PipelineProfile::Baseline, PipelineProfile::PackedAtomics]
    );
    assert_eq!(unbuilt_winner, Some(PipelineProfile::PackedAtomics));

    assert_eq!(failed_tried, reached);
    assert_eq!(failed_run.phase, ProfilePhase::Unfinished);
    assert_eq!(failed_winner, Some(PipelineProfile::PackedAtomics));
}
