use std::time::Duration;

use crate::tune::{
    Budget,
    BudgetRequest,
    KeyValue,
    MIN_CANDIDATE_TIME,
    REWARM,
    SiteKey,
    TUNE_AUTO_CAP,
    TUNE_FULL,
    TUNE_UNKNOWN,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    profile_wins,
    site_min_cost,
    timing_spent,
    tuning_budget,
    tuning_order,
};

const FRAME_TIME: Duration = Duration::from_millis(40);

#[test]
fn a_twenty_minute_encode_gets_about_36_seconds() {
    // 30000 frames at 40 ms is 1200 s, and 3% of that is 36 s.
    let request = BudgetRequest::Automatic { frames: Some(30_000) };
    let budget = tuning_budget(request, FRAME_TIME).expect("above the floor");
    let seconds = budget.total.as_secs_f64();

    assert!((seconds - 36.0).abs() < 0.01, "budget {seconds} s");
    assert!(!budget.full);
}

#[test]
fn a_two_hour_film_is_capped() {
    let request = BudgetRequest::Automatic {
        frames: Some(180_000),
    };
    let budget = tuning_budget(request, FRAME_TIME).expect("above the floor");

    assert_eq!(budget.total, TUNE_AUTO_CAP);
}

#[test]
fn a_short_clip_falls_below_the_floor_and_uses_stored_results() {
    // 2000 frames at 40 ms is 80 s, and 3% of that is 2.4 s.
    let request = BudgetRequest::Automatic { frames: Some(2_000) };
    let budget = tuning_budget(request, FRAME_TIME);

    assert_eq!(budget, None);
}

#[test]
fn an_unknown_length_gets_the_fixed_budget() {
    let request = BudgetRequest::Automatic { frames: None };
    let budget = tuning_budget(request, FRAME_TIME).expect("unknown length always tunes");

    assert_eq!(budget.total, TUNE_UNKNOWN);
}

#[test]
fn explicit_tuning_gets_the_full_budget() {
    let budget = tuning_budget(BudgetRequest::Full, FRAME_TIME).expect("full always tunes");

    assert_eq!(budget.total, TUNE_FULL);
    assert!(budget.full);
}

#[test]
fn an_override_sets_the_budget_for_any_run() {
    let small_request = BudgetRequest::Seconds(5.0);
    let large_request = BudgetRequest::Seconds(900.0);
    let small = tuning_budget(small_request, FRAME_TIME).expect("an override always applies");
    let large = tuning_budget(large_request, FRAME_TIME).expect("an override always applies");

    assert_eq!(small.total, Duration::from_secs(5));
    assert!(!small.full);
    assert!(large.full);
}

#[test]
fn the_minimum_site_cost_counts_rewarm_and_every_candidate() {
    let cost = site_min_cost(3);

    assert_eq!(cost, REWARM + MIN_CANDIDATE_TIME * 3);
}

#[test]
fn affordability_counts_what_is_already_spent() {
    let budget = Budget {
        total: Duration::from_secs(10),
        full: false,
    };

    let spent = Duration::from_secs(8);
    let fits = Duration::from_secs(2);
    let too_much = Duration::from_millis(2001);

    assert!(can_afford(&budget, spent, fits));
    assert!(!can_afford(&budget, spent, too_much));
}

fn value(site: &'static str, per_frame_ms: u64) -> KeyValue {
    let key = SiteKey::builder(site, 64, 64).build();
    let per_frame = Duration::from_millis(per_frame_ms);

    KeyValue {
        key,
        per_frame,
        candidates: 3,
    }
}

#[test]
fn sites_are_tuned_most_valuable_first() {
    let values = vec![value("a", 2), value("b", 10), value("c", 5)];
    let order = tuning_order(values, Duration::from_millis(40), false);
    let sites: Vec<&str> = order.iter().map(|entry| entry.key.site()).collect();

    assert_eq!(sites, vec!["b", "c", "a"]);
}

#[test]
fn sites_under_two_percent_wait_for_the_full_budget() {
    let small = || value("small", 1);
    let large = || value("large", 10);

    let auto_values = vec![small(), large()];
    let full_values = vec![small(), large()];
    let auto_order = tuning_order(auto_values, Duration::from_millis(100), false);
    let full_order = tuning_order(full_values, Duration::from_millis(100), true);

    assert_eq!(auto_order.len(), 1);
    assert_eq!(full_order.len(), 2);
}

fn feed(tracker: &mut WarmUpTracker, frames: &[u64], frame_spacing: Duration) -> WarmUpState {
    let mut elapsed = Duration::ZERO;
    let mut state = WarmUpState::Warming;

    for &millis in frames {
        elapsed += frame_spacing;
        let frame_time = Duration::from_millis(millis);
        state = tracker.push(frame_time, elapsed);
    }

    state
}

#[test]
fn a_steady_pipeline_is_stable_after_two_seconds() {
    let mut tracker = WarmUpTracker::new();
    let frames = [40u64; 60];
    let state = feed(&mut tracker, &frames, Duration::from_millis(40));

    assert_eq!(state, WarmUpState::Stable(Duration::from_millis(40)));
}

#[test]
fn a_steady_pipeline_is_not_stable_before_two_seconds() {
    let mut tracker = WarmUpTracker::new();
    let frames = [40u64; 10];
    let state = feed(&mut tracker, &frames, Duration::from_millis(40));

    assert_eq!(state, WarmUpState::Warming);
}

#[test]
fn a_spread_over_three_percent_keeps_warming() {
    let mut tracker = WarmUpTracker::new();
    let frames: Vec<u64> = (0..60)
        .map(|index| if index % 2 == 0 { 40 } else { 42 })
        .collect();
    let state = feed(&mut tracker, &frames, Duration::from_millis(40));

    assert_eq!(state, WarmUpState::Warming);
}

#[test]
fn an_unsteady_pipeline_is_capped_at_fifteen_seconds() {
    let mut tracker = WarmUpTracker::new();
    let frames: Vec<u64> = (0..400)
        .map(|index| if index % 2 == 0 { 40 } else { 50 })
        .collect();
    let state = feed(&mut tracker, &frames, Duration::from_millis(40));

    assert!(matches!(state, WarmUpState::Capped(_)));
}

#[test]
fn a_profile_wins_only_past_the_margin() {
    let baseline = Duration::from_millis(100);
    let close = Duration::from_millis(98);
    let clear = Duration::from_millis(96);

    assert!(!profile_wins(baseline, close));
    assert!(profile_wins(baseline, clear));
}

#[test]
fn compile_time_is_left_out_of_the_spent_budget() {
    let elapsed = Duration::from_secs(50);
    let compile = Duration::from_secs(45);

    let spent = timing_spent(elapsed, compile);

    assert_eq!(spent, Duration::from_secs(5));
}

#[test]
fn compile_time_past_the_elapsed_time_leaves_nothing_spent() {
    let elapsed = Duration::from_secs(1);
    let compile = Duration::from_secs(2);

    let spent = timing_spent(elapsed, compile);

    assert_eq!(spent, Duration::ZERO);
}

#[test]
fn a_site_is_affordable_once_compile_time_is_left_out() {
    let budget = Budget {
        total: Duration::from_secs(30),
        full: false,
    };
    let elapsed = Duration::from_millis(29_500);
    let compile = Duration::from_secs(20);
    let needed = site_min_cost(3);

    let spent = timing_spent(elapsed, compile);

    assert!(!can_afford(&budget, elapsed, needed));
    assert!(can_afford(&budget, spent, needed));
}
