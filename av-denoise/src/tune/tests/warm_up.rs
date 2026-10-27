use std::time::Duration;

use av_denoise_core::tune::BudgetRequest;

use crate::tune::driver::warm_up_overruns;

const FRAME_TIME: Duration = Duration::from_millis(10);

#[test]
fn a_tiny_explicit_budget_stops_the_warm_up_at_once() {
    let request = BudgetRequest::Seconds(0.5);

    assert!(warm_up_overruns(request, FRAME_TIME, Duration::ZERO));
}

#[test]
fn a_full_budget_lets_the_warm_up_run() {
    let request = BudgetRequest::Full;
    let spent = Duration::from_secs(15);

    assert!(!warm_up_overruns(request, FRAME_TIME, spent));
}

#[test]
fn an_explicit_budget_stops_the_warm_up_once_one_site_no_longer_fits() {
    let request = BudgetRequest::Seconds(5.0);
    let early = Duration::from_secs(1);
    let late = Duration::from_millis(4_500);

    assert!(!warm_up_overruns(request, FRAME_TIME, early));
    assert!(warm_up_overruns(request, FRAME_TIME, late));
}

/// A million frames at 10 ms earn the 120 s automatic cap.
#[test]
fn an_automatic_budget_follows_the_current_frame_time() {
    let request = BudgetRequest::Automatic {
        frames: Some(1_000_000),
    };
    let early = Duration::from_secs(1);
    let late = Duration::from_millis(119_500);

    assert!(!warm_up_overruns(request, FRAME_TIME, early));
    assert!(warm_up_overruns(request, FRAME_TIME, late));
}

/// A clip too short to earn any budget is left to the warm-up's own check.
#[test]
fn an_unaffordable_automatic_budget_is_not_an_overrun() {
    let request = BudgetRequest::Automatic { frames: Some(12) };

    assert!(!warm_up_overruns(request, FRAME_TIME, Duration::ZERO));
}
