use std::time::Duration;

use crate::tune::BudgetRequest;
use crate::tune::driver::pass::warm_up_overruns;

#[test]
fn a_tiny_explicit_budget_stops_the_warm_up_at_once() {
    let request = BudgetRequest::Seconds(0.5);

    assert!(warm_up_overruns(request, Duration::ZERO));
}

#[test]
fn a_full_budget_lets_the_warm_up_run() {
    let request = BudgetRequest::Full;
    let spent = Duration::from_secs(15);

    assert!(!warm_up_overruns(request, spent));
}

#[test]
fn an_explicit_budget_stops_the_warm_up_once_one_site_no_longer_fits() {
    let request = BudgetRequest::Seconds(5.0);
    let early = Duration::from_secs(1);
    let late = Duration::from_millis(4_500);

    assert!(!warm_up_overruns(request, early));
    assert!(warm_up_overruns(request, late));
}
