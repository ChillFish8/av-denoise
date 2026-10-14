use std::time::{Duration, Instant};

use cubecl::profile::{ProfileDuration, ProfileTicks, TimingMethod};

use crate::engine::timing::DeviceTimer;
use crate::tune::{TuneLedger, TunePhase};

fn device_span(length: Duration) -> ProfileDuration {
    let start = Instant::now();
    let end = start + length;
    let future = async move { ProfileTicks::from_start_end(start, end) };

    ProfileDuration::new_device_time(future)
}

#[test]
fn one_frame_in_every_n_is_sampled() {
    let mut timer = DeviceTimer::default();
    timer.set_every(4);

    let mut sampled = Vec::new();
    for _ in 0..9 {
        timer.begin_frame();
        sampled.push(timer.is_sampling());
    }

    let expected = [true, false, false, false, true, false, false, false, true];
    assert_eq!(sampled, expected);
}

#[test]
fn a_timer_left_at_zero_samples_nothing() {
    let mut timer = DeviceTimer::default();

    timer.begin_frame();

    assert!(!timer.is_sampling());
}

#[test]
fn every_span_of_a_sampled_frame_adds_up() {
    let mut timer = DeviceTimer::default();
    timer.set_every(1);

    timer.begin_frame();
    let first = device_span(Duration::from_millis(3));
    timer.record(first);
    let second = device_span(Duration::from_millis(2));
    timer.record(second);

    timer.begin_frame();
    let third = device_span(Duration::from_millis(4));
    timer.record(third);

    let (total, frames) = timer.settle();
    assert_eq!(total, Duration::from_millis(9));
    assert_eq!(frames, 2);
}

#[test]
fn a_device_without_timestamps_samples_nothing() {
    let mut timer = DeviceTimer::default();
    timer.set_every(1);
    timer.set_timing_method(TimingMethod::System);

    timer.begin_frame();

    assert!(!timer.is_sampling());

    let (total, frames) = timer.settle();
    assert_eq!(total, Duration::ZERO);
    assert_eq!(frames, 0);
}

#[test]
fn system_time_spans_are_not_counted() {
    let mut timer = DeviceTimer::default();
    timer.set_every(1);
    timer.begin_frame();

    let start = Instant::now();
    let end = start + Duration::from_millis(5);
    let system = ProfileDuration::new_system_time(start, end);
    timer.record(system);

    let (total, frames) = timer.settle();
    assert_eq!(total, Duration::ZERO);
    assert_eq!(frames, 0);
}

#[test]
fn stopping_sampling_lasts_until_the_next_frame() {
    let mut timer = DeviceTimer::default();
    timer.set_every(1);

    timer.begin_frame();
    timer.stop_sampling();
    let stopped = timer.is_sampling();

    timer.begin_frame();
    let resumed = timer.is_sampling();

    assert!(!stopped);
    assert!(resumed);
}

#[test]
fn a_ledger_out_of_idle_stops_sampling_within_a_frame() {
    let ledger = TuneLedger::new(false);
    let mut timer = DeviceTimer::default();
    timer.set_every(1);
    timer.set_ledger(Some(ledger.clone()));

    timer.begin_frame();
    let idle = timer.is_sampling();

    ledger.set_phase(TunePhase::Census);
    let census = timer.is_sampling();

    ledger.set_phase(TunePhase::Tune);
    let tune = timer.is_sampling();

    ledger.set_phase(TunePhase::Idle);
    let back_to_idle = timer.is_sampling();

    assert!(idle);
    assert!(!census);
    assert!(!tune);
    assert!(back_to_idle);
}
