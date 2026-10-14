use std::time::Duration;

use av_denoise_core::EngineTiming;

use crate::host::PlaneTiming;

fn engine_timing(readback_wait: Duration, device_time: Duration, device_samples: u64) -> EngineTiming {
    EngineTiming {
        readback_wait,
        readback_waits: 4,
        device_time,
        device_samples,
    }
}

#[test]
fn blocked_time_adds_output_waits_and_readback_waits() {
    let engine = engine_timing(Duration::from_millis(3), Duration::ZERO, 0);
    let timing = PlaneTiming::new(Duration::from_millis(50), Duration::from_millis(7), 10, engine);

    assert_eq!(timing.blocked, Duration::from_millis(10));
    assert_eq!(timing.submit, Duration::from_millis(40));
    assert_eq!(timing.frames, 10);
}

#[test]
fn submit_time_saturates_at_zero_when_blocked_time_exceeds_the_calls() {
    let engine = engine_timing(Duration::from_millis(6), Duration::ZERO, 0);
    let timing = PlaneTiming::new(Duration::from_millis(5), Duration::from_millis(4), 1, engine);

    assert_eq!(timing.blocked, Duration::from_millis(10));
    assert_eq!(timing.submit, Duration::ZERO);
}

#[test]
fn device_busy_scales_the_sampled_frames_up_to_every_frame() {
    let engine = engine_timing(Duration::ZERO, Duration::from_millis(6), 3);
    let timing = PlaneTiming::new(Duration::ZERO, Duration::ZERO, 10, engine);

    assert_eq!(timing.device_busy, Some(Duration::from_millis(20)));
}

#[test]
fn device_busy_is_unset_without_samples() {
    let engine = engine_timing(Duration::ZERO, Duration::ZERO, 0);
    let timing = PlaneTiming::new(Duration::ZERO, Duration::ZERO, 10, engine);

    assert_eq!(timing.device_busy, None);
}
