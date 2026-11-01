use std::time::Duration;

use av_denoise_core::{EngineTiming, PassTimings};

use crate::host::{PlanarTiming, PlaneTiming};

fn engine_timing(device_time: Duration, device_samples: u64) -> EngineTiming {
    EngineTiming {
        device_time,
        device_samples,
    }
}

fn pass_timings(luma: Option<EngineTiming>, chroma: Option<EngineTiming>) -> PassTimings {
    PassTimings { luma, chroma }
}

#[test]
fn blocked_time_is_the_output_wait() {
    let calls = Duration::from_millis(50);
    let output_wait = Duration::from_millis(7);
    let engine = engine_timing(Duration::ZERO, 0);
    let passes = pass_timings(Some(engine), None);
    let timing = PlanarTiming::new(calls, output_wait, 10, passes);

    assert_eq!(timing.blocked, output_wait);
    assert_eq!(timing.submit, Duration::from_millis(43));
    assert_eq!(timing.frames, 10);
}

#[test]
fn submit_time_saturates_at_zero_when_blocked_time_exceeds_the_calls() {
    let calls = Duration::from_millis(5);
    let output_wait = Duration::from_millis(10);
    let engine = engine_timing(Duration::ZERO, 0);
    let passes = pass_timings(Some(engine), None);
    let timing = PlanarTiming::new(calls, output_wait, 1, passes);

    assert_eq!(timing.blocked, Duration::from_millis(10));
    assert_eq!(timing.submit, Duration::ZERO);
}

#[test]
fn device_busy_scales_the_sampled_frames_up_to_every_frame() {
    let device_time = Duration::from_millis(6);
    let engine = engine_timing(device_time, 3);
    let timing = PlaneTiming::new(10, engine);

    assert_eq!(timing.device_busy, Some(Duration::from_millis(20)));
}

#[test]
fn device_busy_is_unset_without_samples() {
    let engine = engine_timing(Duration::ZERO, 0);
    let timing = PlaneTiming::new(10, engine);

    assert_eq!(timing.device_busy, None);
}

#[test]
fn each_denoised_plane_keeps_its_own_device_time() {
    let calls = Duration::from_millis(50);
    let output_wait = Duration::from_millis(7);
    let luma_engine = engine_timing(Duration::from_millis(4), 2);
    let chroma_engine = engine_timing(Duration::from_millis(2), 2);
    let passes = pass_timings(Some(luma_engine), Some(chroma_engine));
    let timing = PlanarTiming::new(calls, output_wait, 10, passes);

    let luma = timing.luma.expect("luma is denoised");
    let chroma = timing.chroma.expect("chroma is denoised");

    assert_eq!(luma.device_busy, Some(Duration::from_millis(20)));
    assert_eq!(chroma.device_busy, Some(Duration::from_millis(10)));
}

#[test]
fn a_plane_group_left_uncleaned_has_no_timing() {
    let engine = engine_timing(Duration::ZERO, 0);
    let passes = pass_timings(None, Some(engine));
    let timing = PlanarTiming::new(Duration::ZERO, Duration::ZERO, 10, passes);

    assert_eq!(timing.luma, None);
    assert!(timing.chroma.is_some());
}
