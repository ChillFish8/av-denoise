use std::time::Duration;

use av_denoise_core::EngineTiming;

/// Where one plane's host time went since its denoiser was built.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlaneTiming {
    /// Host time blocked until denoised frames' readbacks landed.
    pub blocked: Duration,
    /// Host time spent building and queuing GPU work, with the blocked time taken out.
    pub submit: Duration,
    /// The GPU time of every frame, estimated from the sampled frames.
    ///
    /// It is `None` when device timing is off or the backend has no device timestamps.
    pub device_busy: Option<Duration>,
    /// Frames emitted.
    pub frames: u64,
}

impl PlaneTiming {
    /// Combines the host's call and output-wait clocks with the engine's counters.
    pub(crate) fn new(calls: Duration, output_wait: Duration, frames: u64, engine: EngineTiming) -> Self {
        let submit = calls.saturating_sub(output_wait);
        let device_busy = estimate_device_busy(engine, frames);

        Self {
            blocked: output_wait,
            submit,
            device_busy,
            frames,
        }
    }
}

/// Scales the sampled frames' device time up to every frame.
fn estimate_device_busy(engine: EngineTiming, frames: u64) -> Option<Duration> {
    if engine.device_samples == 0 {
        return None;
    }

    let per_frame = engine.device_time.div_f64(engine.device_samples as f64);
    let total = per_frame.mul_f64(frames as f64);
    Some(total)
}
