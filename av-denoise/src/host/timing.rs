use std::time::Duration;

use av_denoise_core::{EngineTiming, PassTimings};

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

/// Where each denoised plane's time went.
///
/// One denoiser cleans every plane, so its host time is counted once, on the first denoised plane.
/// Each plane keeps its own device time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlanarTiming {
    pub luma: Option<PlaneTiming>,
    pub chroma: Option<PlaneTiming>,
}

impl PlanarTiming {
    /// Splits the host's clocks and the engine's per-pass counters into per-plane timings.
    pub(crate) fn new(calls: Duration, output_wait: Duration, frames: u64, passes: PassTimings) -> Self {
        let plane = |engine: EngineTiming, holds_host_time: bool| {
            if holds_host_time {
                PlaneTiming::new(calls, output_wait, frames, engine)
            } else {
                PlaneTiming::new(Duration::ZERO, Duration::ZERO, frames, engine)
            }
        };

        let luma = passes.luma.map(|engine| plane(engine, true));
        let chroma_holds_host_time = luma.is_none();
        let chroma = passes.chroma.map(|engine| plane(engine, chroma_holds_host_time));

        Self { luma, chroma }
    }

    /// Each denoised plane's name and timing, in luma, chroma order.
    pub fn planes(&self) -> Vec<(&'static str, PlaneTiming)> {
        let named = [("luma", self.luma), ("chroma", self.chroma)];

        named
            .into_iter()
            .filter_map(|(name, timing)| timing.map(|timing| (name, timing)))
            .collect()
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
