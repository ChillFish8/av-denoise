use std::time::Duration;

use av_denoise_core::{EngineTiming, PassTimings};

/// One denoised plane group's device time since its denoiser was built.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlaneTiming {
    /// The GPU time of every frame, estimated from the sampled frames.
    ///
    /// It is `None` when device timing is off or the backend has no device timestamps.
    pub device_busy: Option<Duration>,
}

impl PlaneTiming {
    pub(crate) fn new(frames: u64, engine: EngineTiming) -> Self {
        let device_busy = estimate_device_busy(engine, frames);
        Self { device_busy }
    }
}

/// Where a denoiser's time went since it was built.
///
/// One denoiser cleans every plane together, so the host time is shared by them. Each denoised plane
/// keeps its own device time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlanarTiming {
    /// Host time blocked until denoised frames' readbacks landed.
    pub blocked: Duration,
    /// Host time spent building and queuing GPU work, with the blocked time taken out.
    pub submit: Duration,
    /// Frames emitted.
    pub frames: u64,
    /// Luma's device time, `None` when luma is not denoised.
    pub luma: Option<PlaneTiming>,
    /// Chroma's device time, `None` when chroma is not denoised.
    pub chroma: Option<PlaneTiming>,
}

impl PlanarTiming {
    /// Combines the host's call and output-wait clocks with the engine's per-pass counters.
    pub(crate) fn new(calls: Duration, output_wait: Duration, frames: u64, passes: PassTimings) -> Self {
        let submit = calls.saturating_sub(output_wait);
        let luma = passes.luma.map(|engine| PlaneTiming::new(frames, engine));
        let chroma = passes.chroma.map(|engine| PlaneTiming::new(frames, engine));

        Self {
            blocked: output_wait,
            submit,
            frames,
            luma,
            chroma,
        }
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
