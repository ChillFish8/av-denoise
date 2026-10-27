use std::collections::VecDeque;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use cubecl::future::DynFut;
use cubecl::prelude::*;
use cubecl::profile::{ProfileDuration, ProfileTicks, TimingMethod};

use crate::tune::gpu::profile_chunk_limit;
use crate::tune::{TuneLedger, TunePhase};

/// Device time one engine has spent since it was built.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineTiming {
    /// Device time summed over the sampled frames.
    pub device_time: Duration,
    /// How many frames had their device time sampled.
    pub device_samples: u64,
}

/// Samples the device time of one frame in every few.
///
/// A sampled frame runs its GPU work inside device-timestamp spans. The spans resolve later, so
/// sampling never waits on the device. A device without timestamp queries samples nothing, because
/// its spans sync the host and cannot tell GPU time from host time. Nothing is sampled while an
/// attached tuning ledger is out of its idle phase, since the tuner profiles those launches itself.
/// A frame whose tune starts partway through counts as sampled but holds only part of its device
/// time.
#[derive(Default)]
pub(crate) struct DeviceTimer {
    /// One frame in this many is sampled. Zero samples none.
    every: u32,
    /// Whether the device times spans with host clocks.
    system_timing: bool,
    ledger: Option<Arc<TuneLedger>>,
    frames: u64,
    sampling: bool,
    /// Whether the current frame has recorded a device span yet.
    frame_recorded: bool,
    pending: VecDeque<DynFut<Option<ProfileTicks>>>,
    total: Duration,
    sampled_frames: u64,
}

impl DeviceTimer {
    pub(crate) fn set_every(&mut self, every: u32) {
        self.every = every;
    }

    /// Latches how the device times spans. Only [TimingMethod::Device] gets sampled.
    pub(crate) fn set_timing_method(&mut self, method: TimingMethod) {
        self.system_timing = method != TimingMethod::Device;
    }

    /// Starts a frame, deciding whether its work is sampled, and collects spans that have landed.
    pub(crate) fn begin_frame(&mut self) {
        self.collect_landed();

        let due = self.every > 0 && self.frames.is_multiple_of(self.every as u64);
        self.sampling = due && !self.system_timing;
        self.frame_recorded = false;
        self.frames += 1;
    }

    /// Attaches the tuning ledger whose phase gates every span.
    pub(crate) fn set_ledger(&mut self, ledger: Option<Arc<TuneLedger>>) {
        self.ledger = ledger;
    }

    /// Whether the next span of the current frame is timed.
    ///
    /// It reads the ledger's phase at every call, so a frame that began while tuning was idle stops
    /// sampling as soon as a census or tune starts.
    pub(crate) fn is_sampling(&self) -> bool {
        if !self.sampling {
            return false;
        }

        self.ledger
            .as_ref()
            .is_none_or(|ledger| ledger.phase() == TunePhase::Idle)
    }

    /// Stops sampling until the next frame starts, so tail work at the end of a stream is untimed.
    pub(crate) fn stop_sampling(&mut self) {
        self.sampling = false;
    }

    /// Keeps one span of the current frame's work.
    pub(crate) fn record(&mut self, duration: ProfileDuration) {
        if duration.timing_method() != TimingMethod::Device {
            return;
        }

        if !self.frame_recorded {
            self.frame_recorded = true;
            self.sampled_frames += 1;
        }

        let future = duration.into_future();
        self.pending.push_back(future);
    }

    /// Waits for every span still in flight and returns the device time and sampled frame count.
    pub(crate) fn settle(&mut self) -> (Duration, u64) {
        while let Some(future) = self.pending.pop_front() {
            let ticks = cubecl::future::block_on(future);
            if let Some(ticks) = ticks {
                self.total += ticks.duration();
            }
        }

        (self.total, self.sampled_frames)
    }

    /// Adds every span at the front of the queue that has landed, without blocking.
    fn collect_landed(&mut self) {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        while let Some(future) = self.pending.front_mut() {
            let poll = future.as_mut().poll(&mut context);
            let Poll::Ready(ticks) = poll else {
                break;
            };

            if let Some(ticks) = ticks {
                self.total += ticks.duration();
            }

            self.pending.pop_front();
        }
    }
}

impl Drop for DeviceTimer {
    /// Lets every span in flight land, since a span's readback holds a mapped staging buffer.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }

        self.settle();
    }
}

/// Code that keeps a [DeviceTimer] for the frames it runs.
pub(crate) trait DeviceTimed {
    fn device_timer(&mut self) -> &mut DeviceTimer;
}

/// Runs `work` on `target`, inside a device-timestamp span when the current frame is sampled.
///
/// A span only times its first compute pass, so `work` must queue no more dispatches than one
/// pass holds and no upload or readback after its first dispatch.
/// [units_per_span](crate::engine::timing::units_per_span) sizes the chunks of a longer segment.
pub(crate) fn timed<T, O>(client: &Client, target: &mut T, work: impl FnOnce(&mut T) -> O + Send) -> O
where
    T: DeviceTimed + Send,
    O: Send,
{
    let sampling = target.device_timer().is_sampling();
    if !sampling {
        return work(target);
    }

    let mut pending_work = Some(work);
    let mut outcome = None;
    let profiled = client.profile(
        || {
            let run = pending_work.take().expect("the profiled work runs once");
            let result = run(target);
            outcome = Some(result);
        },
        "av-denoise frame segment",
    );

    // A span that fails to start never runs the work, so it runs untimed instead.
    if let Some(run) = pending_work.take() {
        let result = run(target);
        outcome = Some(result);
    }

    if let Ok(((), duration)) = profiled {
        target.device_timer().record(duration);
    }

    outcome.expect("the work ran")
}

/// How many units of `dispatches_per_unit` dispatches each fit in one device span.
///
/// The wgpu runtime ends a compute pass after a fixed number of dispatches, and a span only times
/// its first pass.
pub(crate) fn units_per_span(dispatches_per_unit: u32) -> usize {
    let limit = profile_chunk_limit();
    let units = limit / dispatches_per_unit;

    units.max(1) as usize
}
