use std::collections::VecDeque;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::config::{TuningConfig, TuningMode};
use super::device::{DeviceLimits, DeviceProfile};
use super::driver::{TuneOutcome, TuneReport, TuneRequest, is_tuned_on, tune_frames};
use super::frame::{TuningFrame, check_engine_count};
use super::identity::describe;
use super::profile_key::{profile_key, tuning_frame_count};
use super::search::SearchMode;
use super::setup::{KernelSetup, engine_kernels};
use crate::engine::{
    DevicePlane,
    EngineTiming,
    PassEngine,
    PassGeometry,
    PassSpec,
    WindowSpan,
    copy_words_into,
};
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;

/// Where a [TunedGroup] is in its stream.
enum State {
    /// Holding the first frames until there are enough to tune on, each marked when it is context only.
    Buffering { frames: Vec<(TuningFrame, bool)> },
    /// Handing out the frames the buffered frames produced, oldest first.
    ///
    /// Each entry holds every engine's planes back to back, in group order.
    Replaying { ready: VecDeque<Vec<Handle>> },
    /// Forwarding every call to the engines.
    Settled,
}

/// The engines of one group, tuned on the first frames of the stream they denoise.
///
/// With [TuningMode::Auto] and no complete stored result, the group holds the first
/// [tuning_frame_count](crate::tune::tuning_frame_count) frames, tunes on them, then replays them
/// through engines built from the new result. Otherwise it builds those engines straight away.
///
/// Every call takes one list of planes per engine, in the order of the specs, and follows the
/// [PassEngine](crate::PassEngine) contract. A failed tune is logged and the stream carries on with the
/// stored results.
pub struct TunedGroup {
    device: cubecl::Device,
    client: Client,
    profile: DeviceProfile,
    specs: Vec<PassSpec>,
    config: TuningConfig,
    engines: Vec<Box<dyn PassEngine>>,
    state: State,
    report: Option<TuneReport>,
    sample_every: Option<u32>,
    poisoned: bool,
}

impl TunedGroup {
    /// Prepares a group of one engine per spec on `client`, which runs on `device`.
    ///
    /// Empty `specs` are an [Error::InvalidOptions].
    pub fn new(
        device: &cubecl::Device,
        client: &Client,
        specs: Vec<PassSpec>,
        config: TuningConfig,
    ) -> Result<Self, Error> {
        if specs.is_empty() {
            let message = "a tuned group needs at least one engine".to_string();
            return Err(Error::InvalidOptions(message));
        }

        let identity = describe(device, client);
        let limits = DeviceLimits::from_client(client);
        let profile = DeviceProfile { identity, limits };

        let mut group = Self {
            device: device.clone(),
            client: client.clone(),
            profile,
            specs,
            config,
            engines: Vec::new(),
            state: State::Settled,
            report: None,
            sample_every: None,
            poisoned: false,
        };

        let tuned = is_tuned_on(&group.profile.identity, &group.specs, &group.config);
        if !tuned {
            group.state = State::Buffering { frames: Vec::new() };
            return Ok(group);
        }

        group.engines = group.build_engines(group.config.mode)?;

        Ok(group)
    }

    /// Ingests one frame and returns how many frames are ready to emit.
    ///
    /// While buffering it returns 0 until the tuning frame count is reached, then every frame
    /// the buffer produced.
    pub fn push(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<usize, Error> {
        self.check_usable()?;

        match &mut self.state {
            State::Buffering { frames } => {
                let frame = TuningFrame::copy_from(&self.client, &self.specs, planes)?;
                frames.push((frame, false));

                let needed = tuning_frame_count(&self.specs);
                if frames.len() < needed {
                    return Ok(0);
                }

                let buffered = std::mem::take(frames);
                self.settle(buffered)
            },
            State::Replaying { .. } => Err(Error::OutputsPending),
            State::Settled => self.push_engines(planes),
        }
    }

    /// Ingests one frame as context only, producing no output from this call.
    ///
    /// It returns [Error::ContextAfterPush] once the stream has had a real push.
    pub fn push_context(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
        self.check_usable()?;

        match &mut self.state {
            State::Buffering { frames } => {
                let has_pushed = frames.iter().any(|(_, context)| !context);
                if has_pushed {
                    return Err(Error::ContextAfterPush);
                }

                let frame = TuningFrame::copy_from(&self.client, &self.specs, planes)?;
                frames.push((frame, true));

                let needed = tuning_frame_count(&self.specs);
                if frames.len() < needed {
                    return Ok(());
                }

                let buffered = std::mem::take(frames);
                self.settle(buffered)?;

                Ok(())
            },
            State::Replaying { .. } => Err(Error::OutputsPending),
            State::Settled => self.push_context_engines(planes),
        }
    }

    /// Writes the oldest ready frame into `planes`.
    pub fn emit_into(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
        self.check_usable()?;

        match &mut self.state {
            State::Buffering { .. } => Err(Error::NothingToEmit),
            State::Replaying { ready } => {
                check_output_planes(&self.specs, planes)?;

                let frame = ready.pop_front().expect("a replaying group holds a ready frame");
                if ready.is_empty() {
                    self.state = State::Settled;
                }

                copy_frame(&self.client, &self.specs, &frame, planes);

                Ok(())
            },
            State::Settled => self.emit_engines(planes),
        }
    }

    /// Ends the stream and returns how many frames are ready to emit.
    ///
    /// A group still buffering tunes on what it holds first, and the frames those produce come
    /// before the tail frames. With nothing buffered it returns 0 and keeps buffering.
    pub fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;

        match &mut self.state {
            State::Buffering { frames } if frames.is_empty() => Ok(0),
            State::Buffering { frames } => {
                let buffered = std::mem::take(frames);
                let queued = self.settle(buffered)?;
                let tail = self.finish_engines()?;
                Ok(queued + tail)
            },
            State::Replaying { .. } => Err(Error::OutputsPending),
            State::Settled => self.finish_engines(),
        }
    }

    /// Abandons the current stream.
    ///
    /// A group still buffering drops its frames and buffers again from the next push.
    pub fn reset(&mut self) {
        self.poisoned = false;

        if let State::Buffering { frames } = &mut self.state {
            frames.clear();
            return;
        }

        self.state = State::Settled;

        for engine in &mut self.engines {
            engine.reset();
        }
    }

    /// Reads back the film grain measured since the last call, one list per engine.
    ///
    /// The grain of the frames a group tunes on is dropped, and only the stream's own grain is kept.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<Vec<GrainChunk>>, Error> {
        if self.is_buffering() {
            let empty = vec![Vec::new(); self.specs.len()];
            return Ok(empty);
        }

        self.engines
            .iter_mut()
            .map(|engine| engine.drain_grain_chunks())
            .collect()
    }

    /// Device time spent so far, one entry per engine.
    pub fn timing(&mut self) -> Vec<EngineTiming> {
        if self.is_buffering() {
            return vec![EngineTiming::default(); self.specs.len()];
        }

        self.engines.iter_mut().map(|engine| engine.timing()).collect()
    }

    /// Samples the device time of one frame in every `every`, or of none at 0.
    pub fn sample_device_time(&mut self, every: u32) {
        self.sample_every = Some(every);

        for engine in &mut self.engines {
            engine.sample_device_time(every);
        }
    }

    /// The widest window any engine of the group reads.
    pub fn window_span(&self) -> WindowSpan {
        let spans = self.specs.iter().map(PassSpec::window_span);
        let widest = spans.max_by_key(WindowSpan::frame_count);

        widest.expect("a tuned group holds at least one engine")
    }

    /// The most frames any engine of the group holds before it emits one.
    pub fn max_held_frames(&self) -> usize {
        let held = self.specs.iter().map(PassSpec::max_held_frames);

        held.max().expect("a tuned group holds at least one engine")
    }

    pub fn device_profile(&self) -> &DeviceProfile {
        &self.profile
    }

    /// What the group's tuning pass did, once it has run.
    pub fn tune_report(&self) -> Option<&TuneReport> {
        self.report.as_ref()
    }

    fn is_buffering(&self) -> bool {
        matches!(self.state, State::Buffering { .. })
    }

    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }

        Ok(())
    }

    /// Builds an engine per spec from the store's picks under `mode`, with no tuning ledger.
    fn build_engines(&self, mode: TuningMode) -> Result<Vec<Box<dyn PassEngine>>, Error> {
        let config = TuningConfig {
            mode,
            ..self.config.clone()
        };
        let setup = KernelSetup::default();
        let mut engines = Vec::with_capacity(self.specs.len());

        for spec in &self.specs {
            let key = profile_key(spec);
            let kernels = engine_kernels(&setup, &config, &key, &self.profile);
            let mut engine = spec.build(&self.client, kernels)?;

            if let Some(every) = self.sample_every {
                engine.sample_device_time(every);
            }

            engines.push(engine);
        }

        Ok(engines)
    }

    /// Tunes on the buffered `frames`, then replays them through settled engines.
    ///
    /// Returns how many replayed frames are ready. An error leaves the group refusing calls with
    /// [Error::NeedsReset] until [Self::reset].
    fn settle(&mut self, frames: Vec<(TuningFrame, bool)>) -> Result<usize, Error> {
        self.tune(&frames);

        let replayed = self.replay(frames);
        match replayed {
            Ok(ready) if ready.is_empty() => {
                self.state = State::Settled;
                Ok(0)
            },
            Ok(ready) => {
                let count = ready.len();
                self.state = State::Replaying { ready };
                Ok(count)
            },
            Err(error) => {
                self.state = if self.engines.is_empty() {
                    State::Buffering { frames: Vec::new() }
                } else {
                    State::Settled
                };

                self.poisoned = true;
                Err(error)
            },
        }
    }

    /// Runs a tuning pass over `frames` and keeps its report.
    fn tune(&mut self, frames: &[(TuningFrame, bool)]) {
        let tuning_frames: Vec<TuningFrame> = frames.iter().map(|(frame, _)| frame.clone()).collect();
        let request = TuneRequest {
            force: false,
            mode: SearchMode::Families,
        };

        let report = tune_frames(
            &self.device,
            &self.client,
            &self.specs,
            &tuning_frames,
            &self.config,
            &request,
        );

        if let TuneOutcome::Skipped(reason) = &report.outcome {
            tracing::warn!(%reason, "kernel tuning did not run, using the stored results");
        }

        self.report = Some(report);
    }

    /// Builds the settled engines and pushes `frames` through them, keeping every ready frame.
    ///
    /// Each frame is dropped once it is pushed, so the buffer shrinks as the replay outputs grow.
    fn replay(&mut self, frames: Vec<(TuningFrame, bool)>) -> Result<VecDeque<Vec<Handle>>, Error> {
        let mode = self.config.mode.after_tuning();
        self.engines = self.build_engines(mode)?;

        let mut ready = VecDeque::new();

        for (frame, context) in frames {
            let count = self.push_buffered(&frame, context)?;
            drop(frame);

            for _ in 0..count {
                let owned = self.emit_owned()?;
                ready.push_back(owned);
            }
        }

        Ok(ready)
    }

    /// Pushes one buffered frame through the engines, returning how many frames are ready.
    fn push_buffered(&mut self, frame: &TuningFrame, context: bool) -> Result<usize, Error> {
        let lists: Vec<Vec<DevicePlane<'_>>> = self
            .specs
            .iter()
            .enumerate()
            .map(|(index, spec)| frame.planes_for(index, spec.geometry()))
            .collect();
        let planes: Vec<&[DevicePlane<'_>]> = lists.iter().map(Vec::as_slice).collect();

        if context {
            self.push_context_engines(&planes)?;
            return Ok(0);
        }

        self.push_engines(&planes)
    }

    /// Emits one frame from every engine into new handles, every engine's planes back to back.
    fn emit_owned(&mut self) -> Result<Vec<Handle>, Error> {
        let mut owned = Vec::new();

        for (spec, engine) in self.specs.iter().zip(&mut self.engines) {
            let geometry = spec.geometry();
            let handles = output_planes(&self.client, geometry);
            let planes: Vec<DevicePlane<'_>> = handles
                .iter()
                .map(|handle| DevicePlane::new(handle, geometry.width, geometry.height))
                .collect();

            engine.emit_into(&planes)?;
            owned.extend(handles);
        }

        Ok(owned)
    }

    fn push_engines(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<usize, Error> {
        check_engine_count(&self.specs, planes)?;

        let mut ready_counts = Vec::with_capacity(self.engines.len());
        for (engine, engine_planes) in self.engines.iter_mut().zip(planes) {
            let ready = engine.push(engine_planes)?;
            ready_counts.push(ready);
        }

        in_step(&ready_counts)
    }

    fn push_context_engines(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
        check_engine_count(&self.specs, planes)?;

        for (engine, engine_planes) in self.engines.iter_mut().zip(planes) {
            engine.push_context(engine_planes)?;
        }

        Ok(())
    }

    fn emit_engines(&mut self, planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
        check_engine_count(&self.specs, planes)?;

        for (engine, engine_planes) in self.engines.iter_mut().zip(planes) {
            engine.emit_into(engine_planes)?;
        }

        Ok(())
    }

    fn finish_engines(&mut self) -> Result<usize, Error> {
        let mut tail_counts = Vec::with_capacity(self.engines.len());
        for engine in &mut self.engines {
            let tail = engine.finish()?;
            tail_counts.push(tail);
        }

        in_step(&tail_counts)
    }
}

fn check_output_planes(specs: &[PassSpec], planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
    check_engine_count(specs, planes)?;

    for (spec, engine_planes) in specs.iter().zip(planes) {
        let geometry = spec.geometry();
        geometry.check_planes(engine_planes, geometry.output)?;
    }

    Ok(())
}

/// The shared count of `counts`, or an error when the engines disagree.
fn in_step(counts: &[usize]) -> Result<usize, Error> {
    let first = counts.first().copied().unwrap_or(0);
    if counts.iter().all(|count| *count == first) {
        return Ok(first);
    }

    let message = format!("engines are out of step, their frame counts are {counts:?}");
    Err(Error::InvalidOptions(message))
}

/// New handles for one emitted frame of an engine with `geometry`, whole words each.
fn output_planes(client: &Client, geometry: PassGeometry) -> Vec<Handle> {
    let bytes = geometry.output.plane_bytes(geometry.pixels()) as usize;
    let count = geometry.channels.count() as usize;

    (0..count).map(|_| client.empty(bytes)).collect()
}

/// Copies a replayed frame, every engine's planes back to back, into the caller's `planes`.
fn copy_frame(client: &Client, specs: &[PassSpec], frame: &[Handle], planes: &[&[DevicePlane<'_>]]) {
    let mut sources = frame.iter();

    for (spec, engine_planes) in specs.iter().zip(planes) {
        let geometry = spec.geometry();
        let words = geometry.output.plane_bytes(geometry.pixels()) as usize / 4;

        for plane in engine_planes.iter() {
            let source = sources
                .next()
                .expect("a replayed frame holds every engine's planes");
            copy_words_into(client, source, plane.handle(), words);
        }
    }
}
