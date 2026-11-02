use std::collections::VecDeque;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::{CompletedRegion, Nl4dDenoiser, buffer_sizes};
use super::grain::GrainChunk;
use super::options::{Nl4dOptions, nlm_params, resolve_params};
use crate::collab::PATCH_SIZE;
use crate::engine::{
    DevicePlane,
    EdgePadding,
    EgressSource,
    EngineTiming,
    FramePass,
    PassEngine,
    PassGeometry,
    WindowSpan,
    egress,
};
use crate::error::Error;
use crate::motion::SharedMotion;
use crate::nlmeans::ChannelMode;
use crate::nlmeans::denoiser::{check_u32_indexable, lookahead_depth_for};
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::noise::NoiseSnapshot;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
use crate::noise::colour::ColourCells;
use crate::noise::colour::ColourSource;
use crate::tune::{EngineKernels, KernelSelector};

/// The frames an engine at `temporal_radius` holds with a noise look-ahead `lookahead` deep.
fn held_frames(temporal_radius: u32, lookahead: usize) -> usize {
    2 * temporal_radius as usize + lookahead
}

/// Four-dimensional collaborative denoising over GPU planes.
pub(crate) struct Nl4d {
    inner: Nl4dDenoiser,
    client: Client,
    geometry: PassGeometry,
    temporal_radius: u32,
    /// Regions finished but not yet emitted, oldest first.
    pending: VecDeque<CompletedRegion>,
    /// The stream's tail frames still to emit after a finish.
    tail_remaining: usize,
    /// Pushes since the last stream start, used to mark a context-led stream as a continuation.
    pushes: usize,
    /// Whether the current stream has had a real push, after which context frames are refused.
    pushed: bool,
    /// The stream frame of the last emit.
    emitted: Option<u32>,
    poisoned: bool,
}

impl Nl4d {
    /// Builds the engine, returning [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    #[cfg(test)]
    pub fn new(client: &Client, options: Nl4dOptions, geometry: PassGeometry) -> Result<Self, Error> {
        let kernels = EngineKernels::default();
        Self::with_kernels(client, options, geometry, kernels, None, false)
    }

    /// Builds the engine with each launch site's candidate picked from `kernels`.
    ///
    /// Its motion searches start from the camera models in `motion` when given. With `guides_only`,
    /// it runs only its motion and noise reading and its emits write nothing. Returns
    /// [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    pub fn with_kernels(
        client: &Client,
        options: Nl4dOptions,
        geometry: PassGeometry,
        kernels: EngineKernels,
        motion: Option<SharedMotion>,
        guides_only: bool,
    ) -> Result<Self, Error> {
        if geometry.width < PATCH_SIZE || geometry.height < PATCH_SIZE {
            let message = format!(
                "frame dimensions {}x{} must be at least {PATCH_SIZE}x{PATCH_SIZE}",
                geometry.width, geometry.height,
            );
            return Err(Error::InvalidGeometry(message));
        }

        geometry.validate()?;

        let params = resolve_params(&options, geometry.channels)?;

        // Buffer sizing builds the motion block grid, which needs validated motion options.
        let validated = params.validate();
        validated.map_err(Error::InvalidOptions)?;
        let front_validated = params.nlm.validate();
        front_validated.map_err(|error| {
            let message = error.to_string();
            Error::InvalidOptions(message)
        })?;

        let ring_slots = 1 + 2 * u64::from(params.temporal_radius);
        geometry.check_ring_fits(ring_slots)?;

        let reads_luma_motion = motion.as_ref().is_some_and(SharedMotion::reads_luma_motion);
        let sizes = buffer_sizes(
            client,
            &params,
            geometry.width,
            geometry.height,
            reads_luma_motion,
        );
        let indexable = check_u32_indexable(&sizes);
        indexable.map_err(Error::InvalidGeometry)?;

        let selector = KernelSelector::new(client, kernels);
        let inner = Nl4dDenoiser::with_kernels(
            client,
            params,
            geometry.width,
            geometry.height,
            selector,
            guides_only,
        );
        let mut inner = inner.map_err(Error::InvalidOptions)?;
        if let Some(motion) = motion {
            inner.set_shared_motion(motion);
        }

        Ok(Self {
            inner,
            client: client.clone(),
            geometry,
            temporal_radius: options.temporal_radius,
            pending: VecDeque::new(),
            tail_remaining: 0,
            pushes: 0,
            pushed: false,
            emitted: None,
            poisoned: false,
        })
    }

    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }

        Ok(())
    }

    fn check_nothing_pending(&self) -> Result<(), Error> {
        if !self.pending.is_empty() || self.tail_remaining > 0 {
            return Err(Error::OutputsPending);
        }

        Ok(())
    }

    fn check_no_push_yet(&self) -> Result<(), Error> {
        if self.pushed {
            return Err(Error::ContextAfterPush);
        }

        Ok(())
    }

    /// Poisons the engine if `result` is an error.
    fn guard<T>(&mut self, result: Result<T, anyhow::Error>) -> Result<T, Error> {
        if result.is_err() {
            self.poisoned = true;
        }

        result.map_err(Error::Gpu)
    }

    fn restart_stream(&mut self) {
        self.inner.reset_stream();
        self.tail_remaining = 0;
        self.pushes = 0;
        self.pushed = false;
    }

    /// Queues the stream's next tail regions, draining the look-ahead before the tail passes run.
    fn queue_tail_regions(&mut self) -> Result<(), anyhow::Error> {
        let drained = self.inner.drain_lookahead()?;
        if let Some(region) = drained {
            self.pending.push_back(region);
            return Ok(());
        }

        let regions = self.inner.finish_passes()?;
        let count = regions.len();
        let owed = self.tail_remaining;
        anyhow::ensure!(count == owed, "the tail completed {count} of {owed} owed regions");

        self.pending.extend(regions);

        Ok(())
    }

    fn write_out(&self, frame: &Handle, planes: &[DevicePlane<'_>]) {
        let channels = self.geometry.channels;
        // The constructor bounds the ring to `u32` elements, so one plane always fits.
        let pixels = self.geometry.pixels() as u32;
        let source = EgressSource {
            frame,
            pixels,
            channels: channels.count(),
        };
        let placeholder = self.inner.placeholder();

        egress(&self.client, source, planes, self.geometry.output, placeholder);
    }

    /// The most frames an engine built from `options` holds before it emits one.
    ///
    /// A front that measures the noise holds each frame in its noise look-ahead before the frame
    /// enters the ring, on top of the `2 * temporal_radius` frames a region waits for its last pass.
    pub fn held_frames_for(options: &Nl4dOptions, channels: ChannelMode) -> usize {
        let front_params = nlm_params(options, channels);
        let lookahead = lookahead_depth_for(&front_params);

        held_frames(options.temporal_radius, lookahead)
    }

    /// Whether an engine built from `options` pools its grain into colour cells.
    ///
    /// That needs the noise measured, from residuals against a neighbour frame.
    pub fn measures_colour_for(options: &Nl4dOptions, channels: ChannelMode) -> bool {
        let front_params = nlm_params(options, channels);
        let lookahead = lookahead_depth_for(&front_params);

        lookahead > 0 && options.temporal_radius >= 1
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn noise_snapshot_for_test(&self) -> Option<NoiseSnapshot> {
        self.inner.noise_snapshot()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn fail_through_guard_for_test(&mut self) -> Result<(), Error> {
        let failure = Err(anyhow::anyhow!("forced failure"));

        self.guard(failure)
    }
}

impl PassEngine for Nl4d {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let pushed = self.inner.push_planes(planes, self.geometry.input);
        self.guard(pushed)?;

        self.pushed = true;

        let submitted = self.inner.submit_passes();
        let region = self.guard(submitted)?;
        if let Some(region) = region {
            self.pending.push_back(region);
        }

        self.pushes += 1;

        Ok(self.pending.len())
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;
        self.check_no_push_yet()?;

        if self.pushes == 0 {
            self.inner.mark_continuation();
        }

        let pushed = self.inner.push_planes(planes, self.geometry.input);
        self.guard(pushed)?;

        self.pushes += 1;

        Ok(())
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.output)?;

        if self.pending.is_empty() && self.tail_remaining > 0 {
            let queued = self.queue_tail_regions();
            self.guard(queued)?;
        }

        let Some(region) = self.pending.pop_front() else {
            return Err(Error::NothingToEmit);
        };

        let emitted = self.inner.front.slot_frame(region.slot);
        self.emitted = Some(emitted);

        if !self.inner.guides_only() {
            let frame = self.inner.read_region(region);
            self.write_out(&frame, planes);
        }

        if self.tail_remaining > 0 {
            self.tail_remaining -= 1;
            if self.tail_remaining == 0 {
                let held_frames = self.inner.front.held_frames();
                debug_assert_eq!(held_frames, 0, "the tail drained the look-ahead");
                self.restart_stream();
            }
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;

        // The tail runs lazily, one region at a time while the look-ahead drains, since each
        // release reuses the ring slot of the region before it.
        let count = self.inner.finish_count();
        if count == 0 {
            self.restart_stream();
        } else {
            self.tail_remaining = count;
        }

        Ok(count)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.emitted = None;
        self.poisoned = false;
        self.restart_stream();
    }

    fn window_span(&self) -> WindowSpan {
        let held = self.max_held_frames();

        WindowSpan {
            behind: held,
            ahead: held,
            edges: EdgePadding::Shifted,
        }
    }

    fn max_held_frames(&self) -> usize {
        let lookahead = self.inner.lookahead_depth();

        held_frames(self.temporal_radius, lookahead)
    }

    fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error> {
        self.check_usable()?;

        let drained = self.inner.drain_grain_chunks();
        self.guard(drained)
    }

    fn timing(&mut self) -> EngineTiming {
        self.inner.timing()
    }

    fn sample_device_time(&mut self, every: u32) {
        self.inner.sample_device_time(every);
    }
}

impl FramePass for Nl4d {
    fn next_frame(&self) -> u32 {
        self.inner.front.stream_frames()
    }

    fn emitted_frame(&self) -> Option<u32> {
        self.emitted
    }

    fn guides_only(&self) -> bool {
        self.inner.guides_only()
    }

    fn measure_colour(&mut self, source: ColourSource) {
        self.inner.front.measure_colour(source);
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    fn colour_cells_for_test(&self) -> Option<&ColourCells> {
        let noise = self.inner.front.measured_noise()?;
        noise.colour_cells()
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    fn colour_gains_for_test(&self) -> Option<&Handle> {
        let noise = self.inner.front.measured_noise()?;
        noise.colour_gains()
    }
}
