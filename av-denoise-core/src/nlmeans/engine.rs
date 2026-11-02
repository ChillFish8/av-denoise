use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ChannelMode;
use super::denoiser::{GpuOutput, NlmDenoiser, check_u32_indexable, front_buffer_sizes, lookahead_depth_for};
use super::options::{NlmeansAlgorithm, resolve_params};
use super::params::validate_dimensions;
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
use crate::tune::{EngineKernels, KernelSelector};

/// The frames an engine at `temporal_radius` holds with a noise look-ahead `lookahead` deep.
fn held_frames(temporal_radius: u32, lookahead: usize) -> usize {
    temporal_radius as usize + lookahead
}

/// Non-local means over GPU planes.
pub(crate) struct Nlmeans {
    front: NlmDenoiser,
    geometry: PassGeometry,
    /// The finished frame waiting for `emit_into`, if any.
    ready: Option<Handle>,
    /// Tail frames still to produce after `finish`.
    tail_remaining: usize,
    /// Whether the current stream has had a real push, after which context frames are refused.
    pushed: bool,
    /// The stream frame of the last emit.
    emitted: Option<u32>,
    poisoned: bool,
}

impl Nlmeans {
    /// Builds the engine, returning [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub fn new(client: &Client, algorithm: NlmeansAlgorithm, geometry: PassGeometry) -> Result<Self, Error> {
        let kernels = EngineKernels::default();
        Self::with_kernels(client, algorithm, geometry, kernels, None, false)
    }

    /// Builds the engine with each launch site's candidate picked from `kernels`.
    ///
    /// Its motion searches start from the camera models in `motion` when given. With `guides_only`,
    /// it runs only its motion and noise reading and its emits write nothing. Returns
    /// [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    pub fn with_kernels(
        client: &Client,
        algorithm: NlmeansAlgorithm,
        geometry: PassGeometry,
        kernels: EngineKernels,
        motion: Option<SharedMotion>,
        guides_only: bool,
    ) -> Result<Self, Error> {
        geometry.validate()?;

        let dimensions = validate_dimensions(geometry.width, geometry.height);
        dimensions.map_err(|error| {
            let message = error.to_string();
            Error::InvalidGeometry(message)
        })?;

        let params = resolve_params(&algorithm, geometry.channels);
        let validated = params.validate();
        validated.map_err(|error| {
            let message = error.to_string();
            Error::InvalidOptions(message)
        })?;

        let ring_slots = u64::from(params.total_frames());
        geometry.check_ring_fits(ring_slots)?;

        let reads_luma_motion = motion.as_ref().is_some_and(SharedMotion::reads_luma_motion);
        let sizes = front_buffer_sizes(
            client,
            &params,
            geometry.width,
            geometry.height,
            reads_luma_motion,
        );
        let indexable = check_u32_indexable(&sizes.buffers);
        indexable.map_err(Error::InvalidGeometry)?;

        let selector = KernelSelector::new(client, kernels);
        let mut front = NlmDenoiser::with_kernels(client, params, geometry.width, geometry.height, selector);
        if let Some(motion) = motion {
            front.set_shared_motion(motion);
        }

        if guides_only {
            front.set_guides_only();
        }

        Ok(Self {
            front,
            geometry,
            ready: None,
            tail_remaining: 0,
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
        if self.ready.is_some() || self.tail_remaining > 0 {
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

    fn restart_stream(&mut self) {
        self.front.reset_stream_state();
        self.pushed = false;
    }

    /// Poisons the engine if `result` is an error.
    fn guard<T>(&mut self, result: Result<T, anyhow::Error>) -> Result<T, Error> {
        if result.is_err() {
            self.poisoned = true;
        }

        result.map_err(Error::Gpu)
    }

    /// Steps the flush until it yields the next tail frame.
    fn next_tail_frame(&mut self) -> Result<GpuOutput, anyhow::Error> {
        loop {
            let step = self.front.flush_step_gpu()?;
            if let Some(output) = step {
                return Ok(output);
            }
        }
    }

    /// Writes `frame` into `planes`, unless the engine only guides.
    fn write_out(&self, frame: &Handle, planes: &[DevicePlane<'_>]) {
        if self.front.guides_only() {
            return;
        }

        let channels = self.geometry.channels;
        // The constructor bounds the ring to `u32` elements, so one plane always fits.
        let pixels = self.geometry.pixels() as u32;
        let source = EgressSource {
            frame,
            pixels,
            channels: channels.count(),
        };
        let client = self.front.compute_client();
        let placeholder = self.front.placeholder();

        egress(client, source, planes, self.geometry.output, placeholder);
    }

    /// The most frames an engine built from `algorithm` holds before it emits one.
    ///
    /// A front that measures the noise holds each frame in its noise look-ahead before the frame
    /// enters the ring, on top of the `temporal_radius` frames the ring needs ahead of its centre.
    pub fn held_frames_for(algorithm: &NlmeansAlgorithm, channels: ChannelMode) -> usize {
        let params = resolve_params(algorithm, channels);
        let lookahead = lookahead_depth_for(&params);

        held_frames(params.temporal_radius, lookahead)
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn fail_through_guard_for_test(&mut self) -> Result<(), Error> {
        let failure = Err(anyhow::anyhow!("forced failure"));

        self.guard(failure)
    }
}

impl PassEngine for Nlmeans {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let pushed = self.front.push_planes(planes, self.geometry.input);
        self.guard(pushed)?;

        self.pushed = true;

        let submitted = self.front.denoise_submit_gpu();
        let output = self.guard(submitted)?;
        self.ready = output.map(|output| output.handle);

        let has_ready = self.ready.is_some();
        Ok(usize::from(has_ready))
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;
        self.check_no_push_yet()?;

        let pushed = self.front.push_planes(planes, self.geometry.input);
        self.guard(pushed)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.output)?;

        // The window has not moved since the frame was denoised, so its centre is still that frame.
        if let Some(frame) = self.ready.take() {
            let emitted = self.front.centre_frame();
            self.emitted = Some(emitted);
            self.write_out(&frame, planes);
            return Ok(());
        }

        if self.tail_remaining == 0 {
            return Err(Error::NothingToEmit);
        }

        let step = self.next_tail_frame();
        let output = self.guard(step)?;
        let emitted = self.front.centre_frame();
        self.emitted = Some(emitted);
        self.write_out(&output.handle, planes);

        self.tail_remaining -= 1;
        if self.tail_remaining == 0 {
            self.restart_stream();
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;

        self.tail_remaining = self.front.flush_target();
        if self.tail_remaining == 0 {
            self.restart_stream();
        }

        Ok(self.tail_remaining)
    }

    fn reset(&mut self) {
        self.ready = None;
        self.emitted = None;
        self.tail_remaining = 0;
        self.poisoned = false;
        self.restart_stream();
    }

    fn window_span(&self) -> WindowSpan {
        let held = self.max_held_frames();

        WindowSpan {
            behind: held,
            ahead: held,
            edges: EdgePadding::Repeat,
        }
    }

    fn max_held_frames(&self) -> usize {
        let lookahead = self.front.lookahead_depth();

        held_frames(self.front.params.temporal_radius, lookahead)
    }

    fn timing(&mut self) -> EngineTiming {
        self.front.timing()
    }

    fn sample_device_time(&mut self, every: u32) {
        self.front.sample_device_time(every);
    }
}

impl FramePass for Nlmeans {
    fn next_frame(&self) -> u32 {
        self.front.stream_frames()
    }

    fn emitted_frame(&self) -> Option<u32> {
        self.emitted
    }

    fn guides_only(&self) -> bool {
        self.front.guides_only()
    }
}
