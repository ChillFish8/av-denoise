use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::{GpuOutput, NlmDenoiser};
use super::options::{NlmeansAlgorithm, resolve_params};
use super::params::validate_dimensions;
use crate::engine::{DevicePlane, EdgePadding, EgressSource, Engine, Geometry, WindowSpan, egress};
use crate::error::Error;

/// Non-local means over GPU planes.
pub struct Nlmeans<R: Runtime> {
    front: NlmDenoiser<R>,
    geometry: Geometry,
    /// The finished frame waiting for `emit_into`, if any.
    ready: Option<Handle>,
    /// Tail frames still to produce after `finish`.
    tail_remaining: usize,
    poisoned: bool,
}

impl<R: Runtime> Nlmeans<R> {
    /// Builds the engine, returning [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    pub fn new(
        client: &ComputeClient<R>,
        algorithm: NlmeansAlgorithm,
        geometry: Geometry,
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

        let front = NlmDenoiser::new(client, params, geometry.width, geometry.height);

        Ok(Self {
            front,
            geometry,
            ready: None,
            tail_remaining: 0,
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

    fn write_out(&self, frame: &Handle, planes: &[DevicePlane<'_>]) {
        let channels = self.geometry.channels;
        let source = EgressSource {
            frame,
            pixels: self.geometry.pixels(),
            channels: channels.count(),
            stored_ch: channels.storage_count(),
        };
        let client = self.front.compute_client();
        let placeholder = self.front.placeholder();

        egress(client, source, planes, self.geometry.output, placeholder);
    }

    #[cfg(test)]
    pub(crate) fn fail_through_guard_for_test(&mut self) -> Result<(), Error> {
        let failure = Err(anyhow::anyhow!("forced failure"));

        self.guard(failure)
    }
}

impl<R: Runtime> Engine for Nlmeans<R> {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let pushed = self.front.push_planes(planes, self.geometry.input);
        self.guard(pushed)?;

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

        let pushed = self.front.push_planes(planes, self.geometry.input);
        self.guard(pushed)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.output)?;

        if let Some(frame) = self.ready.take() {
            self.write_out(&frame, planes);
            return Ok(());
        }

        if self.tail_remaining == 0 {
            return Err(Error::NothingToEmit);
        }

        let step = self.next_tail_frame();
        let output = self.guard(step)?;
        self.write_out(&output.handle, planes);

        self.tail_remaining -= 1;
        if self.tail_remaining == 0 {
            self.front.reset_stream_state();
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;

        self.tail_remaining = self.front.flush_target();
        if self.tail_remaining == 0 {
            self.front.reset_stream_state();
        }

        Ok(self.tail_remaining)
    }

    fn reset(&mut self) {
        self.ready = None;
        self.tail_remaining = 0;
        self.poisoned = false;
        self.front.reset_stream_state();
    }

    fn window_span(&self) -> WindowSpan {
        let radius = self.front.params.temporal_radius as usize;

        WindowSpan {
            behind: radius,
            ahead: radius,
            edges: EdgePadding::Repeat,
        }
    }

    fn max_held_frames(&self) -> usize {
        self.front.params.temporal_radius as usize
    }
}
