use std::collections::VecDeque;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::{CompletedRegion, Nl4dDenoiser};
use super::grain::GrainChunk;
use super::options::{Nl4dOptions, resolve_params};
use crate::collab::PATCH_SIZE;
use crate::denoiser::OutputFormat;
use crate::engine::{DevicePlane, EdgePadding, EgressSource, Engine, Geometry, WindowSpan, egress};
use crate::error::Error;

/// Four-dimensional collaborative denoising over GPU planes.
pub struct Nl4d<R: Runtime> {
    inner: Nl4dDenoiser<R>,
    client: ComputeClient<R>,
    geometry: Geometry,
    temporal_radius: u32,
    /// Regions finished but not yet emitted, oldest first.
    pending: VecDeque<CompletedRegion>,
    /// Whether the regions in `pending` are a stream's tail.
    finishing: bool,
    /// Pushes since the last stream start, used to mark a context-led stream as a continuation.
    pushes: usize,
    poisoned: bool,
}

impl<R: Runtime> Nl4d<R> {
    /// Builds the engine, returning [Error::InvalidGeometry] or [Error::InvalidOptions] for unusable input.
    pub fn new(client: &ComputeClient<R>, options: Nl4dOptions, geometry: Geometry) -> Result<Self, Error> {
        if geometry.width < PATCH_SIZE || geometry.height < PATCH_SIZE {
            let message = format!(
                "frame dimensions {}x{} must be at least {PATCH_SIZE}x{PATCH_SIZE}",
                geometry.width, geometry.height,
            );
            return Err(Error::InvalidGeometry(message));
        }

        geometry.validate()?;

        let params = resolve_params(&options, geometry.channels)?;
        let inner = Nl4dDenoiser::with_output_format(
            client,
            params,
            geometry.width,
            geometry.height,
            OutputFormat::F32,
        );
        let inner = inner.map_err(Error::InvalidOptions)?;

        Ok(Self {
            inner,
            client: client.clone(),
            geometry,
            temporal_radius: options.temporal_radius,
            pending: VecDeque::new(),
            finishing: false,
            pushes: 0,
            poisoned: false,
        })
    }

    /// Takes the grain chunks measured since the last call.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error> {
        self.check_usable()?;

        let drained = self.inner.drain_grain_chunks();
        let drained = drained.map_err(anyhow::Error::from);

        self.guard(drained)
    }

    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }

        Ok(())
    }

    fn check_nothing_pending(&self) -> Result<(), Error> {
        if !self.pending.is_empty() {
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

    fn restart_stream(&mut self) {
        self.inner.reset_stream();
        self.finishing = false;
        self.pushes = 0;
    }

    fn write_out(&self, frame: &Handle, planes: &[DevicePlane<'_>]) {
        let channels = self.geometry.channels;
        let source = EgressSource {
            frame,
            pixels: self.geometry.pixels(),
            channels: channels.count(),
            stored_ch: channels.storage_count(),
        };
        let placeholder = self.inner.placeholder();

        egress(&self.client, source, planes, self.geometry.output, placeholder);
    }

    #[cfg(test)]
    pub(crate) fn fail_through_guard_for_test(&mut self) -> Result<(), Error> {
        let failure = Err(anyhow::anyhow!("forced failure"));

        self.guard(failure)
    }
}

impl<R: Runtime> Engine for Nl4d<R> {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let pushed = self.inner.push_planes(planes, self.geometry.input);
        self.guard(pushed)?;

        let submitted = self.inner.submit_passes();
        let submitted = submitted.map_err(anyhow::Error::from);
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

        let Some(region) = self.pending.pop_front() else {
            return Err(Error::NothingToEmit);
        };

        let frame = self.inner.read_region(region);
        self.write_out(&frame, planes);

        if self.finishing && self.pending.is_empty() {
            self.restart_stream();
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;
        self.check_nothing_pending()?;

        let finished = self.inner.finish_passes();
        let finished = finished.map_err(anyhow::Error::from);
        let regions = self.guard(finished)?;

        let count = regions.len();
        self.pending.extend(regions);

        if count == 0 {
            self.restart_stream();
        } else {
            self.finishing = true;
        }

        Ok(count)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.poisoned = false;
        self.restart_stream();
    }

    fn window_span(&self) -> WindowSpan {
        let span = 2 * self.temporal_radius as usize;

        WindowSpan {
            behind: span,
            ahead: span,
            edges: EdgePadding::Shifted,
        }
    }

    fn max_held_frames(&self) -> usize {
        2 * self.temporal_radius as usize
    }
}
