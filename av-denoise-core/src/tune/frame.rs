use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{DevicePlane, PassGeometry, PassSpec, copy_words_into};
use crate::error::Error;

/// One frame on the GPU, as each engine of a group reads it.
#[derive(Debug, Clone)]
pub struct TuningFrame {
    planes: Vec<Vec<Handle>>,
}

impl TuningFrame {
    /// A frame whose handles the caller has uploaded, one list per engine in group order.
    ///
    /// Each list holds one handle per channel of its engine, and each handle holds at least the
    /// engine's input plane bytes. [tune_frames](crate::tune::tune_frames) skips a frame set that
    /// breaks this.
    pub fn new(planes: Vec<Vec<Handle>>) -> Self {
        Self { planes }
    }

    /// Copies `planes` on the device into new handles, one list of planes per engine in `specs`.
    ///
    /// Each handle holds the engine's input plane bytes, rounded up to whole words. A list that
    /// does not match its engine's planes is a [Error::PlaneMismatch].
    pub fn copy_from(
        client: &Client,
        specs: &[PassSpec],
        planes: &[&[DevicePlane<'_>]],
    ) -> Result<Self, Error> {
        check_engine_count(specs, planes)?;

        let mut copied = Vec::with_capacity(specs.len());

        for (spec, engine_planes) in specs.iter().zip(planes) {
            let geometry = spec.geometry();
            geometry.check_planes(engine_planes, geometry.input)?;

            let bytes = geometry.input.plane_bytes(geometry.pixels()) as usize;
            let handles: Vec<Handle> = engine_planes
                .iter()
                .map(|plane| copy_plane(client, plane, bytes))
                .collect();

            copied.push(handles);
        }

        Ok(Self::new(copied))
    }

    pub(crate) fn planes_for(&self, engine: usize, geometry: PassGeometry) -> Vec<DevicePlane<'_>> {
        self.planes[engine]
            .iter()
            .map(|handle| DevicePlane::new(handle, geometry.width, geometry.height))
            .collect()
    }
}

/// Checks every frame holds the planes each engine in `specs` reads, with handles large enough.
pub(crate) fn check_frames(specs: &[PassSpec], frames: &[TuningFrame]) -> Result<(), Error> {
    for frame in frames {
        let expected = specs.len();
        let given = frame.planes.len();
        if expected != given {
            let message = format!("expected a tuning frame for {expected} engines, got {given}");
            return Err(Error::PlaneMismatch(message));
        }

        for (index, spec) in specs.iter().enumerate() {
            let geometry = spec.geometry();
            let planes = frame.planes_for(index, geometry);
            geometry.check_planes(&planes, geometry.input)?;
        }
    }

    Ok(())
}

/// Checks there is one list of planes per engine in `specs`.
pub(crate) fn check_engine_count(specs: &[PassSpec], planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
    let expected = specs.len();
    let given = planes.len();
    if expected == given {
        return Ok(());
    }

    let message = format!("expected planes for {expected} engines, got {given}");
    Err(Error::PlaneMismatch(message))
}

fn copy_plane(client: &Client, plane: &DevicePlane<'_>, bytes: usize) -> Handle {
    let words = bytes / 4;
    let destination = client.empty(bytes);
    copy_words_into(client, plane.handle(), &destination, words);

    destination
}
