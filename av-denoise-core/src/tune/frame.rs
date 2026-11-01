use cubecl::prelude::*;
use cubecl::server::Handle;

use super::unit::TuneUnit;
use crate::engine::{DevicePlane, EngineSpec, copy_words_into};
use crate::error::Error;

/// One frame on the GPU, as each engine of a group reads it.
#[derive(Debug, Clone)]
pub struct TuningFrame {
    planes: Vec<Vec<Handle>>,
}

impl TuningFrame {
    /// A frame whose handles the caller has uploaded, one list per engine in group order.
    ///
    /// Each list holds every plane of the frame its engine reads, luma first, and each handle holds
    /// at least that plane's input bytes. [tune_frames](crate::tune::tune_frames) skips a frame set
    /// that breaks this.
    pub fn new(planes: Vec<Vec<Handle>>) -> Self {
        Self { planes }
    }

    /// Copies `planes` on the device into new handles, one list of planes per engine in `specs`.
    ///
    /// Each handle holds its plane's input bytes, rounded up to whole words. A list that does not
    /// match its engine's planes is a [Error::PlaneMismatch].
    pub fn copy_from(
        client: &Client,
        specs: &[EngineSpec],
        planes: &[&[DevicePlane<'_>]],
    ) -> Result<Self, Error> {
        check_engine_count(specs, planes)?;

        let mut copied = Vec::with_capacity(specs.len());

        for (spec, engine_planes) in specs.iter().zip(planes) {
            let unit = TuneUnit(*spec);
            let input = unit.input();
            unit.check_planes(engine_planes, input)?;

            let handles: Vec<Handle> = engine_planes
                .iter()
                .enumerate()
                .map(|(index, plane)| {
                    let bytes = unit.plane_bytes(index, input);
                    copy_plane(client, plane, bytes)
                })
                .collect();

            copied.push(handles);
        }

        Ok(Self::new(copied))
    }

    /// The planes engine `engine` reads, each at the size `unit` gives it.
    pub(crate) fn planes_for(&self, engine: usize, unit: &TuneUnit) -> Vec<DevicePlane<'_>> {
        self.planes[engine]
            .iter()
            .enumerate()
            .map(|(index, handle)| {
                let (width, height) = unit.plane_dims(index);
                DevicePlane::new(handle, width, height)
            })
            .collect()
    }
}

/// Checks every frame holds the planes each engine of `units` reads, with handles large enough.
pub(crate) fn check_frames(units: &[TuneUnit], frames: &[TuningFrame]) -> Result<(), Error> {
    for frame in frames {
        let expected = units.len();
        let given = frame.planes.len();
        if expected != given {
            let message = format!("expected a tuning frame for {expected} engines, got {given}");
            return Err(Error::PlaneMismatch(message));
        }

        for (index, unit) in units.iter().enumerate() {
            let planes = frame.planes_for(index, unit);
            let input = unit.input();
            unit.check_planes(&planes, input)?;
        }
    }

    Ok(())
}

/// Checks there is one list of planes per engine in `specs`.
pub(crate) fn check_engine_count(specs: &[EngineSpec], planes: &[&[DevicePlane<'_>]]) -> Result<(), Error> {
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
