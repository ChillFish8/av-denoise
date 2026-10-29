use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::kernels::copy_words;
use crate::engine::{BLOCK_1D, DevicePlane, EngineSpec, Geometry, MAX_GRID_1D};
use crate::error::Error;

/// One frame on the GPU, as each engine of a group reads it.
#[derive(Debug, Clone)]
pub struct TuningFrame {
    planes: Vec<Vec<Handle>>,
}

impl TuningFrame {
    /// A frame whose handles the caller has uploaded, one list per engine in group order.
    pub fn new(planes: Vec<Vec<Handle>>) -> Self {
        Self { planes }
    }

    /// Copies `planes` on the device into new handles, one list of planes per engine in `specs`.
    ///
    /// Each handle holds the engine's input plane bytes, rounded up to whole words. A list that
    /// does not match its engine's planes is a [Error::PlaneMismatch].
    pub fn copy_from(
        client: &Client,
        specs: &[EngineSpec],
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

    pub(crate) fn planes_for(&self, engine: usize, geometry: Geometry) -> Vec<DevicePlane<'_>> {
        self.planes[engine]
            .iter()
            .map(|handle| DevicePlane::new(handle, geometry.width, geometry.height))
            .collect()
    }
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

/// Copies the first `words` 4-byte words of `source` into `destination` on the device.
pub(crate) fn copy_words_into(client: &Client, source: &Handle, destination: &Handle, words: usize) {
    let groups = (words as u32).div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = groups * BLOCK_1D;

    unsafe {
        copy_words::launch_unchecked(
            client,
            CubeCount::new_1d(groups),
            CubeDim::new_1d(BLOCK_1D),
            BufferArg::from_raw_parts(source.clone(), words),
            BufferArg::from_raw_parts(destination.clone(), words),
            words as u32,
            total_threads,
        );
    }
}
