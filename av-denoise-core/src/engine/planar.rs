use std::fmt::Debug;

use cubecl::prelude::*;

use super::passthrough::PassthroughRing;
use super::{ChromaLayout, DevicePlane, Engine, FramePass, Geometry, PassTimings, WindowSpan};
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;

/// Splits a frame's planes into the luma plane and the chroma planes.
fn split_planes<'p, 'a>(planes: &'p [DevicePlane<'a>]) -> (&'p [DevicePlane<'a>], &'p [DevicePlane<'a>]) {
    let luma_len = planes.len().min(1);
    planes.split_at(luma_len)
}

/// One engine over every plane of a frame, running a luma pass, a chroma pass, or both.
///
/// A plane group without a pass is held in a [PassthroughRing] under the stream frame it came in
/// with, and written out when the cleaned pass emits that same frame.
pub(crate) struct PlanarEngine {
    geometry: Geometry,
    luma: Option<Box<dyn FramePass>>,
    chroma: Option<Box<dyn FramePass>>,
    passthrough: Option<PassthroughRing>,
    poisoned: bool,
}

impl PlanarEngine {
    /// Builds the engine around its passes, which the spec has validated.
    pub(crate) fn new(
        client: &Client,
        geometry: Geometry,
        luma: Option<Box<dyn FramePass>>,
        chroma: Option<Box<dyn FramePass>>,
    ) -> Self {
        let gray = geometry.chroma == ChromaLayout::Gray;
        if gray {
            tracing::warn!("the source has no colour planes, so denoising may be less effective");
        }

        let mut engine = Self {
            geometry,
            luma,
            chroma,
            passthrough: None,
            poisoned: false,
        };

        let passes_through = !gray && engine.luma.is_some() != engine.chroma.is_some();
        if passes_through {
            let ring = engine.passthrough_ring(client);
            engine.passthrough = Some(ring);
        }

        engine
    }

    /// A ring for the uncleaned plane group, one slot longer than the cleaned pass holds frames.
    fn passthrough_ring(&self, client: &Client) -> PassthroughRing {
        let slots = self.lead().max_held_frames() + 1;
        let (index, planes) = if self.luma.is_some() { (1, 2) } else { (0, 1) };
        let (width, height) = self.geometry.plane_dims(index);
        let pixels = u64::from(width) * u64::from(height);
        let bytes = self.geometry.input.plane_bytes(pixels);

        PassthroughRing::new(client, slots, planes, bytes)
    }

    /// The first present pass, luma before chroma.
    fn lead(&self) -> &dyn FramePass {
        let pass = self.luma.as_deref().or(self.chroma.as_deref());
        pass.expect("a validated spec has at least one pass")
    }

    /// The planes of the group no pass cleans.
    fn uncleaned<'p, 'a>(&self, planes: &'p [DevicePlane<'a>]) -> &'p [DevicePlane<'a>] {
        let (luma_planes, chroma_planes) = split_planes(planes);
        if self.luma.is_some() {
            chroma_planes
        } else {
            luma_planes
        }
    }

    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }

        Ok(())
    }

    /// Runs `call` on each pass in luma, chroma order, handing each its own planes.
    ///
    /// The passes move in lockstep, so they must agree on the result. A failure after one pass
    /// already moved, or a disagreement, poisons the engine.
    fn forward<T, F>(&mut self, planes: &[DevicePlane<'_>], mut call: F) -> Result<T, Error>
    where
        T: PartialEq + Debug,
        F: FnMut(&mut dyn FramePass, &[DevicePlane<'_>]) -> Result<T, Error>,
    {
        let (luma_planes, chroma_planes) = split_planes(planes);
        let passes = [
            (self.luma.as_deref_mut(), luma_planes),
            (self.chroma.as_deref_mut(), chroma_planes),
        ];

        let mut agreed: Option<T> = None;
        let mut failure = None;
        for (pass, pass_planes) in passes {
            let Some(pass) = pass else {
                continue;
            };

            let value = match call(pass, pass_planes) {
                Ok(value) => value,
                Err(error) => {
                    failure = Some(error);
                    break;
                },
            };

            match agreed.as_ref() {
                Some(first) if *first != value => {
                    let message = format!("the luma pass returned {first:?} but the chroma pass {value:?}");
                    let error = anyhow::anyhow!(message);
                    failure = Some(Error::Gpu(error));
                    break;
                },
                _ => agreed = Some(value),
            }
        }

        if let Some(error) = failure {
            let partial = agreed.is_some();
            if partial {
                self.poisoned = true;
            }

            return Err(error);
        }

        let value = agreed.expect("a validated spec has at least one pass");
        Ok(value)
    }

    /// Copies the uncleaned planes of stream frame `frame` into the ring.
    fn store_uncleaned(&self, frame: u32, planes: &[DevicePlane<'_>]) {
        let Some(ring) = self.passthrough.as_ref() else {
            return;
        };

        let uncleaned = self.uncleaned(planes);
        ring.store(frame, uncleaned);
    }
}

impl Engine for PlanarEngine {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let frame = self.lead().next_frame();
        let ready = self.forward(planes, |pass, pass_planes| pass.push(pass_planes))?;
        self.store_uncleaned(frame, planes);

        Ok(ready)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.input)?;

        let frame = self.lead().next_frame();
        self.forward(planes, |pass, pass_planes| pass.push_context(pass_planes))?;
        self.store_uncleaned(frame, planes);

        Ok(())
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.output)?;

        self.forward(planes, |pass, pass_planes| pass.emit_into(pass_planes))?;

        let Some(ring) = self.passthrough.as_ref() else {
            return Ok(());
        };

        let lead = self.lead();
        let frame = lead.emitted_frame();
        let frame = frame.expect("a pass names the frame of every emit");
        let newest = lead.next_frame();
        let behind = newest.saturating_sub(frame) as usize;
        debug_assert!(
            behind <= ring.slots(),
            "frame {frame} left the ring before its emit"
        );

        let uncleaned = self.uncleaned(planes);
        ring.load(frame, uncleaned);

        Ok(())
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.check_usable()?;

        self.forward(&[], |pass, _| pass.finish())
    }

    fn reset(&mut self) {
        let passes = [self.luma.as_deref_mut(), self.chroma.as_deref_mut()];
        for pass in passes.into_iter().flatten() {
            pass.reset();
        }

        self.poisoned = false;
    }

    fn window_span(&self) -> WindowSpan {
        self.lead().window_span()
    }

    fn max_held_frames(&self) -> usize {
        self.lead().max_held_frames()
    }

    fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error> {
        self.check_usable()?;

        let Some(luma) = self.luma.as_deref_mut() else {
            return Ok(Vec::new());
        };

        luma.drain_grain_chunks()
    }

    fn timing(&mut self) -> PassTimings {
        let luma = self.luma.as_deref_mut().map(|pass| pass.timing());
        let chroma = self.chroma.as_deref_mut().map(|pass| pass.timing());

        PassTimings { luma, chroma }
    }

    fn sample_device_time(&mut self, every: u32) {
        let passes = [self.luma.as_deref_mut(), self.chroma.as_deref_mut()];
        for pass in passes.into_iter().flatten() {
            pass.sample_device_time(every);
        }
    }
}
