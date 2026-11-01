use std::fmt::Debug;
use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::passthrough::PassthroughRing;
use super::{ChromaLayout, DevicePlane, Engine, FramePass, Geometry, PassTimings, PlaneSource, WindowSpan};
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;

/// The warning a stream of gray frames logs once, as it starts.
pub(crate) const GRAY_WARNING: &str = "the source has no colour planes, so denoising may be less effective";

/// The warning a stream of `geometry` frames logs as it starts, if any.
pub(crate) fn stream_warning(geometry: &Geometry) -> Option<&'static str> {
    let gray = geometry.chroma == ChromaLayout::Gray;
    gray.then_some(GRAY_WARNING)
}

/// The luma plane or the chroma planes of a frame.
#[derive(Debug, Clone, Copy)]
enum PlaneGroup {
    Luma,
    Chroma,
}

impl PlaneGroup {
    /// The indices of the group's planes in a frame of `layout`, empty for chroma on gray.
    fn indices(self, layout: ChromaLayout) -> Range<usize> {
        match self {
            PlaneGroup::Luma => 0..1,
            PlaneGroup::Chroma => 1..layout.plane_count(),
        }
    }
}

/// A frame's planes uploaded from a [PlaneSource] one group at a time, as the passes ask for them.
struct StagedFrame<'s> {
    source: &'s mut dyn PlaneSource,
    geometry: Geometry,
    luma: Option<Vec<Handle>>,
    chroma: Option<Vec<Handle>>,
}

impl<'s> StagedFrame<'s> {
    fn new(source: &'s mut dyn PlaneSource, geometry: Geometry) -> Self {
        Self {
            source,
            geometry,
            luma: None,
            chroma: None,
        }
    }

    /// The handles of `group`, uploaded on first use and checked against the geometry.
    fn handles(&mut self, group: PlaneGroup) -> Result<&[Handle], Error> {
        let indices = group.indices(self.geometry.chroma);
        let slot = match group {
            PlaneGroup::Luma => &mut self.luma,
            PlaneGroup::Chroma => &mut self.chroma,
        };

        if slot.is_none() {
            let uploaded = self.source.upload(indices.clone());
            if uploaded.len() != indices.len() {
                let message = format!("expected {} planes, got {}", indices.len(), uploaded.len());
                return Err(Error::PlaneMismatch(message));
            }

            let planes = self.geometry.device_planes(indices.start, &uploaded);
            let input = self.geometry.input;
            self.geometry.check_plane_sizes(&planes, indices.start, input)?;

            *slot = Some(uploaded);
        }

        let handles = slot.as_deref().expect("the group was just uploaded");
        Ok(handles)
    }
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
    /// Logged once at the start of each stream.
    warning: Option<&'static str>,
    /// Whether the current stream has taken a frame yet.
    stream_started: bool,
    poisoned: bool,
}

impl PlanarEngine {
    /// Builds the engine around its passes, which the spec has validated.
    ///
    /// With `warns` set, a stream that needs a [stream_warning] logs it once, as it starts.
    pub(crate) fn new(
        client: &Client,
        geometry: Geometry,
        luma: Option<Box<dyn FramePass>>,
        chroma: Option<Box<dyn FramePass>>,
        warns: bool,
    ) -> Self {
        let gray = geometry.chroma == ChromaLayout::Gray;
        let warning = if warns { stream_warning(&geometry) } else { None };

        let mut engine = Self {
            geometry,
            luma,
            chroma,
            passthrough: None,
            warning,
            stream_started: false,
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
        let luma = self.luma.as_deref();
        let chroma = self.chroma.as_deref();
        let pass = luma.or(chroma);
        pass.expect("a validated spec has at least one pass")
    }

    /// The plane group no pass cleans.
    fn uncleaned_group(&self) -> PlaneGroup {
        if self.luma.is_some() {
            PlaneGroup::Chroma
        } else {
            PlaneGroup::Luma
        }
    }

    /// The planes of the group no pass cleans.
    fn uncleaned<'p, 'a>(&self, planes: &'p [DevicePlane<'a>]) -> &'p [DevicePlane<'a>] {
        let group = self.uncleaned_group();
        let indices = group.indices(self.geometry.chroma);
        &planes[indices]
    }

    /// Logs the stream's warning when this frame starts a new stream.
    fn start_stream(&mut self) {
        if self.stream_started {
            return;
        }

        self.stream_started = true;
        if let Some(warning) = self.warning {
            tracing::warn!("{warning}");
        }
    }

    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }

        Ok(())
    }

    /// Runs `call` on each pass in luma, chroma order, naming the plane group the pass cleans.
    ///
    /// The passes move in lockstep, so they must agree on the result. A failure after one pass
    /// already moved, or a disagreement, poisons the engine.
    fn forward<T, F>(&mut self, mut call: F) -> Result<T, Error>
    where
        T: PartialEq + Debug,
        F: FnMut(&mut dyn FramePass, PlaneGroup) -> Result<T, Error>,
    {
        let passes = [
            (self.luma.as_deref_mut(), PlaneGroup::Luma),
            (self.chroma.as_deref_mut(), PlaneGroup::Chroma),
        ];

        let mut agreed: Option<T> = None;
        let mut failure = None;
        for (pass, group) in passes {
            let Some(pass) = pass else {
                continue;
            };

            let value = match call(pass, group) {
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

    /// Runs `call` on each pass with its own slice of `planes`.
    fn forward_planes<T, F>(&mut self, planes: &[DevicePlane<'_>], mut call: F) -> Result<T, Error>
    where
        T: PartialEq + Debug,
        F: FnMut(&mut dyn FramePass, &[DevicePlane<'_>]) -> Result<T, Error>,
    {
        let layout = self.geometry.chroma;

        self.forward(|pass, group| {
            let indices = group.indices(layout);
            call(pass, &planes[indices])
        })
    }

    /// Copies the uncleaned planes of stream frame `frame` into the ring.
    fn store_uncleaned(&self, frame: u32, planes: &[DevicePlane<'_>]) {
        let Some(ring) = self.passthrough.as_ref() else {
            return;
        };

        let uncleaned = self.uncleaned(planes);
        ring.store(frame, uncleaned);
    }

    /// Uploads the uncleaned group of stream frame `frame` from `staged` and copies it into the ring.
    fn store_staged(&self, frame: u32, staged: &mut StagedFrame<'_>) -> Result<(), Error> {
        let Some(ring) = self.passthrough.as_ref() else {
            return Ok(());
        };

        let group = self.uncleaned_group();
        let first = group.indices(self.geometry.chroma).start;
        let handles = staged.handles(group)?;
        let planes = self.geometry.device_planes(first, handles);
        ring.store(frame, &planes);

        Ok(())
    }
}

impl Engine for PlanarEngine {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.input)?;
        self.start_stream();

        let frame = self.lead().next_frame();
        let ready = self.forward_planes(planes, |pass, pass_planes| pass.push(pass_planes))?;
        self.store_uncleaned(frame, planes);

        Ok(ready)
    }

    fn push_from(&mut self, source: &mut dyn PlaneSource) -> Result<usize, Error> {
        self.check_usable()?;
        self.start_stream();

        let frame = self.lead().next_frame();
        let geometry = self.geometry;
        let mut staged = StagedFrame::new(source, geometry);
        let ready = self.forward(|pass, group| {
            let first = group.indices(geometry.chroma).start;
            let handles = staged.handles(group)?;
            let planes = geometry.device_planes(first, handles);
            pass.push(&planes)
        })?;

        if let Err(error) = self.store_staged(frame, &mut staged) {
            self.poisoned = true;
            return Err(error);
        }

        Ok(ready)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.input)?;
        self.start_stream();

        let frame = self.lead().next_frame();
        self.forward_planes(planes, |pass, pass_planes| pass.push_context(pass_planes))?;
        self.store_uncleaned(frame, planes);

        Ok(())
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.check_usable()?;
        self.geometry.check_planes(planes, self.geometry.output)?;

        self.forward_planes(planes, |pass, pass_planes| pass.emit_into(pass_planes))?;

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
        self.stream_started = false;

        self.forward(|pass, _| pass.finish())
    }

    fn reset(&mut self) {
        let passes = [self.luma.as_deref_mut(), self.chroma.as_deref_mut()];
        for pass in passes.into_iter().flatten() {
            pass.reset();
        }

        self.stream_started = false;
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
