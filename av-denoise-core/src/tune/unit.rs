use cubecl::prelude::*;

use super::profile_key::profile_key;
use super::selector::EngineKernels;
use crate::engine::{
    DevicePlane,
    Engine,
    EngineSpec,
    EngineTiming,
    PassEngine,
    PassSpec,
    PassTimings,
    SampleFormat,
    WindowSpan,
};
use crate::error::Error;
use crate::nl4d::grain::GrainChunk;
use crate::nlmeans::ChannelMode;

/// What one engine of a tuning set is built from, every plane of a frame or a single pass.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TuneUnit {
    Frame(EngineSpec),
    Pass(PassSpec),
}

impl TuneUnit {
    pub(crate) fn frames(specs: &[EngineSpec]) -> Vec<TuneUnit> {
        specs.iter().copied().map(TuneUnit::Frame).collect()
    }

    pub(crate) fn passes_of(specs: &[PassSpec]) -> Vec<TuneUnit> {
        specs.iter().copied().map(TuneUnit::Pass).collect()
    }

    /// The passes the engine runs, luma before chroma.
    pub(crate) fn passes(&self) -> Vec<PassSpec> {
        match self {
            TuneUnit::Frame(spec) => {
                let (luma, chroma) = spec.passes();
                luma.into_iter().chain(chroma).collect()
            },
            TuneUnit::Pass(pass) => vec![*pass],
        }
    }

    pub(crate) fn is_nl4d(&self) -> bool {
        matches!(
            self,
            TuneUnit::Frame(EngineSpec::Nl4d { .. }) | TuneUnit::Pass(PassSpec::Nl4d { .. })
        )
    }

    /// How many planes each push and emit takes.
    pub(crate) fn plane_count(&self) -> usize {
        match self {
            TuneUnit::Frame(spec) => spec.geometry().chroma.plane_count(),
            TuneUnit::Pass(pass) => pass.geometry().channels.count() as usize,
        }
    }

    /// The width and height of plane `index`.
    pub(crate) fn plane_dims(&self, index: usize) -> (u32, u32) {
        match self {
            TuneUnit::Frame(spec) => spec.geometry().plane_dims(index),
            TuneUnit::Pass(pass) => {
                let geometry = pass.geometry();
                (geometry.width, geometry.height)
            },
        }
    }

    /// The bytes plane `index` takes in `format`.
    pub(crate) fn plane_bytes(&self, index: usize, format: SampleFormat) -> usize {
        let (width, height) = self.plane_dims(index);
        let pixels = u64::from(width) * u64::from(height);

        format.plane_bytes(pixels) as usize
    }

    pub(crate) fn input(&self) -> SampleFormat {
        match self {
            TuneUnit::Frame(spec) => spec.geometry().input,
            TuneUnit::Pass(pass) => pass.geometry().input,
        }
    }

    pub(crate) fn output(&self) -> SampleFormat {
        match self {
            TuneUnit::Frame(spec) => spec.geometry().output,
            TuneUnit::Pass(pass) => pass.geometry().output,
        }
    }

    pub(crate) fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        match self {
            TuneUnit::Frame(spec) => spec.geometry().check_planes(planes, format),
            TuneUnit::Pass(pass) => pass.geometry().check_planes(planes, format),
        }
    }

    /// Builds the engine on `client`, picking each pass's launch site candidates from `kernels`.
    pub(crate) fn build(
        &self,
        client: &Client,
        mut kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<Box<dyn Engine>, Error> {
        match self {
            TuneUnit::Frame(spec) => spec.build(client, kernels),
            TuneUnit::Pass(pass) => {
                let pass_kernels = kernels(pass);
                let engine = pass.build(client, pass_kernels)?;
                let channels = pass.geometry().channels;

                Ok(Box::new(SinglePass { engine, channels }))
            },
        }
    }
}

/// The profile key of every pass of `units`, in unit order and luma before chroma.
pub(crate) fn unit_profile_keys(units: &[TuneUnit]) -> Vec<String> {
    units
        .iter()
        .flat_map(TuneUnit::passes)
        .map(|pass| profile_key(&pass))
        .collect()
}

/// A lone pass driven as an [Engine](crate::engine::Engine) over its own planes.
struct SinglePass {
    engine: Box<dyn PassEngine>,
    channels: ChannelMode,
}

impl Engine for SinglePass {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.engine.push(planes)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.engine.push_context(planes)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.engine.emit_into(planes)
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.engine.finish()
    }

    fn reset(&mut self) {
        self.engine.reset();
    }

    fn window_span(&self) -> WindowSpan {
        self.engine.window_span()
    }

    fn max_held_frames(&self) -> usize {
        self.engine.max_held_frames()
    }

    fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, Error> {
        self.engine.drain_grain_chunks()
    }

    fn timing(&mut self) -> PassTimings {
        let timing: Option<EngineTiming> = Some(self.engine.timing());

        match self.channels {
            ChannelMode::Chroma => PassTimings {
                luma: None,
                chroma: timing,
            },
            ChannelMode::Luma => PassTimings {
                luma: timing,
                chroma: None,
            },
        }
    }

    fn sample_device_time(&mut self, every: u32) {
        self.engine.sample_device_time(every);
    }
}
