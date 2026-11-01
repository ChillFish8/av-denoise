use cubecl::prelude::*;

#[cfg(test)]
use super::PassEngine;
use super::align::StorageAlign;
use super::planar::PlanarEngine;
use super::{ChromaLayout, EdgePadding, Engine, FramePass, Geometry, PassGeometry, WindowSpan};
use crate::error::Error;
#[cfg(test)]
use crate::motion::ModelRing;
use crate::motion::{
    LumaMotionRing,
    MotionCompensationMode,
    MotionCtx,
    PairModelProducer,
    RingRole,
    SharedMotion,
    producer_buffer_sizes,
};
use crate::nl4d::{Nl4d, Nl4dOptions};
use crate::nlmeans::denoiser::check_u32_indexable;
use crate::nlmeans::{ChannelMode, Nlmeans, NlmeansAlgorithm, resolve_params};
use crate::tune::EngineKernels;

/// Model ring slots beyond the frames a pass holds and its temporal window.
const MODEL_RING_MARGIN: u32 = 2;

/// Which engine to build, and for what planes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PassSpec {
    Nlmeans {
        algorithm: NlmeansAlgorithm,
        geometry: PassGeometry,
    },
    Nl4d {
        options: Nl4dOptions,
        geometry: PassGeometry,
    },
}

impl PassSpec {
    /// The most frames the engine this describes holds before it emits one.
    pub fn max_held_frames(&self) -> usize {
        match self {
            PassSpec::Nlmeans { algorithm, geometry } => {
                Nlmeans::held_frames_for(algorithm, geometry.channels)
            },
            PassSpec::Nl4d { options, geometry } => Nl4d::held_frames_for(options, geometry.channels),
        }
    }

    pub fn geometry(&self) -> PassGeometry {
        match self {
            PassSpec::Nlmeans { geometry, .. } | PassSpec::Nl4d { geometry, .. } => *geometry,
        }
    }

    /// The frames before and after a target frame the engine this describes reads.
    pub fn window_span(&self) -> WindowSpan {
        let held = self.max_held_frames();
        let edges = match self {
            PassSpec::Nlmeans { .. } => EdgePadding::Repeat,
            PassSpec::Nl4d { .. } => EdgePadding::Shifted,
        };

        WindowSpan {
            behind: held,
            ahead: held,
            edges,
        }
    }

    /// The motion search the pass runs and its temporal radius, when it compensates motion.
    pub(crate) fn motion(&self) -> Option<(MotionCompensationMode, u32)> {
        let (mode, radius) = match self {
            PassSpec::Nlmeans { algorithm, geometry } => {
                let params = resolve_params(algorithm, geometry.channels);
                (params.motion_compensation, params.temporal_radius)
            },
            PassSpec::Nl4d { options, .. } => (options.motion.into(), options.temporal_radius),
        };

        let active = mode.is_active() && radius > 0;
        active.then_some((mode, radius))
    }

    /// Builds the engine on `client`, picking each launch site's candidate from `kernels`.
    #[cfg(test)]
    pub(crate) fn build(
        &self,
        client: &Client,
        kernels: EngineKernels,
    ) -> Result<Box<dyn PassEngine>, Error> {
        let engine: Box<dyn PassEngine> = self.build_pass(client, kernels, None)?;
        Ok(engine)
    }

    /// Builds the engine as a [FramePass] that names the frame behind each emit.
    ///
    /// `motion` holds the camera models of the frames the pass sees, when they are fitted.
    pub(crate) fn build_pass(
        &self,
        client: &Client,
        kernels: EngineKernels,
        motion: Option<SharedMotion>,
    ) -> Result<Box<dyn FramePass>, Error> {
        match *self {
            PassSpec::Nlmeans { algorithm, geometry } => {
                let engine = Nlmeans::with_kernels(client, algorithm, geometry, kernels, motion)?;
                Ok(Box::new(engine))
            },
            PassSpec::Nl4d { options, geometry } => {
                let engine = Nl4d::with_kernels(client, options, geometry, kernels, motion)?;
                Ok(Box::new(engine))
            },
        }
    }
}

/// Which engine to build, and which planes of the frame it cleans.
///
/// A `Some` option cleans that plane group with those options. Luma is plane 0 and chroma is planes
/// 1 and 2. A plane group left `None` comes out unchanged.
#[derive(Debug, Clone, Copy)]
pub enum EngineSpec {
    Nlmeans {
        luma: Option<NlmeansAlgorithm>,
        chroma: Option<NlmeansAlgorithm>,
        geometry: Geometry,
    },
    Nl4d {
        luma: Option<Nl4dOptions>,
        chroma: Option<Nl4dOptions>,
        geometry: Geometry,
    },
}

impl EngineSpec {
    pub fn geometry(&self) -> Geometry {
        match self {
            EngineSpec::Nlmeans { geometry, .. } | EngineSpec::Nl4d { geometry, .. } => *geometry,
        }
    }

    /// The pass that cleans the luma plane and the pass that cleans the chroma planes.
    ///
    /// Each runs at its own plane size, luma at the frame size and chroma at the chroma size.
    pub(crate) fn passes(&self) -> (Option<PassSpec>, Option<PassSpec>) {
        let geometry = self.geometry();
        let luma_geometry = PassGeometry {
            width: geometry.width,
            height: geometry.height,
            channels: ChannelMode::Luma,
            input: geometry.input,
            output: geometry.output,
        };
        let chroma_dims = geometry.chroma.chroma_dims(geometry.width, geometry.height);
        let chroma_geometry = chroma_dims.map(|(width, height)| PassGeometry {
            width,
            height,
            channels: ChannelMode::Chroma,
            input: geometry.input,
            output: geometry.output,
        });

        match *self {
            EngineSpec::Nlmeans { luma, chroma, .. } => {
                let luma_pass = luma.map(|algorithm| PassSpec::Nlmeans {
                    algorithm,
                    geometry: luma_geometry,
                });
                let chroma_pass = chroma
                    .zip(chroma_geometry)
                    .map(|(algorithm, geometry)| PassSpec::Nlmeans { algorithm, geometry });

                (luma_pass, chroma_pass)
            },
            EngineSpec::Nl4d { luma, chroma, .. } => {
                let luma_pass = luma.map(|options| PassSpec::Nl4d {
                    options,
                    geometry: luma_geometry,
                });
                let chroma_pass = chroma
                    .zip(chroma_geometry)
                    .map(|(options, geometry)| PassSpec::Nl4d { options, geometry });

                (luma_pass, chroma_pass)
            },
        }
    }

    /// Every pass the spec runs, luma before chroma.
    pub(crate) fn pass_specs(&self) -> Vec<PassSpec> {
        let (luma, chroma) = self.passes();
        luma.into_iter().chain(chroma).collect()
    }

    pub(crate) fn is_nl4d(&self) -> bool {
        matches!(self, EngineSpec::Nl4d { .. })
    }

    /// The first pass the spec runs, luma before chroma.
    fn first_pass(&self) -> Option<PassSpec> {
        let (luma, chroma) = self.passes();
        luma.or(chroma)
    }

    /// The most frames the engine this describes holds before it emits one.
    ///
    /// It is 0 for a spec that cleans nothing.
    pub fn max_held_frames(&self) -> usize {
        let pass = self.first_pass();
        pass.map_or(0, |pass| pass.max_held_frames())
    }

    /// The frames before and after a target frame the engine this describes reads.
    pub fn window_span(&self) -> WindowSpan {
        let held = self.max_held_frames();
        let edges = match self {
            EngineSpec::Nlmeans { .. } => EdgePadding::Repeat,
            EngineSpec::Nl4d { .. } => EdgePadding::Shifted,
        };

        WindowSpan {
            behind: held,
            ahead: held,
            edges,
        }
    }

    /// Rejects a spec that cleans nothing, cleans chroma a frame lacks, mixes window spans, or gives
    /// the passes different motion settings.
    ///
    /// The chroma pass takes its window motion from the luma pass, so both need the same motion
    /// search and temporal radius.
    ///
    /// A plane group passed through is copied word for word, so it also needs matching input and
    /// output formats.
    pub fn validate(&self) -> Result<(), Error> {
        let geometry = self.geometry();
        geometry.validate()?;

        let (has_luma, has_chroma) = match self {
            EngineSpec::Nlmeans { luma, chroma, .. } => (luma.is_some(), chroma.is_some()),
            EngineSpec::Nl4d { luma, chroma, .. } => (luma.is_some(), chroma.is_some()),
        };

        if !has_luma && !has_chroma {
            let message = "a spec must clean luma, chroma or both".to_string();
            return Err(Error::InvalidOptions(message));
        }

        let gray = geometry.chroma == ChromaLayout::Gray;
        if has_chroma && gray {
            let message = "a gray frame has no chroma planes to clean".to_string();
            return Err(Error::InvalidGeometry(message));
        }

        let passes_through = !gray && has_luma != has_chroma;
        if passes_through && geometry.input != geometry.output {
            let message = format!(
                "an uncleaned plane needs the same input and output format, got {:?} and {:?}",
                geometry.input, geometry.output,
            );
            return Err(Error::InvalidGeometry(message));
        }

        let (luma, chroma) = self.passes();
        if let (Some(luma), Some(chroma)) = (luma, chroma) {
            let luma_span = luma.window_span();
            let chroma_span = chroma.window_span();
            if luma_span != chroma_span {
                let message =
                    format!("the luma pass reads {luma_span:?} but the chroma pass reads {chroma_span:?}");
                return Err(Error::InvalidOptions(message));
            }

            // The chroma pass takes its window motion from the luma pass.
            let luma_motion = luma.motion();
            let chroma_motion = chroma.motion();
            if luma_motion != chroma_motion {
                let message = format!(
                    "the chroma pass takes its motion from the luma pass, so both need the same motion \
                     search and temporal radius, got {luma_motion:?} and {chroma_motion:?}"
                );
                return Err(Error::InvalidOptions(message));
            }
        }

        Ok(())
    }

    /// The ring the luma pass shares its window motion with the chroma pass through.
    ///
    /// It exists only when the spec runs both passes and both compensate motion.
    pub(crate) fn luma_motion_ring(&self, client: &Client) -> Option<LumaMotionRing> {
        let (luma, chroma) = self.passes();
        let luma_motion = luma.and_then(|pass| pass.motion());
        let chroma_pass = chroma?;
        chroma_pass.motion()?;
        let (mode, radius) = luma_motion?;

        let chroma_geometry = chroma_pass.geometry();
        let align = StorageAlign::from_client(client);
        let chroma_ctx = MotionCtx::new(mode, chroma_geometry.width, chroma_geometry.height, align)?;
        let subsampling = self.geometry().chroma.subsampling();
        let ring = LumaMotionRing::new(client, &chroma_ctx, radius, subsampling);

        Some(ring)
    }

    /// Builds the engine on `client`, picking each pass's launch site candidates from `kernels`.
    ///
    /// A stream of gray frames logs a warning once, as it starts.
    pub(crate) fn build(
        &self,
        client: &Client,
        kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<Box<dyn Engine>, Error> {
        self.build_planar(client, kernels, true)
    }

    /// [Self::build] without the stream warnings, for engines that only time kernels.
    pub(crate) fn build_quiet(
        &self,
        client: &Client,
        kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<Box<dyn Engine>, Error> {
        self.build_planar(client, kernels, false)
    }

    /// [Self::build] that also hands back the camera model ring, when the engine fits one.
    #[cfg(test)]
    pub(crate) fn build_with_models(
        &self,
        client: &Client,
        kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<(Box<dyn Engine>, Option<ModelRing>), Error> {
        let engine = self.build_planar_engine(client, kernels, false)?;
        let models = engine.models().cloned();

        Ok((Box::new(engine), models))
    }

    /// The producer of the frame pair camera models, when any pass compensates motion.
    ///
    /// It fits on the luma plane with the luma pass's search, or the chroma pass's when only that
    /// one searches. Its ring covers the frames the passes hold and their temporal window.
    pub(crate) fn pair_model_producer(&self, client: &Client) -> Result<Option<PairModelProducer>, Error> {
        let (luma, chroma) = self.passes();
        let luma_motion = luma.and_then(|pass| pass.motion());
        let chroma_motion = chroma.and_then(|pass| pass.motion());
        let Some((mode, radius)) = luma_motion.or(chroma_motion) else {
            return Ok(None);
        };

        let validated = mode.validate();
        validated.map_err(|error| {
            let message = error.to_string();
            Error::InvalidOptions(message)
        })?;

        let geometry = self.geometry();
        let held = self.max_held_frames() as u32;
        let window = 2 * radius + 1;
        let depth = held + window + MODEL_RING_MARGIN;

        let sizes = producer_buffer_sizes(client, geometry.width, geometry.height, &mode, depth);
        let indexable = check_u32_indexable(&sizes);
        indexable.map_err(Error::InvalidGeometry)?;

        let producer = PairModelProducer::new(
            client,
            geometry.width,
            geometry.height,
            &mode,
            geometry.input,
            depth,
        );
        Ok(producer)
    }

    fn build_planar(
        &self,
        client: &Client,
        kernels: impl FnMut(&PassSpec) -> EngineKernels,
        warns: bool,
    ) -> Result<Box<dyn Engine>, Error> {
        let engine = self.build_planar_engine(client, kernels, warns)?;
        Ok(Box::new(engine))
    }

    fn build_planar_engine(
        &self,
        client: &Client,
        mut kernels: impl FnMut(&PassSpec) -> EngineKernels,
        warns: bool,
    ) -> Result<PlanarEngine, Error> {
        self.validate()?;

        let geometry = self.geometry();
        let producer = self.pair_model_producer(client)?;
        let ring = producer.as_ref().map(|producer| producer.ring());
        let (chroma_scale_x, chroma_scale_y) = geometry.chroma.chroma_scale();
        let luma_motion_ring = self.luma_motion_ring(client);
        let writer = luma_motion_ring.clone().map(RingRole::Writer);
        let reader = luma_motion_ring.clone().map(RingRole::Reader);
        let luma_motion = ring.map(|ring| SharedMotion {
            ring: ring.clone(),
            scale_x: 1.0,
            scale_y: 1.0,
            luma_motion: writer,
        });
        let chroma_motion = ring.map(|ring| SharedMotion {
            ring: ring.clone(),
            scale_x: chroma_scale_x,
            scale_y: chroma_scale_y,
            luma_motion: reader,
        });

        let (luma, chroma) = self.passes();
        let luma_pass = luma
            .map(|pass| {
                let pass_kernels = kernels(&pass);
                pass.build_pass(client, pass_kernels, luma_motion)
            })
            .transpose()?;
        let chroma_pass = chroma
            .map(|pass| {
                let pass_kernels = kernels(&pass);
                pass.build_pass(client, pass_kernels, chroma_motion)
            })
            .transpose()?;

        let engine = PlanarEngine::new(
            client,
            geometry,
            luma_pass,
            chroma_pass,
            producer,
            luma_motion_ring,
            warns,
        );

        Ok(engine)
    }
}
