use cubecl::prelude::*;

use super::planar::PlanarEngine;
use super::{ChromaLayout, EdgePadding, Engine, FramePass, Geometry, PassEngine, PassGeometry, WindowSpan};
use crate::error::Error;
use crate::nl4d::{Nl4d, Nl4dOptions};
use crate::nlmeans::{ChannelMode, Nlmeans, NlmeansAlgorithm};
use crate::tune::EngineKernels;

/// Which engine to build, and for what planes.
#[derive(Debug, Clone, Copy)]
pub enum PassSpec {
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

    /// Builds the engine on `client`, picking each launch site's candidate from `kernels`.
    #[cfg_attr(not(test), expect(dead_code, reason = "only the tests build a lone pass"))]
    pub(crate) fn build(
        &self,
        client: &Client,
        kernels: EngineKernels,
    ) -> Result<Box<dyn PassEngine>, Error> {
        let engine: Box<dyn PassEngine> = self.build_pass(client, kernels)?;
        Ok(engine)
    }

    /// Builds the engine like [PassSpec::build], as a [FramePass] that names the frame behind each emit.
    pub(crate) fn build_pass(
        &self,
        client: &Client,
        kernels: EngineKernels,
    ) -> Result<Box<dyn FramePass>, Error> {
        match *self {
            PassSpec::Nlmeans { algorithm, geometry } => {
                let engine = Nlmeans::with_kernels(client, algorithm, geometry, kernels)?;
                Ok(Box::new(engine))
            },
            PassSpec::Nl4d { options, geometry } => {
                let engine = Nl4d::with_kernels(client, options, geometry, kernels)?;
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

    /// Rejects a spec that cleans nothing, cleans chroma a frame lacks, or mixes window spans.
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
        }

        Ok(())
    }

    /// Builds the engine on `client`, picking each pass's launch site candidates from `kernels`.
    pub(crate) fn build(
        &self,
        client: &Client,
        mut kernels: impl FnMut(&PassSpec) -> EngineKernels,
    ) -> Result<Box<dyn Engine>, Error> {
        self.validate()?;

        let (luma, chroma) = self.passes();
        let luma_pass = luma
            .map(|pass| {
                let pass_kernels = kernels(&pass);
                pass.build_pass(client, pass_kernels)
            })
            .transpose()?;
        let chroma_pass = chroma
            .map(|pass| {
                let pass_kernels = kernels(&pass);
                pass.build_pass(client, pass_kernels)
            })
            .transpose()?;

        let geometry = self.geometry();
        let engine = PlanarEngine::new(client, geometry, luma_pass, chroma_pass);

        Ok(Box::new(engine))
    }
}
