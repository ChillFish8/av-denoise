use cubecl::prelude::*;

use super::{EdgePadding, PassEngine, PassGeometry, WindowSpan};
use crate::error::Error;
use crate::nl4d::{Nl4d, Nl4dOptions};
use crate::nlmeans::{Nlmeans, NlmeansAlgorithm};
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
    pub(crate) fn build(
        &self,
        client: &Client,
        kernels: EngineKernels,
    ) -> Result<Box<dyn PassEngine>, Error> {
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
