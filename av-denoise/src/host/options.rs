use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    Geometry,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
};

use super::Depth;
use crate::backend::EngineSpec;

/// How a [HostDenoiser](crate::HostDenoiser) should be set up.
///
/// Build one with `DenoiserOptions::builder()`. Every field has a default, so only the parts you care
/// about need naming.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which channels of the frame to denoise.
    #[builder(default = ChannelMode::Yuv)]
    pub channel_mode: ChannelMode,
    /// Whether to clean each frame on its own or across a temporal window.
    ///
    /// This wins over any mode or temporal radius set inside `algorithm`.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Which algorithm to run, along with the settings only that algorithm reads.
    #[builder(default)]
    pub algorithm: Algorithm,
    /// The bit depth of the wire bytes going in and coming out.
    #[builder(default = Depth::Eight)]
    pub depth: Depth,
}

impl DenoiserOptions {
    /// The temporal radius `mode` asks for.
    pub(crate) fn temporal_radius(&self) -> u32 {
        match self.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        }
    }
}

/// Which denoising algorithm to run.
///
/// Each variant carries its own settings, so a knob one algorithm has no use for cannot be set on it.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Algorithm {
    /// The fast NLMeans path, with fixed weighting and no noise measurement.
    Nlmeans(NlmeansOptions),
    /// NLMeans with its weighting matched to the measured noise level.
    NlmeansHq(NlmeansHqOptions),
    /// Groups 8x8 patches across the motion-compensated temporal window.
    Nl4d(Nl4dOptions),
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Nlmeans(NlmeansOptions::default())
    }
}

impl Algorithm {
    /// The engine to build for `options`, at `width` by `height`.
    ///
    /// `options.mode` is copied into the algorithm's own options, so it wins over whatever they hold.
    pub(crate) fn engine_spec(&self, options: &DenoiserOptions, width: u32, height: u32) -> EngineSpec {
        let format = options.depth.sample_format();
        let geometry = Geometry {
            width,
            height,
            channels: options.channel_mode,
            input: format,
            output: format,
        };

        match *self {
            Algorithm::Nlmeans(nlm) => {
                let nlm = NlmeansOptions {
                    mode: options.mode,
                    ..nlm
                };
                let algorithm = NlmeansAlgorithm::Fast(nlm);

                EngineSpec::Nlmeans { algorithm, geometry }
            },
            Algorithm::NlmeansHq(hq) => {
                let nlm = NlmeansOptions {
                    mode: options.mode,
                    ..hq.nlm
                };
                let hq = NlmeansHqOptions { nlm, ..hq };
                let algorithm = NlmeansAlgorithm::Hq(hq);

                EngineSpec::Nlmeans { algorithm, geometry }
            },
            Algorithm::Nl4d(nl4d) => {
                let nl4d = Nl4dOptions {
                    temporal_radius: options.temporal_radius(),
                    ..nl4d
                };

                EngineSpec::Nl4d {
                    options: nl4d,
                    geometry,
                }
            },
        }
    }
}
