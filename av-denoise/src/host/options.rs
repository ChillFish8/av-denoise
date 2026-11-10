use av_denoise_core::{
    ChromaLayout,
    ChromaSiting,
    DenoisingMode,
    EngineSpec,
    Geometry,
    Nl4dOptions,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
};

use super::Depth;
use crate::tune::TuningMode;

/// Which plane group an option is resolved for.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum PlaneGroup {
    Luma,
    Chroma,
}

/// Which planes a caller wants cleaned.
///
/// A plane that is not cleaned comes out exactly as it went in.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ChannelIntent {
    /// Denoise luma only. Chroma passes through.
    Luma,
    /// Denoise chroma only. Luma passes through.
    Chroma,
    /// Denoise luma and chroma, each with its own options.
    ///
    /// Chroma runs at the source's native subsampled resolution.
    LumaChroma,
}

impl ChannelIntent {
    pub(crate) fn cleans_luma(self) -> bool {
        matches!(self, ChannelIntent::Luma | ChannelIntent::LumaChroma)
    }

    pub(crate) fn cleans_chroma(self) -> bool {
        matches!(self, ChannelIntent::Chroma | ChannelIntent::LumaChroma)
    }
}

/// How a [HostDenoiser](crate::HostDenoiser) should be set up.
///
/// Build one with `DenoiserOptions::builder()`. Only `intent` and `chroma` are required. Every other
/// field has a default.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which planes of the frame to denoise.
    pub intent: ChannelIntent,
    /// How the frame stores its colour planes, which sets how many planes each frame has.
    pub chroma: ChromaLayout,
    /// Whether to clean each frame on its own or across a temporal window.
    ///
    /// This wins over any mode or temporal radius set inside `algorithm`.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Which algorithm to run, along with the settings only that algorithm reads.
    #[builder(default)]
    pub algorithm: Algorithm,
    /// Strength override for luma, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub luma_strength: Option<f32>,
    /// Strength override for chroma, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub chroma_strength: Option<f32>,
    /// Luma override for nl4d's `lambda_ht`, which wins over the algorithm's value.
    ///
    /// When this and the algorithm's `lambda_ht` are both unset, the engine picks a calibrated luma
    /// default.
    pub luma_lambda_ht: Option<f32>,
    /// Chroma override for nl4d's `lambda_ht`, which wins over the algorithm's value.
    ///
    /// When this and the algorithm's `lambda_ht` are both unset, the engine picks a calibrated chroma
    /// default.
    pub chroma_lambda_ht: Option<f32>,
    /// The bit depth of the wire bytes going in and coming out.
    #[builder(default = Depth::Eight)]
    pub depth: Depth,
    /// Where the chroma samples sit. It is carried on the engine geometry and never changes the output.
    #[builder(default)]
    pub chroma_siting: ChromaSiting,
    /// Whether the engines use tuned kernels.
    ///
    /// [TuningMode::Cached] and [TuningMode::Auto] both build from stored results.
    #[builder(default = TuningMode::Cached)]
    pub tuning: TuningMode,
    /// Picks only bit-identical candidates and the baseline profile.
    #[builder(default)]
    pub deterministic: bool,
}

impl DenoiserOptions {
    /// The temporal radius `mode` asks for.
    pub(crate) fn temporal_radius(&self) -> u32 {
        match self.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        }
    }

    /// The engine to build for a `width` by `height` frame.
    ///
    /// `mode` is copied into each plane's options, so it wins over whatever `algorithm` holds. Each
    /// plane group then takes its own strength or `lambda_ht` override.
    pub(crate) fn engine_spec(&self, width: u32, height: u32) -> EngineSpec {
        let format = self.depth.sample_format();
        let geometry = Geometry {
            width,
            height,
            chroma: self.chroma,
            input: format,
            output: format,
            chroma_siting: self.chroma_siting,
        };
        let cleans_luma = self.intent.cleans_luma();
        let cleans_chroma = self.intent.cleans_chroma();

        match self.algorithm {
            Algorithm::Nlmeans(nlm) => {
                let luma = cleans_luma.then(|| self.plane_fast(nlm, PlaneGroup::Luma));
                let chroma = cleans_chroma.then(|| self.plane_fast(nlm, PlaneGroup::Chroma));

                EngineSpec::Nlmeans {
                    luma,
                    chroma,
                    geometry,
                }
            },
            Algorithm::NlmeansHq(hq) => {
                let luma = cleans_luma.then(|| self.plane_hq(hq, PlaneGroup::Luma));
                let chroma = cleans_chroma.then(|| self.plane_hq(hq, PlaneGroup::Chroma));

                EngineSpec::Nlmeans {
                    luma,
                    chroma,
                    geometry,
                }
            },
            Algorithm::Nl4d(nl4d) => {
                let luma = cleans_luma.then(|| self.plane_nl4d(nl4d, PlaneGroup::Luma));
                let chroma = cleans_chroma.then(|| self.plane_nl4d(nl4d, PlaneGroup::Chroma));

                EngineSpec::Nl4d {
                    luma,
                    chroma,
                    geometry,
                }
            },
        }
    }

    /// Picks the luma or chroma value of a per-plane setting.
    fn per_plane<T>(group: PlaneGroup, luma: T, chroma: T) -> T {
        match group {
            PlaneGroup::Luma => luma,
            PlaneGroup::Chroma => chroma,
        }
    }

    /// `nlm` with this run's mode and the strength override of `group`.
    fn plane_nlm(&self, nlm: NlmeansOptions, group: PlaneGroup) -> NlmeansOptions {
        let strength = Self::per_plane(group, self.luma_strength, self.chroma_strength);
        let with_strength = with_plane_strength(nlm, strength);

        NlmeansOptions {
            mode: self.mode,
            ..with_strength
        }
    }

    fn plane_fast(&self, nlm: NlmeansOptions, group: PlaneGroup) -> NlmeansAlgorithm {
        let options = self.plane_nlm(nlm, group);
        NlmeansAlgorithm::Fast(options)
    }

    fn plane_hq(&self, hq: NlmeansHqOptions, group: PlaneGroup) -> NlmeansAlgorithm {
        let nlm = self.plane_nlm(hq.nlm, group);
        let options = NlmeansHqOptions { nlm, ..hq };

        NlmeansAlgorithm::Hq(options)
    }

    /// `nl4d` with this run's temporal radius and the `lambda_ht` override of `group`.
    ///
    /// An unset `lambda_ht` stays `None`, so the engine resolves it to the plane's own default.
    /// Grain is only exported from luma.
    fn plane_nl4d(&self, nl4d: Nl4dOptions, group: PlaneGroup) -> Nl4dOptions {
        let plane_lambda_ht = Self::per_plane(group, self.luma_lambda_ht, self.chroma_lambda_ht);

        Nl4dOptions {
            temporal_radius: self.temporal_radius(),
            lambda_ht: plane_lambda_ht.or(nl4d.lambda_ht),
            grain_export: nl4d.grain_export && group == PlaneGroup::Luma,
            grain_report: nl4d.grain_report && group == PlaneGroup::Luma,
            ..nl4d
        }
    }
}

/// Replaces `nlm`'s strength with the per-plane override, when there is one.
fn with_plane_strength(nlm: NlmeansOptions, strength: Option<f32>) -> NlmeansOptions {
    match strength {
        None => nlm,
        Some(strength) => {
            let tuning = NlmTuning {
                strength: Some(strength),
                ..nlm.tuning
            };

            NlmeansOptions { tuning, ..nlm }
        },
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
        let options = NlmeansOptions::default();
        Self::Nlmeans(options)
    }
}
