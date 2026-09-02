use std::collections::VecDeque;

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

use crate::accelerate::Accelerator;
use crate::device::Device;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
#[cfg(test)]
use crate::nlmeans::MotionEstimation;
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    MotionSearch,
    NlmDenoiser,
    NlmParams,
    Pending,
    PrefilterMode,
    TryWait,
    hq_default_strength,
    validate_dimensions,
};
use crate::sniff::sniff_best_accelerator;

/// How a [`Denoiser`] should be set up.
///
/// Build one with `DenoiserOptions::builder()`. Every field has a
/// default, so only the parts you care about need naming.
///
/// Only the settings every algorithm reads live here. Everything else
/// belongs to whichever [`Algorithm`] variant actually uses it.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which channels of the frame to denoise.
    #[builder(default = ChannelMode::Yuv)]
    pub channel_mode: ChannelMode,
    /// Whether to clean each frame on its own or across a temporal
    /// window.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Which algorithm to run, along with the settings only that
    /// algorithm reads.
    #[builder(default)]
    pub algorithm: Algorithm,
}

/// Which denoising algorithm to run.
///
/// Each variant carries its own settings, so a knob one algorithm has no
/// use for cannot be set on it.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Algorithm {
    /// The fast NLMeans path, with fixed weighting and no noise
    /// measurement.
    Nlmeans(NlmeansOptions),
    /// NLMeans with its weighting matched to the measured noise level.
    ///
    /// This also uses a different default `strength`, one that adapts to
    /// the temporal radius and the plane being denoised. See
    /// [`crate::nlmeans::hq_default_strength`].
    NlmeansHq(NlmeansHqOptions),
    /// Groups 8x8 patches across the motion-compensated temporal window
    /// itself, rather than filtering with NLM first and grouping within
    /// one frame afterward.
    ///
    /// No NLM weighting pass ever runs, so none of the NLM knobs appear
    /// on [`Nl4dOptions`].
    Nl4d(Nl4dOptions),
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Nlmeans(NlmeansOptions::default())
    }
}

/// Settings for [`Algorithm::Nlmeans`].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansOptions {
    /// Which reference image the NLM weights are computed against.
    ///
    /// `None`, the default, compares patches on the noisy input
    /// directly. Every other mode costs one extra GPU pass per frame.
    pub prefilter: PrefilterMode,
    /// Whether temporal denoising follows motion between frames.
    ///
    /// `None`, the default, turns motion compensation off. `Mvtools`
    /// warps temporal neighbours into line with the centre frame before
    /// the NLM weighting runs.
    ///
    /// Only has an effect when [`DenoiserOptions::mode`] is
    /// `Temporal { .. }`.
    pub motion_compensation: MotionCompensationMode,
    /// Overrides for the NLM search radius, patch radius, strength, and
    /// self-weight.
    pub tuning: NlmTuning,
}

/// Settings for [`Algorithm::NlmeansHq`].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansHqOptions {
    /// Everything the fast path takes, which HQ takes too.
    pub nlm: NlmeansOptions,
    /// The noise measurement and confidence weighting HQ adds on top.
    pub hq: HqParams,
}

/// Settings for [`Algorithm::Nl4d`].
///
/// nl4d runs the HQ front end only for its machinery, the frame ring,
/// the motion field, and the noise estimate. Nothing weights or averages
/// patches the NLM way, so the NLM knobs are absent here and the fields
/// below are the whole surface.
///
/// The temporal radius comes from [`DenoiserOptions::mode`], which has to
/// be `Temporal { .. }`. Motion tracking is always on, because the
/// grouping kernel reads the motion field and confidence scores it
/// produces.
///
/// `lambda_ht` has a per-plane default. `None` resolves through
/// [`nl4d_default_lambda_ht`] once the plane being denoised is known.
/// `lambda_ht_scale` then multiplies whichever value that resolves to.
///
/// Every other default comes from [`Nl4dParams::default`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl4dOptions {
    /// How motion between frames is tracked.
    pub motion: MotionSearch,
    /// A fixed noise standard deviation in `[0, 1]` units, replacing the
    /// automatic per-frame estimate.
    ///
    /// `None`, the default, measures the noise in each pushed frame and
    /// smooths it over time.
    pub sigma: Option<f32>,
    /// A multiplier applied to the measured noise level before anything
    /// reads it. Defaults to 1.0.
    ///
    /// This does nothing when `sigma` pins the noise level, because the
    /// estimator never runs in that case.
    pub sigma_scale: f32,
    /// A multiplier on the per-block mismatch threshold, which sets how
    /// much extra SAD a block tolerates before its confidence starts to
    /// fall. Defaults to 1.0.
    ///
    /// Higher values tolerate larger mismatches.
    pub thsad_scale: f32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position, in `1..=4`. Defaults to 2.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the centre
    /// frame, in `1..=16`. Defaults to 9.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    /// Higher removes more noise and more fine detail.
    ///
    /// `None` resolves through [`nl4d_default_lambda_ht`], which returns
    /// a different value for luma than for chroma.
    pub lambda_ht: Option<f32>,
    /// A multiplier applied to the resolved `lambda_ht`. Defaults to
    /// 1.0.
    ///
    /// It scales an explicit `lambda_ht` and the calibrated per-plane
    /// default alike, so one value moves both planes together. Has to
    /// be finite and in `[0.1, 10.0]`.
    pub lambda_ht_scale: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Defaults to 0.05. Only
    /// affects how much compute a submit spends, never which candidates
    /// are admitted once they are scored.
    pub c_min: f32,
    /// A multiplier on the mismatch variance a poorly matched temporal
    /// member carries into the hard threshold. Defaults to 1.0.
    ///
    /// The variance grows with the square of this. The mechanism
    /// saturates well before the top of its accepted range, see
    /// [`crate::nl4d::Nl4dParams::mismatch_scale`].
    pub mismatch_scale: f32,
    /// Whether a temporal member's mismatch variance reaches the
    /// hard-threshold shrinkage at all. Defaults to `true`. See
    /// [`crate::nl4d::Nl4dParams::confidence_variance`].
    pub confidence_variance: bool,
    /// Estimates noise fresh from each frame's own window instead of
    /// smoothing it across the whole stream's history. Defaults to
    /// `false`, matching every calibrated preset.
    ///
    /// `av-denoise-vs` turns this on unconditionally, because a
    /// VapourSynth filter has to return the same pixels for a frame no
    /// matter what order frames were requested in, and history-dependent
    /// estimation breaks that guarantee under random access. See
    /// [`HqParams::windowed_noise_estimation`].
    pub windowed_noise_estimation: bool,
}

impl Default for Nl4dOptions {
    fn default() -> Self {
        let defaults = Nl4dParams::default();
        let hq = HqParams::default();
        Self {
            motion: MotionSearch::default(),
            sigma: hq.sigma_override,
            sigma_scale: hq.sigma_scale,
            thsad_scale: hq.thsad_scale,
            refine: defaults.refine,
            spatial_radius: defaults.spatial_radius,
            // Resolved per plane by `nl4d_default_lambda_ht` at
            // construction time, once the plane being denoised is
            // known.
            lambda_ht: None,
            lambda_ht_scale: 1.0,
            c_min: defaults.c_min,
            mismatch_scale: defaults.mismatch_scale,
            confidence_variance: defaults.confidence_variance,
            windowed_noise_estimation: false,
        }
    }
}

impl Nl4dOptions {
    /// The front end's HQ parameters for this configuration.
    ///
    /// `temporal_confidence` is always on, because the grouping kernel
    /// reads the confidence scores it produces. The two strength-related
    /// switches keep their defaults, since nl4d never runs a weighting
    /// pass for them to affect.
    fn to_hq_params(self) -> HqParams {
        HqParams {
            sigma_override: self.sigma,
            sigma_scale: self.sigma_scale,
            thsad_scale: self.thsad_scale,
            temporal_confidence: true,
            windowed_noise_estimation: self.windowed_noise_estimation,
            ..HqParams::default()
        }
    }
}

/// The default `lambda_ht` for nl4d's hard-threshold stage, per plane.
///
/// `lambda_ht` is how many standard deviations of estimated noise a
/// transform coefficient has to clear to survive. Raising it removes more
/// noise and more fine detail with it, so the value is a trade rather
/// than an optimum.
///
/// Luma gets 5.3, picked by eye from rendered comparisons on real grain
/// and deliberately biased toward keeping detail. Higher values remove
/// visibly more noise, but not enough to be worth what they cost in
/// texture.
///
/// `ChannelMode::Yuv` reads the luma value, on the same "a fused pass is
/// dominated by luma" assumption [`hq_default_strength`]
/// makes for its own Yuv case.
///
/// Chroma gets 4.2, picked the same way from the chroma residuals with
/// luma pinned at 5.3.
pub fn nl4d_default_lambda_ht(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => 5.3,
        ChannelMode::Chroma => 4.2,
    }
}

/// Resolves `Nl4dOptions.lambda_ht` for one plane, falling back to
/// [`nl4d_default_lambda_ht`] when the caller left it unset, then
/// applies `lambda_ht_scale`.
///
/// The scale multiplies an explicit value and the calibrated default
/// alike, so it moves both planes together whether or not one of them
/// is pinned.
///
/// The range check lives here rather than in [`Nl4dParams`],
/// which only ever sees the product. A scale of 0 would surface there as
/// a complaint about `lambda_ht`, naming a knob the caller never set.
fn resolve_lambda_ht(opts: &Nl4dOptions, channels: ChannelMode) -> Result<f32, String> {
    if !(opts.lambda_ht_scale.is_finite() && (0.1..=10.0).contains(&opts.lambda_ht_scale)) {
        return Err(format!(
            "lambda_ht_scale must be finite and in [0.1, 10.0], got {}",
            opts.lambda_ht_scale
        ));
    }

    let lambda_ht = opts.lambda_ht.unwrap_or_else(|| nl4d_default_lambda_ht(channels));

    Ok(lambda_ht * opts.lambda_ht_scale)
}

/// Speed vs quality dial.
///
/// Each denoising family reads the same dial and fills in its own knobs
/// from it. For `nlmeans` that is [`nlmeans_variant_for`],
/// [`nlmeans_temporal_radius_for`], and [`nlmeans_search_radius_for`].
/// For `nl4d` it is [`nl4d_temporal_radius_for`] and
/// [`nl4d_spatial_radius_for`].
///
/// Both front ends parse the same names from this one type, so a preset
/// resolves to the same dials everywhere it is used.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Preset {
    /// Fastest and lowest quality.
    Veryfast,
    /// One step up from `veryfast`.
    Fast,
    /// The default, favouring quality over speed.
    #[default]
    Base,
    /// One step down from `veryslow`.
    Slow,
    /// Slowest and highest quality.
    Veryslow,
}

/// Which nlmeans implementation a preset, or an explicit choice, selects.
#[derive(Debug, Copy, Clone, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum NlmeansVariant {
    /// The fast path. Fixed weighting, no noise measurement.
    Fast,
    /// Quality focused. Calibrates its weighting to the noise level,
    /// measured automatically per frame.
    Hq,
}

/// Which [`NlmeansVariant`] a preset runs.
pub fn nlmeans_variant_for(preset: Preset) -> NlmeansVariant {
    match preset {
        Preset::Veryfast => NlmeansVariant::Fast,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => NlmeansVariant::Hq,
    }
}

/// How many neighbouring frames on each side `nlmeans` looks at, at a
/// preset.
pub fn nlmeans_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 0,
        Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How far `nlmeans` looks for similar patches inside a frame, at a
/// preset.
pub fn nlmeans_search_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast | Preset::Base => 2,
        Preset::Slow | Preset::Veryslow => 4,
    }
}

/// How far the temporal window reaches at each preset, for `nl4d`.
///
/// Unlike `nlmeans`, `veryfast` keeps a 1-frame window rather than
/// dropping to 0, because nl4d has nothing to do without neighbouring
/// frames to group against.
pub fn nl4d_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How wide the centre frame's candidate search is at each preset, for
/// `nl4d`.
///
/// `veryfast` shares its temporal radius with `fast`, so this is what
/// separates them. The window covers `(2 * radius + 1)^2` positions, so
/// 6 searches a little over half the candidates 9 does.
///
/// Every preset from `fast` up uses the library default. Widening it
/// further at the slow end costs quadratically and has not been measured
/// to be worth it.
pub fn nl4d_spatial_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 6,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => {
            Nl4dOptions::default().spatial_radius
        },
    }
}

/// Whether a frame is cleaned on its own or alongside its neighbours.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DenoisingMode {
    /// Cleans each frame using only its own pixels.
    Spacial,
    /// Cleans each frame using a window of `2 * radius + 1` frames.
    Temporal { radius: u32 },
}

/// NLM tuning knobs.
///
/// Every field is optional. Whatever is left unset falls back to the
/// library default.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmTuning {
    pub search_radius: Option<u32>,
    pub patch_radius: Option<u32>,
    pub strength: Option<f32>,
    pub self_weight: Option<f32>,
}

impl DenoiserOptions {
    /// Turns this option set into the low-level [`NlmParams`] a backend
    /// denoiser is built from.
    ///
    /// For nl4d this describes the front end only, since nl4d's own
    /// grouping stage is configured from [`Nl4dOptions`] separately in
    /// [`build_engine`].
    ///
    /// Whichever default `strength` applies is folded in here. For the
    /// HQ algorithm that comes from
    /// [`crate::nlmeans::hq_default_strength`].
    ///
    /// This is public so callers building per-plane options, and tests,
    /// can read the resolved values without building a real `Denoiser`.
    #[doc(hidden)]
    pub fn to_nlm_params(&self) -> NlmParams {
        let temporal_radius = match self.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        };

        match self.algorithm {
            Algorithm::Nlmeans(opts) => self.nlm_params_for(opts, None, temporal_radius),
            Algorithm::NlmeansHq(opts) => self.nlm_params_for(opts.nlm, Some(opts.hq), temporal_radius),
            // nl4d never runs a weighting pass, so `strength`,
            // `search_radius`, `patch_radius`, and `self_weight` stay at
            // their library defaults and no prefilter is built.
            Algorithm::Nl4d(opts) => NlmParams {
                channels: self.channel_mode,
                motion_compensation: opts.motion.into(),
                temporal_radius,
                hq: Some(opts.to_hq_params()),
                ..NlmParams::default()
            },
        }
    }

    /// [`Self::to_nlm_params`] for whichever of the two NLM algorithms
    /// is running, with `hq` set only for the quality one.
    fn nlm_params_for(&self, opts: NlmeansOptions, hq: Option<HqParams>, temporal_radius: u32) -> NlmParams {
        // An explicit `strength` always wins, whether it came straight
        // from `NlmTuning` or from a per-plane override the caller
        // already folded in.
        //
        // Otherwise the default depends on `auto_strength`. With it on,
        // HQ reads `strength` as a multiplier on the measured noise
        // level, so it needs its own calibrated default rather than the
        // fast path's absolute FFmpeg-style one. That calibrated default
        // also varies with the temporal radius and with the plane
        // `channel_mode` names, because each per-plane `Denoiser`
        // carries its own channel mode.
        //
        // With auto-strength off, HQ reads `strength` as an absolute
        // value just like the fast path, so it falls back to the same
        // absolute default.
        let strength = opts.tuning.strength.unwrap_or(match hq {
            Some(hq) if hq.auto_strength => hq_default_strength(self.channel_mode, temporal_radius),
            _ => NlmParams::default().strength,
        });

        let defaults = NlmParams::default();
        NlmParams {
            channels: self.channel_mode,
            prefilter: opts.prefilter,
            motion_compensation: opts.motion_compensation,
            temporal_radius,
            hq,
            strength,
            search_radius: opts.tuning.search_radius.unwrap_or(defaults.search_radius),
            patch_radius: opts.tuning.patch_radius.unwrap_or(defaults.patch_radius),
            self_weight: opts.tuning.self_weight.unwrap_or(defaults.self_weight),
        }
    }
}

/// Errors reported by the high-level [`Denoiser`].
#[derive(Debug, thiserror::Error)]
pub enum DenoiserError {
    /// An earlier denoised frame has not been collected yet, so pushing
    /// again would overwrite it in the double-buffered output slot.
    ///
    /// Call [`Denoiser::recv_frame`] or [`Denoiser::try_recv_frame`],
    /// then retry the same `push_frame` call.
    #[error("denoiser queue is full, collect the pending frame before pushing more")]
    QueueFull,
    /// None of the accelerators in the priority list could be started.
    #[error("no accelerator from the priority list is available")]
    NoAcceleratorAvailable,
    /// Anything else, wrapping the internal `anyhow` errors raised by
    /// kernel dispatch and readback.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Either denoiser a `Backend` runtime arm can hold.
///
/// This keeps `Backend`'s own match arms at one line each. Without it,
/// adding a second denoiser type would multiply the runtime arms instead
/// of fanning out once here.
enum Engine<R: Runtime> {
    Nlm(Box<NlmDenoiser<R>>),
    Nl4d(Box<Nl4dDenoiser<R>>),
}

impl<R: Runtime> Engine<R> {
    fn is_nl4d(&self) -> bool {
        matches!(self, Self::Nl4d(_))
    }

    fn push_frame(&mut self, frame: &[f32]) {
        match self {
            Self::Nlm(d) => d.push_frame(frame),
            Self::Nl4d(d) => d.push_frame(frame),
        }
    }

    fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        match self {
            Self::Nlm(d) => d.denoise_submit(),
            // `Nl4dDenoiser::denoise_submit` already returns
            // `DenoiserError` rather than `anyhow::Error`, so this leans
            // on `DenoiserError`'s own `anyhow::Error` conversion instead
            // of re-wrapping it.
            Self::Nl4d(d) => d.denoise_submit().map_err(anyhow::Error::from),
        }
    }

    fn flush(&mut self, sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        match self {
            Self::Nlm(d) => d.flush(sink),
            Self::Nl4d(d) => d.flush(sink).map_err(anyhow::Error::from),
        }
    }

    fn reset_stream(&mut self) {
        match self {
            Self::Nlm(d) => d.reset_stream_state(),
            Self::Nl4d(d) => d.reset_stream(),
        }
    }
}

/// Builds whichever [`Engine`] `algorithm` calls for.
///
/// `Algorithm::Nl4d` carries its own grouping tuning, which is not part
/// of `NlmParams`, so it is read from `algorithm` directly rather than
/// from `params`. This is also where an unset `lambda_ht` picks up its
/// calibrated per-plane default (`resolve_lambda_ht`), the same way
/// `to_nlm_params` resolves HQ's calibrated `strength`, since this is
/// the first point construction has both `opts` and `params.channels`
/// together.
fn build_engine<R: Runtime>(
    client: &ComputeClient<R>,
    algorithm: &Algorithm,
    params: NlmParams,
    width: u32,
    height: u32,
) -> Result<Engine<R>, DenoiserError> {
    match algorithm {
        Algorithm::Nl4d(opts) => {
            // nl4d groups patches across neighbouring frames, so there
            // is nothing for it to do without a temporal window.
            if params.temporal_radius == 0 {
                return Err(DenoiserError::Other(anyhow::anyhow!(
                    "nl4d needs a temporal window, set DenoiserOptions::mode to \
                     DenoisingMode::Temporal"
                )));
            }

            let lambda_ht = resolve_lambda_ht(opts, params.channels)
                .map_err(|e| DenoiserError::Other(anyhow::anyhow!(e)))?;
            let nl4d_params = Nl4dParams {
                temporal_radius: params.temporal_radius,
                nlm: params,
                refine: opts.refine,
                spatial_radius: opts.spatial_radius,
                lambda_ht,
                c_min: opts.c_min,
                mismatch_scale: opts.mismatch_scale,
                confidence_variance: opts.confidence_variance,
            };
            let denoiser = Nl4dDenoiser::new(client, nl4d_params, width, height)
                .map_err(|e| DenoiserError::Other(anyhow::anyhow!(e)))?;
            Ok(Engine::Nl4d(Box::new(denoiser)))
        },
        Algorithm::Nlmeans(_) | Algorithm::NlmeansHq(_) => Ok(Engine::Nlm(Box::new(NlmDenoiser::new(
            client, params, width, height,
        )))),
    }
}

enum Backend {
    #[cfg(feature = "cuda")]
    Cuda(Engine<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(Engine<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(Engine<cubecl::wgpu::WgpuRuntime>),
}

impl Backend {
    fn is_nl4d(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(e) => e.is_nl4d(),
            #[cfg(feature = "rocm")]
            Self::Rocm(e) => e.is_nl4d(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(e) => e.is_nl4d(),
        }
    }
}

enum BackendPending {
    #[cfg(feature = "cuda")]
    Cuda(Pending<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(Pending<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(Pending<cubecl::wgpu::WgpuRuntime>),
}

impl BackendPending {
    fn wait(self) -> Result<Vec<f32>, anyhow::Error> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(p) => p.wait(),
            #[cfg(feature = "rocm")]
            Self::Rocm(p) => p.wait(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(p) => p.wait(),
        }
    }

    /// Polls the readback once. `Ok(Ok(frame))` is a landed frame,
    /// `Ok(Err(self))` is a readback still in flight.
    fn try_wait(self) -> Result<Result<Vec<f32>, Self>, anyhow::Error> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Cuda(p))),
            },
            #[cfg(feature = "rocm")]
            Self::Rocm(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Rocm(p))),
            },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Wgpu(p))),
            },
        }
    }
}

/// How many readbacks the high-level [`Denoiser`] keeps in flight at
/// once.
///
/// This has to match the backend's output-handle count, which is two.
/// Going past it would reuse the oldest pending frame's output handle
/// and quietly corrupt the results.
pub const MAX_PENDING: usize = 2;

/// How many source frames a windowed operation needs behind and ahead
/// of its target frame, target frame itself not counted in either
/// number.
///
/// `reseed` needs exactly `behind + 1 + ahead` frames, oldest first,
/// with the target frame sitting at index `behind`. This is what tells
/// a caller like `reseed` how wide a window to build, and it varies by
/// algorithm because nl4d's own cross-frame accumulator needs more
/// forward context than the NLM algorithms do. See
/// [`Denoiser::window_span`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSpan {
    /// How many frames older than the target the window must include.
    pub behind: usize,
    /// How many frames newer than the target the window must include.
    pub ahead: usize,
}

impl WindowSpan {
    /// The full window size this span describes, target frame
    /// included: `behind + 1 + ahead`.
    pub fn frame_count(&self) -> usize {
        self.behind + 1 + self.ahead
    }
}

/// A stateful denoiser that cleans a stream of frames.
///
/// Push frames in order with [`push_frame`](Self::push_frame) and
/// collect the cleaned ones with [`recv_frame`](Self::recv_frame) or
/// [`try_recv_frame`](Self::try_recv_frame).
///
/// At the end of the stream call [`flush`](Self::flush) to drain
/// whatever temporal context is left.
///
/// Frames are `f32` values in `[0, 1]`, laid out as
/// `width * height * channels`.
///
/// ```no_run
/// use av_denoise_core::accelerate::Accelerator;
/// use av_denoise_core::{ChannelMode, Denoiser, DenoiserOptions, DenoisingMode, Device};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let options = DenoiserOptions::builder()
///     .channel_mode(ChannelMode::Luma)
///     .mode(DenoisingMode::Temporal { radius: 2 })
///     .build();
///
/// let mut denoiser = Denoiser::create(
///     &[Accelerator::Vulkan],
///     &Device::Default,
///     1920,
///     1080,
///     options,
/// )?;
///
/// let frames: Vec<Vec<f32>> = read_my_frames();
/// let mut cleaned: Vec<Vec<f32>> = Vec::new();
///
/// for frame in &frames {
///     denoiser.push_frame(frame)?;
///
///     // Temporal denoising runs a few frames behind the input, so
///     // there is not always one ready to collect.
///     if let Some(out) = denoiser.recv_frame()? {
///         cleaned.push(out);
///     }
/// }
///
/// // Drain the frames still inside the temporal window.
/// denoiser.flush(|out| cleaned.push(out))?;
/// # Ok(())
/// # }
/// # fn read_my_frames() -> Vec<Vec<f32>> { Vec::new() }
/// ```
pub struct Denoiser {
    backend: Backend,
    pending: VecDeque<BackendPending>,
    accelerator: Accelerator,
    width: u32,
    height: u32,
    channels: u32,
    temporal_radius: u32,
    frames_pushed: u32,
}

impl Denoiser {
    /// Tries each accelerator in `accelerators` in order and builds a
    /// denoiser on the first one that works.
    ///
    /// `device` picks a non-default device on the chosen runtime.
    ///
    /// # Thread stack size
    ///
    /// cubecl spawns its own per-device worker thread, named
    /// `DS{U,D}-…`, and runs GPU kernel codegen on it. That thread gets
    /// Rust's default stack, which is `RUST_MIN_STACK` or 2 MiB when
    /// that is unset.
    ///
    /// The windowed NLM kernels unroll their body
    /// `(2 * search_radius + 1)^2` times, so a `search_radius` of about
    /// 5 or more can overflow the 2 MiB default and abort the process.
    ///
    /// Callers using a `search_radius` above 4 should call
    /// [`crate::raise_codegen_stack_limit`] before any cubecl thread
    /// spawns, usually right at the top of `main`.
    pub fn create(
        accelerators: &[Accelerator],
        device: &Device,
        width: u32,
        height: u32,
        options: DenoiserOptions,
    ) -> Result<Self, DenoiserError> {
        let accelerator =
            sniff_best_accelerator(accelerators, device).ok_or(DenoiserError::NoAcceleratorAvailable)?;

        let params = options.to_nlm_params();
        params.validate()?;
        validate_dimensions(width, height)?;

        let channels = params.channels.count();
        let temporal_radius = params.temporal_radius;
        let backend = build_backend(accelerator, device, &options.algorithm, params, width, height)?;

        Ok(Self {
            backend,
            pending: VecDeque::with_capacity(MAX_PENDING),
            accelerator,
            width,
            height,
            channels,
            temporal_radius,
            frames_pushed: 0,
        })
    }

    /// The accelerator [`sniff_best_accelerator`] picked.
    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// The width passed at construction.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height passed at construction.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The temporal radius the resolved parameters run at.
    pub fn temporal_radius(&self) -> u32 {
        self.temporal_radius
    }

    /// How many frames behind and ahead of a target frame this
    /// denoiser needs pushed, in order, to produce that frame's
    /// output through [`PlanarDenoiser::reseed`](crate::PlanarDenoiser::reseed).
    ///
    /// Both NLM algorithms only ever need their own `2 * radius + 1`
    /// sliding window, symmetric around the target frame:
    /// `WindowSpan { behind: radius, ahead: radius }`.
    ///
    /// nl4d's cross-frame accumulator scatters every pass's
    /// contribution across the `2 * radius + 1` frames the pass
    /// reaches, and a frame's own region only starts collecting once
    /// the pass that first reaches it, the one centred `radius` frames
    /// behind it, has run. That earliest pass is itself only real once
    /// the front end's own window is full at that centre, which needs
    /// `radius` more frames behind it again. So nl4d needs the target's
    /// own `radius`-wide neighbourhood doubled on both sides:
    /// `WindowSpan { behind: 2 * radius, ahead: 2 * radius }`.
    pub fn window_span(&self) -> WindowSpan {
        let radius = self.temporal_radius as usize;
        let span = if self.backend.is_nl4d() {
            2 * radius
        } else {
            radius
        };
        WindowSpan {
            behind: span,
            ahead: span,
        }
    }

    /// Uploads one frame into the temporal window.
    ///
    /// `frame` holds `width * height * channels` `f32` values in
    /// `[0, 1]`.
    ///
    /// Once the window is full and the pipeline has room, this also
    /// starts the kernels for the next denoised frame.
    ///
    /// Up to `MAX_PENDING` outputs can be in flight at once, so the GPU
    /// runs one frame's kernels while the previous frame's readback is
    /// still travelling. At that ceiling this returns
    /// [`DenoiserError::QueueFull`], and the caller has to drain a frame
    /// with [`Self::recv_frame`] before pushing more.
    pub fn push_frame(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        // After `temporal_radius` real pushes the leading-edge mirror
        // has primed the window, so the next push produces a pending
        // frame. From then on every push takes a pending slot.
        let window_full = self.frames_pushed > self.temporal_radius;
        if window_full && self.pending.len() >= MAX_PENDING {
            return Err(DenoiserError::QueueFull);
        }

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Cuda(p));
                }
            },
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Rocm(p));
                }
            },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Wgpu(p));
                }
            },
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Uploads one frame into the temporal window without starting a
    /// denoise.
    ///
    /// The ring advances exactly as it does for [`Self::push_frame`], so
    /// the window still fills, but no kernels are submitted and no
    /// output is queued. This is how a caller that can hand over a whole
    /// window at once, rather than a strictly ordered stream, fills the
    /// window in one go and lets only the last push in it submit.
    pub fn push_frame_priming(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.push_frame(frame),
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.push_frame(frame),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.push_frame(frame),
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Drops the current stream and returns to the state a fresh
    /// denoiser starts in, keeping every GPU allocation.
    ///
    /// Anything still in flight is discarded.
    pub fn reset_stream(&mut self) {
        self.pending.clear();
        self.frames_pushed = 0;

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.reset_stream(),
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.reset_stream(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.reset_stream(),
        }
    }

    /// Blocks until the in-flight denoise finishes and returns the
    /// cleaned frame.
    ///
    /// Returns `Ok(None)` when nothing is in flight, which happens while
    /// the temporal window is still filling up.
    pub fn recv_frame(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };
        Ok(Some(pending.wait()?))
    }

    /// Polls the in-flight denoise once.
    ///
    /// Returns `Ok(None)` both when nothing is in flight and when the in-flight readback
    /// has not landed yet, so `None` alone does not tell those two cases apart.
    ///
    /// A caller that needs the frame rather than just checking on it should
    /// use [`Self::recv_frame`] instead.
    ///
    /// This only avoids blocking on the wgpu backends, meaning Vulkan and Metal.
    /// On CUDA and ROCm the readback completes synchronously on its first poll,
    /// so this call blocks until the readback lands there, the same as `recv_frame`.
    pub fn try_recv_frame(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };

        match pending.try_wait()? {
            Ok(frame) => Ok(Some(frame)),
            Err(pending) => {
                self.pending.push_front(pending);
                Ok(None)
            },
        }
    }

    /// Drains the in-flight frames and the trailing temporal tail,
    /// handing each frame it produces to `sink`.
    ///
    /// The tail is padded by repeating the last pushed frame.
    ///
    /// On success the denoiser is ready for a fresh, unrelated stream of
    /// the same size and parameters. Pushing again after a flush starts
    /// a new temporal window from scratch, and flushing more than once
    /// is fine.
    ///
    /// If `flush` returns `Err` the denoiser is in an undefined state
    /// and should be dropped rather than reused.
    pub fn flush(&mut self, mut sink: impl FnMut(Vec<f32>)) -> Result<(), DenoiserError> {
        // Drain the whole pending pipeline, up to MAX_PENDING frames,
        // before submitting the trailing-tail mirrors.
        while let Some(frame) = self.recv_frame()? {
            sink(frame);
        }

        let pixels = (self.width * self.height) as usize;
        let channels = self.channels as usize;
        let scratch_cap = pixels * channels;

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.flush(|slice| {
                let mut v = Vec::with_capacity(scratch_cap);
                v.extend_from_slice(slice);
                sink(v);
            })?,
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.flush(|slice| {
                let mut v = Vec::with_capacity(scratch_cap);
                v.extend_from_slice(slice);
                sink(v);
            })?,
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.flush(|slice| {
                let mut v = Vec::with_capacity(scratch_cap);
                v.extend_from_slice(slice);
                sink(v);
            })?,
        }

        // The backend has already reset its own stream indices. Reset
        // the outer push counter too, so the next push re-arms the
        // window-priming check at the top of `push_frame`.
        self.frames_pushed = 0;

        Ok(())
    }
}

fn build_backend(
    accel: Accelerator,
    device: &Device,
    algorithm: &Algorithm,
    params: NlmParams,
    width: u32,
    height: u32,
) -> Result<Backend, DenoiserError> {
    match accel {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let dev = device.to_cuda()?;
            let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&dev);
            Ok(Backend::Cuda(build_engine(
                &client, algorithm, params, width, height,
            )?))
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let dev = device.to_amd()?;
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&dev);
            Ok(Backend::Rocm(build_engine(
                &client, algorithm, params, width, height,
            )?))
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let dev = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Wgpu(build_engine(
                &client, algorithm, params, width, height,
            )?))
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let dev = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Wgpu(build_engine(
                &client, algorithm, params, width, height,
            )?))
        },
        // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)`
        // widens the `Accelerator` enum to include variants whose
        // backend feature is not enabled. Never reached at runtime.
        #[cfg(docsrs)]
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod options_tests {
    use super::*;

    /// `Algorithm::NlmeansHq` with `hq` overridden and everything else
    /// left at its default.
    fn hq(hq: HqParams) -> Algorithm {
        Algorithm::NlmeansHq(NlmeansHqOptions {
            hq,
            ..NlmeansHqOptions::default()
        })
    }

    /// `Algorithm::Nlmeans` with `tuning` overridden.
    fn fast_tuned(tuning: NlmTuning) -> Algorithm {
        Algorithm::Nlmeans(NlmeansOptions {
            tuning,
            ..NlmeansOptions::default()
        })
    }

    #[test]
    fn nl4d_default_lambda_ht_differs_between_luma_and_chroma() {
        let luma = nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma = nl4d_default_lambda_ht(ChannelMode::Chroma);

        assert!((luma - 5.3).abs() < f32::EPSILON);
        assert!((chroma - 4.2).abs() < f32::EPSILON);
        assert!(
            (chroma - luma).abs() > f32::EPSILON,
            "the two planes should not resolve to the same default"
        );
    }

    #[test]
    fn nl4d_default_lambda_ht_yuv_reads_the_luma_value() {
        let yuv = nl4d_default_lambda_ht(ChannelMode::Yuv);
        let luma = nl4d_default_lambda_ht(ChannelMode::Luma);

        assert!((yuv - luma).abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_lambda_ht_unset_uses_the_per_plane_default() {
        let opts = Nl4dOptions::default();

        let luma = resolve_lambda_ht(&opts, ChannelMode::Luma).expect("the default scale is in range");
        let chroma = resolve_lambda_ht(&opts, ChannelMode::Chroma).expect("the default scale is in range");

        assert!((luma - 5.3).abs() < f32::EPSILON, "got {luma}");
        assert!((chroma - 4.2).abs() < f32::EPSILON, "got {chroma}");
    }

    #[test]
    fn resolve_lambda_ht_explicit_value_overrides_every_plane() {
        let opts = Nl4dOptions {
            lambda_ht: Some(4.4),
            ..Nl4dOptions::default()
        };

        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let got = resolve_lambda_ht(&opts, channels).expect("the default scale is in range");
            assert!(
                (got - 4.4).abs() < f32::EPSILON,
                "channels {channels:?} got {got}"
            );
        }
    }

    #[test]
    fn resolve_lambda_ht_default_scale_leaves_the_value_alone() {
        let opts = Nl4dOptions::default();

        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let got = resolve_lambda_ht(&opts, channels).expect("the default scale is in range");
            let want = nl4d_default_lambda_ht(channels);
            assert!(
                (got - want).abs() < f32::EPSILON,
                "channels {channels:?} got {got}"
            );
        }
    }

    #[test]
    fn resolve_lambda_ht_scale_multiplies_the_per_plane_default() {
        let opts = Nl4dOptions {
            lambda_ht_scale: 1.1,
            ..Nl4dOptions::default()
        };

        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let got = resolve_lambda_ht(&opts, channels).expect("1.1 is in range");
            let want = nl4d_default_lambda_ht(channels) * 1.1;
            assert!(
                (got - want).abs() < 1e-5,
                "channels {channels:?} got {got}, want {want}"
            );
        }
    }

    /// The scale is not limited to the defaults. Pinning one plane and
    /// scaling both is the combination this exists for.
    #[test]
    fn resolve_lambda_ht_scale_multiplies_an_explicit_value() {
        let opts = Nl4dOptions {
            lambda_ht: Some(5.0),
            lambda_ht_scale: 0.9,
            ..Nl4dOptions::default()
        };

        let got = resolve_lambda_ht(&opts, ChannelMode::Luma).expect("0.9 is in range");
        assert!((got - 4.5).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn resolve_lambda_ht_rejects_an_out_of_range_scale() {
        for bad in [0.0, -1.0, 0.05, 10.5, f32::NAN, f32::INFINITY] {
            let opts = Nl4dOptions {
                lambda_ht_scale: bad,
                ..Nl4dOptions::default()
            };
            let err = resolve_lambda_ht(&opts, ChannelMode::Luma).unwrap_err();
            assert!(
                err.contains("lambda_ht_scale"),
                "lambda_ht_scale={bad} should be rejected, got {err}"
            );
        }
    }

    #[test]
    fn the_default_algorithm_is_the_fast_nlmeans_path() {
        let opts = DenoiserOptions::builder().build();
        assert_eq!(opts.algorithm, Algorithm::Nlmeans(NlmeansOptions::default()));
    }

    #[test]
    fn spatial_mode_maps_to_zero_temporal_radius() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Yuv)
            .mode(DenoisingMode::Spacial)
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(params.temporal_radius, 0);
        assert_eq!(params.channels, ChannelMode::Yuv);
    }

    #[test]
    fn temporal_mode_propagates_radius() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 3 })
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(params.temporal_radius, 3);
    }

    #[test]
    fn prefilter_passthrough() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nlmeans(NlmeansOptions {
                prefilter: PrefilterMode::Bilateral {
                    sigma_s: 3.0,
                    sigma_r: 0.02,
                },
                ..NlmeansOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::Bilateral { .. }));
    }

    #[test]
    fn hq_unset_prefilter_defaults_to_none() {
        let opts = DenoiserOptions::builder()
            .algorithm(hq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::None));
    }

    #[test]
    fn fast_unset_prefilter_defaults_to_none() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nlmeans(NlmeansOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::None));
    }

    #[test]
    fn hq_unset_strength_defaults_to_hq_default_strength() {
        // Default channel_mode is Yuv, default mode is Spacial (radius 0).
        let opts = DenoiserOptions::builder()
            .algorithm(hq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        let expected = hq_default_strength(ChannelMode::Yuv, 0);
        assert!((params.strength - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_no_auto_strength_falls_back_to_the_legacy_absolute_default() {
        // `effective_strength_with` only reads `strength` as a
        // multiplier on the measured sigma when `auto_strength` is true.
        // With it false, `strength` is an FFmpeg-style absolute value,
        // so the fallback has to be the fast path's absolute default
        // rather than a calibrated multiplier from
        // `hq_default_strength`.
        let opts = DenoiserOptions::builder()
            .algorithm(hq(HqParams {
                auto_strength: false,
                ..HqParams::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        let expected = NlmParams::default().strength;
        assert!(
            (params.strength - expected).abs() < f32::EPSILON,
            "expected the legacy absolute default {expected}, got {}, which looks like the \
             auto-strength multiplier table leaking through",
            params.strength
        );
    }

    #[test]
    fn hq_luma_r4_uses_measured_table_value() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(hq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_chroma_r4_uses_measured_table_value() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Chroma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(hq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_yuv_r8_uses_measured_table_value() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Yuv)
            .mode(DenoisingMode::Temporal { radius: 8 })
            .algorithm(hq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.30).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_spacial_mode_uses_radius_zero_table_values() {
        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let opts = DenoiserOptions::builder()
                .channel_mode(channels)
                .mode(DenoisingMode::Spacial)
                .algorithm(hq(HqParams::default()))
                .build();
            let params = opts.to_nlm_params();

            let expected = hq_default_strength(channels, 0);
            assert!(
                (params.strength - expected).abs() < f32::EPSILON,
                "for channels {channels:?} expected {expected}, got {}",
                params.strength
            );
        }
    }

    #[test]
    fn hq_explicit_strength_wins_over_the_table_for_every_plane() {
        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let opts = DenoiserOptions::builder()
                .channel_mode(channels)
                .mode(DenoisingMode::Temporal { radius: 4 })
                .algorithm(Algorithm::NlmeansHq(NlmeansHqOptions {
                    nlm: NlmeansOptions {
                        tuning: NlmTuning {
                            strength: Some(0.99),
                            ..NlmTuning::default()
                        },
                        ..NlmeansOptions::default()
                    },
                    hq: HqParams::default(),
                }))
                .build();
            let params = opts.to_nlm_params();

            assert!(
                (params.strength - 0.99).abs() < f32::EPSILON,
                "for channels {channels:?} the explicit strength was overridden by the table"
            );
        }
    }

    #[test]
    fn fast_unset_strength_defaults_to_legacy_default() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nlmeans(NlmeansOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn nl4d_options_default_matches_nl4d_params_default() {
        let opts = Nl4dOptions::default();
        let params = crate::nl4d::Nl4dParams::default();

        assert_eq!(opts.refine, params.refine);
        assert_eq!(opts.spatial_radius, params.spatial_radius);
        assert!((opts.c_min - params.c_min).abs() < f32::EPSILON);
        assert_eq!(opts.confidence_variance, params.confidence_variance);
        // The two `lambda_ht` fields hold different things, so they are
        // not compared. `opts.lambda_ht` stays `None` and is deferred to
        // `nl4d_default_lambda_ht` once the plane is known (see
        // `resolve_lambda_ht_unset_uses_the_per_plane_default` above),
        // while `params.lambda_ht` is a concrete default mirroring the
        // Luma/Yuv value.
        assert_eq!(opts.lambda_ht, None);
        assert!((params.lambda_ht - nl4d_default_lambda_ht(ChannelMode::Yuv)).abs() < f32::EPSILON);
    }

    /// nl4d takes its own noise and confidence knobs rather than a whole
    /// [`HqParams`], so the three it does take have to reach the front
    /// end and the rest have to arrive at their defaults.
    #[test]
    fn nl4d_builds_the_front_ends_hq_params_from_its_own_fields() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 2 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions {
                sigma: Some(0.02),
                sigma_scale: 1.3,
                thsad_scale: 0.8,
                ..Nl4dOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        let hq = params.hq.expect("nl4d always runs the hq front end");
        assert_eq!(hq.sigma_override, Some(0.02));
        assert!((hq.sigma_scale - 1.3).abs() < f32::EPSILON);
        assert!((hq.thsad_scale - 0.8).abs() < f32::EPSILON);
        assert!(
            hq.temporal_confidence,
            "the grouping kernel reads the confidence scores, so this cannot be off"
        );
    }

    /// The temporal radius has one source now, `mode`, so nl4d cannot
    /// disagree with the front end's ring about how wide the window is.
    #[test]
    fn nl4d_reads_its_temporal_radius_from_the_denoising_mode() {
        for radius in [1u32, 4, 8] {
            let opts = DenoiserOptions::builder()
                .mode(DenoisingMode::Temporal { radius })
                .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
                .build();

            assert_eq!(opts.to_nlm_params().temporal_radius, radius);
        }
    }

    /// nl4d never runs an NLM weighting pass, so a prefilter would cost
    /// a GPU pass per frame producing a reference image nothing reads.
    #[test]
    fn nl4d_never_builds_a_prefilter() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 2 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();

        assert!(matches!(opts.to_nlm_params().prefilter, PrefilterMode::None));
    }

    /// Nothing in the nl4d path reads `strength`, so it stays at the
    /// library default rather than picking up HQ's calibrated table.
    #[test]
    fn nl4d_leaves_the_nlm_weighting_knobs_at_their_defaults() {
        let defaults = NlmParams::default();
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - defaults.strength).abs() < f32::EPSILON);
        assert_eq!(params.search_radius, defaults.search_radius);
        assert_eq!(params.patch_radius, defaults.patch_radius);
        assert!((params.self_weight - defaults.self_weight).abs() < f32::EPSILON);
    }

    /// nl4d always tracks motion, so its `MotionSearch` reaches the
    /// front end as an active `Mvtools` mode.
    #[test]
    fn nl4d_motion_search_becomes_an_active_mvtools_mode() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 2 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions {
                motion: MotionSearch {
                    blksize: 32,
                    overlap: 16,
                    search_radius: 6,
                    pyramid_levels: 1,
                    estimation: MotionEstimation::Direct,
                },
                ..Nl4dOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(
            params.motion_compensation,
            MotionCompensationMode::Mvtools {
                blksize: 32,
                overlap: 16,
                search_radius: 6,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            }
        ));
    }

    #[test]
    fn nl4d_motion_search_defaults_match_the_front_ends_own_defaults() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 2 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(
            params.motion_compensation,
            crate::nl4d::Nl4dParams::default().nlm.motion_compensation
        );
    }

    #[test]
    fn motion_compensation_passthrough() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 1 })
            .algorithm(Algorithm::Nlmeans(NlmeansOptions {
                motion_compensation: MotionCompensationMode::Mvtools {
                    blksize: 16,
                    overlap: 8,
                    search_radius: 4,
                    pyramid_levels: 2,
                    estimation: MotionEstimation::Direct,
                },
                ..NlmeansOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(
            params.motion_compensation,
            MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                ..
            }
        ));
    }

    #[test]
    fn motion_compensation_defaults_to_none() {
        let opts = DenoiserOptions::builder().build();
        let params = opts.to_nlm_params();
        assert!(matches!(params.motion_compensation, MotionCompensationMode::None));
    }

    #[test]
    fn nlm_tuning_overrides_individual_fields() {
        let defaults = NlmParams::default();
        let opts = DenoiserOptions::builder()
            .algorithm(fast_tuned(NlmTuning {
                search_radius: Some(7),
                patch_radius: None,
                strength: Some(2.5),
                self_weight: None,
            }))
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(params.search_radius, 7);
        assert_eq!(params.patch_radius, defaults.patch_radius);
        assert!((params.strength - 2.5).abs() < f32::EPSILON);
        assert!((params.self_weight - defaults.self_weight).abs() < f32::EPSILON);
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    fn opts(mode: DenoisingMode) -> DenoiserOptions {
        DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(mode)
            .build()
    }

    fn frame(w: u32, h: u32) -> Vec<f32> {
        vec![0.5f32; (w * h) as usize]
    }

    #[test]
    fn spatial_denoise_roundtrip() {
        let mut d = Denoiser::create(
            &[Accelerator::Vulkan],
            &Device::Default,
            16,
            16,
            opts(DenoisingMode::Spacial),
        )
        .expect("denoiser construction failed");
        assert_eq!(d.selected_accelerator(), Accelerator::Vulkan);

        d.push_frame(&frame(16, 16)).expect("push failed");
        let out = d.recv_frame().expect("recv failed").expect("no frame");
        assert_eq!(out.len(), 16 * 16);
    }

    #[test]
    fn nl4d_algorithm_round_trips_through_the_facade() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 2 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();
        let mut d = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, opts)
            .expect("nl4d denoiser construction failed");
        assert_eq!(d.selected_accelerator(), Accelerator::Vulkan);

        // temporal_radius is 2, so a single push does not fill the
        // window yet, the same convention every temporal algorithm here
        // follows.
        d.push_frame(&frame(16, 16)).expect("push failed");
        assert!(d.recv_frame().expect("recv failed").is_none());

        let mut out = Vec::new();
        d.flush(|f| out.push(f)).expect("flush failed");
        assert_eq!(out.len(), 1, "expected exactly one output for one pushed frame");
        assert_eq!(out[0].len(), 16 * 16);
    }

    /// nl4d groups patches across neighbouring frames, so a spatial
    /// mode leaves it nothing to do. The temporal radius has one source
    /// now, `mode`, so this is the only way to ask for that.
    #[test]
    fn nl4d_rejects_a_spatial_denoising_mode() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Spacial)
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();
        let result = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, opts);

        match result {
            Err(DenoiserError::Other(e)) => assert!(
                e.to_string().contains("temporal window"),
                "unexpected error message: {e}"
            ),
            Err(other) => panic!("expected DenoiserError::Other, got {other:?}"),
            Ok(_) => panic!("expected a rejection, got Ok"),
        }
    }

    /// Both NLM algorithms only need a symmetric `2r+1` window, so
    /// `window_span` must report the same radius on both sides.
    #[test]
    fn window_span_is_symmetric_for_nlmeans() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 3 })
            .algorithm(Algorithm::Nlmeans(NlmeansOptions::default()))
            .build();
        let d = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, opts)
            .expect("denoiser construction failed");

        let span = d.window_span();
        assert_eq!(span.behind, 3, "behind should equal the temporal radius");
        assert_eq!(span.ahead, 3, "ahead should equal the temporal radius");
    }

    /// nl4d's cross-frame accumulator needs the target's own `radius`
    /// neighbourhood doubled on both sides, so both `behind` and
    /// `ahead` must come out to `2 * radius`.
    #[test]
    fn window_span_is_doubled_on_both_sides_for_nl4d() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 3 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
            .build();
        let d = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, opts)
            .expect("nl4d denoiser construction failed");

        let span = d.window_span();
        assert_eq!(span.behind, 6, "behind should equal 2 * the temporal radius");
        assert_eq!(span.ahead, 6, "ahead should equal 2 * the temporal radius");
    }

    #[test]
    fn invalid_params_surface_as_error() {
        let bad = DenoiserOptions::builder()
            .algorithm(Algorithm::Nlmeans(NlmeansOptions {
                tuning: NlmTuning {
                    strength: Some(0.0),
                    ..NlmTuning::default()
                },
                ..NlmeansOptions::default()
            }))
            .build();
        let result = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, bad);

        match result {
            Err(DenoiserError::Other(_)) => {},
            Err(other) => panic!("expected DenoiserError::Other, got {other:?}"),
            Ok(_) => panic!("expected validation error, got Ok"),
        }
    }

    #[test]
    fn tiny_frame_dimensions_surface_as_error() {
        let result = Denoiser::create(
            &[Accelerator::Vulkan],
            &Device::Default,
            2,
            2,
            opts(DenoisingMode::Spacial),
        );

        match result {
            Err(DenoiserError::Other(e)) => {
                assert!(
                    e.to_string().contains("supported minimum"),
                    "unexpected error message: {e}"
                );
            },
            Err(other) => panic!("expected DenoiserError::Other, got {other:?}"),
            Ok(_) => panic!("expected dimension validation error, got Ok"),
        }
    }

    #[test]
    fn push_after_pending_returns_queue_full() {
        let mut d = Denoiser::create(
            &[Accelerator::Vulkan],
            &Device::Default,
            16,
            16,
            opts(DenoisingMode::Spacial),
        )
        .unwrap();

        // The pipeline is two deep because the output handles are
        // double-buffered, so the first two pushes both submit. The
        // third would overwrite the oldest pending frame's output slot,
        // so it is rejected with QueueFull.
        d.push_frame(&frame(16, 16)).unwrap();
        d.push_frame(&frame(16, 16)).unwrap();
        let err = d.push_frame(&frame(16, 16)).expect_err("expected QueueFull");
        assert!(matches!(err, DenoiserError::QueueFull));

        let out = d.recv_frame().unwrap().unwrap();
        assert_eq!(out.len(), 16 * 16);

        // After draining one slot the next push must succeed.
        d.push_frame(&frame(16, 16)).expect("push after drain failed");
    }

    fn frame_filled(w: u32, h: u32, value: f32) -> Vec<f32> {
        vec![value; (w * h) as usize]
    }

    /// Pushes `n` frames of the given value, receiving along the way to
    /// keep the in-flight pipeline below `MAX_PENDING`.
    fn push_n_with_drain(d: &mut Denoiser, n: usize, value: f32, out: &mut Vec<Vec<f32>>) {
        for _ in 0..n {
            loop {
                match d.push_frame(&frame_filled(16, 16, value)) {
                    Ok(()) => break,
                    Err(DenoiserError::QueueFull) => {
                        let f = d
                            .recv_frame()
                            .expect("recv ok")
                            .expect("queue full but recv yielded none");
                        out.push(f);
                    },
                    Err(e) => panic!("unexpected push error: {e:?}"),
                }
            }
        }
    }

    #[test]
    fn flush_leaves_denoiser_reusable_spatial() {
        let mut d = Denoiser::create(
            &[Accelerator::Vulkan],
            &Device::Default,
            16,
            16,
            opts(DenoisingMode::Spacial),
        )
        .unwrap();

        let mut batch_a = Vec::new();
        push_n_with_drain(&mut d, 5, 0.25, &mut batch_a);
        d.flush(|f| batch_a.push(f)).expect("first flush failed");
        assert_eq!(batch_a.len(), 5);

        // After flush the pipeline must be empty.
        assert!(d.recv_frame().unwrap().is_none());

        let mut batch_b = Vec::new();
        push_n_with_drain(&mut d, 5, 0.75, &mut batch_b);
        d.flush(|f| batch_b.push(f)).expect("second flush failed");
        assert_eq!(batch_b.len(), 5);

        for v in batch_b.iter().flatten() {
            assert!((v - 0.75).abs() < 0.1, "batch_b carried state from batch_a: {v}");
        }
        for v in batch_a.iter().flatten() {
            assert!((v - 0.25).abs() < 0.1, "batch_a value unexpectedly drifted: {v}");
        }
    }

    #[test]
    fn flush_leaves_denoiser_reusable_temporal() {
        let mut d = Denoiser::create(
            &[Accelerator::Vulkan],
            &Device::Default,
            16,
            16,
            opts(DenoisingMode::Temporal { radius: 1 }),
        )
        .unwrap();

        let mut batch_a = Vec::new();
        push_n_with_drain(&mut d, 5, 0.25, &mut batch_a);
        d.flush(|f| batch_a.push(f)).expect("first flush failed");
        assert_eq!(batch_a.len(), 5, "expected 5 frames from first batch");

        // The temporal window must be empty after a flush, so the first
        // push of the new stream should not produce a pending frame.
        // With r=1 the window needs 3 frames before `denoise_submit`
        // fires.
        assert!(d.recv_frame().unwrap().is_none());
        d.push_frame(&frame_filled(16, 16, 0.75)).unwrap();
        assert!(
            d.recv_frame().unwrap().is_none(),
            "first push of new temporal stream should not produce output yet"
        );

        // Push 4 more frames (5 total in batch B) with drain.
        let mut batch_b = Vec::new();
        push_n_with_drain(&mut d, 4, 0.75, &mut batch_b);
        d.flush(|f| batch_b.push(f)).expect("second flush failed");
        assert_eq!(batch_b.len(), 5, "expected 5 frames from second batch");

        for v in batch_b.iter().flatten() {
            assert!((v - 0.75).abs() < 0.1, "batch_b carried state from batch_a: {v}");
        }
    }

    #[test]
    fn flush_emits_exactly_n_outputs_for_small_n() {
        // With temporal radius R=2 the window is 5 frames. Pushing fewer
        // than R+1 frames means the window never fills while pushing, so
        // flush must still emit one output per pushed frame rather than
        // R+1 of them.
        for n in 1..=5usize {
            let mut d = Denoiser::create(
                &[Accelerator::Vulkan],
                &Device::Default,
                16,
                16,
                opts(DenoisingMode::Temporal { radius: 2 }),
            )
            .unwrap();

            let mut out = Vec::new();
            push_n_with_drain(&mut d, n, 0.5, &mut out);
            d.flush(|f| out.push(f)).expect("flush failed");
            assert_eq!(
                out.len(),
                n,
                "expected {n} outputs for {n} pushes, got {}",
                out.len()
            );
        }
    }
}
