use std::collections::VecDeque;

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

use crate::accelerate::Accelerator;
use crate::collab::CollabParams;
use crate::device::Device;
use crate::nl3d::{Nl3dDenoiser, Nl3dParams};
#[cfg(test)]
use crate::nlmeans::MotionEstimation;
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    NlmDenoiser,
    NlmParams,
    Pending,
    PrefilterMode,
    hq_default_strength,
    validate_dimensions,
};
use crate::sniff::sniff_best_accelerator;

/// How a [`Denoiser`] should be set up.
///
/// Build one with `DenoiserOptions::builder()`. Every field has a
/// default, so only the parts you care about need naming.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which channels of the frame to denoise.
    #[builder(default = ChannelMode::Yuv)]
    pub channel_mode: ChannelMode,
    /// Whether to clean each frame on its own or across a temporal
    /// window.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Which algorithm variant to run.
    ///
    /// `Nlmeans` is the fast default.
    ///
    /// `NlmeansHq` matches its weighting to the noise level, which it
    /// measures per frame unless `HqParams::sigma_override` pins a fixed
    /// value.
    ///
    /// HQ also uses a different default `strength`, one that adapts to
    /// the temporal radius and the plane being denoised. See
    /// [`crate::nlmeans::hq_default_strength`].
    ///
    /// `Nl3d` runs the same HQ front end, then a collaborative-filter
    /// cleanup pass on top. It reads its front-end strength from the
    /// same table HQ does.
    #[builder(default = Algorithm::Nlmeans)]
    pub algorithm: Algorithm,
    /// Which reference image the NLM weights are computed against.
    ///
    /// `None`, the default, means no prefilter for either algorithm. Set
    /// `PrefilterMode::NlmSpatial` to opt into the NLM spatial pilot.
    pub prefilter: Option<PrefilterMode>,
    /// Whether temporal denoising follows motion between frames.
    ///
    /// `None` turns motion compensation off. `Mvtools` warps temporal
    /// neighbours into line with the centre frame before the NLM
    /// weighting runs.
    ///
    /// Only has an effect when `mode` is `Temporal { .. }`.
    #[builder(default = MotionCompensationMode::None)]
    pub motion_compensation: MotionCompensationMode,
    /// Overrides for the NLM search radius, patch radius, strength, and
    /// self-weight.
    ///
    /// `None` uses the defaults baked into [`NlmParams`].
    pub nlm: Option<NlmTuning>,
}

/// Which denoising algorithm variant to run.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Algorithm {
    /// The fast NLMeans path.
    Nlmeans,
    /// Quality-focused NLMeans with noise-calibrated weighting.
    NlmeansHq(HqParams),
    /// NLMeans followed by a collaborative-filter cleanup pass.
    ///
    /// Always runs the HQ front end, because the collaborative stage
    /// needs the front end's estimated noise sigma to shrink its own
    /// coefficients by.
    Nl3d(Nl3dOptions),
}

/// Tuning for the [`Algorithm::Nl3d`] cascade.
///
/// `hq` configures the NLMeans front end the same way
/// [`Algorithm::NlmeansHq`] does. The remaining fields tune the
/// collaborative filter that runs second.
///
/// `residual_sigma_scale` used to be one shared value across both
/// planes, and everything below this note describes the sweeps that
/// were run while that was still true. It is now resolved per plane,
/// `None` reading through [`nl3d_default_residual_sigma_scale`], and
/// luma is the only plane whose value those sweeps still describe.
/// Chroma's own value and its very different evidence are documented
/// on that function directly.
///
/// # Where the calibrated defaults come from
///
/// `front_strength_scale`, `residual_sigma_scale`, and `lambda_ht` were
/// first swept together on `clean-1080p.mkv` at temporal radius 2, noise
/// levels 4, 6, and 8, one dial at a time with the others pinned, then
/// checked for interaction with a small joint sweep around the chosen
/// point. `k_max` and `tau_match` were left at their pre-existing
/// values, not part of this sweep.
///
/// `residual_sigma_scale` went first, because a separate test
/// (`residual_ratio_matches_measured_residual_within_the_known_independence_gap`
/// in `src/nlmeans/tests/weight_ratio.rs`) had already measured the
/// front end's true residual noise at about 1.92x what the analytic
/// formula predicts. The sweep covered 1.0, 1.4, 1.9, 2.4, and 3.0,
/// bracketed on both sides at all three noise levels, and 1.9 won by
/// having the best worst-case luma XPSNR across the three levels,
/// matching the independent measurement closely. SSIM agreed at every
/// point tested.
///
/// `front_strength_scale` went next, at `residual_sigma_scale` fixed to
/// 1.9. Luma XPSNR forms a broad, nearly flat plateau from 0.3 to 0.6 at
/// every noise level, clearly ahead of 0.8 and 1.0, so the default moved
/// from a 0.8 guess to 0.5, the plateau's centre. SSIM does not disagree
/// anywhere on the plateau.
///
/// `lambda_ht` went last, at the two values above. Luma XPSNR favors 2.0
/// over 2.7 at the two lighter noise levels but the two tie at the
/// heaviest, while luma SSIM favors 2.7 at the two heavier noise levels.
/// 2.7 is kept because it wins the noise level where XPSNR itself is
/// tied and SSIM is not, so lowering it would only help the noise levels
/// that already have the most headroom. This is the one dial where the
/// two metrics point different directions depending on noise level.
///
/// The joint check varied `front_strength_scale` and `lambda_ht`
/// together, one step either side of the chosen point, and found no
/// combination that beat it by more than run-to-run noise, so no
/// interaction large enough to move the calibration was found.
///
/// Radius 4 was checked afterward, with the same three values pinned
/// rather than swept again. It shows the same or larger improvement
/// over the radius-2 front end at every noise level, so the radius-2
/// calibration carries over.
///
/// # Recalibration after the patch-grouping fix
///
/// The collaborative filter's group-ranking kernel had a bug that
/// clamped patch distances at zero before ranking them. Once the noise
/// floor exceeded a candidate's raw distance, which it did for nearly
/// every candidate at `residual_sigma_scale = 1.9`, every candidate tied
/// at exactly zero and the group filled with whichever patches the scan
/// visited first rather than the most similar ones. Every parameter
/// above was originally chosen while this bug was live, so all of them
/// were re-measured once it was fixed, plus `rho`
/// ([`crate::collab::CollabParams::rho`]), a per-frequency noise-shaping
/// correlation that was, at the time, forced on unconditionally for
/// nl3d.
///
/// This round changed the objective on purpose. The original sweep
/// above scored luma XPSNR and SSIM against a clean reference with
/// synthetic noise added, and that reference carries no fine grain or
/// texture of its own, so a configuration that destroys real texture
/// scores no worse than one that only removes noise. The recalibration
/// instead uses the high-frequency energy ratio as its primary target,
/// Laplacian variance in a real-footage face crop divided by the same
/// measurement on the clean source, at a value of 1.0. Below 1.0 means
/// real texture was smoothed away. Above 1.0 means noise was left
/// behind. XPSNR and SSIM are still measured, but only as a guard
/// against a configuration that pushes the ratio far from 1.0 while
/// scoring well, not as the value being maximised.
///
/// `rho` and `residual_sigma_scale` were swept together, since both
/// change how much the collaborative stage shrinks and can trade
/// against each other. `rho` was tried at 0.0, 0.4, 0.65, and 0.8. 0.0
/// is shaping off. 0.8 is the flat-content correlation `rho_window` used
/// to force. 0.65 is the measured correlation on real fine texture.
/// Each `rho` value was crossed with `residual_sigma_scale` at 1.0, 1.4,
/// and 1.9, all twelve combinations, at all three noise levels, on the
/// same `clean-1080p.mkv` source used for the metric-only sweep above.
/// `rho = 0.0` beat every other `rho` value at every
/// `residual_sigma_scale` and every noise level tested, on both the
/// high-frequency ratio and on XPSNR/SSIM simultaneously, with no
/// exception. Raising `rho` monotonically pushed the ratio further above
/// 1.0 (more noise kept) and lowered XPSNR/SSIM at the same time, rather
/// than trading one for the other. Shaping was therefore dropped from
/// nl3d's default entirely, not just tuned down. See
/// `Nl3dDenoiser::new`'s doc comment for where that is done and the
/// measured before/after numbers.
///
/// With `rho` settled at `0.0`, `residual_sigma_scale` was checked
/// across 1.0, 1.4, 1.9, 2.4, and 3.0. 1.9 remained the best value,
/// unchanged from the pre-fix sweep above. It has the best worst-case
/// luma XPSNR and SSIM across the three noise levels, and its
/// high-frequency ratio (0.85 to 1.04 across both faces and all three
/// noise levels) is the closest of any tested value to the 1.0 target.
/// It is bracketed on both sides. 1.4 undershoots the noise removal,
/// with ratios running 0.91 to 1.20. 2.4 and 3.0 overshoot it, with
/// ratios drifting down to 0.82 to 0.88, and XPSNR falls off past 1.9
/// too. `front_strength_scale` and `lambda_ht` were left at their
/// existing values. The `rho` / `residual_sigma_scale` grid already
/// showed no headroom to chase, so neither was re-swept this round.
///
/// The collaborative filter's grouping stage was also checked
/// separately for whether it can tell similar patches from dissimilar
/// ones, since that is a distinct property from how good the final
/// image looks. At the chosen `residual_sigma_scale = 1.9`, admitted
/// patches on flat content average 0.19x the clean-domain distance of a
/// random draw from the same search window, and on fine texture 0.10x,
/// both well under 1.0, meaning admission is finding genuinely closer
/// patches rather than noise-fooled ones. `rho` does not affect
/// grouping. Grouping ranks raw pixel distances, before the DCT
/// transform that `rho` shapes ever runs.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl3dOptions {
    /// The front end's HQ parameters.
    pub hq: HqParams,
    /// A multiplier applied to the front end's calibrated strength
    /// before it runs.
    ///
    /// Sitting below 1.0 leaves the front end filtering gently, so it
    /// leaves structured residual noise behind for the collaborative
    /// stage to remove through grouped shrinkage instead. Calibrated to
    /// 0.5, see the struct-level docs for how.
    pub front_strength_scale: f32,
    /// Largest collaborative stack size, a power of two up to 8.
    /// Defaults to 8.
    pub k_max: u32,
    /// Admission threshold multiplier on the patch noise floor.
    /// Defaults to 3.0.
    pub tau_match: f32,
    /// Hard-threshold multiplier for the collaborative filter's first
    /// stage. Calibrated to 2.7, see the struct-level docs for how.
    pub lambda_ht: f32,
    /// A multiplier on the analytic residual sigma the collaborative
    /// stage shrinks with, per plane.
    ///
    /// `None` resolves through [`nl3d_default_residual_sigma_scale`],
    /// which returns a different calibrated value for luma than for
    /// chroma. Read that function's docs before relying on either
    /// number, since the two rest on very different evidence.
    pub residual_sigma_scale: Option<f32>,
}

impl Default for Nl3dOptions {
    fn default() -> Self {
        let collab = CollabParams::default();
        Self {
            hq: HqParams::default(),
            front_strength_scale: 0.5,
            k_max: collab.k_max,
            tau_match: collab.tau_match,
            lambda_ht: collab.lambda_ht,
            // Resolved per plane by `nl3d_default_residual_sigma_scale`
            // at construction time, once the plane being denoised is
            // known.
            residual_sigma_scale: None,
        }
    }
}

/// The calibrated default `residual_sigma_scale` for the nl3d
/// collaborative stage, per plane.
///
/// # Where the numbers come from
///
/// The two values rest on very different evidence. Luma's is
/// metric-driven. Chroma's is one human's visual judgement on one
/// clip. Treat them accordingly.
///
/// ## Luma, 1.9
///
/// Chosen by a quality-harness sweep that scored a clean reference
/// with synthetic noise added, at three noise levels, picking the
/// value whose worst-case luma XPSNR held up best. Re-swept after a
/// patch-grouping defect in the collaborative stage was fixed, and it
/// held at 1.9 again. A later sweep at finer granularity around 1.9
/// found every adjacent variant scoring within run-to-run noise of
/// each other, differing by under half a code level, below visual
/// threshold. See the struct-level docs on [`Nl3dOptions`] for the
/// full sweep history this value comes from.
///
/// ## Chroma, 8.0
///
/// Chosen by a human reviewing stills and residuals on one clip,
/// `data/brick_source.mkv`, 53 largely static frames. A metric sweep
/// first showed `residual_sigma_scale` has real authority over chroma
/// once it is decoupled from luma, since luma alone had been
/// saturating the shared value and leaving chroma barely filtered (86
/// to 90 percent of its high-frequency content retained, against
/// luma's 45 percent). A human then judged 8.0 to look good and to
/// retain more apparent detail than earlier, less-filtered builds, on
/// the theory that chroma noise had been masking that detail.
///
/// Its upper bound is evidenced directly. At 24.0 the chroma-only
/// residual on one frame renders a character's hair, face, and hands,
/// plus nearby sign text, as legible line art, unambiguous chroma
/// detail loss. Its lower bound is not swept, only that 1.9 left
/// chroma almost unfiltered.
///
/// This value has not been checked on footage with real motion. It
/// rests on one clip and one reviewer, not a metric optimum, and
/// should not be read as being on the same footing as luma's value
/// above.
///
/// # `ChannelMode::Yuv`
///
/// Reads the luma value, the same "a fused pass is dominated by luma"
/// assumption [`crate::nlmeans::hq_default_strength`] makes for its
/// own Yuv case. Yuv was not part of either value's evidence.
pub fn nl3d_default_residual_sigma_scale(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => 1.9,
        ChannelMode::Chroma => 8.0,
    }
}

/// Resolves `Nl3dOptions.residual_sigma_scale` for one plane, falling
/// back to [`nl3d_default_residual_sigma_scale`] when the caller left
/// it unset.
fn resolve_residual_sigma_scale(opts: &Nl3dOptions, channels: ChannelMode) -> f32 {
    opts.residual_sigma_scale
        .unwrap_or_else(|| nl3d_default_residual_sigma_scale(channels))
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
#[derive(Debug, Copy, Clone)]
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

        // An explicit `strength` always wins, whether it came straight
        // from `NlmTuning` or from a per-plane override the caller
        // already folded in.
        //
        // Otherwise the default depends on the algorithm, and for HQ
        // also on `auto_strength`. With auto-strength on, HQ reads
        // `strength` as a multiplier on the measured noise level, so it
        // needs its own calibrated default rather than the fast path's
        // absolute FFmpeg-style one. That calibrated default also varies
        // with the temporal radius and with the plane `channel_mode`
        // names, because each per-plane `Denoiser` carries its own
        // channel mode.
        //
        // With auto-strength off, HQ reads `strength` as an absolute
        // value just like the fast path, so it falls back to the same
        // absolute default.
        // `Nl3d` shares the HQ front end, so it reads `strength` from the
        // same calibrated table `NlmeansHq` does.
        let explicit_strength = self.nlm.and_then(|t| t.strength);
        let strength = explicit_strength.unwrap_or(match self.algorithm {
            // The calibrated table is a multiplier on the measured
            // sigma, so it only applies when `effective_strength_with`
            // will read `strength` that way.
            Algorithm::NlmeansHq(hq) if hq.auto_strength => {
                hq_default_strength(self.channel_mode, temporal_radius)
            },
            Algorithm::Nl3d(opts) if opts.hq.auto_strength => {
                hq_default_strength(self.channel_mode, temporal_radius)
            },
            Algorithm::NlmeansHq(_) | Algorithm::Nlmeans | Algorithm::Nl3d(_) => {
                NlmParams::default().strength
            },
        });

        let mut params = NlmParams {
            channels: self.channel_mode,
            prefilter: self.prefilter.unwrap_or(PrefilterMode::None),
            motion_compensation: self.motion_compensation,
            temporal_radius,
            hq: match self.algorithm {
                Algorithm::Nlmeans => None,
                Algorithm::NlmeansHq(hq) => Some(hq),
                Algorithm::Nl3d(opts) => Some(opts.hq),
            },
            strength,
            // The collaborative stage measures how much noise the front
            // end left behind, which needs the front end's own weight-
            // squared accumulator turned on.
            track_weight_sq: matches!(self.algorithm, Algorithm::Nl3d(_)),
            ..NlmParams::default()
        };
        if let Some(t) = self.nlm {
            if let Some(v) = t.search_radius {
                params.search_radius = v;
            }
            if let Some(v) = t.patch_radius {
                params.patch_radius = v;
            }
            if let Some(v) = t.self_weight {
                params.self_weight = v;
            }
        }
        params
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
// Both variants are boxed so neither denoiser's size dictates every
// `Engine` value's size, whichever variant is actually held.
enum Engine<R: Runtime> {
    Nlm(Box<NlmDenoiser<R>>),
    Nl3d(Box<Nl3dDenoiser<R>>),
}

impl<R: Runtime> Engine<R> {
    fn push_frame(&mut self, frame: &[f32]) {
        match self {
            Self::Nlm(d) => d.push_frame(frame),
            Self::Nl3d(d) => d.push_frame(frame),
        }
    }

    fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        match self {
            Self::Nlm(d) => d.denoise_submit(),
            Self::Nl3d(d) => d.denoise_submit(),
        }
    }

    fn flush(&mut self, sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        match self {
            Self::Nlm(d) => d.flush(sink),
            Self::Nl3d(d) => d.flush(sink),
        }
    }
}

/// Builds whichever [`Engine`] `algorithm` calls for.
///
/// `Algorithm::Nl3d` also carries the collaborative filter's own tuning,
/// which is not part of `NlmParams`, so it is read from `algorithm`
/// directly rather than from `params`. This is also where an unset
/// `residual_sigma_scale` picks up its calibrated per-plane default
/// (`resolve_residual_sigma_scale`), the same way `to_nlm_params`
/// resolves HQ's calibrated `strength`, since this is the first point
/// construction has both `opts` and `params.channels` together.
fn build_engine<R: Runtime>(
    client: &ComputeClient<R>,
    algorithm: &Algorithm,
    params: NlmParams,
    width: u32,
    height: u32,
) -> Result<Engine<R>, DenoiserError> {
    match algorithm {
        Algorithm::Nl3d(opts) => {
            let collab = CollabParams {
                channels: params.channels,
                k_max: opts.k_max,
                tau_match: opts.tau_match,
                lambda_ht: opts.lambda_ht,
                ..CollabParams::default()
            };
            let residual_sigma_scale = resolve_residual_sigma_scale(opts, params.channels);
            let nl3d_params = Nl3dParams {
                nlm: params,
                front_strength_scale: opts.front_strength_scale,
                collab,
                residual_sigma_scale,
            };
            Ok(Engine::Nl3d(Box::new(Nl3dDenoiser::new(
                client,
                nl3d_params,
                width,
                height,
            )?)))
        },
        Algorithm::Nlmeans | Algorithm::NlmeansHq(_) => Ok(Engine::Nlm(Box::new(NlmDenoiser::new(
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
    #[cfg(feature = "cpu")]
    Cpu(Engine<cubecl::cpu::CpuRuntime>),
}

enum BackendPending {
    #[cfg(feature = "cuda")]
    Cuda(Pending<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(Pending<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(Pending<cubecl::wgpu::WgpuRuntime>),
    #[cfg(feature = "cpu")]
    Cpu(Pending<cubecl::cpu::CpuRuntime>),
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
            #[cfg(feature = "cpu")]
            Self::Cpu(p) => p.wait(),
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
/// use av_denoise::accelerate::Accelerator;
/// use av_denoise::{ChannelMode, Denoiser, DenoiserOptions, DenoisingMode, Device};
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
    /// Callers using a `search_radius` above 4 should set
    /// `RUST_MIN_STACK` to at least 16 MiB before any cubecl thread
    /// spawns, usually right at the top of `main`.
    ///
    /// ```no_run
    /// if std::env::var_os("RUST_MIN_STACK").is_none() {
    ///     // SAFETY: single-threaded at startup.
    ///     unsafe { std::env::set_var("RUST_MIN_STACK", "16777216") };
    /// }
    /// ```
    pub fn create(
        accelerators: &[Accelerator],
        device: &Device,
        width: u32,
        height: u32,
        options: DenoiserOptions,
    ) -> Result<Self, DenoiserError> {
        let accelerator =
            sniff_best_accelerator(accelerators).ok_or(DenoiserError::NoAcceleratorAvailable)?;

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
            #[cfg(feature = "cpu")]
            Backend::Cpu(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Cpu(p));
                }
            },
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
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

    /// Collects the in-flight denoise if one is ready.
    ///
    /// This can still block for a moment while the runtime confirms the
    /// readback has landed. When the kernels have already finished the
    /// wait is effectively nothing.
    pub fn try_recv_frame(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
        self.recv_frame()
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
            #[cfg(feature = "cpu")]
            Backend::Cpu(d) => d.flush(|slice| {
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
        #[cfg(feature = "cpu")]
        Accelerator::Cpu => {
            let dev = device.to_cpu()?;
            let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Cpu(build_engine(
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

    #[test]
    fn nl3d_default_residual_sigma_scale_differs_between_luma_and_chroma() {
        let luma = nl3d_default_residual_sigma_scale(ChannelMode::Luma);
        let chroma = nl3d_default_residual_sigma_scale(ChannelMode::Chroma);

        assert!((luma - 1.9).abs() < f32::EPSILON);
        assert!((chroma - 8.0).abs() < f32::EPSILON);
        assert!(
            (chroma - luma).abs() > 1.0,
            "luma ({luma}) and chroma ({chroma}) must resolve to clearly different defaults"
        );
    }

    #[test]
    fn nl3d_default_residual_sigma_scale_yuv_reads_the_luma_value() {
        let yuv = nl3d_default_residual_sigma_scale(ChannelMode::Yuv);
        let luma = nl3d_default_residual_sigma_scale(ChannelMode::Luma);

        assert!((yuv - luma).abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_residual_sigma_scale_unset_uses_the_per_plane_default() {
        let opts = Nl3dOptions::default();

        let luma = resolve_residual_sigma_scale(&opts, ChannelMode::Luma);
        let chroma = resolve_residual_sigma_scale(&opts, ChannelMode::Chroma);

        assert!((luma - 1.9).abs() < f32::EPSILON, "got {luma}");
        assert!((chroma - 8.0).abs() < f32::EPSILON, "got {chroma}");
    }

    #[test]
    fn resolve_residual_sigma_scale_explicit_value_overrides_every_plane() {
        let opts = Nl3dOptions {
            residual_sigma_scale: Some(3.3),
            ..Nl3dOptions::default()
        };

        for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
            let got = resolve_residual_sigma_scale(&opts, channels);
            assert!(
                (got - 3.3).abs() < f32::EPSILON,
                "channels {channels:?} got {got}"
            );
        }
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
            .prefilter(PrefilterMode::Bilateral {
                sigma_s: 3.0,
                sigma_r: 0.02,
            })
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::Bilateral { .. }));
    }

    #[test]
    fn hq_unset_prefilter_defaults_to_none() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::NlmeansHq(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale: 1.0,
            }))
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::None));
    }

    #[test]
    fn hq_explicit_none_prefilter_is_respected() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::NlmeansHq(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale: 1.0,
            }))
            .prefilter(PrefilterMode::None)
            .build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::None));
    }

    #[test]
    fn fast_unset_prefilter_defaults_to_none() {
        let opts = DenoiserOptions::builder().algorithm(Algorithm::Nlmeans).build();
        let params = opts.to_nlm_params();

        assert!(matches!(params.prefilter, PrefilterMode::None));
    }

    #[test]
    fn hq_unset_strength_defaults_to_hq_default_strength() {
        // Default channel_mode is Yuv, default mode is Spacial (radius 0).
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::NlmeansHq(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale: 1.0,
            }))
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
            .algorithm(Algorithm::NlmeansHq(HqParams {
                auto_strength: false,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale: 1.0,
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
            .algorithm(Algorithm::NlmeansHq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_chroma_r4_uses_measured_table_value() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Chroma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::NlmeansHq(HqParams::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn hq_yuv_r8_uses_measured_table_value() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Yuv)
            .mode(DenoisingMode::Temporal { radius: 8 })
            .algorithm(Algorithm::NlmeansHq(HqParams::default()))
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
                .algorithm(Algorithm::NlmeansHq(HqParams::default()))
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
                .algorithm(Algorithm::NlmeansHq(HqParams::default()))
                .nlm(NlmTuning {
                    search_radius: None,
                    patch_radius: None,
                    strength: Some(0.99),
                    self_weight: None,
                })
                .build();
            let params = opts.to_nlm_params();

            assert!(
                (params.strength - 0.99).abs() < f32::EPSILON,
                "for channels {channels:?} the explicit strength was overridden by the table"
            );
        }
    }

    #[test]
    fn hq_explicit_strength_is_respected() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::NlmeansHq(HqParams {
                auto_strength: true,
                noise_floor: true,
                sigma_override: None,
                temporal_confidence: true,
                thsad_scale: 1.0,
                sigma_scale: 1.0,
            }))
            .nlm(NlmTuning {
                search_radius: None,
                patch_radius: None,
                strength: Some(1.0),
                self_weight: None,
            })
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn fast_unset_strength_defaults_to_legacy_default() {
        let opts = DenoiserOptions::builder().algorithm(Algorithm::Nlmeans).build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn nl3d_sets_track_weight_sq() {
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nl3d(Nl3dOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        assert!(
            params.track_weight_sq,
            "nl3d needs the front end's weight-squared sums tracked"
        );
    }

    #[test]
    fn nl3d_hq_field_is_populated_from_nl3d_options() {
        let hq = HqParams {
            auto_strength: false,
            ..HqParams::default()
        };
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nl3d(Nl3dOptions {
                hq,
                ..Nl3dOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(params.hq, Some(hq));
    }

    #[test]
    fn nl3d_uses_the_hq_strength_table_when_auto_strength_is_on() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::Nl3d(Nl3dOptions::default()))
            .build();
        let params = opts.to_nlm_params();

        let expected = hq_default_strength(ChannelMode::Luma, 4);
        assert!((params.strength - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn nl3d_explicit_strength_wins_over_the_table() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::Nl3d(Nl3dOptions::default()))
            .nlm(NlmTuning {
                search_radius: None,
                patch_radius: None,
                strength: Some(0.99),
                self_weight: None,
            })
            .build();
        let params = opts.to_nlm_params();

        assert!((params.strength - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn motion_compensation_passthrough() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 1 })
            .motion_compensation(MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Direct,
            })
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
            .nlm(NlmTuning {
                search_radius: Some(7),
                patch_radius: None,
                strength: Some(2.5),
                self_weight: None,
            })
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
    fn nl3d_algorithm_round_trips_through_the_facade() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Spacial)
            .algorithm(Algorithm::Nl3d(Nl3dOptions::default()))
            .build();
        let mut d = Denoiser::create(&[Accelerator::Vulkan], &Device::Default, 16, 16, opts)
            .expect("nl3d denoiser construction failed");
        assert_eq!(d.selected_accelerator(), Accelerator::Vulkan);

        d.push_frame(&frame(16, 16)).expect("push failed");
        let out = d.recv_frame().expect("recv failed").expect("no frame");
        assert_eq!(out.len(), 16 * 16);
    }

    #[test]
    fn invalid_params_surface_as_error() {
        let bad = DenoiserOptions::builder()
            .nlm(NlmTuning {
                search_radius: None,
                patch_radius: None,
                strength: Some(0.0),
                self_weight: None,
            })
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

// A smoke test that catches outright breakage on the CPU backend.
#[cfg(all(test, feature = "cpu"))]
mod cpu_smoke_tests {
    use super::*;

    #[test]
    fn cpu_backend_denoises_a_frame() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Spacial)
            .build();
        let mut d = Denoiser::create(&[Accelerator::Cpu], &Device::Default, 16, 16, opts)
            .expect("denoiser construction failed");
        assert_eq!(d.selected_accelerator(), Accelerator::Cpu);

        d.push_frame(&vec![0.5f32; 16 * 16]).expect("push failed");
        let out = d.recv_frame().expect("recv failed").expect("no frame");
        assert_eq!(out.len(), 16 * 16);
    }
}
