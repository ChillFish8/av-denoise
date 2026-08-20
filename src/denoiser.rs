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
    /// `Nl4d` drives the same HQ front end for its machinery, the frame
    /// ring, the motion field, and the noise estimate, rather than
    /// filtering with it first. It reads its front-end strength from
    /// the same table HQ does.
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
    /// Groups 8x8 patches across the motion-compensated temporal window
    /// itself, rather than filtering with NLM first and grouping within
    /// one frame afterward.
    ///
    /// Always runs the HQ front end with an active motion-compensated
    /// ring and temporal confidence on, because the grouping kernel
    /// reads the motion field and confidence scores that front end
    /// builds.
    Nl4d(Nl4dOptions),
}

/// Tuning for the [`Algorithm::Nl4d`] cascade.
///
/// `hq` configures the front end the same way [`Algorithm::NlmeansHq`]
/// does. The remaining fields tune the temporal grouping and shrinkage
/// that run on top of it, mirroring [`crate::nl4d::Nl4dParams`] one for
/// one.
///
/// `lambda_ht` has a per-plane calibrated default. `None` resolves
/// through [`nl4d_default_lambda_ht`] once the plane being denoised is
/// known. A caller wanting luma and chroma to shrink by different
/// amounts explicitly sets `--luma-lambda-ht` and `--chroma-lambda-ht`
/// on the CLI, resolved per plane in `CliOptions::algorithm_for`
/// (`src/bin/ingest.rs`).
///
/// `temporal_radius`, `refine`, `spatial_radius`, and `c_min` have not
/// been calibrated by a sweep. Their defaults are carried straight
/// over from [`crate::nl4d::Nl4dParams::default`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl4dOptions {
    /// The front end's HQ parameters.
    pub hq: HqParams,
    /// How many frames on each side of the centre frame the temporal
    /// search reaches into, in `1..=8`. Defaults to 2.
    ///
    /// This has to equal the temporal radius [`DenoisingMode`] resolves
    /// to for the same [`DenoiserOptions`], because the front end's
    /// frame ring and the outer [`Denoiser`]'s own push/flush bookkeeping
    /// both depend on that radius agreeing everywhere it is read.
    pub temporal_radius: u32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position, in `1..=4`. Defaults to 2.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the centre
    /// frame, in `1..=16`. Defaults to 9.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    ///
    /// `None` resolves through [`nl4d_default_lambda_ht`], which
    /// returns a different calibrated value for luma than for chroma.
    /// Read that function's docs before relying on either number,
    /// since the two rest on very different evidence.
    pub lambda_ht: Option<f32>,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Defaults to 0.05. Only
    /// affects how much compute a submit spends, never which candidates
    /// are admitted once they are scored.
    pub c_min: f32,
    /// Whether a temporal member's mismatch variance reaches the
    /// hard-threshold shrinkage at all. Defaults to `true`. See
    /// [`crate::nl4d::Nl4dParams::confidence_variance`].
    pub confidence_variance: bool,
}

impl Default for Nl4dOptions {
    fn default() -> Self {
        let defaults = Nl4dParams::default();
        Self {
            hq: HqParams::default(),
            temporal_radius: defaults.temporal_radius,
            refine: defaults.refine,
            spatial_radius: defaults.spatial_radius,
            // Resolved per plane by `nl4d_default_lambda_ht` at
            // construction time, once the plane being denoised is
            // known.
            lambda_ht: None,
            c_min: defaults.c_min,
            confidence_variance: defaults.confidence_variance,
        }
    }
}

/// The calibrated default `lambda_ht` for the nl4d temporal grouping's
/// hard-threshold stage, per plane.
///
/// # Luma and `ChannelMode::Yuv`, 5.3
///
/// Chosen by a human from a rendered ladder on real brick grain
/// (`data/nl4d_calibration/README.md`), luma `lambda_ht` pinned across
/// 5.0, 5.3, 5.6, 7.0, 8.5, 10.0, and 12.0 at temporal radius 2, each
/// rung measured against a frame-22 Laplacian high-frequency-energy
/// ratio and checked by eye against crops and 16x-amplified removed
/// residuals. Every rung from 5.6 up removes noticeably more
/// high-frequency energy than the `bm3dhip` reference does at its own
/// matched strength, and 5.6 and 7.0 remove more of it than 5.3 for
/// only a fairly small further loss of detail. The reviewer picked 5.3
/// anyway, recorded verbatim in that document:
///
/// > "09_nl4d_lam5p3 Is probably the best for a default, although 5.6
/// > and 7.0 reduce the noise more for faily minimal detail loss, we
/// > should opt on the side of detail retention given the additional
/// > noise reduction is not great enough to warrent it."
///
/// The numbers alone do not carry that reasoning. A metric-only pick
/// could just as well have landed on 5.6 or 7.0, so the deliberate
/// bias toward detail retention is the reason 5.3 is the value here
/// rather than either of those. `ChannelMode::Yuv` reads the same
/// value, the same "a fused pass is dominated by luma" assumption
/// [`crate::nlmeans::hq_default_strength`] makes for its own Yuv case.
/// Yuv was not part of the ladder's own evidence.
///
/// The ladder behind 5.3 was rendered on a build that scattered only
/// centre-frame members, so it compensated for the lost cross-frame
/// averaging with a harsher `lambda_ht` than a filter aggregating
/// every member would need. Cross-frame aggregation has since landed,
/// and every member now writes back into the frame it was matched
/// from, so that compensation no longer applies. The value was left at
/// 5.3 because a human checked it on the cross-frame build and
/// confirmed it still held, recorded verbatim:
///
/// > "no, I confirmed manually that 5.3 was fine."
///
/// That is a confirmation of this one value on this one build, not a
/// re-run of the ladder and not a fresh calibration.
///
/// # Chroma, 4.2
///
/// The prior value here was 3.6, chosen on metric optima alone (see
/// `data/nl4d_chroma_calibration/README.md`) and never checked by eye.
///
/// It was re-swept on the cross-frame-aggregation build with luma
/// pinned at 5.3, across a nine-rung ladder deliberately extended to
/// 8.5 so it bracketed and exceeded luma parity, since the assumption
/// that chroma wants less aggression than luma had never itself been
/// checked against a residual look. A human judged the frame-22 crops
/// and the 16x-amplified U-plane removed-residuals from that ladder
/// and picked 4.2.
///
/// Two sets of numbers accompanied that judgement without deciding
/// it. Encoded size under a realistic delivery encode saves 57.21
/// percent against the undenoised source at 4.2, against 56.87
/// percent at the prior 3.6 and 57.73 percent at luma parity 5.3.
/// U-plane residual structure correlation rises smoothly across the
/// ladder with no threshold, 0.341 at 3.6, 0.358 at 4.2, and 0.394 at
/// 5.3, so there is no cliff in that number either and the choice is
/// a continuum trade a human still has to make.
///
/// The full evidence, every rendered arm, the residual crops, and the
/// numbers above, lives in `data/nl4d_chroma_recal/README.md`.
pub fn nl4d_default_lambda_ht(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => 5.3,
        ChannelMode::Chroma => 4.2,
    }
}

/// Resolves `Nl4dOptions.lambda_ht` for one plane, falling back to
/// [`nl4d_default_lambda_ht`] when the caller left it unset.
fn resolve_lambda_ht(opts: &Nl4dOptions, channels: ChannelMode) -> f32 {
    opts.lambda_ht.unwrap_or_else(|| nl4d_default_lambda_ht(channels))
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
        // `Nl4d` shares the HQ front end, so it reads `strength` from
        // the same calibrated table `NlmeansHq` does.
        let explicit_strength = self.nlm.and_then(|t| t.strength);
        let strength = explicit_strength.unwrap_or(match self.algorithm {
            // The calibrated table is a multiplier on the measured
            // sigma, so it only applies when `effective_strength_with`
            // will read `strength` that way.
            Algorithm::NlmeansHq(hq) if hq.auto_strength => {
                hq_default_strength(self.channel_mode, temporal_radius)
            },
            Algorithm::Nl4d(opts) if opts.hq.auto_strength => {
                hq_default_strength(self.channel_mode, temporal_radius)
            },
            Algorithm::NlmeansHq(_) | Algorithm::Nlmeans | Algorithm::Nl4d(_) => {
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
                Algorithm::Nl4d(opts) => Some(opts.hq),
            },
            strength,
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
    Nl4d(Box<Nl4dDenoiser<R>>),
}

impl<R: Runtime> Engine<R> {
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
}

/// Builds whichever [`Engine`] `algorithm` calls for.
///
/// `Algorithm::Nl4d` carries its own cascade tuning, which is not part
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
            let lambda_ht = resolve_lambda_ht(opts, params.channels);
            let nl4d_params = Nl4dParams {
                nlm: params,
                temporal_radius: opts.temporal_radius,
                refine: opts.refine,
                spatial_radius: opts.spatial_radius,
                lambda_ht,
                c_min: opts.c_min,
                confidence_variance: opts.confidence_variance,
            };
            let denoiser = Nl4dDenoiser::new(client, nl4d_params, width, height)
                .map_err(|e| DenoiserError::Other(anyhow::anyhow!(e)))?;
            Ok(Engine::Nl4d(Box::new(denoiser)))
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
        // `Nl4dDenoiser` forces its own `temporal_radius` onto the front
        // end's ring regardless of what `params.temporal_radius` says
        // (see `Nl4dParams::validate`'s doc comment), so the outer
        // push/flush bookkeeping below has to read the value nl4d will
        // actually run with, not the one `to_nlm_params` derived from
        // `mode`. Every other algorithm still runs at exactly
        // `params.temporal_radius`.
        let temporal_radius = match &options.algorithm {
            Algorithm::Nl4d(opts) => opts.temporal_radius,
            _ => params.temporal_radius,
        };
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

    #[test]
    fn nl4d_default_lambda_ht_differs_between_luma_and_chroma() {
        let luma = nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma = nl4d_default_lambda_ht(ChannelMode::Chroma);

        assert!((luma - 5.3).abs() < f32::EPSILON);
        assert!((chroma - 4.2).abs() < f32::EPSILON);
        assert!(
            (chroma - luma).abs() > f32::EPSILON,
            "luma ({luma}) and chroma ({chroma}) must resolve to different defaults"
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

        let luma = resolve_lambda_ht(&opts, ChannelMode::Luma);
        let chroma = resolve_lambda_ht(&opts, ChannelMode::Chroma);

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
            let got = resolve_lambda_ht(&opts, channels);
            assert!(
                (got - 4.4).abs() < f32::EPSILON,
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
    fn nl4d_options_default_matches_nl4d_params_default() {
        let opts = Nl4dOptions::default();
        let params = crate::nl4d::Nl4dParams::default();

        assert_eq!(opts.temporal_radius, params.temporal_radius);
        assert_eq!(opts.refine, params.refine);
        assert_eq!(opts.spatial_radius, params.spatial_radius);
        assert!((opts.c_min - params.c_min).abs() < f32::EPSILON);
        // `lambda_ht` is no longer compared here. `opts.lambda_ht` stays
        // `None`, deferred to `nl4d_default_lambda_ht` once the plane is
        // known (`resolve_lambda_ht_unset_uses_the_per_plane_default`
        // above), while `params.lambda_ht` is `Nl4dParams`'s own
        // independent concrete default, which mirrors the Luma/Yuv
        // calibrated value.
        assert_eq!(opts.lambda_ht, None);
        assert!((params.lambda_ht - nl4d_default_lambda_ht(ChannelMode::Yuv)).abs() < f32::EPSILON);
    }

    #[test]
    fn nl4d_hq_field_is_populated_from_nl4d_options() {
        let hq = HqParams {
            auto_strength: false,
            ..HqParams::default()
        };
        let opts = DenoiserOptions::builder()
            .algorithm(Algorithm::Nl4d(Nl4dOptions {
                hq,
                ..Nl4dOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        assert_eq!(params.hq, Some(hq));
    }

    #[test]
    fn nl4d_uses_the_hq_strength_table_when_auto_strength_is_on() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions {
                temporal_radius: 4,
                ..Nl4dOptions::default()
            }))
            .build();
        let params = opts.to_nlm_params();

        let expected = hq_default_strength(ChannelMode::Luma, 4);
        assert!((params.strength - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn nl4d_explicit_strength_wins_over_the_table() {
        let opts = DenoiserOptions::builder()
            .mode(DenoisingMode::Temporal { radius: 4 })
            .algorithm(Algorithm::Nl4d(Nl4dOptions {
                temporal_radius: 4,
                ..Nl4dOptions::default()
            }))
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
    fn nl4d_algorithm_round_trips_through_the_facade() {
        let opts = DenoiserOptions::builder()
            .channel_mode(ChannelMode::Luma)
            .mode(DenoisingMode::Temporal { radius: 2 })
            .motion_compensation(MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            })
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
