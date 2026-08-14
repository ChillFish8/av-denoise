use std::collections::VecDeque;

use cubecl::Runtime;

use crate::accelerate::Accelerator;
use crate::device::Device;
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

/// User-facing denoiser configuration. Build with `DenoiserOptions::builder()`.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which channels of the frame to denoise.
    #[builder(default = ChannelMode::Yuv)]
    pub channel_mode: ChannelMode,
    /// Spatial-only or temporal denoising.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Algorithm variant. `Nlmeans` is the fast default. `NlmeansHq`
    /// adapts weighting to the noise level, measured automatically per
    /// frame unless `HqParams::sigma_override` pins a fixed value. Its
    /// default `strength` multiplier is also different, and adapts to
    /// the temporal radius and the plane being denoised, see
    /// [`crate::nlmeans::hq_default_strength`].
    #[builder(default = Algorithm::Nlmeans)]
    pub algorithm: Algorithm,
    /// Reference clip source for NLM weight computation. `None` (the
    /// default) uses no prefilter for either algorithm. Set
    /// `PrefilterMode::NlmSpatial` explicitly to opt into the NLM
    /// spatial pilot.
    pub prefilter: Option<PrefilterMode>,
    /// Motion-compensation mode for temporal denoising. `None`
    /// disables MC; `Mvtools` warps temporal neighbours into spatial
    /// alignment with the centre frame before NLM weighting. Only
    /// takes effect when `mode` is `Temporal { .. }`.
    #[builder(default = MotionCompensationMode::None)]
    pub motion_compensation: MotionCompensationMode,
    /// Override NLM tuning (search/patch radius, strength, self-weight).
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
}

/// Standard spatial or temporal-aware denoising.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DenoisingMode {
    /// Spatial-only denoising (single frame).
    Spacial,
    /// Temporal-aware denoising over a `2 * radius + 1` window.
    Temporal { radius: u32 },
}

/// NLM tuning knobs. All optional; missing fields fall back to library
/// defaults.
#[derive(Debug, Copy, Clone)]
pub struct NlmTuning {
    pub search_radius: Option<u32>,
    pub patch_radius: Option<u32>,
    pub strength: Option<f32>,
    pub self_weight: Option<f32>,
}

impl DenoiserOptions {
    /// Resolve this option set into the low-level [`NlmParams`] a
    /// backend `Denoiser` is actually built from, folding in whichever
    /// default `strength` applies (see [`crate::nlmeans::hq_default_strength`]
    /// for the HQ algorithm). Exposed (rather than kept private) so
    /// callers building per-plane options, and tests, can inspect the
    /// resolved values without constructing a real `Denoiser`.
    #[doc(hidden)]
    pub fn to_nlm_params(&self) -> NlmParams {
        let temporal_radius = match self.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        };

        // An explicit `strength` (whether from `NlmTuning` directly or a
        // per-plane override already folded into it by the caller) always
        // wins. Otherwise the default depends on the algorithm and, for
        // HQ, on `auto_strength`. With auto-strength on, HQ's `strength`
        // is a multiplier on the measured noise level, so it needs its
        // own calibrated default rather than the fast path's absolute
        // FFmpeg-style default. That default also depends on the
        // temporal radius and which plane `self.channel_mode` names,
        // since each per-plane `Denoiser` carries its own channel mode.
        // With auto-strength off, HQ's `strength` is used verbatim as an
        // absolute value, same as the fast path, so it falls back to the
        // same absolute default.
        let explicit_strength = self.nlm.and_then(|t| t.strength);
        let strength = explicit_strength.unwrap_or(match self.algorithm {
            // The calibrated table is a multiplier on measured sigma, so
            // it only applies when `effective_strength_with` will treat
            // `strength` that way. With auto-strength off, `strength` is
            // used verbatim as an FFmpeg-style absolute value, so the
            // fallback has to be the same absolute default the fast path
            // uses.
            Algorithm::NlmeansHq(hq) if hq.auto_strength => {
                hq_default_strength(self.channel_mode, temporal_radius)
            },
            Algorithm::NlmeansHq(_) | Algorithm::Nlmeans => NlmParams::default().strength,
        });

        let mut params = NlmParams {
            channels: self.channel_mode,
            prefilter: self.prefilter.unwrap_or(PrefilterMode::None),
            motion_compensation: self.motion_compensation,
            temporal_radius,
            hq: match self.algorithm {
                Algorithm::Nlmeans => None,
                Algorithm::NlmeansHq(hq) => Some(hq),
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

/// Errors surfaced from the high-level [`Denoiser`].
#[derive(Debug, thiserror::Error)]
pub enum DenoiserError {
    /// A previous denoised frame hasn't been collected yet and the
    /// internal double-buffered output slot would alias. Call
    /// [`Denoiser::recv_frame`] or [`Denoiser::try_recv_frame`] first,
    /// then retry the same `push_frame` call.
    #[error("denoiser queue is full; collect the pending frame before pushing more")]
    QueueFull,
    /// None of the accelerators in the priority list could be initialised.
    #[error("no accelerator from the priority list is available")]
    NoAcceleratorAvailable,
    /// Catch-all wrapping internal `anyhow` errors from kernel dispatch
    /// and readback.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

enum Backend {
    #[cfg(feature = "cuda")]
    Cuda(NlmDenoiser<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(NlmDenoiser<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(NlmDenoiser<cubecl::wgpu::WgpuRuntime>),
    #[cfg(feature = "cpu")]
    Cpu(NlmDenoiser<cubecl::cpu::CpuRuntime>),
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

/// Maximum number of outstanding `Pending` readbacks the high-level
/// [`Denoiser`] keeps in flight. Must equal the backend's output-handle
/// count, which is two output handles. Exceeding this aliases the oldest
/// pending's output handle and silently corrupts results.
pub const MAX_PENDING: usize = 2;

/// High-level stateful denoiser. Push frames in order with
/// [`push_frame`](Self::push_frame); collect denoised frames with
/// [`recv_frame`](Self::recv_frame) or
/// [`try_recv_frame`](Self::try_recv_frame); call [`flush`](Self::flush)
/// at end-of-stream to drain any remaining temporal context.
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
    /// Probe each accelerator in `accelerators` in order and build a
    /// denoiser on the first one that's available. `device` lets the
    /// caller pick a non-default device for the chosen runtime.
    ///
    /// # Thread stack size
    ///
    /// cubecl spawns an internal per-device worker thread (named
    /// `DS{U,D}-…`) on which GPU kernel codegen runs. It uses Rust's
    /// default thread stack (`RUST_MIN_STACK`, or 2 MiB if unset). The
    /// windowed NLM kernels here contain `(2·search_radius + 1)²`-times
    /// `#[unroll]`ed bodies, so large `search_radius` (≳ 5) values can
    /// overflow that 2 MiB default and abort the process.
    ///
    /// Callers planning to use `search_radius > 4` should set
    /// `RUST_MIN_STACK` to at least 16 MiB before any cubecl thread
    /// spawns (typically at the very top of `main`), e.g.:
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
        let backend = build_backend(accelerator, device, params, width, height)?;

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

    /// The accelerator picked by [`sniff_best_accelerator`].
    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// Width passed at construction.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height passed at construction.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Upload one frame into the temporal window. `frame` must contain
    /// `width * height * channels` `f32` values in `[0, 1]`. Once the
    /// window is full and the in-flight pipeline has room, this also
    /// kicks off the kernels for the next denoised frame.
    ///
    /// Up to `MAX_PENDING` outputs may be in flight simultaneously:
    /// the GPU runs frame N+1's kernels while frame N's readback is in
    /// flight. Returns [`DenoiserError::QueueFull`] once that ceiling
    /// is reached; the caller must drain via [`Self::recv_frame`] before
    /// pushing more.
    pub fn push_frame(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        // After `temporal_radius` real pushes the leading-edge mirror has
        // primed the window, so the next push will set a pending. From
        // that point on, every push consumes a pending slot.
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

    /// Block until the in-flight denoise completes and return the
    /// denoised frame. Returns `Ok(None)` if nothing is in flight
    /// (e.g. the temporal window isn't full yet).
    pub fn recv_frame(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };
        Ok(Some(pending.wait()?))
    }

    /// Drain the in-flight denoise if one is ready. **May block
    /// briefly** while the runtime confirms the readback has landed;
    /// on workloads where kernels are already complete, the wait is
    /// effectively immediate.
    pub fn try_recv_frame(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
        self.recv_frame()
    }

    /// Drain in-flight frames and the trailing temporal tail (padded by duplicating
    /// the last pushed frame), handing each produced frame to `sink`.
    ///
    /// On success the denoiser is left ready to accept a fresh,
    /// independent stream of the same dimensions and parameters.
    /// Pushing more frames after `flush` starts a new temporal window from
    /// scratch. May be called multiple times.
    ///
    /// If `flush` returns `Err`, the denoiser is left in an undefined
    /// state and should be dropped rather than reused.
    pub fn flush(&mut self, mut sink: impl FnMut(Vec<f32>)) -> Result<(), DenoiserError> {
        // Drain the full pending pipeline (up to MAX_PENDING frames) before
        // submitting the trailing-tail mirrors.
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

        // Backend has already reset its own stream indices. Reset the
        // outer push counter too so the next push re-arms the
        // window-priming logic at the top of `push_frame`.
        self.frames_pushed = 0;

        Ok(())
    }
}

fn build_backend(
    accel: Accelerator,
    device: &Device,
    params: NlmParams,
    width: u32,
    height: u32,
) -> Result<Backend, DenoiserError> {
    match accel {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let dev = device.to_cuda()?;
            let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&dev);
            Ok(Backend::Cuda(NlmDenoiser::new(&client, params, width, height)))
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let dev = device.to_amd()?;
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&dev);
            Ok(Backend::Rocm(NlmDenoiser::new(&client, params, width, height)))
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let dev = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Wgpu(NlmDenoiser::new(&client, params, width, height)))
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let dev = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Wgpu(NlmDenoiser::new(&client, params, width, height)))
        },
        #[cfg(feature = "cpu")]
        Accelerator::Cpu => {
            let dev = device.to_cpu()?;
            let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&dev);
            Ok(Backend::Cpu(NlmDenoiser::new(&client, params, width, height)))
        },
        // Match-exhaustiveness placeholder for docs.rs, where `cfg(docsrs)`
        // widens the `Accelerator` enum to include variants whose backend
        // feature is not actually enabled. Never reached at runtime.
        #[cfg(docsrs)]
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod options_tests {
    use super::*;

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
        // `effective_strength_with` only treats `strength` as a multiplier
        // on the measured sigma when `auto_strength` is true. With it
        // false, `strength` is used verbatim as an FFmpeg-style absolute
        // strength, so the fallback must be the fast path's absolute
        // default, not a calibrated multiplier from `hq_default_strength`.
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
            "expected the legacy absolute default {expected}, got {} (looks like the \
             auto-strength multiplier table leaked through)",
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
                "channels={channels:?}: expected {expected}, got {}",
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
                "channels={channels:?}: explicit strength was overridden by the table"
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

        // Depth-2 pipeline: the first two pushes both submit successfully
        // (output handles are double-buffered). The third would alias the
        // oldest pending's output slot and is rejected with QueueFull.
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

    /// Push `n` frames of the given value, interleaving `recv_frame` to keep
    /// the in-flight pipeline below `MAX_PENDING`.
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

        // The temporal window must be empty after flush: the first push of
        // the new stream should not yield a pending immediately. With r=1
        // the window needs 3 frames before `denoise_submit` fires.
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
        // than R+1 frames means the window never fills during pushes —
        // flush must still emit exactly N outputs (one per pushed frame),
        // not R+1.
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

// Used to just catch fires on the CPU backend
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
