mod convert;
mod reseed;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    GrainChunk,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    WindowSpan,
};

pub use self::convert::{
    f32_to_plane,
    interleave_uv_to_f32,
    interleave_yuv_to_f32,
    plane_to_f32,
    unpack_uv_from_f32,
};
pub use self::reseed::ReseedWindow;
use crate::backend::Device;
use crate::backend::accelerate::Accelerator;
use crate::host::{Algorithm, DenoiserError, DenoiserOptions, Depth, HostDenoiser};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// Halved axes round up, so an odd dimension keeps the extra sample,
    /// matching what y4m and ffmpeg do.
    pub fn chroma_dims(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            Subsampling::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
            Subsampling::Yuv422 => (w.div_ceil(2), h),
            Subsampling::Yuv444 => (w, h),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameLayout {
    pub width: u32,
    pub height: u32,
    pub subsampling: Subsampling,
    pub depth: Depth,
}

impl FrameLayout {
    pub fn luma_pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn chroma_dims(&self) -> (u32, u32) {
        self.subsampling.chroma_dims(self.width, self.height)
    }

    pub fn chroma_pixels(&self) -> usize {
        let (w, h) = self.chroma_dims();
        (w as usize) * (h as usize)
    }

    /// Wire size of the luma plane.
    pub fn luma_bytes(&self) -> usize {
        self.luma_pixels() * self.depth.bytes_per_sample()
    }

    /// Wire size of one chroma plane.
    pub fn chroma_bytes(&self) -> usize {
        self.chroma_pixels() * self.depth.bytes_per_sample()
    }

    /// A full black luma plane, used when no luma source is available.
    pub fn black_luma_plane(&self) -> Vec<u8> {
        fill_plane(self.luma_pixels(), 0, self.depth)
    }

    /// A full neutral chroma plane, used when a source has no chroma.
    pub fn neutral_chroma_plane(&self) -> Vec<u8> {
        fill_plane(self.chroma_pixels(), self.depth.neutral_chroma(), self.depth)
    }
}

/// Builds a plane of `samples` copies of `value` in wire-byte form.
pub fn fill_plane(samples: usize, value: u16, depth: Depth) -> Vec<u8> {
    match depth.bytes_per_sample() {
        1 => vec![value as u8; samples],
        _ => {
            let word = value.to_le_bytes();
            let mut out = Vec::with_capacity(samples * 2);
            for _ in 0..samples {
                out.extend_from_slice(&word);
            }
            out
        },
    }
}

/// A planar YUV frame holding little-endian wire bytes.
///
/// Plane lengths come from [`FrameLayout`], so `y.len()` is
/// `layout.luma_bytes()` and both `u.len()` and `v.len()` are
/// `layout.chroma_bytes()`.
#[derive(Debug, Clone)]
pub struct Planes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Which planes a caller wants cleaned, once `--channel-mode` (or the
/// equivalent host option) has been resolved.
///
/// This is separate from the library's [`ChannelMode`] because this layer
/// may run more than one `HostDenoiser` in lockstep, one for luma and one for
/// chroma. It may also run a single fused three-channel denoiser instead.
/// Which of those applies depends on the caller's channel selection and
/// the source's chroma subsampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ChannelIntent {
    /// Denoise luma only. Chroma passes through.
    Luma,
    /// Denoise chroma only. Luma passes through.
    Chroma,
    /// Denoise both luma and chroma as two independent denoisers.
    /// Chroma runs at the source's native subsampled resolution.
    LumaChroma,
    /// A single library `HostDenoiser` running the fused three-channel
    /// kernel. Needs a YUV444 source, which is checked at ingest setup
    /// time.
    YuvFused,
}

impl ChannelIntent {
    /// Rejects the intent if the source's subsampling cannot support it.
    pub fn validate_for_source(self, layout: FrameLayout) -> Result<(), anyhow::Error> {
        match self {
            ChannelIntent::YuvFused if layout.subsampling != Subsampling::Yuv444 => {
                anyhow::bail!(
                    "--channel-mode yuv requires a YUV444 source, got {:?}. Convert the input first, for example with `ffmpeg -pix_fmt yuv444p`",
                    layout.subsampling
                );
            },
            _ => Ok(()),
        }
    }
}

/// The per-plane option set a caller resolves once and passes into
/// [`PlanarDenoiser::create`].
#[derive(Debug, Clone)]
pub struct PlaneOptions {
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub intent: ChannelIntent,
    pub mode: DenoisingMode,
    /// Which denoising algorithm to run, along with the settings only
    /// that algorithm reads.
    pub algorithm: Algorithm,
    /// Per-plane strength override for the luma denoiser. Takes
    /// precedence over the algorithm's own `tuning.strength` when set.
    /// Only has an effect on the two NLM algorithms.
    pub luma_strength: Option<f32>,
    /// Per-plane strength override for the chroma denoiser. Takes
    /// precedence over the algorithm's own `tuning.strength` when set.
    /// Only has an effect on the two NLM algorithms.
    pub chroma_strength: Option<f32>,
    /// Per-plane override for `lambda_ht`, luma. Takes precedence over
    /// `algorithm`'s value when set, which itself falls back to a
    /// calibrated per-plane default when nothing at all is set. Only
    /// has an effect when `algorithm` is `Algorithm::Nl4d`, where it
    /// pins the temporal grouping stage's hard threshold.
    pub luma_lambda_ht: Option<f32>,
    /// Per-plane override for `lambda_ht`, chroma. Takes precedence over
    /// `algorithm`'s value when set, which itself falls back to a
    /// calibrated per-plane default when nothing at all is set. Only
    /// has an effect when `algorithm` is `Algorithm::Nl4d`, where it
    /// pins the temporal grouping stage's hard threshold.
    pub chroma_lambda_ht: Option<f32>,
}

impl PlaneOptions {
    /// Resolves `self.algorithm` for one plane, folding in the per-plane
    /// overrides that apply to whichever algorithm `self.algorithm` is.
    ///
    /// For the two NLM algorithms that is `strength`. For `Nl4d` it is
    /// `lambda_ht`, since nl4d has no NLM weighting pass for a strength
    /// to affect.
    ///
    /// `Nl4d`'s `lambda_ht` stays `Option<f32>` all the way through
    /// this method. When neither a per-plane flag nor the matching
    /// shared flag was set, the result is `None`, deferred to
    /// `nl4d_default_lambda_ht` at construction, once the plane being
    /// denoised is known there too. That is what gives luma and chroma
    /// different values when a caller passes no flags at all.
    fn algorithm_for(&self, channels: ChannelMode) -> Algorithm {
        let per_plane = |luma, chroma| match channels {
            ChannelMode::Luma => luma,
            ChannelMode::Chroma => chroma,
            ChannelMode::Yuv => None,
        };

        match self.algorithm {
            Algorithm::Nl4d(nl4d) => Algorithm::Nl4d(Nl4dOptions {
                // Left unresolved when unset, since the calibrated
                // default depends on the plane, which
                // `nl4d_default_lambda_ht` resolves at construction.
                lambda_ht: per_plane(self.luma_lambda_ht, self.chroma_lambda_ht).or(nl4d.lambda_ht),
                grain_export: nl4d.grain_export && channels != ChannelMode::Chroma,
                ..nl4d
            }),
            Algorithm::Nlmeans(nlm) => {
                let strength = per_plane(self.luma_strength, self.chroma_strength);
                Algorithm::Nlmeans(with_plane_strength(nlm, strength))
            },
            Algorithm::NlmeansHq(opts) => {
                let strength = per_plane(self.luma_strength, self.chroma_strength);
                Algorithm::NlmeansHq(NlmeansHqOptions {
                    nlm: with_plane_strength(opts.nlm, strength),
                    ..opts
                })
            },
        }
    }

    /// `depth` is the source's wire depth, which every denoiser
    /// quantises to on the GPU.
    fn denoiser_options(&self, channels: ChannelMode, depth: Depth) -> DenoiserOptions {
        DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .algorithm(self.algorithm_for(channels))
            .depth(depth)
            .build()
    }
}

/// `nlm` with `strength` replaced by the per-plane override, when there
/// is one. An unset override leaves the shared value alone.
fn with_plane_strength(nlm: NlmeansOptions, strength: Option<f32>) -> NlmeansOptions {
    match strength {
        None => nlm,
        Some(strength) => NlmeansOptions {
            tuning: NlmTuning {
                strength: Some(strength),
                ..nlm.tuning
            },
            ..nlm
        },
    }
}

/// Reads the result of a `PlanarDenoiser::push` call for the
/// push-then-drain-then-retry loop.
///
/// `Ok(false)` means the push landed. `Ok(true)` means the queue was
/// full, so the caller should drain one output and push again.
///
/// Any error other than `QueueFull` is passed on rather than discarded.
pub fn push_needs_retry(result: Result<(), DenoiserError>) -> Result<bool, anyhow::Error> {
    match result {
        Ok(()) => Ok(false),
        Err(DenoiserError::QueueFull) => Ok(true),
        Err(other) => Err(other.into()),
    }
}

/// Turns one denoised frame's per-plane buffers into `N` planes.
fn into_array<const N: usize>(planes: Vec<Vec<u8>>) -> [Vec<u8>; N] {
    let count = planes.len();
    let array = planes.try_into();
    array.unwrap_or_else(|_| panic!("expected {N} planes from a HostDenoiser, got {count}"))
}

fn into_yuv(planes: Vec<Vec<u8>>) -> Planes {
    let [y, u, v] = into_array(planes);
    Planes { y, u, v }
}

fn into_uv(planes: Vec<Vec<u8>>) -> (Vec<u8>, Vec<u8>) {
    let [u, v] = into_array(planes);
    (u, v)
}

fn into_luma(planes: Vec<Vec<u8>>) -> Vec<u8> {
    let [y] = into_array(planes);
    y
}

/// The push a [`PlanarDenoiser`] runs against each enabled half, either
/// [HostDenoiser::push] or [HostDenoiser::push_priming].
type WirePush = fn(&mut HostDenoiser, &[&[u8]]) -> Result<(), DenoiserError>;

/// Wraps the luma and chroma `HostDenoiser` instances needed for one
/// subsampled YUV source.
///
/// The caller pushes planar frames in and gets planar frames out. The
/// luma and chroma split is invisible from the outside.
pub struct PlanarDenoiser {
    layout: FrameLayout,
    luma: Option<HostDenoiser>,
    chroma: Option<HostDenoiser>,
    /// Set when the intent is `YuvFused`, in which case `luma` and
    /// `chroma` are both unset.
    yuv: Option<HostDenoiser>,
    // Source planes queued for passthrough when the matching denoiser is
    // disabled. Only the disabled side's queue is ever filled. Entries
    // are popped one per frame the enabled side emits, so temporal
    // delays stay aligned.
    luma_passthrough: VecDeque<Vec<u8>>,
    chroma_passthrough: VecDeque<(Vec<u8>, Vec<u8>)>,
    /// The temporal radius every owned denoiser runs at, resolved from
    /// `opts.mode` at construction.
    temporal_radius: u32,
}

impl PlanarDenoiser {
    pub fn create(opts: &PlaneOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
        let (chroma_w, chroma_h) = layout.chroma_dims();

        if chroma_w == 0 || chroma_h == 0 {
            anyhow::bail!(
                "frame dimensions {}x{} are too small for subsampling {:?}",
                layout.width,
                layout.height,
                layout.subsampling
            );
        }

        opts.intent.validate_for_source(layout)?;

        let (denoise_luma, denoise_chroma, denoise_yuv) = match opts.intent {
            ChannelIntent::Luma => (true, false, false),
            ChannelIntent::Chroma => (false, true, false),
            ChannelIntent::LumaChroma => (true, true, false),
            ChannelIntent::YuvFused => (false, false, true),
        };

        let luma = denoise_luma
            .then(|| {
                HostDenoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    layout.width,
                    layout.height,
                    opts.denoiser_options(ChannelMode::Luma, layout.depth),
                )
            })
            .transpose()?;

        let chroma = denoise_chroma
            .then(|| {
                HostDenoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    chroma_w,
                    chroma_h,
                    opts.denoiser_options(ChannelMode::Chroma, layout.depth),
                )
            })
            .transpose()?;

        let yuv = denoise_yuv
            .then(|| {
                HostDenoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    layout.width,
                    layout.height,
                    opts.denoiser_options(ChannelMode::Yuv, layout.depth),
                )
            })
            .transpose()?;

        let temporal_radius = match opts.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        };

        Ok(Self {
            layout,
            luma,
            chroma,
            yuv,
            luma_passthrough: VecDeque::new(),
            chroma_passthrough: VecDeque::new(),
            temporal_radius,
        })
    }

    /// The temporal radius the underlying denoisers run at.
    pub fn temporal_radius(&self) -> u32 {
        self.temporal_radius
    }

    /// Pushes one planar frame.
    ///
    /// On `QueueFull` the caller should receive one frame and then retry
    /// the whole call. Any other error is passed on unchanged.
    ///
    /// The denoiser push runs before either passthrough queue is
    /// touched, so a retry replays the whole frame cleanly instead of
    /// queueing the disabled side's plane twice.
    ///
    /// # Why a retry cannot duplicate a frame
    ///
    /// In `LumaChroma` mode `luma` and `chroma` are both real
    /// `HostDenoiser`s with their own queues. A retry pushes again into
    /// whichever half already succeeded, which would duplicate that
    /// half's frame if the two could ever sit at different fill levels.
    ///
    /// They cannot. Both are built from the same `opts.mode`, so they
    /// share a temporal radius and a `MAX_PENDING` ceiling. Every
    /// successful push or receive moves both on by exactly one frame,
    /// and a failed push moves neither, because the `QueueFull` check
    /// runs before anything changes.
    ///
    /// So the two halves always enter this function with the same frame
    /// count and the same pending depth, and the `QueueFull` check
    /// inside `HostDenoiser::push` answers the same way for each. If the luma
    /// push succeeds then the chroma push succeeds too, which makes the
    /// duplicate unreachable.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push)
    }

    /// Uploads one planar frame into the temporal window without starting
    /// a denoise.
    ///
    /// Mirrors [`Self::push`], down to queueing the disabled side's
    /// passthrough plane, but no output is ever produced for this call.
    /// This is how the reseed paths fill a window's leading frames before
    /// its real pushes start.
    fn push_priming(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push_priming)
    }

    /// Shared body of [`Self::push`] and [`Self::push_priming`].
    ///
    /// `push_frame` is [HostDenoiser::push] for a real push or
    /// [HostDenoiser::push_priming] for a priming one, run
    /// against whichever of `yuv`, `luma`, and `chroma` is enabled.
    ///
    /// The planes go over as wire bytes, so the normalisation and the
    /// channel interleave both happen on the GPU.
    fn push_with(&mut self, planes: &Planes, push_frame: WirePush) -> Result<(), DenoiserError> {
        if let Some(d) = self.yuv.as_mut() {
            push_frame(d, &[&planes.y, &planes.u, &planes.v])?;
            return Ok(());
        }

        if let Some(d) = self.luma.as_mut() {
            push_frame(d, &[&planes.y])?;
        }

        if let Some(d) = self.chroma.as_mut() {
            push_frame(d, &[&planes.u, &planes.v])?;
        }

        if self.luma.is_none() {
            self.luma_passthrough.push_back(planes.y.clone());
        }

        if self.chroma.is_none() {
            self.chroma_passthrough
                .push_back((planes.u.clone(), planes.v.clone()));
        }

        Ok(())
    }

    /// Blocks until each enabled half emits one frame, then reassembles
    /// them into a planar frame.
    ///
    /// Returns `Ok(None)` if neither half had pending output.
    pub fn recv(&mut self) -> Result<Option<Planes>, anyhow::Error> {
        if let Some(d) = self.yuv.as_mut() {
            let received = d.recv()?;
            let planes = received.map(into_yuv);
            return Ok(planes);
        }

        let luma_out = self
            .luma
            .as_mut()
            .map(|d| d.recv())
            .transpose()?
            .flatten()
            .map(into_luma);

        let chroma_out = self
            .chroma
            .as_mut()
            .map(|d| d.recv())
            .transpose()?
            .flatten()
            .map(into_uv);

        // A disabled side has no HostDenoiser to query. When the enabled side
        // produced output, pop the matching source plane from the
        // disabled side's passthrough queue instead.
        let luma_passthrough = if self.luma.is_none() && chroma_out.is_some() {
            self.luma_passthrough.pop_front()
        } else {
            None
        };

        let chroma_passthrough = if self.chroma.is_none() && luma_out.is_some() {
            self.chroma_passthrough.pop_front()
        } else {
            None
        };

        if luma_out.is_none() && chroma_out.is_none() {
            return Ok(None);
        }

        let planes = self.assemble(luma_out, chroma_out, luma_passthrough, chroma_passthrough);

        Ok(Some(planes))
    }

    /// Reads back the luma grain chunks measured since the last call, in frame order.
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, anyhow::Error> {
        let source = self.luma.as_mut().or(self.yuv.as_mut());
        let Some(denoiser) = source else {
            return Ok(Vec::new());
        };

        let chunks = denoiser.drain_grain_chunks()?;

        Ok(chunks)
    }

    /// Drains the temporal tail of both halves.
    ///
    /// `sink` is called once per emitted planar frame.
    pub fn flush(&mut self, mut sink: impl FnMut(Planes)) -> Result<(), anyhow::Error> {
        if let Some(d) = self.yuv.as_mut() {
            d.flush(|planes| {
                let frame = into_yuv(planes);
                sink(frame);
            })?;
            return Ok(());
        }

        let mut luma_buf: Vec<Vec<u8>> = Vec::new();
        let mut chroma_buf: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        if let Some(d) = self.luma.as_mut() {
            d.flush(|planes| {
                let luma = into_luma(planes);
                luma_buf.push(luma);
            })?;
        }

        if let Some(d) = self.chroma.as_mut() {
            d.flush(|planes| {
                let chroma = into_uv(planes);
                chroma_buf.push(chroma);
            })?;
        }

        // The two halves run in lockstep, so they flush the same number
        // of frames. For each emitted frame the disabled side, if there
        // is one, pops the matching source plane from its passthrough
        // queue.
        let count = luma_buf.len().max(chroma_buf.len());

        for i in 0..count {
            let y = if let Some(buf) = luma_buf.get_mut(i) {
                std::mem::take(buf)
            } else if let Some(src) = self.luma_passthrough.pop_front() {
                src
            } else {
                self.layout.black_luma_plane()
            };

            let (u, v) = if let Some(pair) = chroma_buf.get_mut(i) {
                std::mem::take(pair)
            } else if let Some((src_u, src_v)) = self.chroma_passthrough.pop_front() {
                (src_u, src_v)
            } else {
                (
                    self.layout.neutral_chroma_plane(),
                    self.layout.neutral_chroma_plane(),
                )
            };

            sink(Planes { y, u, v });
        }

        if !self.luma_passthrough.is_empty() || !self.chroma_passthrough.is_empty() {
            tracing::warn!(
                luma_remaining = self.luma_passthrough.len(),
                chroma_remaining = self.chroma_passthrough.len(),
                "passthrough queue not fully drained after flush",
            );
            self.luma_passthrough.clear();
            self.chroma_passthrough.clear();
        }

        Ok(())
    }

    /// The number of frames behind and ahead of a target frame a
    /// [`Self::reseed`] window must supply, for whichever algorithm this
    /// `PlanarDenoiser` runs.
    ///
    /// Every owned `HostDenoiser` was built from the same algorithm, so any
    /// one of them answers for all of them.
    pub fn window_span(&self) -> WindowSpan {
        self.yuv
            .as_ref()
            .or(self.luma.as_ref())
            .or(self.chroma.as_ref())
            .expect("PlanarDenoiser always keeps at least one HostDenoiser")
            .window_span()
    }

    fn assemble(
        &self,
        luma: Option<Vec<u8>>,
        chroma: Option<(Vec<u8>, Vec<u8>)>,
        luma_passthrough: Option<Vec<u8>>,
        chroma_passthrough: Option<(Vec<u8>, Vec<u8>)>,
    ) -> Planes {
        let y = match (luma, luma_passthrough) {
            (Some(v), _) => v,
            (None, Some(src)) => src,
            (None, None) => self.layout.black_luma_plane(),
        };

        let (u, v) = match (chroma, chroma_passthrough) {
            (Some(pair), _) => pair,
            (None, Some(src)) => src,
            (None, None) => (
                self.layout.neutral_chroma_plane(),
                self.layout.neutral_chroma_plane(),
            ),
        };

        Planes { y, u, v }
    }
}

#[cfg(test)]
mod cli_options_tests {
    use av_denoise_core::NlmeansAlgorithm;

    use super::*;
    use crate::backend::EngineSpec;

    /// A `PlaneOptions` with every field at a neutral default, so each test
    /// only overrides what it cares about.
    ///
    /// `mode` and `algorithm` are the two fields every test below sets
    /// for itself.
    fn base_opts(
        mode: DenoisingMode,
        algorithm: Algorithm,
        luma_strength: Option<f32>,
        chroma_strength: Option<f32>,
    ) -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode,
            algorithm,
            luma_strength,
            chroma_strength,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        }
    }

    #[test]
    fn luma_strength_alone_overrides_only_the_luma_plane() {
        let opts = base_opts(DenoisingMode::Spacial, Algorithm::default(), Some(0.7), None);

        let luma = expect_nlmeans(opts.denoiser_options(ChannelMode::Luma, Depth::Eight).algorithm);
        let chroma = expect_nlmeans(opts.denoiser_options(ChannelMode::Chroma, Depth::Eight).algorithm);

        assert!(
            matches!(luma.tuning.strength, Some(s) if (s - 0.7).abs() < f32::EPSILON),
            "expected luma tuning.strength = Some(0.7), got {:?}",
            luma.tuning.strength
        );
        assert_eq!(
            chroma.tuning.strength, None,
            "chroma plane should carry no override so the table default applies"
        );
    }

    #[test]
    fn both_per_plane_strengths_set_independently() {
        let opts = base_opts(DenoisingMode::Spacial, Algorithm::default(), Some(0.7), Some(0.3));

        let luma = expect_nlmeans(opts.denoiser_options(ChannelMode::Luma, Depth::Eight).algorithm);
        let chroma = expect_nlmeans(opts.denoiser_options(ChannelMode::Chroma, Depth::Eight).algorithm);

        assert!(
            matches!(luma.tuning.strength, Some(s) if (s - 0.7).abs() < f32::EPSILON),
            "expected luma tuning.strength = Some(0.7), got {:?}",
            luma.tuning.strength
        );
        assert!(
            matches!(chroma.tuning.strength, Some(s) if (s - 0.3).abs() < f32::EPSILON),
            "expected chroma tuning.strength = Some(0.3), got {:?}",
            chroma.tuning.strength
        );
    }

    #[test]
    fn no_overrides_hq_leaves_strength_to_the_per_plane_table() {
        let opts = base_opts(
            DenoisingMode::Temporal { radius: 4 },
            Algorithm::NlmeansHq(NlmeansHqOptions::default()),
            None,
            None,
        );

        for channels in [ChannelMode::Luma, ChannelMode::Chroma] {
            let options = opts.denoiser_options(channels, Depth::Eight);
            let spec = options.algorithm.engine_spec(&options, 16, 16);

            let EngineSpec::Nlmeans {
                algorithm: NlmeansAlgorithm::Hq(hq),
                geometry,
            } = spec
            else {
                panic!("expected an HQ nlmeans spec for {channels:?}, got {spec:?}");
            };

            assert_eq!(geometry.channels, channels);
            assert_eq!(hq.nlm.mode, DenoisingMode::Temporal { radius: 4 });
            assert_eq!(
                hq.nlm.tuning.strength, None,
                "{channels:?} should use the calibrated table"
            );
        }
    }

    /// A `PlaneOptions` running `Algorithm::Nl4d`, with every field at a
    /// neutral default except the two per-plane `lambda_ht` overrides
    /// under test.
    fn nl4d_opts(luma_lambda_ht: Option<f32>, chroma_lambda_ht: Option<f32>) -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: 2 },
            algorithm: Algorithm::Nl4d(Nl4dOptions::default()),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht,
            chroma_lambda_ht,
        }
    }

    /// Unwraps an `Algorithm::Nlmeans`, panicking with the whole value
    /// on any other variant.
    fn expect_nlmeans(algorithm: Algorithm) -> NlmeansOptions {
        match algorithm {
            Algorithm::Nlmeans(n) => n,
            other => panic!("expected Algorithm::Nlmeans, got {other:?}"),
        }
    }

    /// Unwraps an `Algorithm::Nl4d`, panicking with the whole value on
    /// any other variant.
    fn expect_nl4d(algorithm: Algorithm) -> Nl4dOptions {
        match algorithm {
            Algorithm::Nl4d(n) => n,
            other => panic!("expected Algorithm::Nl4d, got {other:?}"),
        }
    }

    /// The routing property that matters most for a shared field: an
    /// override aimed at one plane must never leak into the other
    /// instance. `luma_lambda_ht` set alone must change nothing about
    /// the chroma instance, and vice versa in the sibling test below.
    #[test]
    fn luma_lambda_ht_alone_overrides_only_the_luma_instance_for_nl4d() {
        let opts = nl4d_opts(Some(4.0), None);

        let luma = expect_nl4d(opts.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(opts.algorithm_for(ChannelMode::Chroma));

        assert!((luma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
        assert_eq!(
            chroma.lambda_ht,
            Nl4dOptions::default().lambda_ht,
            "chroma should stay unresolved here (None), deferred to its own per-plane \
             default at construction, got {:?}",
            chroma.lambda_ht
        );
    }

    #[test]
    fn chroma_lambda_ht_alone_overrides_only_the_chroma_instance_for_nl4d() {
        let opts = nl4d_opts(None, Some(4.0));

        let luma = expect_nl4d(opts.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(opts.algorithm_for(ChannelMode::Chroma));

        assert_eq!(
            luma.lambda_ht,
            Nl4dOptions::default().lambda_ht,
            "luma should stay unresolved here (None), deferred to its own per-plane \
             default at construction, got {:?}",
            luma.lambda_ht
        );
        assert!((chroma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn both_planes_lambda_ht_set_independently_for_nl4d() {
        let opts = nl4d_opts(Some(2.0), Some(3.5));

        let luma = expect_nl4d(opts.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(opts.algorithm_for(ChannelMode::Chroma));

        assert!((luma.lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((chroma.lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);

        // Every other field stays shared between the two instances even
        // though lambda_ht diverges.
        assert_eq!(luma.refine, chroma.refine);
        assert_eq!(luma.spatial_radius, chroma.spatial_radius);
        assert!((luma.c_min - chroma.c_min).abs() < f32::EPSILON);
    }

    #[test]
    fn unset_nl4d_overrides_resolve_to_different_lambda_ht_per_plane_end_to_end() {
        let opts = nl4d_opts(None, None);

        let luma = expect_nl4d(opts.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(opts.algorithm_for(ChannelMode::Chroma));

        // Neither plane has anything set anywhere, so both stay
        // unresolved at this layer...
        assert_eq!(luma.lambda_ht, None);
        assert_eq!(chroma.lambda_ht, None);

        // ...but resolving each through the same function construction
        // uses (`nl4d_default_lambda_ht`) gives
        // luma and chroma different values, which is the whole point of
        // a caller passing no flags at all getting both per-plane
        // defaults.
        let luma_default = crate::nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma_default = crate::nl4d_default_lambda_ht(ChannelMode::Chroma);
        assert!((luma_default - 4.158).abs() < f32::EPSILON);
        assert!((chroma_default - 3.234).abs() < f32::EPSILON);
        assert!((chroma_default - luma_default).abs() > f32::EPSILON);
    }
}

// Feature-gated because every test here builds its `PlaneOptions` from
// `chroma_only_opts`, which names the `Vulkan` accelerator variant. That
// variant only exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod passthrough_retry_tests {
    use super::*;
    use crate::accelerate::Accelerator;
    use crate::{Algorithm, DenoisingMode};

    /// Chroma-only intent, so `luma` is the disabled passthrough half and
    /// `chroma` is the one that can report `QueueFull`.
    ///
    /// That is what drives the retry loop in `push_with_drain`.
    fn chroma_only_opts() -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: ChannelIntent::Chroma,
            mode: DenoisingMode::Spacial,
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        }
    }

    fn fake_planes(layout: FrameLayout) -> Planes {
        Planes {
            y: fill_plane(layout.luma_pixels(), layout.depth.neutral_chroma(), layout.depth),
            u: layout.neutral_chroma_plane(),
            v: layout.neutral_chroma_plane(),
        }
    }

    #[test]
    fn queue_full_retry_does_not_double_queue_the_passthrough_plane() {
        let layout = FrameLayout {
            width: 16,
            height: 16,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let mut wd =
            PlanarDenoiser::create(&chroma_only_opts(), layout).expect("denoiser construction failed");
        let planes = fake_planes(layout);

        // Spatial mode runs a depth-2 pipeline, so the first two pushes
        // land directly. See `push_after_pending_returns_queue_full` in
        // `src/host/tests/denoiser.rs`.
        wd.push(&planes).expect("first push should land");
        wd.push(&planes).expect("second push should land");

        // Third push hits QueueFull on the chroma half.
        let err = wd.push(&planes).expect_err("expected QueueFull");
        assert!(
            matches!(err, DenoiserError::QueueFull),
            "expected QueueFull, got {err:?}"
        );

        // Mirror the retry loop in `push_with_drain`. Drain one output,
        // then retry the whole `push()` call for the same frame.
        wd.recv().expect("recv after drain failed");
        wd.push(&planes).expect("retry push should land after drain");

        // The chroma denoiser accepted three frames, two directly and
        // one on the retry, and `recv` popped one back off. The disabled
        // luma half's passthrough queue must track that one for one, and
        // must not count the frame whose first attempt hit `QueueFull`
        // twice.
        assert_eq!(
            wd.luma_passthrough.len(),
            2,
            "expected exactly one passthrough entry per chroma frame actually accepted, got {}",
            wd.luma_passthrough.len()
        );
    }
}

// Feature-gated because every test here builds its `PlaneOptions` from
// `luma_chroma_opts`, which names the `Vulkan` accelerator variant. That
// variant only exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod lumachroma_lockstep_tests {
    use super::*;
    use crate::accelerate::Accelerator;
    use crate::{Algorithm, DenoisingMode};

    /// Runs `luma` and `chroma` as two real `HostDenoiser`s in spatial mode.
    ///
    /// Spatial mode passes a uniform-valued plane through unchanged, as
    /// the `uniform_*_passthrough` tests in `av-denoise-core/src/nlmeans/tests` show. The
    /// test can therefore give each plane its own marker value and spot
    /// the two halves drifting apart.
    fn luma_chroma_opts() -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Spacial,
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        }
    }

    /// A uniform-valued frame whose luma and chroma planes each encode
    /// `idx` with a different formula.
    ///
    /// If the round trip ever pairs luma from one push with chroma from
    /// another, the two encodings disagree and the test catches it.
    fn marked_planes(layout: FrameLayout, idx: u8) -> Planes {
        let chroma_pixels = layout.chroma_pixels();
        let y_val = 10 + idx;
        let uv_val = 200 - idx;

        Planes {
            y: fill_plane(layout.luma_pixels(), y_val as u16, layout.depth),
            u: fill_plane(chroma_pixels, uv_val as u16, layout.depth),
            v: fill_plane(chroma_pixels, uv_val as u16, layout.depth),
        }
    }

    #[test]
    fn distinct_u_and_v_planes_come_back_in_order() {
        let layout = FrameLayout {
            width: 16,
            height: 16,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let options = luma_chroma_opts();
        let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

        let chroma_pixels = layout.chroma_pixels();
        let planes = Planes {
            y: fill_plane(layout.luma_pixels(), 100, layout.depth),
            u: fill_plane(chroma_pixels, 60, layout.depth),
            v: fill_plane(chroma_pixels, 190, layout.depth),
        };
        denoiser.push(&planes).expect("push failed");

        let received = denoiser.recv().expect("recv failed");
        let out = received.expect("spatial mode emits one frame per push");

        for &sample in &out.u {
            assert!(sample.abs_diff(60) <= 2, "U sample {sample}, expected about 60");
        }

        for &sample in &out.v {
            assert!(sample.abs_diff(190) <= 2, "V sample {sample}, expected about 190");
        }
    }

    #[test]
    fn queue_full_retries_never_desync_luma_and_chroma() {
        let layout = FrameLayout {
            width: 16,
            height: 16,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let mut wd =
            PlanarDenoiser::create(&luma_chroma_opts(), layout).expect("denoiser construction failed");

        // More pushes than the depth-2 pipeline holds, so this drives
        // several `QueueFull`-then-retry cycles.
        const N: u8 = 6;
        let mut outputs: Vec<Planes> = Vec::new();

        for idx in 0..N {
            let planes = marked_planes(layout, idx);

            // Mirror the retry loop in `push_with_drain` exactly, which
            // is the sequence the CLI workers run.
            if push_needs_retry(wd.push(&planes)).expect("push_needs_retry") {
                if let Some(out) = wd.recv().expect("recv failed") {
                    outputs.push(out);
                }

                wd.push(&planes).expect("retry push should land after drain");
            }
        }

        wd.flush(|out| outputs.push(out)).expect("flush failed");

        assert_eq!(
            outputs.len(),
            N as usize,
            "expected exactly one output frame per input frame, got {}",
            outputs.len()
        );

        for out in &outputs {
            let y_val = out.y[0];
            let uv_val = out.u[0];
            let idx_from_y = y_val - 10;
            let idx_from_uv = 200 - uv_val;

            assert_eq!(
                idx_from_y, idx_from_uv,
                "luma marker {y_val} (frame {idx_from_y}) and chroma marker {uv_val} \
                 (frame {idx_from_uv}) disagree, so the luma and chroma pushes have drifted apart"
            );
        }
    }
}

#[cfg(test)]
mod push_needs_retry_tests {
    use super::*;

    #[test]
    fn ok_means_no_retry() {
        let outcome = push_needs_retry(Ok(())).expect("Ok(()) must not itself error");
        assert!(!outcome, "a landed push must not ask the caller to retry");
    }

    #[test]
    fn queue_full_signals_retry() {
        let outcome =
            push_needs_retry(Err(DenoiserError::QueueFull)).expect("QueueFull must not itself error");
        assert!(outcome, "QueueFull must still trigger the retry-after-drain path");
    }

    #[test]
    fn non_queue_full_errors_propagate_instead_of_being_swallowed() {
        let synthetic = DenoiserError::Other(anyhow::anyhow!("synthetic readback failure"));

        let outcome = push_needs_retry(Err(synthetic));

        assert!(
            outcome.is_err(),
            "a non-QueueFull push error must propagate instead of being silently treated as success"
        );
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn layout(depth: Depth) -> FrameLayout {
        FrameLayout {
            width: 4,
            height: 4,
            subsampling: Subsampling::Yuv420,
            depth,
        }
    }

    #[test]
    fn byte_lengths_scale_with_depth() {
        assert_eq!(layout(Depth::Eight).luma_bytes(), 16);
        assert_eq!(layout(Depth::Ten).luma_bytes(), 32);
        assert_eq!(layout(Depth::Eight).chroma_bytes(), 4);
        assert_eq!(layout(Depth::Ten).chroma_bytes(), 8);
    }

    #[test]
    fn neutral_chroma_fill_is_correct_at_each_depth() {
        let eight = layout(Depth::Eight).neutral_chroma_plane();
        assert_eq!(eight, vec![128u8; 4]);

        // 512 little-endian is [0x00, 0x02], repeated per sample.
        let ten = layout(Depth::Ten).neutral_chroma_plane();
        assert_eq!(ten, vec![0x00, 0x02, 0x00, 0x02, 0x00, 0x02, 0x00, 0x02]);

        // 2048 little-endian is [0x00, 0x08].
        let twelve = layout(Depth::Twelve).neutral_chroma_plane();
        assert_eq!(twelve.len(), 8);
        assert_eq!(&twelve[0..2], &[0x00, 0x08]);
    }

    #[test]
    fn black_luma_fill_is_zero_at_the_right_length() {
        assert_eq!(layout(Depth::Eight).black_luma_plane(), vec![0u8; 16]);
        assert_eq!(layout(Depth::Ten).black_luma_plane(), vec![0u8; 32]);
    }
}

#[cfg(test)]
mod chroma_dims_tests {
    use super::*;

    #[test]
    fn yuv420_even_dims_halve() {
        assert_eq!(Subsampling::Yuv420.chroma_dims(1920, 1080), (960, 540));
    }

    #[test]
    fn yuv420_odd_width_rounds_up() {
        assert_eq!(Subsampling::Yuv420.chroma_dims(1919, 1080), (960, 540));
    }

    #[test]
    fn yuv420_odd_height_rounds_up() {
        assert_eq!(Subsampling::Yuv420.chroma_dims(1920, 1079), (960, 540));
    }

    #[test]
    fn yuv420_odd_both_dims_round_up() {
        assert_eq!(Subsampling::Yuv420.chroma_dims(1919, 1079), (960, 540));
    }

    #[test]
    fn yuv422_even_width_halves() {
        assert_eq!(Subsampling::Yuv422.chroma_dims(1920, 1080), (960, 1080));
    }

    #[test]
    fn yuv422_odd_width_rounds_up() {
        assert_eq!(Subsampling::Yuv422.chroma_dims(1919, 1080), (960, 1080));
    }

    #[test]
    fn yuv444_passes_even_dims_through() {
        assert_eq!(Subsampling::Yuv444.chroma_dims(1920, 1080), (1920, 1080));
    }

    #[test]
    fn yuv444_passes_odd_dims_through() {
        assert_eq!(Subsampling::Yuv444.chroma_dims(1919, 1079), (1919, 1079));
    }
}
