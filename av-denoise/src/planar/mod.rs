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
    /// The chroma plane size for a `width` by `height` frame.
    ///
    /// Halved axes round up, so an odd dimension keeps the extra sample, matching what y4m and ffmpeg do.
    pub fn chroma_dims(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Subsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
            Subsampling::Yuv422 => (width.div_ceil(2), height),
            Subsampling::Yuv444 => (width, height),
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
        let (chroma_width, chroma_height) = self.chroma_dims();
        (chroma_width as usize) * (chroma_height as usize)
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
        let samples = self.luma_pixels();
        fill_plane(samples, 0, self.depth)
    }

    /// A full neutral chroma plane, used when a source has no chroma.
    pub fn neutral_chroma_plane(&self) -> Vec<u8> {
        let samples = self.chroma_pixels();
        let neutral = self.depth.neutral_chroma();
        fill_plane(samples, neutral, self.depth)
    }
}

/// Builds a plane of `samples` copies of `value` in wire-byte form.
pub fn fill_plane(samples: usize, value: u16, depth: Depth) -> Vec<u8> {
    match depth.bytes_per_sample() {
        1 => vec![value as u8; samples],
        _ => {
            let word = value.to_le_bytes();
            let mut plane = Vec::with_capacity(samples * 2);
            for _ in 0..samples {
                plane.extend_from_slice(&word);
            }

            plane
        },
    }
}

/// A planar YUV frame holding little-endian wire bytes.
///
/// Plane lengths come from [FrameLayout], so `y.len()` is `layout.luma_bytes()` and both `u.len()`
/// and `v.len()` are `layout.chroma_bytes()`.
#[derive(Debug, Clone)]
pub struct Planes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Which planes a caller wants cleaned.
///
/// This is separate from [ChannelMode] because a [PlanarDenoiser] may run two `HostDenoiser`s in
/// lockstep, one for luma and one for chroma, or a single fused three-channel one. Which applies
/// depends on the caller's channel selection and the source's chroma subsampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ChannelIntent {
    /// Denoise luma only. Chroma passes through.
    Luma,
    /// Denoise chroma only. Luma passes through.
    Chroma,
    /// Denoise luma and chroma as two independent denoisers.
    ///
    /// Chroma runs at the source's native subsampled resolution.
    LumaChroma,
    /// A single `HostDenoiser` running the fused three-channel kernel, which needs a YUV444 source.
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

/// The per-plane option set a caller resolves once and passes into [PlanarDenoiser::create].
#[derive(Debug, Clone)]
pub struct PlaneOptions {
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub intent: ChannelIntent,
    pub mode: DenoisingMode,
    /// Which denoising algorithm to run, along with the settings only that algorithm reads.
    pub algorithm: Algorithm,
    /// Strength override for the luma denoiser, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub luma_strength: Option<f32>,
    /// Strength override for the chroma denoiser, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub chroma_strength: Option<f32>,
    /// Luma override for nl4d's `lambda_ht`, the temporal grouping stage's hard threshold.
    ///
    /// It wins over `algorithm`'s value, which itself falls back to a calibrated per-plane default
    /// when neither is set.
    pub luma_lambda_ht: Option<f32>,
    /// Chroma override for nl4d's `lambda_ht`, the temporal grouping stage's hard threshold.
    ///
    /// It wins over `algorithm`'s value, which itself falls back to a calibrated per-plane default
    /// when neither is set.
    pub chroma_lambda_ht: Option<f32>,
}

impl PlaneOptions {
    /// Resolves `self.algorithm` for one plane, folding in that plane's overrides.
    ///
    /// The NLM algorithms take a strength override. Nl4d takes a `lambda_ht` override instead, because
    /// it has no NLM weighting pass for a strength to affect.
    ///
    /// An unset nl4d `lambda_ht` stays `None`, so construction resolves it with
    /// [nl4d_default_lambda_ht](crate::nl4d_default_lambda_ht) once the plane is known. That is what
    /// gives luma and chroma different values when a caller sets nothing.
    fn algorithm_for(&self, channels: ChannelMode) -> Algorithm {
        let per_plane = |luma, chroma| match channels {
            ChannelMode::Luma => luma,
            ChannelMode::Chroma => chroma,
            ChannelMode::Yuv => None,
        };

        match self.algorithm {
            Algorithm::Nl4d(nl4d) => {
                let plane_lambda_ht = per_plane(self.luma_lambda_ht, self.chroma_lambda_ht);
                let options = Nl4dOptions {
                    lambda_ht: plane_lambda_ht.or(nl4d.lambda_ht),
                    grain_export: nl4d.grain_export && channels != ChannelMode::Chroma,
                    ..nl4d
                };

                Algorithm::Nl4d(options)
            },
            Algorithm::Nlmeans(nlm) => {
                let strength = per_plane(self.luma_strength, self.chroma_strength);
                let options = with_plane_strength(nlm, strength);

                Algorithm::Nlmeans(options)
            },
            Algorithm::NlmeansHq(hq) => {
                let strength = per_plane(self.luma_strength, self.chroma_strength);
                let nlm = with_plane_strength(hq.nlm, strength);
                let options = NlmeansHqOptions { nlm, ..hq };

                Algorithm::NlmeansHq(options)
            },
        }
    }

    /// The options for one plane's denoiser at the source's wire `depth`.
    ///
    /// Every denoiser quantises to `depth` on the GPU.
    fn denoiser_options(&self, channels: ChannelMode, depth: Depth) -> DenoiserOptions {
        let algorithm = self.algorithm_for(channels);

        DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .algorithm(algorithm)
            .depth(depth)
            .build()
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

/// Reads the result of a [PlanarDenoiser::push] for the push, drain and retry loop.
///
/// `Ok(false)` means the push landed. `Ok(true)` means the queue was full, so the caller should drain
/// one output and push again. Any error other than `QueueFull` is passed on rather than discarded.
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
    let [u_plane, v_plane] = into_array(planes);
    (u_plane, v_plane)
}

fn into_luma(planes: Vec<Vec<u8>>) -> Vec<u8> {
    let [y_plane] = into_array(planes);
    y_plane
}

/// The push a [PlanarDenoiser] runs against each enabled half, either [HostDenoiser::push] or
/// [HostDenoiser::push_priming].
type WirePush = fn(&mut HostDenoiser, &[&[u8]]) -> Result<(), DenoiserError>;

/// Wraps the luma and chroma `HostDenoiser` instances needed for one subsampled YUV source.
///
/// The caller pushes planar frames in and gets planar frames out. The luma and chroma split is
/// invisible from the outside.
pub struct PlanarDenoiser {
    layout: FrameLayout,
    luma: Option<HostDenoiser>,
    chroma: Option<HostDenoiser>,
    /// Set when the intent is `YuvFused`, in which case `luma` and `chroma` are both unset.
    yuv: Option<HostDenoiser>,
    // Source planes queued for the disabled side, if there is one. One entry is popped per frame the
    // enabled side emits, so temporal delays stay aligned.
    luma_passthrough: VecDeque<Vec<u8>>,
    chroma_passthrough: VecDeque<(Vec<u8>, Vec<u8>)>,
    temporal_radius: u32,
}

impl PlanarDenoiser {
    pub fn create(options: &PlaneOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
        let (chroma_width, chroma_height) = layout.chroma_dims();

        if chroma_width == 0 || chroma_height == 0 {
            anyhow::bail!(
                "frame dimensions {}x{} are too small for subsampling {:?}",
                layout.width,
                layout.height,
                layout.subsampling
            );
        }

        options.intent.validate_for_source(layout)?;

        let (denoise_luma, denoise_chroma, denoise_yuv) = match options.intent {
            ChannelIntent::Luma => (true, false, false),
            ChannelIntent::Chroma => (false, true, false),
            ChannelIntent::LumaChroma => (true, true, false),
            ChannelIntent::YuvFused => (false, false, true),
        };

        let luma = denoise_luma
            .then(|| {
                let luma_options = options.denoiser_options(ChannelMode::Luma, layout.depth);
                HostDenoiser::create(
                    &options.accelerators,
                    &options.device,
                    layout.width,
                    layout.height,
                    luma_options,
                )
            })
            .transpose()?;

        let chroma = denoise_chroma
            .then(|| {
                let chroma_options = options.denoiser_options(ChannelMode::Chroma, layout.depth);
                HostDenoiser::create(
                    &options.accelerators,
                    &options.device,
                    chroma_width,
                    chroma_height,
                    chroma_options,
                )
            })
            .transpose()?;

        let yuv = denoise_yuv
            .then(|| {
                let yuv_options = options.denoiser_options(ChannelMode::Yuv, layout.depth);
                HostDenoiser::create(
                    &options.accelerators,
                    &options.device,
                    layout.width,
                    layout.height,
                    yuv_options,
                )
            })
            .transpose()?;

        let temporal_radius = match options.mode {
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
    /// On `QueueFull` the caller should receive one frame and then retry the whole call. Any other
    /// error is passed on unchanged. The denoiser push runs before either passthrough queue is touched,
    /// so a retry replays the frame cleanly instead of queueing the disabled side's plane twice.
    ///
    /// # Why a retry cannot duplicate a frame
    ///
    /// In `LumaChroma` mode a retry pushes again into whichever half already succeeded, which would
    /// duplicate that half's frame if the two halves could sit at different fill levels. They cannot.
    /// Both share a temporal radius and a [MAX_PENDING](crate::MAX_PENDING) ceiling, and every
    /// successful push or receive moves both on by exactly one frame. A failed push moves neither,
    /// because the `QueueFull` check runs before anything changes. So if the luma push succeeds then
    /// the chroma push succeeds too.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push)
    }

    /// Uploads one planar frame into the temporal window without starting a denoise.
    ///
    /// It queues the disabled side's passthrough plane like [Self::push], but never produces output.
    fn push_priming(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push_priming)
    }

    /// Runs `push_frame` against whichever of `yuv`, `luma` and `chroma` is enabled.
    ///
    /// The planes go over as wire bytes, so the normalisation and the channel interleave both happen
    /// on the GPU.
    fn push_with(&mut self, planes: &Planes, push_frame: WirePush) -> Result<(), DenoiserError> {
        if let Some(denoiser) = self.yuv.as_mut() {
            push_frame(denoiser, &[&planes.y, &planes.u, &planes.v])?;
            return Ok(());
        }

        if let Some(denoiser) = self.luma.as_mut() {
            push_frame(denoiser, &[&planes.y])?;
        }

        if let Some(denoiser) = self.chroma.as_mut() {
            push_frame(denoiser, &[&planes.u, &planes.v])?;
        }

        if self.luma.is_none() {
            self.luma_passthrough.push_back(planes.y.clone());
        }

        if self.chroma.is_none() {
            let chroma_pair = (planes.u.clone(), planes.v.clone());
            self.chroma_passthrough.push_back(chroma_pair);
        }

        Ok(())
    }

    /// Blocks until each enabled half emits one frame, then reassembles them into a planar frame.
    ///
    /// Returns `Ok(None)` if neither half had pending output.
    pub fn recv(&mut self) -> Result<Option<Planes>, anyhow::Error> {
        if let Some(denoiser) = self.yuv.as_mut() {
            let received = denoiser.recv()?;
            let planes = received.map(into_yuv);
            return Ok(planes);
        }

        let luma_out = self
            .luma
            .as_mut()
            .map(|denoiser| denoiser.recv())
            .transpose()?
            .flatten()
            .map(into_luma);

        let chroma_out = self
            .chroma
            .as_mut()
            .map(|denoiser| denoiser.recv())
            .transpose()?
            .flatten()
            .map(into_uv);

        // A disabled side has no HostDenoiser to query, so it pops its matching source plane when the
        // enabled side produced output.
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
        if let Some(denoiser) = self.yuv.as_mut() {
            denoiser.flush(|planes| {
                let frame = into_yuv(planes);
                sink(frame);
            })?;
            return Ok(());
        }

        let mut luma_frames: Vec<Vec<u8>> = Vec::new();
        let mut chroma_frames: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        if let Some(denoiser) = self.luma.as_mut() {
            denoiser.flush(|planes| {
                let luma = into_luma(planes);
                luma_frames.push(luma);
            })?;
        }

        if let Some(denoiser) = self.chroma.as_mut() {
            denoiser.flush(|planes| {
                let chroma = into_uv(planes);
                chroma_frames.push(chroma);
            })?;
        }

        // The two halves run in lockstep, so they flush the same number of frames. For each frame the
        // disabled side, if there is one, pops the matching source plane from its passthrough queue.
        let count = luma_frames.len().max(chroma_frames.len());

        for i in 0..count {
            let y_plane = if let Some(frame) = luma_frames.get_mut(i) {
                std::mem::take(frame)
            } else if let Some(source) = self.luma_passthrough.pop_front() {
                source
            } else {
                self.layout.black_luma_plane()
            };

            let (u_plane, v_plane) = if let Some(pair) = chroma_frames.get_mut(i) {
                std::mem::take(pair)
            } else if let Some((source_u, source_v)) = self.chroma_passthrough.pop_front() {
                (source_u, source_v)
            } else {
                (
                    self.layout.neutral_chroma_plane(),
                    self.layout.neutral_chroma_plane(),
                )
            };

            sink(Planes {
                y: y_plane,
                u: u_plane,
                v: v_plane,
            });
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

    /// The number of frames behind and ahead of a target frame a [Self::reseed] window must supply.
    ///
    /// Every owned `HostDenoiser` was built from the same algorithm, so any one of them answers for all
    /// of them.
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
        let y_plane = match (luma, luma_passthrough) {
            (Some(plane), _) => plane,
            (None, Some(source)) => source,
            (None, None) => self.layout.black_luma_plane(),
        };

        let (u_plane, v_plane) = match (chroma, chroma_passthrough) {
            (Some(pair), _) => pair,
            (None, Some(source)) => source,
            (None, None) => (
                self.layout.neutral_chroma_plane(),
                self.layout.neutral_chroma_plane(),
            ),
        };

        Planes {
            y: y_plane,
            u: u_plane,
            v: v_plane,
        }
    }
}

#[cfg(test)]
mod cli_options_tests {
    use av_denoise_core::NlmeansAlgorithm;

    use super::*;
    use crate::backend::EngineSpec;

    /// A `PlaneOptions` with every field other than the four arguments at a neutral default.
    fn base_options(
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
        let algorithm = Algorithm::default();
        let plane_options = base_options(DenoisingMode::Spacial, algorithm, Some(0.7), None);

        let luma_options = plane_options.denoiser_options(ChannelMode::Luma, Depth::Eight);
        let chroma_options = plane_options.denoiser_options(ChannelMode::Chroma, Depth::Eight);
        let luma = expect_nlmeans(luma_options.algorithm);
        let chroma = expect_nlmeans(chroma_options.algorithm);

        assert!(
            matches!(luma.tuning.strength, Some(strength) if (strength - 0.7).abs() < f32::EPSILON),
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
        let algorithm = Algorithm::default();
        let plane_options = base_options(DenoisingMode::Spacial, algorithm, Some(0.7), Some(0.3));

        let luma_options = plane_options.denoiser_options(ChannelMode::Luma, Depth::Eight);
        let chroma_options = plane_options.denoiser_options(ChannelMode::Chroma, Depth::Eight);
        let luma = expect_nlmeans(luma_options.algorithm);
        let chroma = expect_nlmeans(chroma_options.algorithm);

        assert!(
            matches!(luma.tuning.strength, Some(strength) if (strength - 0.7).abs() < f32::EPSILON),
            "expected luma tuning.strength = Some(0.7), got {:?}",
            luma.tuning.strength
        );
        assert!(
            matches!(chroma.tuning.strength, Some(strength) if (strength - 0.3).abs() < f32::EPSILON),
            "expected chroma tuning.strength = Some(0.3), got {:?}",
            chroma.tuning.strength
        );
    }

    #[test]
    fn no_overrides_hq_leaves_strength_to_the_per_plane_table() {
        let hq_options = NlmeansHqOptions::default();
        let plane_options = base_options(
            DenoisingMode::Temporal { radius: 4 },
            Algorithm::NlmeansHq(hq_options),
            None,
            None,
        );

        for channels in [ChannelMode::Luma, ChannelMode::Chroma] {
            let options = plane_options.denoiser_options(channels, Depth::Eight);
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

    /// A `PlaneOptions` running `Algorithm::Nl4d` with only the two `lambda_ht` overrides set.
    fn nl4d_options(luma_lambda_ht: Option<f32>, chroma_lambda_ht: Option<f32>) -> PlaneOptions {
        let default_nl4d = Nl4dOptions::default();

        PlaneOptions {
            accelerators: vec![],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: 2 },
            algorithm: Algorithm::Nl4d(default_nl4d),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht,
            chroma_lambda_ht,
        }
    }

    fn expect_nlmeans(algorithm: Algorithm) -> NlmeansOptions {
        match algorithm {
            Algorithm::Nlmeans(options) => options,
            other => panic!("expected Algorithm::Nlmeans, got {other:?}"),
        }
    }

    fn expect_nl4d(algorithm: Algorithm) -> Nl4dOptions {
        match algorithm {
            Algorithm::Nl4d(options) => options,
            other => panic!("expected Algorithm::Nl4d, got {other:?}"),
        }
    }

    #[test]
    fn luma_lambda_ht_alone_overrides_only_the_luma_instance_for_nl4d() {
        let plane_options = nl4d_options(Some(4.0), None);
        let default_lambda_ht = Nl4dOptions::default().lambda_ht;

        let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
        let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
        let luma = expect_nl4d(luma_algorithm);
        let chroma = expect_nl4d(chroma_algorithm);

        assert!((luma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
        assert_eq!(
            chroma.lambda_ht, default_lambda_ht,
            "chroma should stay unresolved here (None), deferred to its own per-plane \
             default at construction, got {:?}",
            chroma.lambda_ht
        );
    }

    #[test]
    fn chroma_lambda_ht_alone_overrides_only_the_chroma_instance_for_nl4d() {
        let plane_options = nl4d_options(None, Some(4.0));
        let default_lambda_ht = Nl4dOptions::default().lambda_ht;

        let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
        let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
        let luma = expect_nl4d(luma_algorithm);
        let chroma = expect_nl4d(chroma_algorithm);

        assert_eq!(
            luma.lambda_ht, default_lambda_ht,
            "luma should stay unresolved here (None), deferred to its own per-plane \
             default at construction, got {:?}",
            luma.lambda_ht
        );
        assert!((chroma.lambda_ht.unwrap() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn both_planes_lambda_ht_set_independently_for_nl4d() {
        let plane_options = nl4d_options(Some(2.0), Some(3.5));

        let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
        let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
        let luma = expect_nl4d(luma_algorithm);
        let chroma = expect_nl4d(chroma_algorithm);

        assert!((luma.lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((chroma.lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);

        // Every other field stays shared between the two instances even though lambda_ht diverges.
        assert_eq!(luma.refine, chroma.refine);
        assert_eq!(luma.spatial_radius, chroma.spatial_radius);
        assert!((luma.c_min - chroma.c_min).abs() < f32::EPSILON);
    }

    #[test]
    fn unset_nl4d_overrides_resolve_to_different_lambda_ht_per_plane_end_to_end() {
        let plane_options = nl4d_options(None, None);

        let luma_algorithm = plane_options.algorithm_for(ChannelMode::Luma);
        let chroma_algorithm = plane_options.algorithm_for(ChannelMode::Chroma);
        let luma = expect_nl4d(luma_algorithm);
        let chroma = expect_nl4d(chroma_algorithm);

        // Neither plane has anything set, so both stay unresolved at this layer.
        assert_eq!(luma.lambda_ht, None);
        assert_eq!(chroma.lambda_ht, None);

        // Construction resolves each through `nl4d_default_lambda_ht`, which gives luma and chroma
        // different values.
        let luma_default = crate::nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma_default = crate::nl4d_default_lambda_ht(ChannelMode::Chroma);
        assert!((luma_default - 4.158).abs() < f32::EPSILON);
        assert!((chroma_default - 3.234).abs() < f32::EPSILON);
        assert!((chroma_default - luma_default).abs() > f32::EPSILON);
    }
}

// Gated on `vulkan` because `chroma_only_options` names the `Vulkan` accelerator variant.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod passthrough_retry_tests {
    use super::*;
    use crate::accelerate::Accelerator;
    use crate::{Algorithm, DenoisingMode};

    /// Chroma-only intent, so `luma` is the disabled passthrough half and `chroma` is the one that can
    /// report `QueueFull`.
    fn chroma_only_options() -> PlaneOptions {
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
        let luma_pixels = layout.luma_pixels();
        let neutral = layout.depth.neutral_chroma();

        Planes {
            y: fill_plane(luma_pixels, neutral, layout.depth),
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
        let options = chroma_only_options();
        let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");
        let planes = fake_planes(layout);

        // Spatial mode runs a depth-2 pipeline, so the first two pushes land directly.
        denoiser.push(&planes).expect("first push should land");
        denoiser.push(&planes).expect("second push should land");

        // The third push hits QueueFull on the chroma half.
        let err = denoiser.push(&planes).expect_err("expected QueueFull");
        assert!(
            matches!(err, DenoiserError::QueueFull),
            "expected QueueFull, got {err:?}"
        );

        // Drain one output, then retry the whole push for the same frame.
        denoiser.recv().expect("recv after drain failed");
        denoiser
            .push(&planes)
            .expect("retry push should land after drain");

        // Chroma accepted three frames and `recv` popped one, so the luma passthrough queue must hold
        // two and must not count the frame whose first attempt hit `QueueFull` twice.
        assert_eq!(
            denoiser.luma_passthrough.len(),
            2,
            "expected exactly one passthrough entry per chroma frame actually accepted, got {}",
            denoiser.luma_passthrough.len()
        );
    }
}

// Gated on `vulkan` because `luma_chroma_options` names the `Vulkan` accelerator variant.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod lumachroma_lockstep_tests {
    use super::*;
    use crate::accelerate::Accelerator;
    use crate::{Algorithm, DenoisingMode};

    /// Runs `luma` and `chroma` as two real `HostDenoiser`s in spatial mode.
    ///
    /// Spatial mode passes a uniform-valued plane through unchanged, so each plane can carry its own
    /// marker value and the two halves drifting apart shows up.
    fn luma_chroma_options() -> PlaneOptions {
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

    /// A uniform-valued frame whose luma and chroma planes encode `frame_index` with different formulas.
    ///
    /// Pairing luma from one push with chroma from another makes the two encodings disagree.
    fn marked_planes(layout: FrameLayout, frame_index: u8) -> Planes {
        let luma_pixels = layout.luma_pixels();
        let chroma_pixels = layout.chroma_pixels();
        let luma_marker = 10 + frame_index;
        let chroma_marker = 200 - frame_index;

        Planes {
            y: fill_plane(luma_pixels, luma_marker as u16, layout.depth),
            u: fill_plane(chroma_pixels, chroma_marker as u16, layout.depth),
            v: fill_plane(chroma_pixels, chroma_marker as u16, layout.depth),
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
        let options = luma_chroma_options();
        let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

        let luma_pixels = layout.luma_pixels();
        let chroma_pixels = layout.chroma_pixels();
        let planes = Planes {
            y: fill_plane(luma_pixels, 100, layout.depth),
            u: fill_plane(chroma_pixels, 60, layout.depth),
            v: fill_plane(chroma_pixels, 190, layout.depth),
        };
        denoiser.push(&planes).expect("push failed");

        let received = denoiser.recv().expect("recv failed");
        let denoised = received.expect("spatial mode emits one frame per push");

        for &sample in &denoised.u {
            assert!(sample.abs_diff(60) <= 2, "U sample {sample}, expected about 60");
        }

        for &sample in &denoised.v {
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
        let options = luma_chroma_options();
        let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

        // More pushes than the depth-2 pipeline holds, so this drives several `QueueFull` retries.
        const FRAME_COUNT: u8 = 6;
        let mut outputs: Vec<Planes> = Vec::new();

        for frame_index in 0..FRAME_COUNT {
            let planes = marked_planes(layout, frame_index);

            // The push, drain and retry sequence a streaming caller runs.
            let pushed = denoiser.push(&planes);
            let needs_retry = push_needs_retry(pushed).expect("push_needs_retry");
            if needs_retry {
                if let Some(denoised) = denoiser.recv().expect("recv failed") {
                    outputs.push(denoised);
                }

                denoiser
                    .push(&planes)
                    .expect("retry push should land after drain");
            }
        }

        denoiser
            .flush(|denoised| outputs.push(denoised))
            .expect("flush failed");

        assert_eq!(
            outputs.len(),
            FRAME_COUNT as usize,
            "expected exactly one output frame per input frame, got {}",
            outputs.len()
        );

        for denoised in &outputs {
            let luma_marker = denoised.y[0];
            let chroma_marker = denoised.u[0];
            let index_from_luma = luma_marker - 10;
            let index_from_chroma = 200 - chroma_marker;

            assert_eq!(
                index_from_luma, index_from_chroma,
                "luma marker {luma_marker} (frame {index_from_luma}) and chroma marker {chroma_marker} \
                 (frame {index_from_chroma}) disagree, so the luma and chroma pushes have drifted apart"
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
        let cause = anyhow::anyhow!("synthetic readback failure");
        let synthetic = DenoiserError::Other(cause);

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
        let eight_bit = layout(Depth::Eight);
        let ten_bit = layout(Depth::Ten);

        assert_eq!(eight_bit.luma_bytes(), 16);
        assert_eq!(ten_bit.luma_bytes(), 32);
        assert_eq!(eight_bit.chroma_bytes(), 4);
        assert_eq!(ten_bit.chroma_bytes(), 8);
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
        let eight = layout(Depth::Eight).black_luma_plane();
        let ten = layout(Depth::Ten).black_luma_plane();

        assert_eq!(eight, vec![0u8; 16]);
        assert_eq!(ten, vec![0u8; 32]);
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
