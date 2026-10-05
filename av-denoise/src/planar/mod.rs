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
    /// Whether to clean each frame on its own or across a temporal window.
    ///
    /// This wins over `NlmeansOptions.mode` and `Nl4dOptions.temporal_radius` inside `algorithm`.
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
        self.check_planes(planes)?;

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

    /// Rejects a frame whose plane lengths do not match the layout.
    ///
    /// A half that rejects a plane stays usable, so checking every plane before either half is pushed keeps
    /// luma and chroma from drifting a frame apart.
    fn check_planes(&self, planes: &Planes) -> Result<(), DenoiserError> {
        let luma_bytes = self.layout.luma_bytes();
        let chroma_bytes = self.layout.chroma_bytes();
        let expected = [
            ("Y", planes.y.len(), luma_bytes),
            ("U", planes.u.len(), chroma_bytes),
            ("V", planes.v.len(), chroma_bytes),
        ];

        for (name, length, expected_length) in expected {
            if length != expected_length {
                let message = format!("plane {name} holds {length} bytes, expected {expected_length}");
                let error = av_denoise_core::Error::PlaneMismatch(message);
                return Err(DenoiserError::Engine(error));
            }
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
