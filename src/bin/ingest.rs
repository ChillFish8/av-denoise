use std::collections::VecDeque;

use av_denoise::accelerate::Accelerator;
use av_denoise::{
    Algorithm,
    ChannelMode,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    Depth,
    Device,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    pub fn chroma_dims(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            Subsampling::Yuv420 => (w / 2, h / 2),
            Subsampling::Yuv422 => (w / 2, h),
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
pub(crate) fn fill_plane(samples: usize, value: u16, depth: Depth) -> Vec<u8> {
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

/// Which planes the binary was asked to clean, once `--channel-mode` has
/// been resolved.
///
/// This is separate from the library's [`ChannelMode`] because the binary
/// may run more than one `Denoiser` in lockstep, one for luma and one for
/// chroma. It may also run a single fused three-channel denoiser instead.
/// Which of those applies depends on the user's `--channel-mode` and the
/// source's chroma subsampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryChannelIntent {
    /// Denoise luma only. Chroma passes through.
    Luma,
    /// Denoise chroma only. Luma passes through.
    Chroma,
    /// Denoise both luma and chroma as two independent denoisers.
    /// Chroma runs at the source's native subsampled resolution.
    LumaChroma,
    /// A single library `Denoiser` running the fused three-channel
    /// kernel. Needs a YUV444 source, which is checked at ingest setup
    /// time.
    YuvFused,
}

impl BinaryChannelIntent {
    /// Rejects the intent if the source's subsampling cannot support it.
    pub fn validate_for_source(self, layout: FrameLayout) -> Result<(), anyhow::Error> {
        match self {
            BinaryChannelIntent::YuvFused if layout.subsampling != Subsampling::Yuv444 => {
                anyhow::bail!(
                    "--channel-mode yuv requires a YUV444 source, got {:?}. Convert the input first, for example with `ffmpeg -pix_fmt yuv444p`",
                    layout.subsampling
                );
            },
            _ => Ok(()),
        }
    }
}

/// CLI-shaped option set forwarded from `main` into ingest modules.
#[derive(Debug, Clone)]
pub struct CliOptions {
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub intent: BinaryChannelIntent,
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
    /// Per-plane override for `mismatch_scale`, luma. Takes precedence
    /// over `algorithm`'s value when set. Only has an effect when
    /// `algorithm` is `Algorithm::Nl4d`.
    pub luma_mismatch_scale: Option<f32>,
    /// Per-plane override for `mismatch_scale`, chroma. Takes precedence
    /// over `algorithm`'s value when set. Only has an effect when
    /// `algorithm` is `Algorithm::Nl4d`.
    pub chroma_mismatch_scale: Option<f32>,
    /// Draws the denoising progress bar for file input.
    pub progress: bool,
}

impl CliOptions {
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
                // Unlike `lambda_ht` this has one default for both
                // planes, so an unset override simply leaves the shared
                // value in place rather than deferring to construction.
                mismatch_scale: per_plane(self.luma_mismatch_scale, self.chroma_mismatch_scale)
                    .unwrap_or(nl4d.mismatch_scale),
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

    fn denoiser_options(&self, channels: ChannelMode) -> DenoiserOptions {
        DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .algorithm(self.algorithm_for(channels))
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

/// Reads the result of a `WorkerDenoiser::push` call for the
/// push-then-drain-then-retry loop that `file_mode.rs` and
/// `stream_mode.rs` both use.
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

/// Wraps the luma and chroma `Denoiser` instances needed for one
/// subsampled YUV source.
///
/// The caller pushes planar frames in and gets planar frames out. The
/// luma and chroma split is invisible from the outside.
pub struct WorkerDenoiser {
    layout: FrameLayout,
    luma: Option<Denoiser>,
    chroma: Option<Denoiser>,
    /// Set when the intent is `YuvFused`, in which case `luma` and
    /// `chroma` are both unset.
    yuv: Option<Denoiser>,
    // Source planes queued for passthrough when the matching denoiser is
    // disabled. Only the disabled side's queue is ever filled. Entries
    // are popped one per frame the enabled side emits, so temporal
    // delays stay aligned.
    luma_passthrough: VecDeque<Vec<u8>>,
    chroma_passthrough: VecDeque<(Vec<u8>, Vec<u8>)>,
}

impl WorkerDenoiser {
    pub fn create(opts: &CliOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
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
            BinaryChannelIntent::Luma => (true, false, false),
            BinaryChannelIntent::Chroma => (false, true, false),
            BinaryChannelIntent::LumaChroma => (true, true, false),
            BinaryChannelIntent::YuvFused => (false, false, true),
        };

        let luma = denoise_luma
            .then(|| {
                Denoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    layout.width,
                    layout.height,
                    opts.denoiser_options(ChannelMode::Luma),
                )
            })
            .transpose()?;

        let chroma = denoise_chroma
            .then(|| {
                Denoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    chroma_w,
                    chroma_h,
                    opts.denoiser_options(ChannelMode::Chroma),
                )
            })
            .transpose()?;

        let yuv = denoise_yuv
            .then(|| {
                Denoiser::create(
                    &opts.accelerators,
                    &opts.device,
                    layout.width,
                    layout.height,
                    opts.denoiser_options(ChannelMode::Yuv),
                )
            })
            .transpose()?;

        Ok(Self {
            layout,
            luma,
            chroma,
            yuv,
            luma_passthrough: VecDeque::new(),
            chroma_passthrough: VecDeque::new(),
        })
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
    /// `Denoiser`s with their own queues. A retry pushes again into
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
    /// inside `push_frame` answers the same way for each. If the luma
    /// push succeeds then the chroma push succeeds too, which makes the
    /// duplicate unreachable.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        if let Some(d) = self.yuv.as_mut() {
            let buf = interleave_yuv_to_f32(&planes.y, &planes.u, &planes.v, self.layout.depth);
            d.push_frame(&buf)?;
            return Ok(());
        }

        if let Some(d) = self.luma.as_mut() {
            let buf = plane_to_f32(&planes.y, self.layout.depth);
            d.push_frame(&buf)?;
        }

        if let Some(d) = self.chroma.as_mut() {
            let buf = interleave_uv_to_f32(&planes.u, &planes.v, self.layout.depth);
            d.push_frame(&buf)?;
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
            return match d.recv_frame()? {
                Some(packed) => Ok(Some(unpack_yuv_from_f32(
                    &packed,
                    self.layout.luma_pixels(),
                    self.layout.depth,
                ))),
                None => Ok(None),
            };
        }

        let luma_out = self.luma.as_mut().map(|d| d.recv_frame()).transpose()?.flatten();

        let chroma_out = self
            .chroma
            .as_mut()
            .map(|d| d.recv_frame())
            .transpose()?
            .flatten();

        // A disabled side has no Denoiser to query. When the enabled side
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

    /// Drains the temporal tail of both halves.
    ///
    /// `sink` is called once per emitted planar frame.
    pub fn flush(&mut self, mut sink: impl FnMut(Planes)) -> Result<(), anyhow::Error> {
        if let Some(d) = self.yuv.as_mut() {
            let pixels = self.layout.luma_pixels();
            let depth = self.layout.depth;
            d.flush(|packed| sink(unpack_yuv_from_f32(&packed, pixels, depth)))?;
            return Ok(());
        }

        let chroma_pixels = self.layout.chroma_pixels();

        let mut luma_buf: Vec<Vec<f32>> = Vec::new();
        let mut chroma_buf: Vec<Vec<f32>> = Vec::new();

        if let Some(d) = self.luma.as_mut() {
            d.flush(|v| luma_buf.push(v))?;
        }

        if let Some(d) = self.chroma.as_mut() {
            d.flush(|v| chroma_buf.push(v))?;
        }

        // The two halves run in lockstep, so they flush the same number
        // of frames. For each emitted frame the disabled side, if there
        // is one, pops the matching source plane from its passthrough
        // queue.
        let count = luma_buf.len().max(chroma_buf.len());

        for i in 0..count {
            let y = if let Some(buf) = luma_buf.get(i) {
                f32_to_plane(buf, self.layout.depth)
            } else if let Some(src) = self.luma_passthrough.pop_front() {
                src
            } else {
                self.layout.black_luma_plane()
            };

            let (u, v) = if let Some(packed) = chroma_buf.get(i) {
                unpack_uv_from_f32(packed, chroma_pixels, self.layout.depth)
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

    fn assemble(
        &self,
        luma: Option<Vec<f32>>,
        chroma: Option<Vec<f32>>,
        luma_passthrough: Option<Vec<u8>>,
        chroma_passthrough: Option<(Vec<u8>, Vec<u8>)>,
    ) -> Planes {
        let chroma_pixels = self.layout.chroma_pixels();

        let y = match (luma, luma_passthrough) {
            (Some(v), _) => f32_to_plane(&v, self.layout.depth),
            (None, Some(src)) => src,
            (None, None) => self.layout.black_luma_plane(),
        };

        let (u, v) = match (chroma, chroma_passthrough) {
            (Some(packed), _) => unpack_uv_from_f32(&packed, chroma_pixels, self.layout.depth),
            (None, Some(src)) => src,
            (None, None) => (
                self.layout.neutral_chroma_plane(),
                self.layout.neutral_chroma_plane(),
            ),
        };

        Planes { y, u, v }
    }
}

/// Reads and writes samples in one wire format.
///
/// The implementor is chosen once per conversion, which keeps the
/// per-sample path free of depth branches.
trait SampleCodec {
    const BYTES: usize;

    fn read(plane: &[u8], i: usize) -> u16;
    fn write(plane: &mut [u8], i: usize, value: u16);
}

/// One byte per sample.
struct Narrow;

impl SampleCodec for Narrow {
    const BYTES: usize = 1;

    #[inline(always)]
    fn read(plane: &[u8], i: usize) -> u16 {
        plane[i] as u16
    }

    #[inline(always)]
    fn write(plane: &mut [u8], i: usize, value: u16) {
        plane[i] = value as u8;
    }
}

/// Two bytes per sample, little-endian.
struct Wide;

impl SampleCodec for Wide {
    const BYTES: usize = 2;

    #[inline(always)]
    fn read(plane: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([plane[2 * i], plane[2 * i + 1]])
    }

    #[inline(always)]
    fn write(plane: &mut [u8], i: usize, value: u16) {
        plane[2 * i..2 * i + 2].copy_from_slice(&value.to_le_bytes());
    }
}

/// Quantises a normalised value to a native-depth sample.
#[inline(always)]
fn quantise(v: f32, max: f32) -> u16 {
    (v.clamp(0.0, 1.0) * max + 0.5) as u16
}

/// Converts one wire-byte plane to normalised f32.
fn plane_to_f32(plane: &[u8], depth: Depth) -> Vec<f32> {
    let max = depth.max_value();

    fn run<C: SampleCodec>(plane: &[u8], max: f32) -> Vec<f32> {
        let samples = plane.len() / C::BYTES;
        (0..samples).map(|i| C::read(plane, i) as f32 / max).collect()
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(plane, max),
        _ => run::<Wide>(plane, max),
    }
}

/// Reverse of [`plane_to_f32`].
fn f32_to_plane(plane: &[f32], depth: Depth) -> Vec<u8> {
    let max = depth.max_value();

    fn run<C: SampleCodec>(plane: &[f32], max: f32) -> Vec<u8> {
        let mut out = vec![0u8; plane.len() * C::BYTES];
        for (i, &v) in plane.iter().enumerate() {
            C::write(&mut out, i, quantise(v, max));
        }
        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(plane, max),
        _ => run::<Wide>(plane, max),
    }
}

/// Interleaves equal-length Y, U, and V planes from a YUV444 source into
/// `[Y0, U0, V0, Y1, U1, V1, ...]` as f32 in `[0, 1]`.
///
/// This is the layout the library's fused three-channel kernel expects.
fn interleave_yuv_to_f32(y: &[u8], u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(y.len(), u.len());
    debug_assert_eq!(u.len(), v.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(y: &[u8], u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
        let pixels = y.len() / C::BYTES;
        let mut out = Vec::with_capacity(pixels * 3);

        for i in 0..pixels {
            out.push(C::read(y, i) as f32 / max);
            out.push(C::read(u, i) as f32 / max);
            out.push(C::read(v, i) as f32 / max);
        }

        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(y, u, v, max),
        _ => run::<Wide>(y, u, v, max),
    }
}

/// Reverse of [`interleave_yuv_to_f32`].
fn unpack_yuv_from_f32(packed: &[f32], pixels: usize, depth: Depth) -> Planes {
    debug_assert_eq!(packed.len(), 3 * pixels);

    let max = depth.max_value();

    fn run<C: SampleCodec>(packed: &[f32], pixels: usize, max: f32) -> Planes {
        let mut y = vec![0u8; pixels * C::BYTES];
        let mut u = vec![0u8; pixels * C::BYTES];
        let mut v = vec![0u8; pixels * C::BYTES];

        for (i, chunk) in packed.chunks_exact(3).enumerate() {
            C::write(&mut y, i, quantise(chunk[0], max));
            C::write(&mut u, i, quantise(chunk[1], max));
            C::write(&mut v, i, quantise(chunk[2], max));
        }

        Planes { y, u, v }
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(packed, pixels, max),
        _ => run::<Wide>(packed, pixels, max),
    }
}

/// Interleaves separate U and V planes into `[U, V, U, V, ...]` as f32
/// in `[0, 1]`.
fn interleave_uv_to_f32(u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(u.len(), v.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
        let pixels = u.len() / C::BYTES;
        let mut out = Vec::with_capacity(pixels * 2);

        for i in 0..pixels {
            out.push(C::read(u, i) as f32 / max);
            out.push(C::read(v, i) as f32 / max);
        }

        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(u, v, max),
        _ => run::<Wide>(u, v, max),
    }
}

/// Reverse of [`interleave_uv_to_f32`].
fn unpack_uv_from_f32(packed: &[f32], chroma_pixels: usize, depth: Depth) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(packed.len(), 2 * chroma_pixels);

    let max = depth.max_value();

    fn run<C: SampleCodec>(packed: &[f32], chroma_pixels: usize, max: f32) -> (Vec<u8>, Vec<u8>) {
        let mut u = vec![0u8; chroma_pixels * C::BYTES];
        let mut v = vec![0u8; chroma_pixels * C::BYTES];

        for (i, chunk) in packed.chunks_exact(2).enumerate() {
            C::write(&mut u, i, quantise(chunk[0], max));
            C::write(&mut v, i, quantise(chunk[1], max));
        }

        (u, v)
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(packed, chroma_pixels, max),
        _ => run::<Wide>(packed, chroma_pixels, max),
    }
}

#[cfg(test)]
mod converter_tests {
    use super::*;

    /// Encodes native-depth samples into wire bytes, the inverse of what
    /// the converters read.
    fn wire(samples: &[u16], depth: Depth) -> Vec<u8> {
        match depth.bytes_per_sample() {
            1 => samples.iter().map(|&s| s as u8).collect(),
            _ => samples.iter().flat_map(|&s| s.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn plane_round_trips_boundary_codes_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let samples: Vec<u16> = vec![0, 1, 16, 64, 235, max / 2, max - 1, max]
                .into_iter()
                .filter(|&s| s <= max)
                .collect();

            let bytes = wire(&samples, depth);
            let restored = f32_to_plane(&plane_to_f32(&bytes, depth), depth);

            assert_eq!(restored, bytes, "plane round trip failed at {depth:?}");
        }
    }

    /// Samples above 8 bits are little-endian on the wire regardless of
    /// host endianness.
    #[test]
    fn high_depth_samples_are_little_endian() {
        // 1023 = 0x03FF -> [0xFF, 0x03]
        let bytes = wire(&[1023, 0, 512], Depth::Ten);
        assert_eq!(bytes, vec![0xFF, 0x03, 0x00, 0x00, 0x00, 0x02]);

        let f = plane_to_f32(&bytes, Depth::Ten);
        assert!(
            (f[0] - 1.0).abs() < 1e-6,
            "0x03FF should normalize to 1.0, got {}",
            f[0]
        );
        assert_eq!(f[1], 0.0);
    }

    #[test]
    fn uv_interleave_round_trips_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let u_samples = vec![0, max / 4, max];
            let v_samples = vec![max, max / 2, 1];

            let u_bytes = wire(&u_samples, depth);
            let v_bytes = wire(&v_samples, depth);

            let packed = interleave_uv_to_f32(&u_bytes, &v_bytes, depth);
            assert_eq!(packed.len(), 6, "packed UV length wrong at {depth:?}");

            let (ru, rv) = unpack_uv_from_f32(&packed, 3, depth);
            assert_eq!(ru, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(rv, v_bytes, "V round trip failed at {depth:?}");
        }
    }

    #[test]
    fn yuv_interleave_round_trips_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let y_samples = vec![0, max / 3, max];
            let u_samples = vec![max, 0, max / 2];
            let v_samples = vec![max / 4, max, 0];

            let y_bytes = wire(&y_samples, depth);
            let u_bytes = wire(&u_samples, depth);
            let v_bytes = wire(&v_samples, depth);

            let packed = interleave_yuv_to_f32(&y_bytes, &u_bytes, &v_bytes, depth);
            assert_eq!(packed.len(), 9, "packed YUV length wrong at {depth:?}");

            let out = unpack_yuv_from_f32(&packed, 3, depth);
            assert_eq!(out.y, y_bytes, "Y round trip failed at {depth:?}");
            assert_eq!(out.u, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(out.v, v_bytes, "V round trip failed at {depth:?}");
        }
    }

    #[test]
    fn quantise_matches_the_clamping_form_including_nan() {
        fn reference(v: f32, max: f32) -> u16 {
            (v.clamp(0.0, 1.0) * max + 0.5) as u16
        }

        let max = 1023.0;
        let cases = [
            -1.0,
            -0.001,
            0.0,
            0.5,
            0.999,
            1.0,
            1.001,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        for v in cases {
            assert_eq!(quantise(v, max), reference(v, max), "mismatch at {v}");
        }
    }

    /// Limited-range codes normalise to matching values at every depth,
    /// which is the property the whole design rests on.
    ///
    /// The match is within one 8-bit code level rather than exact. ITU
    /// defines the limited-range endpoints as exact multiples, so 16
    /// becomes 64 and 235 becomes 940, but full scale is not a multiple,
    /// because 255 becomes 1023. That leaves 235/255 and 940/1023
    /// differing by 0.0027, roughly 0.69 of an 8-bit step.
    ///
    /// Agreement below one step is the real property here.
    ///
    /// `normalized_scale_is_identical_across_depths` in
    /// `src/nlmeans/mod.rs` pins the same property on the library's own
    /// normalise helper.
    #[test]
    fn limited_range_codes_agree_across_depths() {
        /// One 8-bit code level, the precision the endpoints agree to.
        const TOL: f32 = 1.0 / 255.0;

        let eight = plane_to_f32(&wire(&[16, 235], Depth::Eight), Depth::Eight);
        let ten = plane_to_f32(&wire(&[64, 940], Depth::Ten), Depth::Ten);

        for (a, b) in eight.iter().zip(ten.iter()) {
            assert!((a - b).abs() < TOL, "8-bit {a} vs 10-bit {b}");
        }
    }
}

#[cfg(test)]
mod cli_options_tests {
    use av_denoise::nlmeans::NlmParams;

    use super::*;

    /// A `CliOptions` with every field at a neutral default, so each test
    /// only overrides what it cares about.
    ///
    /// `mode` and `algorithm` are the two fields every test below sets
    /// for itself.
    fn base_opts(
        mode: DenoisingMode,
        algorithm: Algorithm,
        luma_strength: Option<f32>,
        chroma_strength: Option<f32>,
    ) -> CliOptions {
        CliOptions {
            accelerators: vec![],
            device: Device::Default,
            intent: BinaryChannelIntent::LumaChroma,
            mode,
            algorithm,
            luma_strength,
            chroma_strength,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
            progress: false,
        }
    }

    #[test]
    fn luma_strength_alone_overrides_only_the_luma_plane() {
        let opts = base_opts(DenoisingMode::Spacial, Algorithm::default(), Some(0.7), None);

        let luma = expect_nlmeans(opts.denoiser_options(ChannelMode::Luma).algorithm);
        let chroma = expect_nlmeans(opts.denoiser_options(ChannelMode::Chroma).algorithm);

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

        let luma = expect_nlmeans(opts.denoiser_options(ChannelMode::Luma).algorithm);
        let chroma = expect_nlmeans(opts.denoiser_options(ChannelMode::Chroma).algorithm);

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
    fn no_overrides_hq_resolves_through_to_nlm_params_to_the_measured_tables() {
        // Radius 4 in the measured tables is luma 0.35 and chroma
        // 0.70 (see the table docs in `src/nlmeans/params.rs`).
        let opts = base_opts(
            DenoisingMode::Temporal { radius: 4 },
            Algorithm::NlmeansHq(NlmeansHqOptions::default()),
            None,
            None,
        );

        let luma_params: NlmParams = opts.denoiser_options(ChannelMode::Luma).to_nlm_params();
        let chroma_params: NlmParams = opts.denoiser_options(ChannelMode::Chroma).to_nlm_params();

        assert!(
            (luma_params.strength - 0.35).abs() < f32::EPSILON,
            "expected luma strength 0.35 at r4, got {}",
            luma_params.strength
        );
        assert!(
            (chroma_params.strength - 0.70).abs() < f32::EPSILON,
            "expected chroma strength 0.70 at r4, got {}",
            chroma_params.strength
        );
    }

    /// A `CliOptions` running `Algorithm::Nl4d`, with every field at a
    /// neutral default except the two per-plane `lambda_ht` overrides
    /// under test.
    fn nl4d_opts(luma_lambda_ht: Option<f32>, chroma_lambda_ht: Option<f32>) -> CliOptions {
        CliOptions {
            accelerators: vec![],
            device: Device::Default,
            intent: BinaryChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: 2 },
            algorithm: Algorithm::Nl4d(Nl4dOptions::default()),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht,
            chroma_lambda_ht,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
            progress: false,
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

    /// A `CliOptions` running `Algorithm::Nl4d` with a shared
    /// `mismatch_scale` and the two per-plane overrides under test.
    fn nl4d_mismatch_opts(
        shared: f32,
        luma_mismatch_scale: Option<f32>,
        chroma_mismatch_scale: Option<f32>,
    ) -> CliOptions {
        CliOptions {
            algorithm: Algorithm::Nl4d(Nl4dOptions {
                mismatch_scale: shared,
                ..Nl4dOptions::default()
            }),
            luma_mismatch_scale,
            chroma_mismatch_scale,
            ..nl4d_opts(None, None)
        }
    }

    /// The same routing property the `lambda_ht` pair is checked for,
    /// applied to `mismatch_scale`. An override aimed at one plane must
    /// leave the other on the shared value, which for this field is a
    /// resolved number rather than a deferred `None`.
    #[test]
    fn a_per_plane_mismatch_scale_overrides_only_its_own_instance_for_nl4d() {
        let luma_only = nl4d_mismatch_opts(2.0, Some(8.0), None);
        let luma = expect_nl4d(luma_only.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(luma_only.algorithm_for(ChannelMode::Chroma));
        assert!((luma.mismatch_scale - 8.0).abs() < f32::EPSILON);
        assert!(
            (chroma.mismatch_scale - 2.0).abs() < f32::EPSILON,
            "chroma should keep the shared value, got {}",
            chroma.mismatch_scale
        );

        let chroma_only = nl4d_mismatch_opts(2.0, None, Some(8.0));
        let luma = expect_nl4d(chroma_only.algorithm_for(ChannelMode::Luma));
        let chroma = expect_nl4d(chroma_only.algorithm_for(ChannelMode::Chroma));
        assert!((chroma.mismatch_scale - 8.0).abs() < f32::EPSILON);
        assert!(
            (luma.mismatch_scale - 2.0).abs() < f32::EPSILON,
            "luma should keep the shared value, got {}",
            luma.mismatch_scale
        );
    }

    /// A fused Yuv pass has no plane to pick, so neither override
    /// applies and the shared value stands.
    #[test]
    fn a_yuv_instance_ignores_both_per_plane_mismatch_scales() {
        let opts = nl4d_mismatch_opts(2.0, Some(8.0), Some(4.0));
        let yuv = expect_nl4d(opts.algorithm_for(ChannelMode::Yuv));

        assert!((yuv.mismatch_scale - 2.0).abs() < f32::EPSILON);
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
        // uses (`nl4d_default_lambda_ht`, see `src/denoiser.rs`) gives
        // luma and chroma different values, which is the whole point of
        // a caller passing no flags at all getting both calibrated
        // defaults.
        let luma_default = av_denoise::nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma_default = av_denoise::nl4d_default_lambda_ht(ChannelMode::Chroma);
        assert!((luma_default - 5.3).abs() < f32::EPSILON);
        assert!((chroma_default - 4.2).abs() < f32::EPSILON);
        assert!((chroma_default - luma_default).abs() > f32::EPSILON);
    }
}

// Feature-gated because every test here builds its `CliOptions` from
// `chroma_only_opts`, which names the `Vulkan` accelerator variant. That
// variant only exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod passthrough_retry_tests {
    use av_denoise::accelerate::Accelerator;
    use av_denoise::{Algorithm, DenoisingMode};

    use super::*;

    /// Chroma-only intent, so `luma` is the disabled passthrough half and
    /// `chroma` is the one that can report `QueueFull`.
    ///
    /// That is what drives the retry loop in `push_with_drain` and in
    /// `stream_mode.rs`.
    fn chroma_only_opts() -> CliOptions {
        CliOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: BinaryChannelIntent::Chroma,
            mode: DenoisingMode::Spacial,
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
            progress: false,
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
            WorkerDenoiser::create(&chroma_only_opts(), layout).expect("denoiser construction failed");
        let planes = fake_planes(layout);

        // Spatial mode runs a depth-2 pipeline, so the first two pushes
        // land directly. See `push_after_pending_returns_queue_full` in
        // `src/denoiser.rs`.
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

// Feature-gated because every test here builds its `CliOptions` from
// `luma_chroma_opts`, which names the `Vulkan` accelerator variant. That
// variant only exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod lumachroma_lockstep_tests {
    use av_denoise::accelerate::Accelerator;
    use av_denoise::{Algorithm, DenoisingMode};

    use super::*;

    /// Runs `luma` and `chroma` as two real `Denoiser`s in spatial mode.
    ///
    /// Spatial mode passes a uniform-valued plane through unchanged, as
    /// the `uniform_*_passthrough` tests in `src/nlmeans/tests` show. The
    /// test can therefore give each plane its own marker value and spot
    /// the two halves drifting apart.
    fn luma_chroma_opts() -> CliOptions {
        CliOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: BinaryChannelIntent::LumaChroma,
            mode: DenoisingMode::Spacial,
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
            progress: false,
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
    fn queue_full_retries_never_desync_luma_and_chroma() {
        let layout = FrameLayout {
            width: 16,
            height: 16,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let mut wd =
            WorkerDenoiser::create(&luma_chroma_opts(), layout).expect("denoiser construction failed");

        // More pushes than the depth-2 pipeline holds, so this drives
        // several `QueueFull`-then-retry cycles.
        const N: u8 = 6;
        let mut outputs: Vec<Planes> = Vec::new();

        for idx in 0..N {
            let planes = marked_planes(layout, idx);

            // Mirror the retry loop in `push_with_drain` exactly, which
            // is the sequence `file_mode.rs` and `stream_mode.rs` run.
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

/// Maps our [`Subsampling`] and [`Depth`] onto the [`y4m::Colorspace`]
/// used to read the input and write the output header.
pub fn subsampling_to_y4m(s: Subsampling, depth: Depth) -> y4m::Colorspace {
    match (s, depth) {
        (Subsampling::Yuv420, Depth::Eight) => y4m::Colorspace::C420,
        (Subsampling::Yuv420, Depth::Ten) => y4m::Colorspace::C420p10,
        (Subsampling::Yuv420, Depth::Twelve) => y4m::Colorspace::C420p12,
        (Subsampling::Yuv422, Depth::Eight) => y4m::Colorspace::C422,
        (Subsampling::Yuv422, Depth::Ten) => y4m::Colorspace::C422p10,
        (Subsampling::Yuv422, Depth::Twelve) => y4m::Colorspace::C422p12,
        (Subsampling::Yuv444, Depth::Eight) => y4m::Colorspace::C444,
        (Subsampling::Yuv444, Depth::Ten) => y4m::Colorspace::C444p10,
        (Subsampling::Yuv444, Depth::Twelve) => y4m::Colorspace::C444p12,
    }
}

/// Maps a [`y4m::Colorspace`] back onto our [`Subsampling`] and
/// [`Depth`].
///
/// Grayscale and any other unsupported colorspace are rejected with an
/// error naming what is required instead.
pub fn subsampling_from_y4m(c: y4m::Colorspace) -> Result<(Subsampling, Depth), anyhow::Error> {
    let sub = match c {
        y4m::Colorspace::C420
        | y4m::Colorspace::C420jpeg
        | y4m::Colorspace::C420paldv
        | y4m::Colorspace::C420mpeg2
        | y4m::Colorspace::C420p10
        | y4m::Colorspace::C420p12 => Subsampling::Yuv420,
        y4m::Colorspace::C422 | y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => Subsampling::Yuv422,
        y4m::Colorspace::C444 | y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => Subsampling::Yuv444,
        other => anyhow::bail!("unsupported y4m colorspace {other:?}, need 4:2:0, 4:2:2, or 4:4:4"),
    };

    let depth = Depth::from_bits(c.get_bit_depth())?;

    Ok((sub, depth))
}

/// Pulls the `X`-prefixed vendor extension params out of a decoded y4m
/// header's raw params bytes, `XCOLORRANGE=LIMITED` being the common one.
///
/// The leading `X` is stripped so the result can go straight to
/// [`y4m::EncoderBuilder::append_vendor_extension`], which adds the `X`
/// back when it writes the output header.
///
/// This is how whatever colorspace and range tags the source declared
/// reach the output instead of being dropped.
///
/// A token that [`y4m::VendorExtensionString`] rejects, which means one
/// containing a space, is skipped rather than failing the run.
pub fn y4m_vendor_extensions(raw_params: &[u8]) -> Vec<y4m::VendorExtensionString> {
    raw_params
        .split(|&b| b == b' ')
        .filter(|tok| tok.first() == Some(&b'X'))
        .filter_map(|tok| y4m::VendorExtensionString::new(tok[1..].to_vec()).ok())
        .collect()
}

#[cfg(test)]
mod colorspace_tests {
    use super::*;

    #[test]
    fn colorspace_round_trips_every_supported_combination() {
        let combos = [
            (Subsampling::Yuv420, Depth::Eight),
            (Subsampling::Yuv420, Depth::Ten),
            (Subsampling::Yuv420, Depth::Twelve),
            (Subsampling::Yuv422, Depth::Eight),
            (Subsampling::Yuv422, Depth::Ten),
            (Subsampling::Yuv422, Depth::Twelve),
            (Subsampling::Yuv444, Depth::Eight),
            (Subsampling::Yuv444, Depth::Ten),
            (Subsampling::Yuv444, Depth::Twelve),
        ];

        for (sub, depth) in combos {
            let cs = subsampling_to_y4m(sub, depth);
            let (rsub, rdepth) = subsampling_from_y4m(cs).expect("should map back");

            assert_eq!(rsub, sub, "subsampling lost for {cs:?}");
            assert_eq!(rdepth, depth, "depth lost for {cs:?}");
        }
    }

    #[test]
    fn ten_bit_420_maps_to_c420p10() {
        // `y4m::Colorspace` derives only `Debug, Clone, Copy`, not
        // `PartialEq`, so `assert_eq!` won't compile here.
        assert!(matches!(
            subsampling_to_y4m(Subsampling::Yuv420, Depth::Ten),
            y4m::Colorspace::C420p10
        ));
    }

    #[test]
    fn eight_bit_420_variants_all_map_to_yuv420_eight() {
        for cs in [
            y4m::Colorspace::C420,
            y4m::Colorspace::C420jpeg,
            y4m::Colorspace::C420paldv,
            y4m::Colorspace::C420mpeg2,
        ] {
            let (sub, depth) = subsampling_from_y4m(cs).expect("should map");
            assert_eq!(sub, Subsampling::Yuv420);
            assert_eq!(depth, Depth::Eight);
        }
    }

    #[test]
    fn grayscale_colorspaces_are_rejected_with_a_clear_message() {
        for cs in [y4m::Colorspace::Cmono, y4m::Colorspace::Cmono12] {
            let err = subsampling_from_y4m(cs).expect_err("grayscale should be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("{cs:?}")),
                "error should name the offending colorspace, got {msg}"
            );
        }
    }
}
