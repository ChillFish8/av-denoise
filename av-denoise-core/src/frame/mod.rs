use std::collections::VecDeque;

use crate::accelerate::Accelerator;
use crate::{
    Algorithm,
    ChannelMode,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    Depth,
    Device,
    FrameOutput,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    OutputFormat,
    WindowSpan,
};

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
/// may run more than one `Denoiser` in lockstep, one for luma and one for
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
    /// A single library `Denoiser` running the fused three-channel
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
    /// Per-plane override for `mismatch_scale`, luma. Takes precedence
    /// over `algorithm`'s value when set. Only has an effect when
    /// `algorithm` is `Algorithm::Nl4d`.
    pub luma_mismatch_scale: Option<f32>,
    /// Per-plane override for `mismatch_scale`, chroma. Takes precedence
    /// over `algorithm`'s value when set. Only has an effect when
    /// `algorithm` is `Algorithm::Nl4d`.
    pub chroma_mismatch_scale: Option<f32>,
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

    /// `depth` is the source's wire depth, which every denoiser
    /// quantises to on the GPU.
    fn denoiser_options(&self, channels: ChannelMode, depth: Depth) -> DenoiserOptions {
        DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .algorithm(self.algorithm_for(channels))
            .output_format(OutputFormat::Wire { depth })
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

/// Pops up to `count` entries off the front of `queue`, discarding them.
fn drop_leading<T>(queue: &mut VecDeque<T>, count: usize) {
    for _ in 0..count.min(queue.len()) {
        queue.pop_front();
    }
}

/// Reads the result of a `PlanarDenoiser::push` call for the
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

/// Unwraps a denoised frame from one of the `Denoiser`s
/// [`PlanarDenoiser`] builds.
///
/// Those are always built in [`crate::OutputFormat::Wire`], so the other
/// variant never reaches here.
fn expect_wire(out: FrameOutput) -> Vec<u8> {
    out.into_wire()
        .expect("PlanarDenoiser builds every Denoiser in wire output format")
}

/// Splits a fused YUV444 wire frame into its three planes.
///
/// The pack kernel leaves a three-channel frame interleaved, so this is
/// the byte-level counterpart of the host converter it replaced. That one
/// lives in `converter_tests` now, as the oracle this is checked against.
fn split_yuv_wire(wire: &[u8], depth: Depth) -> Planes {
    let bytes = depth.bytes_per_sample();
    let pixels = wire.len() / (3 * bytes);

    let mut y = Vec::with_capacity(pixels * bytes);
    let mut u = Vec::with_capacity(pixels * bytes);
    let mut v = Vec::with_capacity(pixels * bytes);

    for pixel in wire.chunks_exact(3 * bytes) {
        y.extend_from_slice(&pixel[..bytes]);
        u.extend_from_slice(&pixel[bytes..2 * bytes]);
        v.extend_from_slice(&pixel[2 * bytes..]);
    }

    Planes { y, u, v }
}

/// Splits a chroma wire frame into its U and V planes.
///
/// The pack kernel writes U's whole region first and V's after it, so
/// each plane is one contiguous half of the buffer.
fn split_uv_wire(wire: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let (u, v) = wire.split_at(wire.len() / 2);
    (u.to_vec(), v.to_vec())
}

/// The push a [`PlanarDenoiser`] runs against each enabled half, either
/// [`Denoiser::push_frame_wire`] or
/// [`Denoiser::push_frame_wire_priming`].
type WirePush = fn(&mut Denoiser, &[&[u8]], Depth) -> Result<(), DenoiserError>;

/// Wraps the luma and chroma `Denoiser` instances needed for one
/// subsampled YUV source.
///
/// The caller pushes planar frames in and gets planar frames out. The
/// luma and chroma split is invisible from the outside.
pub struct PlanarDenoiser {
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
                Denoiser::create(
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
                Denoiser::create(
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
                Denoiser::create(
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
    /// inside `push_frame_wire` answers the same way for each. If the luma
    /// push succeeds then the chroma push succeeds too, which makes the
    /// duplicate unreachable.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, Denoiser::push_frame_wire)
    }

    /// Uploads one planar frame into the temporal window without starting
    /// a denoise.
    ///
    /// Mirrors [`Self::push`], down to queueing the disabled side's
    /// passthrough plane, but no output is ever produced for this call.
    /// This is how [`Self::reseed`] fills the window from an explicit
    /// window of frames before the one real push that starts a denoise.
    fn push_priming(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, Denoiser::push_frame_wire_priming)
    }

    /// Shared body of [`Self::push`] and [`Self::push_priming`].
    ///
    /// `push_frame` is [`Denoiser::push_frame_wire`] for a real push or
    /// [`Denoiser::push_frame_wire_priming`] for a priming one, run
    /// against whichever of `yuv`, `luma`, and `chroma` is enabled.
    ///
    /// The planes go over as wire bytes, so the normalisation and the
    /// channel interleave both happen on the GPU.
    fn push_with(&mut self, planes: &Planes, push_frame: WirePush) -> Result<(), DenoiserError> {
        let depth = self.layout.depth;

        if let Some(d) = self.yuv.as_mut() {
            push_frame(d, &[&planes.y, &planes.u, &planes.v], depth)?;
            return Ok(());
        }

        if let Some(d) = self.luma.as_mut() {
            push_frame(d, &[&planes.y], depth)?;
        }

        if let Some(d) = self.chroma.as_mut() {
            push_frame(d, &[&planes.u, &planes.v], depth)?;
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
                Some(packed) => Ok(Some(split_yuv_wire(&expect_wire(packed), self.layout.depth))),
                None => Ok(None),
            };
        }

        let luma_out = self
            .luma
            .as_mut()
            .map(|d| d.recv_frame())
            .transpose()?
            .flatten()
            .map(expect_wire);

        let chroma_out = self
            .chroma
            .as_mut()
            .map(|d| d.recv_frame())
            .transpose()?
            .flatten()
            .map(expect_wire);

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
            let depth = self.layout.depth;
            d.flush(|packed| sink(split_yuv_wire(&expect_wire(packed), depth)))?;
            return Ok(());
        }

        let mut luma_buf: Vec<Vec<u8>> = Vec::new();
        let mut chroma_buf: Vec<Vec<u8>> = Vec::new();

        if let Some(d) = self.luma.as_mut() {
            d.flush(|v| luma_buf.push(expect_wire(v)))?;
        }

        if let Some(d) = self.chroma.as_mut() {
            d.flush(|v| chroma_buf.push(expect_wire(v)))?;
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

            let (u, v) = if let Some(packed) = chroma_buf.get(i) {
                split_uv_wire(packed)
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
    /// Every owned `Denoiser` was built from the same algorithm, so any
    /// one of them answers for all of them.
    pub fn window_span(&self) -> WindowSpan {
        self.yuv
            .as_ref()
            .or(self.luma.as_ref())
            .or(self.chroma.as_ref())
            .expect("PlanarDenoiser always keeps at least one Denoiser")
            .window_span()
    }

    /// Denoises the target frame of an explicit window, sized and
    /// shaped exactly as [`Self::window_span`] reports for whichever
    /// algorithm this `PlanarDenoiser` runs.
    ///
    /// This abandons whatever stream was running and starts a new one
    /// from the window, keeping every GPU allocation. When it returns,
    /// the stream sits exactly where it would be had the window been
    /// pushed frame by frame, so the caller can carry on with
    /// [`Self::push`] and [`Self::recv`] for the frame after the target.
    ///
    /// Callers clamp the window's indices at the clip's ends, matching
    /// how the streaming path repeats the first and last frames.
    ///
    /// # Why the window is wider than `2r+1` for some algorithms
    ///
    /// The two NLM algorithms produce one output per submit once their
    /// own `2r+1`-frame window is full, so a symmetric window centred
    /// on the target frame is enough.
    ///
    /// nl4d scatters every pass's contribution across the `2r+1`
    /// frames that pass reaches, and a target frame's own region only
    /// starts collecting contributions once the earliest pass able to
    /// reach it, the one centred `r` frames behind the target, has
    /// actually run, which itself needs the front end's own window
    /// full at that earlier centre. Both of those requirements push
    /// the target's own `r`-wide neighbourhood back by another `r`, on
    /// both sides, which is exactly what [`Self::window_span`] reports
    /// through nl4d's doubled `behind` and `ahead`. This is bit-exact
    /// with the streaming path because every frame the window supplies
    /// is real, distinct content, run through the same sequence of
    /// passes streaming would have run to reach the target frame.
    pub fn reseed(&mut self, window: &[Planes]) -> Result<Planes, anyhow::Error> {
        let span = self.window_span();
        let expected = span.frame_count();
        if window.len() != expected {
            anyhow::bail!("reseed needs a window of {expected} frames, got {}", window.len());
        }

        self.luma_passthrough.clear();
        self.chroma_passthrough.clear();

        for d in [self.yuv.as_mut(), self.luma.as_mut(), self.chroma.as_mut()]
            .into_iter()
            .flatten()
        {
            d.reset_stream();
        }

        // Prime the first `2 * temporal_radius` frames, filling the
        // underlying denoiser's own window without submitting anything,
        // exactly as streaming would have primed it. This count comes
        // from the front end's own window size, not from `span`, so it
        // stays the same for every algorithm. Every remaining frame is
        // then a real push, one submit per frame.
        let radius = self.temporal_radius as usize;
        let priming_count = 2 * radius;
        let (head, tail) = window.split_at(priming_count);
        for planes in head {
            self.push_priming(planes)?;
        }

        // Priming queues one passthrough entry per frame, just as a
        // real push does. `nlmeans`'s single real push, below, always
        // emits and pairs with the target's own entry once `radius` of
        // these leading ones are out of the way, exactly as before.
        //
        // nl4d's real pushes below emit more than once: nl4d's own
        // gate gives every push once its own window is full a real
        // output, but only the last `ahead - behind + 1` of them
        // complete a region as new as the target's, the earlier ones
        // complete regions further behind it that this call has no use
        // for. Draining after every real push, not only the last,
        // keeps the pending queue from ever holding more than one
        // frame at a time, and it walks the passthrough queue forward
        // by exactly one entry per region completed, so by the time
        // the target's own region completes, its entry is the one at
        // the front to pop. The same `radius` leading drop lines that
        // front up correctly beforehand for both algorithms, because
        // nl4d's own gate width is `radius` regardless of how wide
        // `span` is.
        drop_leading(&mut self.luma_passthrough, radius);
        drop_leading(&mut self.chroma_passthrough, radius);

        let mut result = None;
        for planes in tail {
            self.push(planes)?;
            if let Some(out) = self.recv()? {
                result = Some(out);
            }
        }

        result.ok_or_else(|| anyhow::anyhow!("a full window produced no frame, this is a bug"))
    }

    fn assemble(
        &self,
        luma: Option<Vec<u8>>,
        chroma: Option<Vec<u8>>,
        luma_passthrough: Option<Vec<u8>>,
        chroma_passthrough: Option<(Vec<u8>, Vec<u8>)>,
    ) -> Planes {
        let y = match (luma, luma_passthrough) {
            (Some(v), _) => v,
            (None, Some(src)) => src,
            (None, None) => self.layout.black_luma_plane(),
        };

        let (u, v) = match (chroma, chroma_passthrough) {
            (Some(packed), _) => split_uv_wire(&packed),
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
///
/// `gpu_unpack_wire` does this on the device now. This host version is
/// the oracle that kernel is checked against.
pub fn plane_to_f32(plane: &[u8], depth: Depth) -> Vec<f32> {
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
pub fn f32_to_plane(plane: &[f32], depth: Depth) -> Vec<u8> {
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
///
/// `gpu_unpack_wire` does this on the device now. This host version is
/// the oracle that kernel is checked against.
pub fn interleave_yuv_to_f32(y: &[u8], u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
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

/// Interleaves separate U and V planes into `[U, V, U, V, ...]` as f32
/// in `[0, 1]`.
///
/// `gpu_unpack_wire` does this on the device now. This host version is
/// the oracle that kernel is checked against.
pub fn interleave_uv_to_f32(u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
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
pub fn unpack_uv_from_f32(packed: &[f32], chroma_pixels: usize, depth: Depth) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(packed.len(), 2 * chroma_pixels);

    let max = depth.max_value();

    fn run<C: SampleCodec>(packed: &[f32], chroma_pixels: usize, max: f32) -> (Vec<u8>, Vec<u8>) {
        let mut u = vec![0u8; chroma_pixels * C::BYTES];
        let mut v = vec![0u8; chroma_pixels * C::BYTES];

        for (i, chunk) in packed.as_chunks::<2>().0.iter().enumerate() {
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

    /// Reverse of [`interleave_yuv_to_f32`], and the oracle
    /// `split_yuv_wire` is checked against.
    ///
    /// Production splits a fused YUV frame from the wire bytes the GPU
    /// already quantised. This host version stays because cubecl can
    /// compile a kernel to nothing without reporting an error, and a
    /// kernel compared against itself compares zeros to zeros.
    fn unpack_yuv_from_f32(packed: &[f32], pixels: usize, depth: Depth) -> Planes {
        debug_assert_eq!(packed.len(), 3 * pixels);

        let max = depth.max_value();

        fn run<C: SampleCodec>(packed: &[f32], pixels: usize, max: f32) -> Planes {
            let mut y = vec![0u8; pixels * C::BYTES];
            let mut u = vec![0u8; pixels * C::BYTES];
            let mut v = vec![0u8; pixels * C::BYTES];

            for (i, chunk) in packed.as_chunks::<3>().0.iter().enumerate() {
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
    fn split_uv_wire_matches_unpack_uv_from_f32() {
        let u_src = [0.0, 0.25, 0.5, 1.0];
        let v_src = [1.0, 0.75, 0.5, 0.0];

        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let packed: Vec<f32> = u_src.iter().zip(&v_src).flat_map(|(&u, &v)| [u, v]).collect();
            let (want_u, want_v) = unpack_uv_from_f32(&packed, 4, depth);

            // The kernel lays a chroma frame out as U's whole region
            // followed by V's.
            let wire: Vec<u8> = f32_to_plane(&u_src, depth)
                .into_iter()
                .chain(f32_to_plane(&v_src, depth))
                .collect();

            let (u, v) = split_uv_wire(&wire);
            assert_eq!(u, want_u, "U disagreed at {depth:?}");
            assert_eq!(v, want_v, "V disagreed at {depth:?}");
        }
    }

    #[test]
    fn split_yuv_wire_matches_unpack_yuv_from_f32() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let packed: Vec<f32> = (0..9).map(|i| i as f32 / 9.0).collect();

            let want = unpack_yuv_from_f32(&packed, 3, depth);
            let got = split_yuv_wire(&f32_to_plane(&packed, depth), depth);

            assert_eq!(got.y, want.y, "Y disagreed at {depth:?}");
            assert_eq!(got.u, want.u, "U disagreed at {depth:?}");
            assert_eq!(got.v, want.v, "V disagreed at {depth:?}");
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
    use super::*;
    use crate::nlmeans::NlmParams;

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
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
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
    fn no_overrides_hq_resolves_through_to_nlm_params_to_the_measured_tables() {
        // Radius 4 in the measured tables is luma 0.35 and chroma
        // 0.70 (see the table docs in `src/nlmeans/params.rs`).
        let opts = base_opts(
            DenoisingMode::Temporal { radius: 4 },
            Algorithm::NlmeansHq(NlmeansHqOptions::default()),
            None,
            None,
        );

        let luma_params: NlmParams = opts
            .denoiser_options(ChannelMode::Luma, Depth::Eight)
            .to_nlm_params();
        let chroma_params: NlmParams = opts
            .denoiser_options(ChannelMode::Chroma, Depth::Eight)
            .to_nlm_params();

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
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
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

    /// A `PlaneOptions` running `Algorithm::Nl4d` with a shared
    /// `mismatch_scale` and the two per-plane overrides under test.
    fn nl4d_mismatch_opts(
        shared: f32,
        luma_mismatch_scale: Option<f32>,
        chroma_mismatch_scale: Option<f32>,
    ) -> PlaneOptions {
        PlaneOptions {
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
        let luma_default = crate::nl4d_default_lambda_ht(ChannelMode::Luma);
        let chroma_default = crate::nl4d_default_lambda_ht(ChannelMode::Chroma);
        assert!((luma_default - 5.3).abs() < f32::EPSILON);
        assert!((chroma_default - 4.2).abs() < f32::EPSILON);
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
    /// That is what drives the retry loop in `push_with_drain` and in
    /// `stream_mode.rs`.
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
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
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

// Feature-gated because every test here builds its `PlaneOptions` from
// `luma_chroma_opts`, which names the `Vulkan` accelerator variant. That
// variant only exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
#[cfg(test)]
mod lumachroma_lockstep_tests {
    use super::*;
    use crate::accelerate::Accelerator;
    use crate::{Algorithm, DenoisingMode};

    /// Runs `luma` and `chroma` as two real `Denoiser`s in spatial mode.
    ///
    /// Spatial mode passes a uniform-valued plane through unchanged, as
    /// the `uniform_*_passthrough` tests in `src/nlmeans/tests` show. The
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
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
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
            PlanarDenoiser::create(&luma_chroma_opts(), layout).expect("denoiser construction failed");

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

#[cfg(test)]
mod tests;
