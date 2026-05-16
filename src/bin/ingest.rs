//! Common ingestion types shared by the file (ffms2) and stdin (y4m) pipelines.
//!
//! Defines the planar 8-bit YUV frame representation passed between the
//! decoder and the worker, and a [`WorkerDenoiser`] that hides the
//! Luma/Chroma split required when the source is chroma-subsampled.

use std::collections::VecDeque;

use av_denoise::accelerate::Accelerator;
use av_denoise::{
    ChannelMode,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    Device,
    NlmTuning,
    PrefilterMode,
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
}

/// Planar 8-bit YUV frame. Plane lengths are determined by [`FrameLayout`]:
/// `y.len() == width*height`, `u.len() == v.len() == chroma_w*chroma_h`.
#[derive(Debug, Clone)]
pub struct Planes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Resolved channel-denoising intent driven by the binary CLI.
///
/// Distinct from the library's [`ChannelMode`] because the binary can
/// run *multiple* library `Denoiser`s in lockstep (luma + chroma split)
/// or a single fused 3-channel denoiser, depending on the user's
/// `--channel-mode` choice and the source's chroma subsampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryChannelIntent {
    /// Denoise luma only; chroma passes through.
    Luma,
    /// Denoise chroma only; luma passes through.
    Chroma,
    /// Denoise both luma and chroma as two independent denoisers.
    /// Chroma runs at the source's native subsampled resolution.
    LumaChroma,
    /// Single library `Denoiser` running the fused 3-channel kernel.
    /// Requires a YUV444 source — validated at ingest setup time.
    YuvFused,
}

impl BinaryChannelIntent {
    /// Reject the intent if the source's subsampling is incompatible.
    pub fn validate_for_source(self, layout: FrameLayout) -> Result<(), anyhow::Error> {
        match self {
            BinaryChannelIntent::YuvFused if layout.subsampling != Subsampling::Yuv444 => {
                anyhow::bail!(
                    "--channel-mode yuv requires a YUV444 source (got {:?}); convert the input first (e.g. ffmpeg -pix_fmt yuv444p)",
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
    pub prefilter: PrefilterMode,
    pub nlm_tuning: Option<NlmTuning>,
    /// Per-plane strength override for the luma denoiser. Takes
    /// precedence over `nlm_tuning.strength` when set.
    pub luma_strength: Option<f32>,
    /// Per-plane strength override for the chroma denoiser. Takes
    /// precedence over `nlm_tuning.strength` when set.
    pub chroma_strength: Option<f32>,
}

impl CliOptions {
    fn denoiser_options(&self, channels: ChannelMode) -> DenoiserOptions {
        let b = DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .prefilter(self.prefilter);

        let strength_override = match channels {
            ChannelMode::Luma => self.luma_strength,
            ChannelMode::Chroma => self.chroma_strength,
            ChannelMode::Yuv => None,
        };

        let tuning = match (self.nlm_tuning, strength_override) {
            (Some(base), Some(s)) => Some(NlmTuning {
                strength: Some(s),
                ..base
            }),
            (None, Some(s)) => Some(NlmTuning {
                search_radius: None,
                patch_radius: None,
                strength: Some(s),
                self_weight: None,
            }),
            (Some(base), None) => Some(base),
            (None, None) => None,
        };

        match tuning {
            Some(t) => b.nlm(t).build(),
            None => b.build(),
        }
    }
}

/// Wraps the Luma and Chroma `Denoiser` instances needed for a single
/// subsampled YUV source. The caller pushes planar frames in and gets
/// planar frames out; the Luma/Chroma split is invisible.
pub struct WorkerDenoiser {
    layout: FrameLayout,
    luma: Option<Denoiser>,
    chroma: Option<Denoiser>,
    /// Set when intent is `YuvFused`; mutually exclusive with `luma`/`chroma`.
    yuv: Option<Denoiser>,
    // Source planes queued for passthrough when the corresponding denoiser
    // is disabled. Only the *disabled* side's queue is ever populated. Popped
    // 1:1 with the enabled side's output so temporal delays stay aligned.
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

    /// Push one planar frame. On `QueueFull` the caller should `recv` first
    /// and retry — the error propagates upwards unchanged.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        if let Some(d) = self.yuv.as_mut() {
            let buf = interleave_yuv_to_f32(&planes.y, &planes.u, &planes.v);
            d.push_frame(&buf)?;
            return Ok(());
        }

        if let Some(d) = self.luma.as_mut() {
            let buf = u8_plane_to_f32(&planes.y);
            d.push_frame(&buf)?;
        } else {
            self.luma_passthrough.push_back(planes.y.clone());
        }

        if let Some(d) = self.chroma.as_mut() {
            let buf = interleave_uv_to_f32(&planes.u, &planes.v);
            d.push_frame(&buf)?;
        } else {
            self.chroma_passthrough
                .push_back((planes.u.clone(), planes.v.clone()));
        }

        Ok(())
    }

    /// Block until each enabled half emits one frame; reassemble a planar frame.
    /// Returns `Ok(None)` if neither half had pending output.
    pub fn recv(&mut self) -> Result<Option<Planes>, anyhow::Error> {
        if let Some(d) = self.yuv.as_mut() {
            return match d.recv_frame()? {
                Some(packed) => Ok(Some(unpack_yuv_from_f32(&packed, self.layout.luma_pixels()))),
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

        // A side that's disabled has no Denoiser to query; if the *enabled*
        // side produced output, pop the matching source-plane frame from the
        // disabled side's passthrough queue.
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

    /// Drain temporal tails for both halves. `sink` is called once per
    /// emitted planar frame.
    pub fn flush(&mut self, mut sink: impl FnMut(Planes)) -> Result<(), anyhow::Error> {
        if let Some(d) = self.yuv.as_mut() {
            let pixels = self.layout.luma_pixels();
            d.flush(|packed| sink(unpack_yuv_from_f32(&packed, pixels)))?;
            return Ok(());
        }

        let luma_pixels = self.layout.luma_pixels();
        let chroma_pixels = self.layout.chroma_pixels();

        let mut luma_buf: Vec<Vec<f32>> = Vec::new();
        let mut chroma_buf: Vec<Vec<f32>> = Vec::new();

        if let Some(d) = self.luma.as_mut() {
            d.flush(|v| luma_buf.push(v))?;
        }

        if let Some(d) = self.chroma.as_mut() {
            d.flush(|v| chroma_buf.push(v))?;
        }

        // The two halves run in lock-step, so the number of flushed frames
        // matches. For each emitted frame, the disabled side (if any) pops
        // the matching source plane from its passthrough queue.
        let count = luma_buf.len().max(chroma_buf.len());

        for i in 0..count {
            let y = if let Some(buf) = luma_buf.get(i) {
                f32_to_u8_plane(buf)
            } else if let Some(src) = self.luma_passthrough.pop_front() {
                src
            } else {
                vec![0u8; luma_pixels]
            };

            let (u, v) = if let Some(packed) = chroma_buf.get(i) {
                unpack_uv_from_f32(packed, chroma_pixels)
            } else if let Some((src_u, src_v)) = self.chroma_passthrough.pop_front() {
                (src_u, src_v)
            } else {
                (vec![128u8; chroma_pixels], vec![128u8; chroma_pixels])
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
        let luma_pixels = self.layout.luma_pixels();
        let chroma_pixels = self.layout.chroma_pixels();

        let y = match (luma, luma_passthrough) {
            (Some(v), _) => f32_to_u8_plane(&v),
            (None, Some(src)) => src,
            (None, None) => vec![0u8; luma_pixels],
        };

        let (u, v) = match (chroma, chroma_passthrough) {
            (Some(packed), _) => unpack_uv_from_f32(&packed, chroma_pixels),
            (None, Some(src)) => src,
            (None, None) => (vec![128u8; chroma_pixels], vec![128u8; chroma_pixels]),
        };

        Planes { y, u, v }
    }
}

fn u8_plane_to_f32(plane: &[u8]) -> Vec<f32> {
    plane.iter().map(|&b| b as f32 / 255.0).collect()
}

fn f32_to_u8_plane(plane: &[f32]) -> Vec<u8> {
    plane
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// Interleave Y/U/V planes (all equal length, i.e. YUV444) into
/// `[Y0,U0,V0, Y1,U1,V1, …]` f32 in `[0, 1]` — the layout the library's
/// fused 3-channel kernel expects.
fn interleave_yuv_to_f32(y: &[u8], u: &[u8], v: &[u8]) -> Vec<f32> {
    debug_assert_eq!(y.len(), u.len());
    debug_assert_eq!(u.len(), v.len());

    let mut out = Vec::with_capacity(y.len() * 3);

    for ((&yy, &uu), &vv) in y.iter().zip(u.iter()).zip(v.iter()) {
        out.push(yy as f32 / 255.0);
        out.push(uu as f32 / 255.0);
        out.push(vv as f32 / 255.0);
    }

    out
}

/// Reverse of `interleave_yuv_to_f32`: take a `[Y,U,V,Y,U,V,…]` f32
/// buffer (length `3 * pixels`) and split into three u8 planes.
fn unpack_yuv_from_f32(packed: &[f32], pixels: usize) -> Planes {
    debug_assert_eq!(packed.len(), 3 * pixels);

    let mut y = Vec::with_capacity(pixels);
    let mut u = Vec::with_capacity(pixels);
    let mut v = Vec::with_capacity(pixels);

    for chunk in packed.chunks_exact(3) {
        y.push((chunk[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        u.push((chunk[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        v.push((chunk[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }

    Planes { y, u, v }
}

/// Interleave separate U and V planes into [U,V,U,V,...] f32 in [0, 1].
fn interleave_uv_to_f32(u: &[u8], v: &[u8]) -> Vec<f32> {
    debug_assert_eq!(u.len(), v.len());

    let mut out = Vec::with_capacity(u.len() * 2);

    for (&uu, &vv) in u.iter().zip(v.iter()) {
        out.push(uu as f32 / 255.0);
        out.push(vv as f32 / 255.0);
    }

    out
}

/// Reverse of `interleave_uv_to_f32`: take a packed [U,V,U,V,...] f32 buffer
/// and split into two u8 planes.
fn unpack_uv_from_f32(packed: &[f32], chroma_pixels: usize) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(packed.len(), 2 * chroma_pixels);

    let mut u = Vec::with_capacity(chroma_pixels);
    let mut v = Vec::with_capacity(chroma_pixels);

    for chunk in packed.chunks_exact(2) {
        u.push((chunk[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        v.push((chunk[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }

    (u, v)
}

/// Map our [`Subsampling`] enum onto the y4m [`y4m::Colorspace`] used for
/// both reading the input and writing the output header.
pub fn subsampling_to_y4m(s: Subsampling) -> y4m::Colorspace {
    match s {
        Subsampling::Yuv420 => y4m::Colorspace::C420,
        Subsampling::Yuv422 => y4m::Colorspace::C422,
        Subsampling::Yuv444 => y4m::Colorspace::C444,
    }
}

pub fn subsampling_from_y4m(c: y4m::Colorspace) -> Result<Subsampling, anyhow::Error> {
    match c {
        y4m::Colorspace::C420
        | y4m::Colorspace::C420jpeg
        | y4m::Colorspace::C420paldv
        | y4m::Colorspace::C420mpeg2 => Ok(Subsampling::Yuv420),
        y4m::Colorspace::C422 => Ok(Subsampling::Yuv422),
        y4m::Colorspace::C444 => Ok(Subsampling::Yuv444),
        other => anyhow::bail!("unsupported y4m colorspace {other:?}; need 4:2:0, 4:2:2, or 4:4:4 8-bit"),
    }
}
