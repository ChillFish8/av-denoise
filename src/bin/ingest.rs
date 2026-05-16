//! Common ingestion types shared by the file (ffms2) and stdin (y4m) pipelines.
//!
//! Defines the planar 8-bit YUV frame representation passed between the
//! decoder and the worker, and a [`WorkerDenoiser`] that hides the
//! Luma/Chroma split required when the source is chroma-subsampled.

use av_denoise::accelerate::Accelerator;
use av_denoise::{
    ChannelMode,
    Denoiser,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    Device,
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

/// CLI-shaped option set forwarded from `main` into ingest modules.
#[derive(Debug, Clone)]
pub struct CliOptions {
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub channel_mode: ChannelMode,
    pub mode: DenoisingMode,
    pub prefilter: PrefilterMode,
}

impl CliOptions {
    fn denoiser_options(&self, channels: ChannelMode) -> DenoiserOptions {
        DenoiserOptions::builder()
            .channel_mode(channels)
            .mode(self.mode)
            .prefilter(self.prefilter)
            .build()
    }
}

/// Wraps the Luma and Chroma `Denoiser` instances needed for a single
/// subsampled YUV source. The caller pushes planar frames in and gets
/// planar frames out; the Luma/Chroma split is invisible.
pub struct WorkerDenoiser {
    layout: FrameLayout,
    luma: Option<Denoiser>,
    chroma: Option<Denoiser>,
}

impl WorkerDenoiser {
    pub fn new(opts: &CliOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
        let (chroma_w, chroma_h) = layout.chroma_dims();

        if chroma_w == 0 || chroma_h == 0 {
            anyhow::bail!(
                "frame dimensions {}x{} are too small for subsampling {:?}",
                layout.width,
                layout.height,
                layout.subsampling
            );
        }

        let (denoise_luma, denoise_chroma) = match opts.channel_mode {
            ChannelMode::Luma => (true, false),
            ChannelMode::Chroma => (false, true),
            ChannelMode::Yuv => (true, true),
        };

        let luma = if denoise_luma {
            Some(Denoiser::new(
                &opts.accelerators,
                &opts.device,
                layout.width,
                layout.height,
                opts.denoiser_options(ChannelMode::Luma),
            )?)
        } else {
            None
        };

        let chroma = if denoise_chroma {
            Some(Denoiser::new(
                &opts.accelerators,
                &opts.device,
                chroma_w,
                chroma_h,
                opts.denoiser_options(ChannelMode::Chroma),
            )?)
        } else {
            None
        };

        Ok(Self { layout, luma, chroma })
    }

    /// Push one planar frame. On `QueueFull` the caller should `recv` first
    /// and retry — the error propagates upwards unchanged.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        if let Some(d) = self.luma.as_mut() {
            let buf = u8_plane_to_f32(&planes.y);
            d.push_frame(&buf)?;
        }

        if let Some(d) = self.chroma.as_mut() {
            let buf = interleave_uv_to_f32(&planes.u, &planes.v);
            d.push_frame(&buf)?;
        }

        Ok(())
    }

    /// Block until each enabled half emits one frame; reassemble a planar frame.
    /// Returns `Ok(None)` if neither half had pending output.
    pub fn recv(&mut self) -> Result<Option<Planes>, anyhow::Error> {
        let luma_out = self.luma.as_mut().map(|d| d.recv_frame()).transpose()?.flatten();

        let chroma_out = self
            .chroma
            .as_mut()
            .map(|d| d.recv_frame())
            .transpose()?
            .flatten();

        if luma_out.is_none() && chroma_out.is_none() {
            return Ok(None);
        }

        Ok(Some(self.assemble(luma_out, chroma_out)))
    }

    /// Drain temporal tails for both halves. `sink` is called once per
    /// emitted planar frame.
    pub fn flush(&mut self, mut sink: impl FnMut(Planes)) -> Result<(), anyhow::Error> {
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
        // matches. If only one half is enabled the other yields a passthrough
        // plane filled with neutral chroma / black luma.
        let count = luma_buf.len().max(chroma_buf.len());

        for i in 0..count {
            let y = luma_buf
                .get(i)
                .map(|v| f32_to_u8_plane(v))
                .unwrap_or_else(|| vec![0u8; luma_pixels]);

            let (u, v) = if let Some(packed) = chroma_buf.get(i) {
                unpack_uv_from_f32(packed, chroma_pixels)
            } else {
                (vec![128u8; chroma_pixels], vec![128u8; chroma_pixels])
            };

            sink(Planes { y, u, v });
        }

        Ok(())
    }

    fn assemble(&self, luma: Option<Vec<f32>>, chroma: Option<Vec<f32>>) -> Planes {
        let luma_pixels = self.layout.luma_pixels();
        let chroma_pixels = self.layout.chroma_pixels();

        let y = match luma {
            Some(v) => f32_to_u8_plane(&v),
            None => vec![0u8; luma_pixels],
        };

        let (u, v) = match chroma {
            Some(packed) => unpack_uv_from_f32(&packed, chroma_pixels),
            None => (vec![128u8; chroma_pixels], vec![128u8; chroma_pixels]),
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
