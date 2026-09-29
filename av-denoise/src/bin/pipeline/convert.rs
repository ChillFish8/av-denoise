use std::sync::Arc;

use av_denoise::{FrameLayout, Planes, Subsampling};
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;

/// A decoded frame at whichever sample width the source uses.
pub enum DecodedFrame {
    Eight(Arc<Frame<u8>>),
    Wide(Arc<Frame<u16>>),
}

/// A sample type the pipeline can decode, detect scenes on and denoise.
pub trait SourcePixel: Pixel {
    fn into_decoded(frame: Arc<Frame<Self>>) -> DecodedFrame;

    /// Returns `None` when the frame holds the other sample width.
    fn from_decoded(frame: DecodedFrame) -> Option<Arc<Frame<Self>>>;

    fn to_planes(frame: &Frame<Self>, layout: FrameLayout) -> Result<Planes, anyhow::Error>;
}

impl SourcePixel for u8 {
    fn into_decoded(frame: Arc<Frame<Self>>) -> DecodedFrame {
        DecodedFrame::Eight(frame)
    }

    fn from_decoded(frame: DecodedFrame) -> Option<Arc<Frame<Self>>> {
        match frame {
            DecodedFrame::Eight(frame) => Some(frame),
            DecodedFrame::Wide(_) => None,
        }
    }

    fn to_planes(frame: &Frame<Self>, layout: FrameLayout) -> Result<Planes, anyhow::Error> {
        planes_from_v_frame_u8(frame, layout)
    }
}

impl SourcePixel for u16 {
    fn into_decoded(frame: Arc<Frame<Self>>) -> DecodedFrame {
        DecodedFrame::Wide(frame)
    }

    fn from_decoded(frame: DecodedFrame) -> Option<Arc<Frame<Self>>> {
        match frame {
            DecodedFrame::Wide(frame) => Some(frame),
            DecodedFrame::Eight(_) => None,
        }
    }

    fn to_planes(frame: &Frame<Self>, layout: FrameLayout) -> Result<Planes, anyhow::Error> {
        planes_from_v_frame_u16(frame, layout)
    }
}

/// Checks each plane's byte length against the layout, failing with an error naming which plane
/// is wrong, the length found and the length expected.
pub fn check_plane_lens(planes: &Planes, layout: FrameLayout) -> Result<(), anyhow::Error> {
    for (name, got, expected) in [
        ("y", planes.y.len(), layout.luma_bytes()),
        ("u", planes.u.len(), layout.chroma_bytes()),
        ("v", planes.v.len(), layout.chroma_bytes()),
    ] {
        if got != expected {
            anyhow::bail!("{name} plane is {got} bytes, expected {expected} from the frame layout");
        }
    }
    Ok(())
}

pub fn planes_from_v_frame_u8(
    frame: &v_frame::frame::Frame<u8>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u8(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

pub fn planes_from_v_frame_u16(
    frame: &v_frame::frame::Frame<u16>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u16(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

pub fn collect_plane_u8(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height);

    for row in plane.rows() {
        out.extend_from_slice(row);
    }

    out
}

pub fn collect_plane_u16(plane: &v_frame::plane::Plane<u16>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height * 2);

    for row in plane.rows() {
        for &s in row {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }

    out
}

pub fn subsampling_from_av_decoders(
    cs: v_frame::chroma::ChromaSubsampling,
) -> Result<Subsampling, anyhow::Error> {
    use v_frame::chroma::ChromaSubsampling;

    match cs {
        ChromaSubsampling::Yuv420 => Ok(Subsampling::Yuv420),
        ChromaSubsampling::Yuv422 => Ok(Subsampling::Yuv422),
        ChromaSubsampling::Yuv444 => Ok(Subsampling::Yuv444),
        other => {
            anyhow::bail!("unsupported chroma subsampling {other:?}, need 4:2:0, 4:2:2, or 4:4:4")
        },
    }
}
