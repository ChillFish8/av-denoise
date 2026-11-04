use std::io::Read;
use std::path::Path;

use av_decoders::{Decoder, DecoderImpl, VideoDetails};
use av_denoise::{ChromaSiting, Depth, FrameLayout, Subsampling};
use avtbx_bestsource::IndexCache;

use super::bestsource::{BestSourceReader, open_bestsource};
use super::convert::subsampling_from_av_decoders;
use crate::cli::InputSource;
use crate::y4m_format::{siting_from_y4m, subsampling_from_y4m, y4m_vendor_extensions};

/// What the rest of a run needs to know about its input.
#[derive(Clone)]
pub struct SourceInfo {
    /// The decoder's description.
    pub details: VideoDetails,
    pub layout: FrameLayout,
    /// Carried through to the output header when the input declared one.
    pub pixel_aspect: Option<y4m::Ratio>,
    pub vendor_extensions: Vec<y4m::VendorExtensionString>,
    /// Frames the run should emit, when the input knows its own length.
    pub estimated_frames: Option<usize>,
}

/// The reader behind an open input.
pub enum FrameReader {
    File(BestSourceReader),
    Y4m(Decoder),
}

/// An open input, ready to decode.
pub struct OpenedSource {
    pub reader: FrameReader,
    pub info: SourceInfo,
}

pub fn open_source(input: &InputSource, cache: IndexCache) -> Result<OpenedSource, anyhow::Error> {
    match input {
        InputSource::File(path) => open_file(path, cache),
        InputSource::Stdin | InputSource::Fd(_) => {
            let reader = input.open_reader()?;
            open_y4m(reader)
        },
    }
}

pub fn open_file(path: &Path, cache: IndexCache) -> Result<OpenedSource, anyhow::Error> {
    let (reader, info) = open_bestsource(path, cache)?;

    // Only 8-bit 4:2:0 output carries a siting, so nothing else is logged as forwarded.
    let layout = info.layout;
    let carries_siting = layout.subsampling == Subsampling::Yuv420 && layout.depth == Depth::Eight;
    let logged_siting = if carries_siting {
        layout.chroma_siting
    } else {
        ChromaSiting::Unknown
    };

    log_forwarded_colour(info.pixel_aspect, &info.vendor_extensions, logged_siting);

    Ok(OpenedSource {
        reader: FrameReader::File(reader),
        info,
    })
}

fn log_forwarded_colour(
    pixel_aspect: Option<y4m::Ratio>,
    vendor_extensions: &[y4m::VendorExtensionString],
    chroma_siting: ChromaSiting,
) {
    let siting_unknown = chroma_siting == ChromaSiting::Unknown;
    if pixel_aspect.is_none() && vendor_extensions.is_empty() && siting_unknown {
        return;
    }

    let aspect = pixel_aspect.map(|ratio| (ratio.num, ratio.den));
    let range = vendor_extensions.first().map(|extension| {
        let value = extension.value();
        String::from_utf8_lossy(value).into_owned()
    });

    tracing::info!(
        pixel_aspect = ?aspect,
        color_range = ?range,
        chroma_siting = ?chroma_siting,
        "forwarding colour tags from the source",
    );
}

/// Maps an FFmpeg colour range onto the y4m `XCOLORRANGE` tag, without its leading `X`.
///
/// `1` is limited and `2` is full.
pub fn color_range_extension(range: i32) -> Option<y4m::VendorExtensionString> {
    let tag: &[u8] = match range {
        1 => b"COLORRANGE=LIMITED",
        2 => b"COLORRANGE=FULL",
        _ => return None,
    };

    let value = tag.to_vec();

    y4m::VendorExtensionString::new(value).ok()
}

/// Opens a y4m stream.
///
/// The colourspace is checked before the stream reaches av-decoders, which panics on one it
/// does not know.
pub fn open_y4m(reader: Box<dyn Read>) -> Result<OpenedSource, anyhow::Error> {
    let y4m_decoder = y4m::decode(reader)?;
    let colorspace = y4m_decoder.get_colorspace();
    subsampling_from_y4m(colorspace)?;

    let pixel_aspect = y4m_decoder.get_pixel_aspect();
    let raw_params = y4m_decoder.get_raw_params();
    let vendor_extensions = y4m_vendor_extensions(raw_params);
    let chroma_siting = siting_from_y4m(colorspace);

    let decoder_impl = DecoderImpl::Y4m(y4m_decoder);
    let decoder = Decoder::from_decoder_impl(decoder_impl)?;
    let details = *decoder.get_video_details();
    let mut layout = layout_from_details(&details)?;
    layout.chroma_siting = chroma_siting;

    let info = SourceInfo {
        details,
        layout,
        pixel_aspect: Some(pixel_aspect),
        vendor_extensions,
        estimated_frames: None,
    };

    Ok(OpenedSource {
        reader: FrameReader::Y4m(decoder),
        info,
    })
}

pub fn layout_from_details(details: &VideoDetails) -> Result<FrameLayout, anyhow::Error> {
    let depth = Depth::from_bits(details.bit_depth)?;
    let subsampling = subsampling_from_av_decoders(details.chroma_sampling)?;

    Ok(FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling,
        depth,
        chroma_siting: ChromaSiting::Unknown,
    })
}
