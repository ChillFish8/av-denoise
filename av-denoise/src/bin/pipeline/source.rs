use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use av_decoders::{Decoder, DecoderImpl, VideoDetails};
use av_denoise::{Depth, FrameLayout};

use super::convert::subsampling_from_av_decoders;
use crate::cli::InputSource;
use crate::frame_index;
use crate::y4m_format::{subsampling_from_y4m, y4m_vendor_extensions};

/// What the rest of a run needs to know about its input.
#[derive(Clone)]
pub struct SourceInfo {
    pub details: VideoDetails,
    pub layout: FrameLayout,
    /// Carried through to the output header when the input declared one.
    pub pixel_aspect: Option<y4m::Ratio>,
    pub vendor_extensions: Vec<y4m::VendorExtensionString>,
    /// Frames the run should emit, when the input knows its own length.
    pub estimated_frames: Option<usize>,
}

/// An open input, ready to decode.
pub struct OpenedSource {
    pub decoder: Decoder,
    /// Decoder frame numbers that carry no picture of their own.
    pub phantom: BTreeSet<usize>,
    pub info: SourceInfo,
}

pub fn open_source(input: &InputSource) -> Result<OpenedSource, anyhow::Error> {
    match input {
        InputSource::File(path) => open_file(path),
        InputSource::Stdin | InputSource::Fd(_) => {
            let reader = input.open_reader()?;
            open_y4m(reader)
        },
    }
}

pub fn open_file(path: &Path) -> Result<OpenedSource, anyhow::Error> {
    let mut decoder = Decoder::from_file(path)?;
    let details = *decoder.get_video_details();
    let layout = layout_from_details(&details)?;

    // Only inspects the metadata ffms2 already built, so it costs nothing.
    let phantom = frame_index::read_index(&mut decoder)
        .map(|index| frame_index::phantom_indices(&index))
        .unwrap_or_default();

    if !phantom.is_empty() {
        tracing::info!(
            dropped = phantom.len(),
            "the decoder reports frames that carry no picture of their own, dropping them",
        );
    }

    let estimated_frames = details
        .total_frames
        .map(|total| total.saturating_sub(phantom.len()));

    let info = SourceInfo {
        details,
        layout,
        pixel_aspect: None,
        vendor_extensions: Vec::new(),
        estimated_frames,
    };

    Ok(OpenedSource {
        decoder,
        phantom,
        info,
    })
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

    let decoder_impl = DecoderImpl::Y4m(y4m_decoder);
    let decoder = Decoder::from_decoder_impl(decoder_impl)?;
    let details = *decoder.get_video_details();
    let layout = layout_from_details(&details)?;

    let info = SourceInfo {
        details,
        layout,
        pixel_aspect: Some(pixel_aspect),
        vendor_extensions,
        estimated_frames: None,
    };

    Ok(OpenedSource {
        decoder,
        phantom: BTreeSet::new(),
        info,
    })
}

fn layout_from_details(details: &VideoDetails) -> Result<FrameLayout, anyhow::Error> {
    let depth = Depth::from_bits(details.bit_depth)?;
    let subsampling = subsampling_from_av_decoders(details.chroma_sampling)?;

    Ok(FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling,
        depth,
    })
}
