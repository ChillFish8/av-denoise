use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use av_decoders::{Decoder, DecoderImpl, VideoDetails};
use av_denoise::{Depth, FrameLayout};
use ffms2_sys::{FFMS_ColorRanges, FFMS_GetVideoProperties};

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

    let colour = read_file_colour(&mut decoder);
    let vendor_extensions: Vec<y4m::VendorExtensionString> = colour.range.into_iter().collect();

    log_forwarded_colour(colour.pixel_aspect, &vendor_extensions);

    let info = SourceInfo {
        details,
        layout,
        pixel_aspect: colour.pixel_aspect,
        vendor_extensions,
        estimated_frames,
    };

    Ok(OpenedSource {
        decoder,
        phantom,
        info,
    })
}

fn log_forwarded_colour(pixel_aspect: Option<y4m::Ratio>, vendor_extensions: &[y4m::VendorExtensionString]) {
    if pixel_aspect.is_none() && vendor_extensions.is_empty() {
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
        "forwarding colour tags from the source",
    );
}

/// The colour tags ffms2 reports for a video track.
pub struct FileColour {
    pub range: Option<y4m::VendorExtensionString>,
    pub pixel_aspect: Option<y4m::Ratio>,
}

/// Maps an ffms2 colour range onto the y4m `XCOLORRANGE` tag, without its leading `X`.
pub fn color_range_extension(range: i32) -> Option<y4m::VendorExtensionString> {
    let limited = FFMS_ColorRanges::FFMS_CR_MPEG as i32;
    let full = FFMS_ColorRanges::FFMS_CR_JPEG as i32;

    let tag: &[u8] = match range {
        value if value == limited => b"COLORRANGE=LIMITED",
        value if value == full => b"COLORRANGE=FULL",
        _ => return None,
    };

    let value = tag.to_vec();

    y4m::VendorExtensionString::new(value).ok()
}

/// Turns an ffms2 sample aspect ratio into a y4m pixel aspect.
///
/// ffms2 reports 0 for an unset ratio, so anything but a positive pair gives none.
pub fn pixel_aspect_from_sar(numerator: i32, denominator: i32) -> Option<y4m::Ratio> {
    if numerator <= 0 || denominator <= 0 {
        return None;
    }

    let ratio = y4m::Ratio::new(numerator as usize, denominator as usize);

    Some(ratio)
}

/// Reads the colour range and sample aspect ratio of the track behind `decoder`.
///
/// Both come back empty when the decoder is not backed by ffms2.
pub fn read_file_colour(decoder: &mut Decoder) -> FileColour {
    let empty = FileColour {
        range: None,
        pixel_aspect: None,
    };

    let Some(ffms2) = decoder.get_ffms2_impl() else {
        return empty;
    };

    // SAFETY: a live `Ffms2Decoder` holds a non-null video source, and the properties it
    // returns belong to that source, so they stay valid while the decoder does.
    let properties_ptr = unsafe { FFMS_GetVideoProperties(ffms2.video_source) };

    if properties_ptr.is_null() {
        return empty;
    }

    // SAFETY: checked non-null just above.
    let properties = unsafe { &*properties_ptr };
    let range = color_range_extension(properties.ColorRange);
    let pixel_aspect = pixel_aspect_from_sar(properties.SARNum, properties.SARDen);

    FileColour { range, pixel_aspect }
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
