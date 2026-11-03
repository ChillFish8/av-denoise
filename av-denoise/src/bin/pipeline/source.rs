use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use av_decoders::{Decoder, DecoderImpl, Rational32, VideoDetails};
use av_denoise::{ChromaSiting, Depth, FrameLayout, Subsampling};
use ffms2_sys::{FFMS_ErrorInfo, FFMS_GetFrame, FFMS_GetVideoProperties, FFMS_VideoSource};

use super::convert::subsampling_from_av_decoders;
use super::frame_rate::{check_constant_rate, variable_rate_error};
use super::probe::nominal_frame_rate;
use crate::cli::InputSource;
use crate::frame_index::{self, IndexEntry};
use crate::y4m_format::{
    siting_from_chroma_location,
    siting_from_y4m,
    subsampling_from_y4m,
    y4m_vendor_extensions,
};

/// What the rest of a run needs to know about its input.
#[derive(Clone)]
pub struct SourceInfo {
    /// The decoder's description, with the frame rate a file declares in place of ffms2's estimate.
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
    let mut details = *decoder.get_video_details();
    details.frame_rate = file_frame_rate(path, details.frame_rate);

    let mut layout = layout_from_details(&details)?;

    // Only inspects the metadata ffms2 already built, so it costs nothing.
    let index = frame_index::read_index(&mut decoder);
    let phantom = index
        .as_deref()
        .map(frame_index::phantom_indices)
        .unwrap_or_default();

    if !phantom.is_empty() {
        tracing::info!(
            dropped = phantom.len(),
            "the decoder reports frames that carry no picture of their own, dropping them",
        );
    }

    check_file_timing(&mut decoder, index.as_deref(), &phantom, details.frame_rate)?;

    let estimated_frames = details
        .total_frames
        .map(|total| total.saturating_sub(phantom.len()));

    let colour = read_file_colour(&mut decoder);
    let vendor_extensions: Vec<y4m::VendorExtensionString> = colour.range.into_iter().collect();
    layout.chroma_siting = colour.chroma_siting;

    // Only 8-bit 4:2:0 output carries a siting, so nothing else is logged as forwarded.
    let carries_siting = layout.subsampling == Subsampling::Yuv420 && layout.depth == Depth::Eight;
    let logged_siting = if carries_siting {
        colour.chroma_siting
    } else {
        ChromaSiting::Unknown
    };

    log_forwarded_colour(colour.pixel_aspect, &vendor_extensions, logged_siting);

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

/// The frame rate the file at `path` declares, or the rate ffms2 `measured` when none can be read.
///
/// ffms2 averages its rate over every packet, so it drifts on files with phantom packets,
/// timestamp gaps or few frames.
pub fn file_frame_rate(path: &Path, measured: Rational32) -> Rational32 {
    let Some(nominal) = nominal_frame_rate(path) else {
        tracing::debug!(
            numerator = *measured.numer(),
            denominator = *measured.denom(),
            "no nominal frame rate could be read, keeping the rate ffms2 measured",
        );
        return measured;
    };

    nominal
}

/// Rejects a file whose kept frames do not run at the constant `declared` rate.
///
/// The check is skipped when ffms2 cannot describe the file's timestamps.
fn check_file_timing(
    decoder: &mut Decoder,
    index: Option<&[IndexEntry]>,
    phantom: &BTreeSet<usize>,
    declared: Rational32,
) -> Result<(), anyhow::Error> {
    let tick_seconds = frame_index::read_tick_seconds(decoder);
    let Some((index, tick_seconds)) = index.zip(tick_seconds) else {
        tracing::debug!("the frame timestamps could not be read, so the frame rate is not checked");
        return Ok(());
    };

    let mut kept: Vec<i64> = index
        .iter()
        .enumerate()
        .filter(|(position, _)| !phantom.contains(position))
        .map(|(_, entry)| entry.pts)
        .collect();
    kept.sort_unstable();

    check_constant_rate(&kept, tick_seconds, declared).map_err(variable_rate_error)
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

/// The colour tags ffms2 reports for a video track.
pub struct FileColour {
    pub range: Option<y4m::VendorExtensionString>,
    pub pixel_aspect: Option<y4m::Ratio>,
    pub chroma_siting: ChromaSiting,
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

/// Reads the colour range, sample aspect ratio and chroma siting of the track behind `decoder`.
///
/// All three come back empty when the decoder is not backed by ffms2.
pub fn read_file_colour(decoder: &mut Decoder) -> FileColour {
    let empty = FileColour {
        range: None,
        pixel_aspect: None,
        chroma_siting: ChromaSiting::Unknown,
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

    let chroma_siting = first_frame_siting(ffms2.video_source);

    FileColour {
        range,
        pixel_aspect,
        chroma_siting,
    }
}

/// Reads the chroma siting off the first decoded frame, since ffms2 reports it per frame.
fn first_frame_siting(video_source: *mut FFMS_VideoSource) -> ChromaSiting {
    // SAFETY: an all-zero error info is the empty one ffms2 expects, with no message buffer.
    let mut error = unsafe { std::mem::zeroed::<FFMS_ErrorInfo>() };

    // SAFETY: `video_source` comes from a live `Ffms2Decoder`, so it is non-null and valid.
    let frame_ptr = unsafe { FFMS_GetFrame(video_source, 0, std::ptr::addr_of_mut!(error)) };

    if frame_ptr.is_null() {
        return ChromaSiting::Unknown;
    }

    // SAFETY: checked non-null just above, and ffms2 keeps the frame alive until the next call.
    let location = unsafe { (*frame_ptr).ChromaLocation };

    siting_from_chroma_location(location)
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
        decoder,
        phantom: BTreeSet::new(),
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
