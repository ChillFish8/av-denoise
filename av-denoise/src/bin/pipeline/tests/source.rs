use std::io::{Cursor, Read};

use av_denoise::{Depth, Subsampling};

use crate::pipeline::convert::SourcePixel;
use crate::pipeline::source::{color_range_extension, open_y4m, pixel_aspect_from_sar};

fn y4m_bytes(colorspace: y4m::Colorspace, extension: Option<&str>, frames: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut builder = y4m::encode(4, 4, y4m::Ratio::new(25, 1))
        .with_colorspace(colorspace)
        .with_pixel_aspect(y4m::Ratio::new(4, 3));

    if let Some(extension) = extension {
        let vendor = y4m::VendorExtensionString::new(extension.as_bytes().to_vec())
            .expect("the extension has no spaces");
        builder = builder.append_vendor_extension(vendor);
    }

    let mut encoder = builder.write_header(&mut bytes).expect("header should write");
    let sample_bytes = colorspace.get_bytes_per_sample();
    let luma = vec![0u8; 16 * sample_bytes];
    let chroma_samples = match colorspace {
        y4m::Colorspace::C422p10 => 8,
        _ => 4,
    };
    let chroma = vec![0u8; chroma_samples * sample_bytes];

    for _ in 0..frames {
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

fn reader(bytes: Vec<u8>) -> Box<dyn Read> {
    Box::new(Cursor::new(bytes))
}

#[test]
fn a_pipe_keeps_its_vendor_extensions_and_pixel_aspect() {
    let bytes = y4m_bytes(y4m::Colorspace::C420, Some("COLORRANGE=LIMITED"), 1);
    let opened = open_y4m(reader(bytes)).expect("a 4:2:0 pipe opens");

    let extensions: Vec<&[u8]> = opened
        .info
        .vendor_extensions
        .iter()
        .map(|extension| extension.value())
        .collect();
    let aspect = opened.info.pixel_aspect.expect("a pipe carries its pixel aspect");

    assert_eq!(extensions, vec![b"COLORRANGE=LIMITED".as_slice()]);
    assert_eq!((aspect.num, aspect.den), (4, 3));
}

#[test]
fn a_ten_bit_pipe_reports_its_layout() {
    let bytes = y4m_bytes(y4m::Colorspace::C422p10, None, 1);
    let opened = open_y4m(reader(bytes)).expect("a 10-bit 4:2:2 pipe opens");

    assert_eq!(opened.info.layout.width, 4);
    assert_eq!(opened.info.layout.subsampling, Subsampling::Yuv422);
    assert_eq!(opened.info.layout.depth, Depth::Ten);
}

#[test]
fn ten_bit_stream_round_trips_header_and_plane_sizes() {
    let bytes = y4m_bytes(y4m::Colorspace::C420p10, None, 2);
    let mut opened = open_y4m(reader(bytes)).expect("a 10-bit 4:2:0 pipe opens");
    let layout = opened.info.layout;

    assert_eq!(layout.subsampling, Subsampling::Yuv420);
    assert_eq!(layout.depth, Depth::Ten);

    let frame = opened
        .decoder
        .read_video_frame::<u16>()
        .expect("the pipe holds a frame");
    let planes = u16::to_planes(&frame, layout).expect("the frame matches its layout");

    assert_eq!(planes.y.len(), 4 * 4 * 2);
    assert_eq!(planes.u.len(), 2 * 2 * 2);
    assert_eq!(planes.v.len(), 2 * 2 * 2);
}

#[test]
fn a_pipe_has_no_phantom_frames_or_frame_estimate() {
    let bytes = y4m_bytes(y4m::Colorspace::C420, None, 3);
    let opened = open_y4m(reader(bytes)).expect("a 4:2:0 pipe opens");

    assert!(opened.phantom.is_empty());
    assert_eq!(opened.info.estimated_frames, None);
}

#[test]
fn a_mono_pipe_is_rejected_without_panicking() {
    let mut bytes = Vec::new();
    y4m::encode(4, 4, y4m::Ratio::new(25, 1))
        .with_colorspace(y4m::Colorspace::Cmono)
        .write_header(&mut bytes)
        .expect("header should write");

    let err = match open_y4m(reader(bytes)) {
        Ok(_) => panic!("a mono pipe must be rejected"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("Cmono"), "got {err}");
}

#[test]
fn a_limited_range_maps_to_the_limited_tag() {
    let extension = color_range_extension(1).expect("MPEG range is tagged");

    assert_eq!(extension.value(), b"COLORRANGE=LIMITED");
}

#[test]
fn a_full_range_maps_to_the_full_tag() {
    let extension = color_range_extension(2).expect("JPEG range is tagged");

    assert_eq!(extension.value(), b"COLORRANGE=FULL");
}

#[test]
fn an_unspecified_or_unknown_range_adds_no_tag() {
    for range in [0, 3, -1] {
        assert!(
            color_range_extension(range).is_none(),
            "range {range} must add no tag"
        );
    }
}

#[test]
fn a_positive_sar_becomes_the_pixel_aspect() {
    let aspect = pixel_aspect_from_sar(32, 27).expect("32:27 is a valid SAR");

    assert_eq!((aspect.num, aspect.den), (32, 27));
}

#[test]
fn an_unset_or_invalid_sar_gives_no_pixel_aspect() {
    for (numerator, denominator) in [(0, 0), (0, 1), (1, 0), (-4, 3)] {
        let aspect = pixel_aspect_from_sar(numerator, denominator);

        assert!(aspect.is_none(), "SAR {numerator}:{denominator} must give none");
    }
}
