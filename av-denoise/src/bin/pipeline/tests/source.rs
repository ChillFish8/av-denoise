use std::io::{Cursor, Read};

use av_denoise::{Depth, Subsampling};

use crate::pipeline::source::open_y4m;

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
