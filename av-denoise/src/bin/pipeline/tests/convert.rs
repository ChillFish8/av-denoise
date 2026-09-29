use av_denoise::{Depth, FrameLayout, Subsampling};

use crate::pipeline::convert::{collect_plane_u16, planes_from_v_frame_u8, planes_from_v_frame_u16};

/// A 10-bit v_frame plane serialises to little-endian wire bytes at
/// twice the sample count.
#[test]
fn collect_plane_u16_writes_little_endian_bytes() {
    use std::num::{NonZeroU8, NonZeroUsize};

    use v_frame::chroma::ChromaSubsampling;
    use v_frame::frame::{Frame, FrameBuilder};

    let mut frame: Frame<u16> = FrameBuilder::new(
        NonZeroUsize::new(2).expect("width is non-zero"),
        NonZeroUsize::new(2).expect("height is non-zero"),
        ChromaSubsampling::Yuv420,
        NonZeroU8::new(10).expect("depth is non-zero"),
    )
    .build()
    .expect("a 2x2 10-bit frame builds");

    frame
        .y_plane
        .copy_from_slice(&[0u16, 1, 512, 1023])
        .expect("four samples fill a 2x2 plane");

    let bytes = collect_plane_u16(&frame.y_plane);

    assert_eq!(bytes.len(), 8, "4 samples at 2 bytes each");
    assert_eq!(
        bytes,
        vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0xFF, 0x03],
        "samples must be little-endian"
    );
}

#[test]
fn planes_from_v_frame_u8_matching_layout_succeeds() {
    use std::num::{NonZeroU8, NonZeroUsize};

    use v_frame::chroma::ChromaSubsampling;
    use v_frame::frame::FrameBuilder;

    let layout = FrameLayout {
        width: 2,
        height: 2,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let frame: v_frame::frame::Frame<u8> = FrameBuilder::new(
        NonZeroUsize::new(2).expect("width is non-zero"),
        NonZeroUsize::new(2).expect("height is non-zero"),
        ChromaSubsampling::Yuv420,
        NonZeroU8::new(8).expect("depth is non-zero"),
    )
    .build()
    .expect("a 2x2 8-bit frame builds");

    let planes = planes_from_v_frame_u8(&frame, layout).expect("matching layout should not error");

    assert_eq!(planes.y.len(), layout.luma_bytes());
    assert_eq!(planes.u.len(), layout.chroma_bytes());
    assert_eq!(planes.v.len(), layout.chroma_bytes());
}

#[test]
fn planes_from_v_frame_u16_matching_layout_succeeds() {
    use std::num::{NonZeroU8, NonZeroUsize};

    use v_frame::chroma::ChromaSubsampling;
    use v_frame::frame::FrameBuilder;

    let layout = FrameLayout {
        width: 2,
        height: 2,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Ten,
    };
    let frame: v_frame::frame::Frame<u16> = FrameBuilder::new(
        NonZeroUsize::new(2).expect("width is non-zero"),
        NonZeroUsize::new(2).expect("height is non-zero"),
        ChromaSubsampling::Yuv420,
        NonZeroU8::new(10).expect("depth is non-zero"),
    )
    .build()
    .expect("a 2x2 10-bit frame builds");

    let planes = planes_from_v_frame_u16(&frame, layout).expect("matching layout should not error");

    assert_eq!(planes.y.len(), layout.luma_bytes());
    assert_eq!(planes.u.len(), layout.chroma_bytes());
    assert_eq!(planes.v.len(), layout.chroma_bytes());
}

#[test]
fn planes_from_v_frame_u8_mismatched_layout_errors() {
    use std::num::{NonZeroU8, NonZeroUsize};

    use v_frame::chroma::ChromaSubsampling;
    use v_frame::frame::FrameBuilder;

    let frame: v_frame::frame::Frame<u8> = FrameBuilder::new(
        NonZeroUsize::new(2).expect("width is non-zero"),
        NonZeroUsize::new(2).expect("height is non-zero"),
        ChromaSubsampling::Yuv420,
        NonZeroU8::new(8).expect("depth is non-zero"),
    )
    .build()
    .expect("a 2x2 8-bit frame builds");

    let layout = FrameLayout {
        width: 4,
        height: 4,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };

    let err = planes_from_v_frame_u8(&frame, layout).expect_err("a smaller frame should not pass");
    let msg = err.to_string();

    assert!(msg.contains('y'), "error should name the plane: {msg}");
    assert!(
        msg.contains('4'),
        "error should name the 2x2 plane's length (4): {msg}"
    );
    assert!(
        msg.contains("16"),
        "error should name the layout's expected length (16): {msg}"
    );
}
