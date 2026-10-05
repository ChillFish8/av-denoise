use std::num::{NonZeroU8, NonZeroUsize};
use std::sync::Arc;

use av_denoise::{Depth, FrameLayout, Subsampling};
use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::{Frame, FrameBuilder};

use crate::pipeline::convert::{
    SourcePixel,
    collect_plane_u16,
    planes_from_v_frame_u8,
    planes_from_v_frame_u16,
};

#[test]
fn collect_plane_u16_writes_little_endian_bytes() {
    let width = NonZeroUsize::new(2).expect("width is non-zero");
    let height = NonZeroUsize::new(2).expect("height is non-zero");
    let bit_depth = NonZeroU8::new(10).expect("depth is non-zero");
    let mut frame: Frame<u16> = FrameBuilder::new(width, height, ChromaSubsampling::Yuv420, bit_depth)
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
    let layout = FrameLayout {
        width: 2,
        height: 2,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let width = NonZeroUsize::new(2).expect("width is non-zero");
    let height = NonZeroUsize::new(2).expect("height is non-zero");
    let bit_depth = NonZeroU8::new(8).expect("depth is non-zero");
    let frame: Frame<u8> = FrameBuilder::new(width, height, ChromaSubsampling::Yuv420, bit_depth)
        .build()
        .expect("a 2x2 8-bit frame builds");

    let planes = planes_from_v_frame_u8(&frame, layout).expect("matching layout should not error");

    assert_eq!(planes.y.len(), layout.luma_bytes());
    assert_eq!(planes.u.len(), layout.chroma_bytes());
    assert_eq!(planes.v.len(), layout.chroma_bytes());
}

#[test]
fn planes_from_v_frame_u16_matching_layout_succeeds() {
    let layout = FrameLayout {
        width: 2,
        height: 2,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Ten,
    };
    let width = NonZeroUsize::new(2).expect("width is non-zero");
    let height = NonZeroUsize::new(2).expect("height is non-zero");
    let bit_depth = NonZeroU8::new(10).expect("depth is non-zero");
    let frame: Frame<u16> = FrameBuilder::new(width, height, ChromaSubsampling::Yuv420, bit_depth)
        .build()
        .expect("a 2x2 10-bit frame builds");

    let planes = planes_from_v_frame_u16(&frame, layout).expect("matching layout should not error");

    assert_eq!(planes.y.len(), layout.luma_bytes());
    assert_eq!(planes.u.len(), layout.chroma_bytes());
    assert_eq!(planes.v.len(), layout.chroma_bytes());
}

#[test]
fn planes_from_v_frame_u8_mismatched_layout_errors() {
    let width = NonZeroUsize::new(2).expect("width is non-zero");
    let height = NonZeroUsize::new(2).expect("height is non-zero");
    let bit_depth = NonZeroU8::new(8).expect("depth is non-zero");
    let frame: Frame<u8> = FrameBuilder::new(width, height, ChromaSubsampling::Yuv420, bit_depth)
        .build()
        .expect("a 2x2 8-bit frame builds");

    let layout = FrameLayout {
        width: 4,
        height: 4,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };

    let err = planes_from_v_frame_u8(&frame, layout).expect_err("a smaller frame should not pass");
    let message = err.to_string();

    assert!(message.contains('y'), "error should name the plane: {message}");
    assert!(
        message.contains('4'),
        "error should name the 2x2 plane's length (4): {message}"
    );
    assert!(
        message.contains("16"),
        "error should name the layout's expected length (16): {message}"
    );
}

#[test]
fn a_decoded_frame_only_unwraps_to_its_own_depth() {
    let width = NonZeroUsize::new(2).expect("width is non-zero");
    let height = NonZeroUsize::new(2).expect("height is non-zero");
    let bit_depth = NonZeroU8::new(8).expect("depth is non-zero");
    let frame: Frame<u8> = FrameBuilder::new(width, height, ChromaSubsampling::Yuv420, bit_depth)
        .build()
        .expect("a 2x2 8-bit frame builds");
    let shared = Arc::new(frame);
    let decoded = u8::into_decoded(shared);
    let unwrapped = u16::from_decoded(decoded);

    assert!(unwrapped.is_none());
}
