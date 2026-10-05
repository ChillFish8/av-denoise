use super::*;

#[test]
fn yuv420_even_dims_halve() {
    assert_eq!(Subsampling::Yuv420.chroma_dims(1920, 1080), (960, 540));
}

#[test]
fn yuv420_odd_width_rounds_up() {
    assert_eq!(Subsampling::Yuv420.chroma_dims(1919, 1080), (960, 540));
}

#[test]
fn yuv420_odd_height_rounds_up() {
    assert_eq!(Subsampling::Yuv420.chroma_dims(1920, 1079), (960, 540));
}

#[test]
fn yuv420_odd_both_dims_round_up() {
    assert_eq!(Subsampling::Yuv420.chroma_dims(1919, 1079), (960, 540));
}

#[test]
fn yuv422_even_width_halves() {
    assert_eq!(Subsampling::Yuv422.chroma_dims(1920, 1080), (960, 1080));
}

#[test]
fn yuv422_odd_width_rounds_up() {
    assert_eq!(Subsampling::Yuv422.chroma_dims(1919, 1080), (960, 1080));
}

#[test]
fn yuv444_passes_even_dims_through() {
    assert_eq!(Subsampling::Yuv444.chroma_dims(1920, 1080), (1920, 1080));
}

#[test]
fn yuv444_passes_odd_dims_through() {
    assert_eq!(Subsampling::Yuv444.chroma_dims(1919, 1079), (1919, 1079));
}
