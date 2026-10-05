use super::*;

fn layout(depth: Depth) -> FrameLayout {
    FrameLayout {
        width: 4,
        height: 4,
        subsampling: Subsampling::Yuv420,
        depth,
    }
}

#[test]
fn byte_lengths_scale_with_depth() {
    let eight_bit = layout(Depth::Eight);
    let ten_bit = layout(Depth::Ten);

    assert_eq!(eight_bit.luma_bytes(), 16);
    assert_eq!(ten_bit.luma_bytes(), 32);
    assert_eq!(eight_bit.chroma_bytes(), 4);
    assert_eq!(ten_bit.chroma_bytes(), 8);
}

#[test]
fn neutral_chroma_fill_is_correct_at_each_depth() {
    let eight = layout(Depth::Eight).neutral_chroma_plane();
    assert_eq!(eight, vec![128u8; 4]);

    // 512 little-endian is [0x00, 0x02], repeated per sample.
    let ten = layout(Depth::Ten).neutral_chroma_plane();
    assert_eq!(ten, vec![0x00, 0x02, 0x00, 0x02, 0x00, 0x02, 0x00, 0x02]);

    // 2048 little-endian is [0x00, 0x08].
    let twelve = layout(Depth::Twelve).neutral_chroma_plane();
    assert_eq!(twelve.len(), 8);
    assert_eq!(&twelve[0..2], &[0x00, 0x08]);
}

#[test]
fn black_luma_fill_is_zero_at_the_right_length() {
    let eight = layout(Depth::Eight).black_luma_plane();
    let ten = layout(Depth::Ten).black_luma_plane();

    assert_eq!(eight, vec![0u8; 16]);
    assert_eq!(ten, vec![0u8; 32]);
}
