use av_denoise_vs::frames::{pack_plane, unpack_plane_into, window_indices};

/// A strided buffer whose padding bytes are all 0xAA, so a bug that
/// reads padding is visible rather than silently plausible.
fn strided(width_bytes: usize, height: usize, stride: usize) -> Vec<u8> {
    let mut buf = vec![0xAAu8; stride * height];
    for y in 0..height {
        for x in 0..width_bytes {
            buf[y * stride + x] = (y * width_bytes + x) as u8;
        }
    }
    buf
}

#[test]
fn packing_drops_the_row_padding() {
    let (w, h, stride) = (5usize, 3usize, 8usize);
    let packed = pack_plane(&strided(w, h, stride), stride, w, h);

    assert_eq!(packed.len(), w * h);
    assert!(!packed.contains(&0xAA), "padding leaked into the packed buffer");
    for y in 0..h {
        for x in 0..w {
            assert_eq!(packed[y * w + x], (y * w + x) as u8);
        }
    }
}

#[test]
fn packing_round_trips_through_unpacking() {
    let (w, h, stride) = (5usize, 3usize, 8usize);
    let src = strided(w, h, stride);
    let packed = pack_plane(&src, stride, w, h);

    let mut dst = vec![0xAAu8; stride * h];
    unpack_plane_into(&mut dst, stride, w, h, &packed);

    assert_eq!(dst, src, "round trip changed the buffer");
}

#[test]
fn an_unpadded_stride_is_a_straight_copy() {
    let (w, h) = (5usize, 3usize);
    let src = strided(w, h, w);
    assert_eq!(pack_plane(&src, w, w, h), src);
}

#[test]
fn window_indices_clamp_at_the_clip_start() {
    assert_eq!(window_indices(0, 2, 2, 11), vec![0, 0, 0, 1, 2]);
    assert_eq!(window_indices(1, 2, 2, 11), vec![0, 0, 1, 2, 3]);
}

#[test]
fn window_indices_clamp_at_the_clip_end() {
    assert_eq!(window_indices(11, 2, 2, 11), vec![9, 10, 11, 11, 11]);
    assert_eq!(window_indices(10, 2, 2, 11), vec![8, 9, 10, 11, 11]);
}

#[test]
fn window_indices_are_untouched_mid_clip() {
    assert_eq!(window_indices(5, 2, 2, 11), vec![3, 4, 5, 6, 7]);
}

#[test]
fn a_short_clip_clamps_from_both_ends_at_once() {
    // Three frames, span two on each side, so every index saturates
    // somewhere.
    assert_eq!(window_indices(1, 2, 2, 2), vec![0, 0, 1, 2, 2]);
}

#[test]
fn a_zero_span_window_is_the_frame_itself() {
    assert_eq!(window_indices(4, 0, 0, 11), vec![4]);
}

/// nl4d's window is wider ahead of the target than behind it, so
/// `window_indices` must honour `behind` and `ahead` independently
/// rather than assuming a symmetric window, the property that made
/// `window_indices` exist in the first place.
#[test]
fn an_asymmetric_span_is_not_forced_symmetric() {
    assert_eq!(window_indices(5, 2, 4, 11), vec![3, 4, 5, 6, 7, 8, 9]);
}

/// The asymmetric case clamps at the clip start too: `behind` alone
/// saturates while `ahead` still reaches forward normally.
#[test]
fn an_asymmetric_span_clamps_at_the_clip_start() {
    assert_eq!(window_indices(1, 2, 4, 11), vec![0, 0, 1, 2, 3, 4, 5]);
}

/// The asymmetric case clamps at the clip end too: `ahead` saturates
/// while `behind` still reaches back normally.
#[test]
fn an_asymmetric_span_clamps_at_the_clip_end() {
    assert_eq!(window_indices(9, 2, 4, 11), vec![7, 8, 9, 10, 11, 11, 11]);
}

#[test]
fn a_two_byte_depth_packs_by_bytes_not_samples() {
    // 4 samples of 16-bit data is 8 bytes per row, stride 12.
    let (width_bytes, h, stride) = (8usize, 2usize, 12usize);
    let packed = pack_plane(&strided(width_bytes, h, stride), stride, width_bytes, h);
    assert_eq!(packed.len(), width_bytes * h);
    assert!(!packed.contains(&0xAA));
}
