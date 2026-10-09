use av_denoise_vs::frames::{pack_plane, shifted_window_range, unpack_plane_into, window_indices};

/// A strided buffer whose padding bytes are all 0xAA, so a bug that reads padding is visible rather
/// than silently plausible.
fn strided(width_bytes: usize, height: usize, stride: usize) -> Vec<u8> {
    let mut buffer = vec![0xAAu8; stride * height];
    for y in 0..height {
        for x in 0..width_bytes {
            buffer[y * stride + x] = (y * width_bytes + x) as u8;
        }
    }

    buffer
}

#[test]
fn packing_drops_the_row_padding() {
    let (width, height, stride) = (5usize, 3usize, 8usize);
    let src = strided(width, height, stride);
    let packed = pack_plane(&src, stride, width, height);

    assert_eq!(packed.len(), width * height);
    assert!(!packed.contains(&0xAA), "padding leaked into the packed buffer");
    for y in 0..height {
        for x in 0..width {
            assert_eq!(packed[y * width + x], (y * width + x) as u8);
        }
    }
}

#[test]
fn packing_round_trips_through_unpacking() {
    let (width, height, stride) = (5usize, 3usize, 8usize);
    let src = strided(width, height, stride);
    let packed = pack_plane(&src, stride, width, height);

    let mut dst = vec![0xAAu8; stride * height];
    unpack_plane_into(&mut dst, stride, width, height, &packed);

    assert_eq!(dst, src, "round trip changed the buffer");
}

#[test]
fn an_unpadded_stride_is_a_straight_copy() {
    let (width, height) = (5usize, 3usize);
    let src = strided(width, height, width);
    let packed = pack_plane(&src, width, width, height);

    assert_eq!(packed, src);
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
    // Three frames with a span of two on each side, so every index saturates somewhere.
    assert_eq!(window_indices(1, 2, 2, 2), vec![0, 0, 1, 2, 2]);
}

#[test]
fn a_zero_span_window_is_the_frame_itself() {
    assert_eq!(window_indices(4, 0, 0, 11), vec![4]);
}

/// nl4d's window is wider ahead of the target than behind it.
#[test]
fn an_asymmetric_span_is_not_forced_symmetric() {
    assert_eq!(window_indices(5, 2, 4, 11), vec![3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn an_asymmetric_span_clamps_at_the_clip_start() {
    assert_eq!(window_indices(1, 2, 4, 11), vec![0, 0, 1, 2, 3, 4, 5]);
}

#[test]
fn an_asymmetric_span_clamps_at_the_clip_end() {
    assert_eq!(window_indices(9, 2, 4, 11), vec![7, 8, 9, 10, 11, 11, 11]);
}

#[test]
fn a_two_byte_depth_packs_by_bytes_not_samples() {
    // 4 samples of 16-bit data is 8 bytes per row, with a stride of 12.
    let (width_bytes, height, stride) = (8usize, 2usize, 12usize);
    let src = strided(width_bytes, height, stride);
    let packed = pack_plane(&src, stride, width_bytes, height);

    assert_eq!(packed.len(), width_bytes * height);
    assert!(!packed.contains(&0xAA));
}

#[test]
fn shifted_windows_stop_at_the_clip_start() {
    assert_eq!(shifted_window_range(0, 4, 4, 11), 0..=4);
    assert_eq!(shifted_window_range(2, 4, 4, 11), 0..=6);
}

#[test]
fn shifted_windows_stop_at_the_clip_end() {
    assert_eq!(shifted_window_range(11, 4, 4, 11), 7..=11);
    assert_eq!(shifted_window_range(9, 4, 4, 11), 5..=11);
}

#[test]
fn shifted_windows_are_full_mid_clip() {
    assert_eq!(shifted_window_range(6, 4, 4, 20), 2..=10);
}

#[test]
fn a_short_clip_shifted_window_is_the_whole_clip() {
    assert_eq!(shifted_window_range(1, 4, 4, 2), 0..=2);
}
