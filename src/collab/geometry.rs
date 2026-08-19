/// Number of reference patches along one axis.
///
/// `dim` must be at least `PATCH_SIZE`. Denoiser construction validates
/// that before this runs.
pub fn refs_along(dim: u32) -> u32 {
    (dim - super::PATCH_SIZE).div_ceil(super::STEP) + 1
}

/// Top-left pixel of reference index `i` along one axis. The last
/// reference clamps so its patch stays inside the frame.
pub fn ref_pos(i: u32, dim: u32) -> u32 {
    (i * super::STEP).min(dim - super::PATCH_SIZE)
}

/// Total reference count for a frame.
pub fn ref_count(width: u32, height: u32) -> usize {
    refs_along(width) as usize * refs_along(height) as usize
}

/// Length of the member-position buffer, one packed `u32` per slot.
pub fn member_buf_len(width: u32, height: u32, k_max: u32) -> usize {
    ref_count(width, height) * k_max as usize
}

/// Length of the member-frame buffer, one `u32` ring slot per slot.
///
/// Identical to [`member_buf_len`]. A distinct name documents that this
/// buffer holds the physical frame slot each member was matched in,
/// rather than a packed position.
pub fn member_frame_buf_len(width: u32, height: u32, k_max: u32) -> usize {
    member_buf_len(width, height, k_max)
}

/// Length of the member-sigma buffer, one extra-variance `f32` per slot.
///
/// Identical to [`member_buf_len`]. A distinct name documents that this
/// buffer holds [`crate::collab::kernels::group_temporal::collab_group_temporal`]'s
/// per-member mismatch variance rather than a packed position.
pub fn member_sig2_buf_len(width: u32, height: u32, k_max: u32) -> usize {
    member_buf_len(width, height, k_max)
}

/// Length of the filtered-patch debug buffer in `Vector<f32, N>` lines,
/// one whole group of `k_max` patches per reference.
///
/// The filters only fill this when their `emit_filtered` flag is set,
/// which tests do and the pipeline does not, so nothing sizes a real
/// allocation off this outside of tests.
pub fn filtered_buf_len(width: u32, height: u32, k_max: u32) -> usize {
    ref_count(width, height) * k_max as usize * super::PATCH_AREA as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_cover_1080p_exactly() {
        // (1920 - 8) / 4 + 1 = 479, (1080 - 8) / 4 + 1 = 269.
        assert_eq!(refs_along(1920), 479);
        assert_eq!(refs_along(1080), 269);
    }

    #[test]
    fn last_ref_clamps_inside_the_frame() {
        let w = 21; // not a multiple of STEP past PATCH_SIZE
        let n = refs_along(w);
        assert_eq!(ref_pos(n - 1, w), w - 8);
        for i in 0..n {
            assert!(ref_pos(i, w) + 8 <= w);
        }
    }

    #[test]
    fn every_pixel_is_covered_by_one_to_three_refs_per_axis() {
        // Regular spacing gives 2 covering references per axis, and the
        // single clamped edge gap can add a third. It never reaches a
        // fourth, so the bound here is 1..=3, giving at most 9 (3 x 3)
        // covering references per pixel in 2D.
        for dim in [8u32, 9, 16, 21, 64] {
            for x in 0..dim {
                let n = refs_along(dim);
                let covering = (0..n)
                    .filter(|&i| {
                        let p = ref_pos(i, dim);
                        p <= x && x < p + 8
                    })
                    .count();
                assert!((1..=3).contains(&covering), "dim={dim} x={x} covering={covering}");
            }
        }
    }

    #[test]
    fn ref_count_is_the_product_of_the_per_axis_counts() {
        assert_eq!(
            ref_count(1920, 1080),
            refs_along(1920) as usize * refs_along(1080) as usize
        );
    }

    #[test]
    fn member_buf_len_scales_with_k_max() {
        // refs_along(16) = (16 - 8) / 4 + 1 = 3, so ref_count(16, 16) is
        // 3 * 3 = 9, and member_buf_len(16, 16, 8) is 9 * 8 = 72.
        assert_eq!(member_buf_len(16, 16, 8), 72);
    }

    #[test]
    fn member_frame_buf_len_matches_member_buf_len() {
        assert_eq!(member_frame_buf_len(16, 16, 8), member_buf_len(16, 16, 8));
    }

    #[test]
    fn filtered_buf_len_scales_with_patch_area_and_k_max() {
        // ref_count(16, 16) is 9, as above, so filtered_buf_len(16, 16,
        // 8) is 9 * 8 * PATCH_AREA (64) = 4608.
        assert_eq!(filtered_buf_len(16, 16, 8), 4608);
        assert_eq!(filtered_buf_len(16, 16, 1), 576);
    }
}
