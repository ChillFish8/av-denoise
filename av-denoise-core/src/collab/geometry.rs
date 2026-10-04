use super::{PATCH_SIZE, STEP};

/// Number of reference patches along one axis.
///
/// `dim` must be at least `PATCH_SIZE`.
pub fn refs_along(dim: u32) -> u32 {
    (dim - PATCH_SIZE).div_ceil(STEP) + 1
}

/// Cubes along x for the `collab_fused` kernel.
///
/// Each cube runs eight 8-lane groups with one reference patch each. The count rounds up, so the
/// last cube of a row runs dead groups past the end.
pub fn fused_cubes_x(width: u32) -> u32 {
    refs_along(width).div_ceil(8)
}

/// Top-left pixel of reference index `i` along one axis.
///
/// The last reference clamps so its patch stays inside the frame.
pub fn ref_pos(i: u32, dim: u32) -> u32 {
    (i * STEP).min(dim - PATCH_SIZE)
}

pub fn ref_count(width: u32, height: u32) -> usize {
    refs_along(width) as usize * refs_along(height) as usize
}

/// Columns and rows of a strength map.
///
/// The map holds one entry per 8x8 quarter of the noise estimator's 16x16 blocks, so a ragged
/// frame edge rounds up to a whole block.
pub fn strength_map_dims(width: u32, height: u32) -> (u32, u32) {
    let cols = 2 * width.div_ceil(16);
    let rows = 2 * height.div_ceil(16);

    (cols, rows)
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
    fn fused_cubes_cover_every_reference() {
        // 1920 gives 479 references, so the last of the 60 cubes runs seven live groups.
        assert_eq!(fused_cubes_x(1920), 60);

        for dim in [8u32, 9, 21, 64, 100, 104, 128, 1280, 1920, 3840] {
            let cubes = fused_cubes_x(dim);
            let refs = refs_along(dim);
            assert!(cubes * 8 >= refs, "dim={dim} leaves references uncovered");
            assert!((cubes - 1) * 8 < refs, "dim={dim} launches a wholly dead cube");
        }
    }

    #[test]
    fn last_ref_clamps_inside_the_frame() {
        // Not a multiple of STEP past PATCH_SIZE.
        let width = 21;
        let ref_total = refs_along(width);
        assert_eq!(ref_pos(ref_total - 1, width), width - 8);

        for i in 0..ref_total {
            assert!(ref_pos(i, width) + 8 <= width);
        }
    }

    #[test]
    fn every_pixel_is_covered_by_one_to_three_refs_per_axis() {
        // Regular spacing gives 2 covering references per axis, and the clamped edge gap can add a
        // third.
        for dim in [8u32, 9, 16, 21, 64] {
            for x in 0..dim {
                let ref_total = refs_along(dim);
                let covering = (0..ref_total)
                    .filter(|&i| {
                        let pos = ref_pos(i, dim);
                        pos <= x && x < pos + 8
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
    fn strength_map_dims_round_up_to_whole_blocks() {
        assert_eq!(strength_map_dims(1920, 1080), (240, 136));
        assert_eq!(strength_map_dims(960, 540), (120, 68));
        assert_eq!(strength_map_dims(70, 54), (10, 8));
        assert_eq!(strength_map_dims(16, 16), (2, 2));
    }
}
